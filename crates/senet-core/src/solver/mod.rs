//! Solver: computes V(me, opp) = P(player to throw wins | optimal play) for every
//! position of the 5-piece game by parallel Gauss-Seidel value iteration, to a
//! numerical tolerance (docs/ACCURACY.md).
//!
//! Bellman equation (RULES.md turn structure):
//!   V(s)   = sum_t p_t * Q_t(s)
//!   Q_t(s) = max over legal moves of R(after), or 1 - V(flip s) if there is no move
//!   R(s')  = 1                    if the mover has borne off every piece
//!          = V(s')                if t grants another throw (same player throws again)
//!          = 1 - V(flip s')       otherwise
//!
//! Bearing off is irreversible, so layers are solved in groups {(w,b),(b,w)} of
//! increasing w+b; a group only depends on itself and on groups with one fewer piece.
//! Within a group, states are swept in decreasing order of total pips (sum of all
//! occupied square numbers): ordinary forward moves strictly increase it, so most
//! successors already hold this sweep's values and few sweeps are needed.

pub mod meta;
mod storage;

pub use storage::Layers;
use storage::{LayerData, SolveLock, check_probabilities, f32_ckpt_path, u24_ckpt_path, u24_path};

use crate::atomic::tmp_path;
use crate::board::{Bits, MAX_PIECES, Pos, Rules, THROW_PROBS};
use crate::db::{layer_path, map_layer};
use crate::index::{
    ALL29, BINOM, N_USABLE, all_layers, colex_rank, colex_rank_among, compact, expand, index_compact, index_of,
    layer_size, position_of,
};
use crate::invalid_input;
use crate::movegen::{MAX_MOVES, MoveList, gen_moves};
use crate::rng::Rng;
use rayon::prelude::*;
use std::io;
use std::path::{Path, PathBuf};
use std::time::Instant;

/// A reference to the value of a successor position: a win for the mover (`w == 0`), or
/// the stored value of position `idx` of layer `(w, b)`, complemented if `flip`.
#[derive(Clone, Copy)]
struct Succ {
    w: u8,
    b: u8,
    flip: bool,
    idx: u64,
}

const WIN: Succ = Succ { w: 0, b: 0, flip: false, idx: 0 };

#[inline(always)]
fn succ_of(after: Pos, extra: bool) -> Succ {
    if after.me == 0 {
        return WIN;
    }
    let (me, opp) = if extra { (after.me, after.opp) } else { (after.opp, after.me) };
    let mc = compact(me);
    let oc = compact(opp);
    let w = mc.count_ones() as usize;
    let b = oc.count_ones() as usize;
    Succ { w: w as u8, b: b as u8, flip: !extra, idx: index_compact(mc, oc, w, b) }
}

#[inline(always)]
fn succ_value(layers: &Layers, s: Succ) -> f64 {
    if s.w == 0 {
        return 1.0;
    }
    let v = layers.layer(s.w as usize, s.b as usize).get(s.idx);
    if s.flip { 1.0 - v } else { v }
}

/// Successors of one position for every throw, excluding forfeits.
struct Gathered {
    succ: [Succ; 5 * MAX_MOVES],
    /// Number of successors for each throw 1..=5, stored consecutively in `succ`.
    counts: [u8; 6],
    n: usize,
    /// Probability of throwing something with no legal move (turn passes).
    forfeit: f64,
}

impl Gathered {
    const EMPTY: Gathered = Gathered { succ: [WIN; 5 * MAX_MOVES], counts: [0; 6], n: 0, forfeit: 0.0 };
}

/// Gathers the successors of `pos` into `g`, replacing what it held. (A sweep reuses its
/// `Gathered`s: making one for every position costs a few percent of the sweep.)
#[inline(always)]
fn gather(rules: &Rules, pos: Pos, g: &mut Gathered) {
    (g.counts, g.n, g.forfeit) = ([0; 6], 0, 0.0);
    let mut ml = MoveList::new();
    for t in 1..=5u8 {
        gen_moves(rules, pos, t, &mut ml);
        if ml.is_empty() {
            g.forfeit += THROW_PROBS[t as usize];
            continue;
        }
        let extra = rules.extra_throw(t);
        for m in ml.iter() {
            g.succ[g.n] = succ_of(m.after, extra);
            g.n += 1;
        }
        g.counts[t as usize] = ml.len() as u8;
    }
}

/// Asks for all the values `g` refers to at once, so that the reads that follow wait for
/// them together: about a sixth off a sweep. Asking one position ahead, or for both cache
/// lines of a 24-bit value that spans two, measured no faster.
#[inline(always)]
fn prefetch_all(layers: &Layers, g: &Gathered) {
    for s in &g.succ[..g.n] {
        if s.w != 0 {
            layers.layer(s.w as usize, s.b as usize).prefetch(s.idx);
        }
    }
}

/// Expected value over the throws that have a legal move (best move each time).
#[inline(always)]
fn moving_part(layers: &Layers, g: &Gathered) -> f64 {
    let mut v = 0.0f64;
    let mut succ = g.succ[..g.n].iter();
    for (t, &count) in g.counts.iter().enumerate().skip(1) {
        let best = succ.by_ref().take(count as usize).map(|&s| succ_value(layers, s)).fold(0.0, f64::max);
        v += THROW_PROBS[t] * best;
    }
    v
}

/// The Bellman update T(V)(pos) for one position, using stored values, or NaN if a value
/// it reads is not a probability (`moving_part` takes maxima that skip NaN). The audit
/// (`residual_check`) uses it on values read from files.
#[inline(always)]
pub(crate) fn bellman(rules: &Rules, layers: &Layers, pos: Pos) -> f64 {
    let mut g = Gathered::EMPTY;
    gather(rules, pos, &mut g);
    prefetch_all(layers, &g);
    let is_probability = |v: f64| (0.0..=1.0).contains(&v);
    if !g.succ[..g.n].iter().all(|&s| is_probability(succ_value(layers, s))) {
        return f64::NAN;
    }
    let a = moving_part(layers, &g);
    if g.forfeit > 0.0 {
        let (w, b, i) = index_of(pos.flip());
        let mirror = layers.layer(w, b).get(i);
        if !is_probability(mirror) {
            return f64::NAN;
        }
        a + g.forfeit * (1.0 - mirror)
    } else {
        a
    }
}

/// Joint update of a position s and its mirror s' = flip(s). With A, A' the expected
/// values over throws that have a legal move and F, F' the forfeit probabilities:
///   V = A + F (1 - V'),  V' = A' + F' (1 - V)
/// which is solved exactly, so forfeit ping-pong costs no extra sweeps. The system always
/// has a solution: the side with the highest piece on the board can move it with a 1
/// (with a 2 or 3 from 29 or 28), so one of F, F' is at most 3/4. `[g1, g2]` is scratch
/// space.
#[inline(always)]
fn pair_update(rules: &Rules, layers: &Layers, s: Pos, [g1, g2]: &mut [Gathered; 2]) -> (f64, f64) {
    gather(rules, s, g1);
    gather(rules, s.flip(), g2);
    prefetch_all(layers, g1);
    prefetch_all(layers, g2);
    let (a1, f1) = (moving_part(layers, g1), g1.forfeit);
    let (a2, f2) = (moving_part(layers, g2), g2.forfeit);
    let den = 1.0 - f1 * f2;
    debug_assert!(den >= 0.25, "{s:?}");
    ((a1 + f1 * (1.0 - a2 - f2)) / den, (a2 + f2 * (1.0 - a1 - f1)) / den)
}

/// Largest pip sum of a side: the top `MAX_PIECES` usable squares.
const MAX_PIP: usize = 30 + 29 + 28 + 26 + 25;
/// Largest total pip sum of a position.
const MAX_TOTAL_PIP: usize = MAX_PIP + 24 + 23 + 22 + 21 + 20;

/// All k-subsets of the 29 compact squares for each k, sorted by pip sum (the sum of
/// their square numbers).
struct Combos {
    /// Compact masks, sorted by (pip sum, mask).
    masks: Vec<Vec<u32>>,
    /// Colex rank of each mask, parallel to `masks`.
    ranks: Vec<Vec<u64>>,
    /// Pip sum of each mask, parallel to `masks`.
    pips: Vec<Vec<u16>>,
    /// Start offset of each pip sum in `masks` (length `MAX_TOTAL_PIP + 2`).
    bucket: Vec<Vec<u32>>,
}

impl Combos {
    fn build() -> Combos {
        // Index 0 (the empty subset) is unused.
        let mut c = Combos { masks: vec![vec![]], ranks: vec![vec![]], pips: vec![vec![]], bucket: vec![vec![]] };
        for (k, &count) in BINOM[N_USABLE as usize].iter().enumerate().take(MAX_PIECES + 1).skip(1) {
            let mut v: Vec<(u16, u32)> = k_subsets(k).map(|m| (Bits(expand(m)).sum::<u32>() as u16, m)).collect();
            assert_eq!(v.len() as u64, count);
            v.sort_unstable();
            let mut bucket = vec![0u32; MAX_TOTAL_PIP + 2];
            for &(p, _) in &v {
                bucket[p as usize + 1] += 1;
            }
            for i in 1..bucket.len() {
                bucket[i] += bucket[i - 1];
            }
            c.masks.push(v.iter().map(|&(_, m)| m).collect());
            c.ranks.push(v.iter().map(|&(_, m)| colex_rank(m)).collect());
            c.pips.push(v.iter().map(|&(p, _)| p).collect());
            c.bucket.push(bucket);
        }
        c
    }

    /// Indices into `masks[k]` of the masks with pip sum `pip`.
    fn bucket_range(&self, k: usize, pip: usize) -> std::ops::Range<usize> {
        if pip > MAX_TOTAL_PIP {
            return 0..0;
        }
        self.bucket[k][pip] as usize..self.bucket[k][pip + 1] as usize
    }
}

/// All k-subsets of the compact squares as masks, in increasing order (Gosper's hack).
fn k_subsets(k: usize) -> impl Iterator<Item = u32> {
    std::iter::successors(Some((1u32 << k) - 1), |&c| {
        let u = c & c.wrapping_neg();
        let r = c + u;
        Some((((r ^ c) >> 2) / u) | r)
    })
    .take_while(|&c| c <= ALL29)
}

/// One Gauss-Seidel sweep over the group of layers `(w, b)` and `(b, w)`, in decreasing
/// total-pip order. Every position of layer `(w, b)` is updated jointly with its mirror
/// in `(b, w)` (for `w == b`, each mirror pair once). Swaps keep the total pip count
/// unchanged, so swap fights are cycles inside one level: each level is re-iterated (up
/// to `inner` times) until its own largest change drops below `inner_tol` before moving
/// down. Returns the largest change seen on the first pass over each level and where it
/// occurred.
fn sweep(
    rules: &Rules,
    layers: &Layers,
    combos: &Combos,
    w: usize,
    b: usize,
    inner: u32,
    inner_tol: f64,
) -> (f64, Pos) {
    let d1 = layers.layer(w, b);
    let d2 = layers.layer(b, w);
    let kb = BINOM[N_USABLE as usize - w][b];
    let kw = BINOM[N_USABLE as usize - b][w];
    let pip_lo_b = combos.pips[b][0] as usize;
    let pip_hi_b = *combos.pips[b].last().expect("non-empty") as usize;
    let mut worst = (0.0f64, Pos::default());
    for total in (pip_lo_b..=MAX_TOTAL_PIP).rev() {
        // Mover combos whose pip sum leaves room for an opponent combo.
        let first = combos.bucket[w][total.saturating_sub(pip_hi_b).min(MAX_TOTAL_PIP + 1)] as usize;
        let last = combos.bucket[w][(total - pip_lo_b).min(MAX_TOTAL_PIP) + 1] as usize;
        if first >= last {
            continue;
        }
        let pass = || {
            (first..last)
                .into_par_iter()
                .with_min_len(16)
                .map(|i| {
                    let mc = combos.masks[w][i];
                    let me = expand(mc);
                    let base1 = combos.ranks[w][i] * kb;
                    let mut local = (0.0f64, Pos::default());
                    let mut scratch = [Gathered::EMPTY; 2];
                    for j in combos.bucket_range(b, total - combos.pips[w][i] as usize) {
                        let oc = combos.masks[b][j];
                        if oc & mc != 0 || (w == b && oc < mc) {
                            continue;
                        }
                        let idx1 = base1 + colex_rank_among(oc, mc);
                        let idx2 = combos.ranks[b][j] * kw + colex_rank_among(mc, oc);
                        let pos = Pos { me, opp: expand(oc) };
                        let (v1, v2) = pair_update(rules, layers, pos, &mut scratch);
                        let change = d1.store(idx1, v1).max(d2.store(idx2, v2));
                        if change > local.0 {
                            local = (change, pos);
                        }
                    }
                    local
                })
                .reduce(|| (0.0, Pos::default()), |x, y| if x.0 >= y.0 { x } else { y })
        };
        let first_pass = pass();
        if first_pass.0 > worst.0 {
            worst = first_pass;
        }
        let mut change = first_pass.0;
        for _ in 1..inner {
            if change <= inner_tol {
                break;
            }
            change = pass().0;
        }
    }
    worst
}

/// Statistics of one solved group (recorded in `meta.json`, see `meta`).
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct GroupStats {
    pub layers: Vec<(usize, usize)>,
    pub states: u64,
    pub sweeps: u32,
    pub final_max_delta: f64,
    pub seconds: f64,
    /// The solve continued from a checkpoint: `sweeps` and `seconds` count only its own.
    #[serde(default)]
    pub resumed: bool,
}

#[derive(Clone, Copy, Debug)]
pub struct SolveConfig {
    /// A group is solved once a full sweep changes no value by more than this.
    pub tol: f64,
    /// Give up on a group (saving a checkpoint) after this many sweeps.
    pub max_sweeps: u32,
    /// Maximum passes over each total-pip level per sweep (1 = plain Gauss-Seidel).
    pub inner_iters: u32,
    /// Groups with at least this many pieces in total are solved in 24-bit storage (3
    /// bytes per position instead of 4) and read their inputs from 24-bit copies; this
    /// keeps the 5v5 group within about 23 GB of memory.
    pub compact_from_sum: usize,
    /// Save a checkpoint of the group being solved every this many seconds.
    pub checkpoint_secs: f64,
    /// Solve the groups with at most this many pieces in total.
    pub max_sum: usize,
    /// Report progress on stderr.
    pub verbose: bool,
}

impl SolveConfig {
    /// The default settings. The published database's groups of 7 to 10 pieces were solved
    /// with this `tol`, its smaller groups with 1e-7.
    pub const DEFAULT: SolveConfig = SolveConfig {
        tol: 3e-7,
        max_sweeps: 1000,
        inner_iters: 1,
        compact_from_sum: 2 * MAX_PIECES,
        checkpoint_secs: 1800.0,
        max_sum: 2 * MAX_PIECES,
        verbose: false,
    };
}

impl Default for SolveConfig {
    fn default() -> Self {
        SolveConfig::DEFAULT
    }
}

/// Groups in solving order: (hi, lo) with hi >= lo, by increasing hi + lo.
pub fn groups(max_sum: usize) -> Vec<(usize, usize)> {
    (2..=max_sum.min(2 * MAX_PIECES))
        .flat_map(|sum| (1..=sum / 2).map(move |lo| (sum - lo, lo)))
        .filter(|&(hi, _)| hi <= MAX_PIECES)
        .collect()
}

fn group_layers(hi: usize, lo: usize) -> Vec<(usize, usize)> {
    if hi == lo { vec![(hi, lo)] } else { vec![(hi, lo), (lo, hi)] }
}

/// Solves all groups with up to `cfg.max_sum` pieces in total into `dir`, skipping groups
/// whose layer files already exist and resuming from checkpoints. Records each group in
/// `dir/meta.json` (see `meta`) and returns the statistics of the groups solved by this call.
/// Fails if another solve is using `dir`.
///
/// Disk space: besides the smaller layers (14.1 GB), the 5v5 group needs its checkpoint
/// (15.1 GB) and the 24-bit copies of its inputs (7.6 GB), which are removed before its
/// layer (20.2 GB) is written: 36.8 GB at most, for a database of 34.2 GB.
pub fn solve_all(rules: &Rules, dir: &Path, cfg: &SolveConfig) -> io::Result<Vec<GroupStats>> {
    if *rules != Rules::KENDALL5 {
        return Err(invalid_input("the solver is defined for the KENDALL5 ruleset only"));
    }
    std::fs::create_dir_all(dir)?;
    let _lock = SolveLock::take(dir)?;
    meta::Meta::load(dir)?; // a malformed record fails now, not after hours of solving
    remove_leftovers(dir)?;
    let t0 = Instant::now();
    let combos = Combos::build();
    if cfg.verbose {
        eprintln!("combination tables built in {:.1}s", t0.elapsed().as_secs_f64());
    }
    let mut layers = Layers::new();
    let mut stats = Vec::new();
    for (hi, lo) in groups(cfg.max_sum) {
        let group = group_layers(hi, lo);
        if group.iter().all(|&(w, b)| layer_path(dir, w, b).exists()) {
            if cfg.verbose {
                eprintln!("group {group:?}: already solved, skipping");
            }
            remove_working_files(dir, &group)?; // as an interruption may leave them
            meta::record_if_missing(dir, &group)?;
            continue;
        }
        let compact = hi + lo >= cfg.compact_from_sum;
        prepare_inputs(dir, &mut layers, &group, compact)?;
        let resumed = alloc_or_resume(dir, &mut layers, &group, compact)?;
        let states: u64 = group.iter().map(|&(w, b)| layer_size(w, b)).sum();
        if cfg.verbose {
            eprintln!(
                "group {group:?}: {states} states ({:.2} GB resident){}",
                layers.resident_bytes() as f64 / 1e9,
                if resumed { ", resumed from checkpoint" } else { "" }
            );
        }
        let st = GroupStats { resumed, ..solve_group(rules, dir, cfg, &layers, &combos, &group)? };
        finish_group(dir, &mut layers, &group)?;
        meta::record(dir, meta::GroupRecord::solved(&st, cfg.tol))?;
        if cfg.verbose {
            eprintln!("  done: {} sweeps, {:.1}s", st.sweeps, st.seconds);
        }
        stats.push(st);
    }
    Ok(stats)
}

/// Iterates sweeps over a group whose layers are resident until it converges.
fn solve_group(
    rules: &Rules,
    dir: &Path,
    cfg: &SolveConfig,
    layers: &Layers,
    combos: &Combos,
    group: &[(usize, usize)],
) -> io::Result<GroupStats> {
    let (w, b) = group[0];
    let t0 = Instant::now();
    let mut last_checkpoint = Instant::now();
    let mut sweeps = 0;
    loop {
        let ts = Instant::now();
        let (change, worst_pos) = sweep(rules, layers, combos, w, b, cfg.inner_iters, cfg.tol);
        sweeps += 1;
        if cfg.verbose {
            let start = rules.start();
            let start_info = match index_of(start) {
                (sw, sb, i) if (sw, sb) == (w, b) => format!("  V(start)={:.7}", layers.layer(w, b).get(i)),
                _ => String::new(),
            };
            eprintln!(
                "  sweep {sweeps:3}: max delta {change:.3e} at {worst_pos:?}  ({:.1}s){start_info}",
                ts.elapsed().as_secs_f64()
            );
        }
        if change <= cfg.tol {
            let states = group.iter().map(|&(w, b)| layer_size(w, b)).sum();
            let seconds = t0.elapsed().as_secs_f64();
            let layers = group.to_vec();
            return Ok(GroupStats { layers, states, sweeps, final_max_delta: change, seconds, resumed: false });
        }
        if sweeps >= cfg.max_sweeps {
            save_checkpoint(dir, layers, group)?;
            return Err(io::Error::other(format!(
                "group {group:?} did not converge to {:e} in {sweeps} sweeps (last max change {change:.3e}); \
                 a checkpoint was saved and a rerun continues from it",
                cfg.tol
            )));
        }
        if last_checkpoint.elapsed().as_secs_f64() > cfg.checkpoint_secs {
            // A checkpoint only saves time after an interruption: solve on without it.
            match save_checkpoint(dir, layers, group) {
                Ok(()) if cfg.verbose => eprintln!("  checkpoint saved"),
                Ok(()) => {}
                Err(e) => eprintln!("  warning: no checkpoint saved ({e}); solving on"),
            }
            last_checkpoint = Instant::now();
        }
    }
}

/// The layers a group's successors can fall in, other than the group itself: the mover
/// bore off a piece and either throws again or the turn passes.
fn group_inputs(group: &[(usize, usize)]) -> Vec<(usize, usize)> {
    let mut inputs: Vec<_> =
        group.iter().filter(|&&(w, _)| w > 1).flat_map(|&(w, b)| [(w - 1, b), (b, w - 1)]).collect();
    inputs.sort_unstable();
    inputs.dedup();
    inputs
}

/// Makes exactly the inputs of `group` resident (mapped from their `.f32` files, or from
/// fresh 24-bit copies if `compact`), dropping every other layer.
fn prepare_inputs(dir: &Path, layers: &mut Layers, group: &[(usize, usize)], compact: bool) -> io::Result<()> {
    let inputs = group_inputs(group);
    for (w, b) in all_layers() {
        let current = layers.take(w, b);
        if !inputs.contains(&(w, b)) {
            continue;
        }
        let data = match current {
            Some(d @ LayerData::F32Map(_)) if !compact => d,
            _ => {
                let path = layer_path(dir, w, b);
                let mapped = LayerData::F32Map(map_layer(&path, w, b)?);
                check_probabilities(&mapped, layer_size(w, b), &path)?;
                if compact { mapped.to_u24_map(&u24_path(dir, w, b))? } else { mapped }
            }
        };
        layers.set(w, b, data);
    }
    Ok(())
}

/// Allocates the layers of `group`, starting from their checkpoints where present.
/// Returns whether any checkpoint was used. Fails on a checkpoint in the other storage.
fn alloc_or_resume(dir: &Path, layers: &mut Layers, group: &[(usize, usize)], compact: bool) -> io::Result<bool> {
    let mut resumed = false;
    for &(w, b) in group {
        let n = layer_size(w, b);
        let (checkpoint, other) = if compact {
            (u24_ckpt_path(dir, w, b), f32_ckpt_path(dir, w, b))
        } else {
            (f32_ckpt_path(dir, w, b), u24_ckpt_path(dir, w, b))
        };
        if other.exists() {
            let (theirs, ours) = if compact { ("float32", "24-bit") } else { ("24-bit", "float32") };
            return Err(invalid_input(format!(
                "{} is a checkpoint in {theirs} storage, but this solve stores group {group:?} in {ours} \
                 storage. Solve with the storage it was saved in (see --compact-from), or delete it to start over",
                other.display()
            )));
        }
        let data = if checkpoint.exists() {
            resumed = true;
            LayerData::load_checkpoint(&checkpoint, n, compact)
                .map_err(|e| io::Error::new(e.kind(), format!("{e} (delete the checkpoint to start over)")))?
        } else {
            LayerData::new_active(n, 0.5, compact)?
        };
        layers.set(w, b, data);
    }
    Ok(resumed)
}

fn save_checkpoint(dir: &Path, layers: &Layers, group: &[(usize, usize)]) -> io::Result<()> {
    for &(w, b) in group {
        let data = layers.layer(w, b);
        let path = match data {
            LayerData::Active24(_) => u24_ckpt_path(dir, w, b),
            _ => f32_ckpt_path(dir, w, b),
        };
        data.save_checkpoint(&path)?;
    }
    Ok(())
}

/// Writes the solved layers of `group` to the database, keeping only the `.f32` inputs
/// resident. The group's checkpoints and the 24-bit copies of its inputs are removed
/// first, so that the disk never holds them and the new layers at once; a crash while the
/// layers are written (about a minute for the 5v5 layer) loses the group's solve.
fn finish_group(dir: &Path, layers: &mut Layers, group: &[(usize, usize)]) -> io::Result<()> {
    for (w, b) in all_layers() {
        match layers.take(w, b) {
            Some(LayerData::U24Map(_)) | None => {}
            Some(d) => layers.set(w, b, d),
        }
    }
    remove_working_files(dir, group)?;
    for &(w, b) in group {
        layers.layer(w, b).write_f32(&layer_path(dir, w, b))?;
    }
    for &(w, b) in group {
        layers.take(w, b);
    }
    Ok(())
}

/// Removes those of `paths` that exist.
fn remove_files(paths: impl IntoIterator<Item = PathBuf>) -> io::Result<()> {
    for path in paths {
        match std::fs::remove_file(&path) {
            Err(e) if e.kind() != io::ErrorKind::NotFound => {
                return Err(io::Error::new(e.kind(), format!("removing {}: {e}", path.display())));
            }
            _ => {}
        }
    }
    Ok(())
}

/// Removes the files that only the solve of `group` uses: its checkpoints and the 24-bit
/// copies of its inputs.
fn remove_working_files(dir: &Path, group: &[(usize, usize)]) -> io::Result<()> {
    let checkpoints = group.iter().flat_map(|&(w, b)| [f32_ckpt_path(dir, w, b), u24_ckpt_path(dir, w, b)]);
    remove_files(checkpoints.chain(group_inputs(group).into_iter().map(|(w, b)| u24_path(dir, w, b))))
}

/// Removes what an interrupted solve can leave that no solve uses: partly written files and
/// 24-bit input copies (each group makes its own).
fn remove_leftovers(dir: &Path) -> io::Result<()> {
    let files = all_layers().flat_map(|(w, b)| {
        let written = [layer_path(dir, w, b), f32_ckpt_path(dir, w, b), u24_ckpt_path(dir, w, b), u24_path(dir, w, b)];
        written.into_iter().map(|path| tmp_path(&path)).chain([u24_path(dir, w, b)])
    });
    remove_files(files.chain([tmp_path(&dir.join(meta::FILE))]))
}

/// Bellman residual of solved layer `(w, b)`: max |T(V)(s) - V(s)| over `samples` random
/// positions, or over every position if `samples` is at least the layer size. Verifies a
/// database independently of how it was produced. A value that is not a probability, at a
/// sampled position or at a position its moves lead to, counts as an infinite residual
/// (maxima would skip a NaN). `layers` must also hold the inputs of `(w, b)`'s group.
pub fn residual_check(rules: &Rules, layers: &Layers, w: usize, b: usize, samples: u64, seed: u64) -> io::Result<f64> {
    if !(1..=MAX_PIECES).contains(&w) || !(1..=MAX_PIECES).contains(&b) {
        return Err(invalid_input(format!("no layer ({w}, {b})")));
    }
    let group = group_layers(w.max(b), w.min(b));
    let mut needed = group.iter().copied().chain(group_inputs(&group));
    if let Some((mw, mb)) = needed.find(|&(w, b)| !layers.has(w, b)) {
        return Err(invalid_input(format!("layer ({mw}, {mb}) is not loaded")));
    }
    let n = layer_size(w, b);
    let data = layers.layer(w, b);
    let residual = |idx: u64| {
        let r = (bellman(rules, layers, position_of(w, b, idx)) - data.get(idx)).abs();
        if r.is_nan() { f64::INFINITY } else { r }
    };
    Ok(if samples >= n {
        (0..n).into_par_iter().map(residual).reduce(|| 0.0, f64::max)
    } else {
        (0..samples)
            .into_par_iter()
            .map(|k| residual(Rng::new(seed ^ k.wrapping_mul(0x9E37_79B9_7F4A_7C15)).below(n)))
            .reduce(|| 0.0, f64::max)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn combos_complete() {
        let c = Combos::build();
        for (k, &count) in BINOM[29].iter().enumerate().take(MAX_PIECES + 1).skip(1) {
            assert_eq!(c.masks[k].len() as u64, count);
            assert_eq!(*c.bucket[k].last().unwrap() as u64, count);
        }
        assert_eq!(c.pips[MAX_PIECES].last().map(|&p| p as usize), Some(MAX_PIP));
    }

    #[test]
    fn group_order_and_inputs() {
        assert_eq!(groups(4), [(1, 1), (2, 1), (3, 1), (2, 2)]);
        assert_eq!(groups(99).len(), 15);
        assert_eq!(group_inputs(&group_layers(3, 1)), [(1, 2), (2, 1)]);
        assert_eq!(group_inputs(&group_layers(5, 4)), [(3, 5), (4, 4), (5, 3)]);
        assert_eq!(group_inputs(&group_layers(1, 1)), []);
    }

    fn temp_dir(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("senet_solver_{name}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        dir
    }

    fn assert_solved(dir: &Path, max_sum: usize) -> Layers {
        let rules = Rules::KENDALL5;
        let layers = Layers::open(dir, max_sum).unwrap();
        for (w, b) in all_layers().filter(|&(w, b)| w + b <= max_sum) {
            let r = residual_check(&rules, &layers, w, b, u64::MAX, 1).unwrap();
            assert!(r < 1e-6, "layer ({w},{b}) residual {r}");
        }
        layers
    }

    #[test]
    fn solve_small_in_both_storages_and_check_residual() {
        let f32_dir = temp_dir("f32");
        let u24_dir = temp_dir("u24");
        let cfg = SolveConfig { tol: 1e-7, max_sum: 4, ..SolveConfig::DEFAULT };
        solve_all(&Rules::KENDALL5, &f32_dir, &cfg).unwrap();
        solve_all(&Rules::KENDALL5, &u24_dir, &SolveConfig { compact_from_sum: 3, ..cfg }).unwrap();
        let a = assert_solved(&f32_dir, 4);
        let b = assert_solved(&u24_dir, 4);
        for (w, l) in all_layers().filter(|&(w, b)| w + b <= 4) {
            for i in 0..layer_size(w, l) {
                let (x, y) = (a.layer(w, l).get(i), b.layer(w, l).get(i));
                assert!((x - y).abs() < 2e-6, "({w},{l})[{i}]: {x} vs {y}");
            }
        }
        // Only the database and meta.json remain.
        let mut names: Vec<String> =
            std::fs::read_dir(&u24_dir).unwrap().map(|e| e.unwrap().file_name().into_string().unwrap()).collect();
        names.sort();
        assert_eq!(names, ["L11.f32", "L12.f32", "L13.f32", "L21.f32", "L22.f32", "L31.f32", "meta.json"]);
        // Mover on 30 vs opponent on 1: needs a 1 (prob 1/4) each try, but the opponent
        // needs many throws; the mover should be a strong favourite.
        let (w, l, i) = index_of(Pos::from_squares(&[30], &[1]).unwrap());
        let v = a.layer(w, l).get(i);
        assert!(v > 0.8 && v <= 1.0, "V = {v}");
        assert!(residual_check(&Rules::KENDALL5, &a, 3, 2, 10, 1).is_err(), "layer (3, 2) is not loaded");
        assert!(residual_check(&Rules::KENDALL5, &a, 0, 1, 10, 1).is_err());
        drop((a, b));
        let _ = std::fs::remove_dir_all(&f32_dir);
        let _ = std::fs::remove_dir_all(&u24_dir);
    }

    #[test]
    fn corrupt_values_fail_the_residual_check() {
        let dir = temp_dir("nan");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(layer_path(&dir, 1, 1), f32::NAN.to_le_bytes().repeat(layer_size(1, 1) as usize)).unwrap();
        let layers = Layers::open(&dir, 2).unwrap();
        for samples in [u64::MAX, 10] {
            let r = residual_check(&Rules::KENDALL5, &layers, 1, 1, samples, 1).unwrap();
            assert_eq!(r, f64::INFINITY, "a layer of NaNs ({samples} samples)");
        }
        drop(layers);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn corrupt_successors_fail_the_residual_check() {
        let rules = Rules::KENDALL5;
        let dir = temp_dir("nan_successor");
        solve_all(&rules, &dir, &SolveConfig { tol: 1e-7, max_sum: 3, ..SolveConfig::DEFAULT }).unwrap();
        // No position of layer (2, 1) has its best move to position 21 of layer (1, 2), so
        // best-move maxima that skipped a NaN there would leave every residual tiny.
        let path = layer_path(&dir, 1, 2);
        let mut bytes = std::fs::read(&path).unwrap();
        bytes[4 * 21..][..4].copy_from_slice(&f32::NAN.to_le_bytes());
        std::fs::write(&path, bytes).unwrap();
        let layers = Layers::open(&dir, 3).unwrap();
        assert_eq!(residual_check(&rules, &layers, 2, 1, u64::MAX, 1).unwrap(), f64::INFINITY);
        drop(layers);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn corrupt_inputs_stop_the_solve() {
        let rules = Rules::KENDALL5;
        let cfg = SolveConfig { tol: 1e-7, max_sum: 2, ..SolveConfig::DEFAULT };
        let nans = f32::NAN.to_le_bytes().repeat(layer_size(1, 1) as usize);
        let dir = temp_dir("corrupt");
        std::fs::create_dir_all(&dir).unwrap();
        // A checkpoint of NaNs is refused (it would otherwise pass for converged).
        std::fs::write(f32_ckpt_path(&dir, 1, 1), &nans).unwrap();
        let err = solve_all(&rules, &dir, &cfg).unwrap_err().to_string();
        assert!(err.contains("is not a probability"), "{err}");
        // So is a solved layer of NaNs that the next group reads.
        std::fs::remove_file(f32_ckpt_path(&dir, 1, 1)).unwrap();
        solve_all(&rules, &dir, &cfg).unwrap();
        std::fs::write(layer_path(&dir, 1, 1), &nans).unwrap();
        let err = solve_all(&rules, &dir, &SolveConfig { max_sum: 3, ..cfg }).unwrap_err().to_string();
        assert!(err.contains("L11.f32: value NaN at index 0 is not a probability"), "{err}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn non_convergence_saves_a_checkpoint_and_resumes() {
        let dir = temp_dir("resume");
        let cfg = SolveConfig { tol: 1e-7, max_sum: 3, ..SolveConfig::DEFAULT };
        let err = solve_all(&Rules::KENDALL5, &dir, &SolveConfig { max_sweeps: 2, ..cfg }).unwrap_err();
        assert!(err.to_string().contains("did not converge"), "{err}");
        assert!(f32_ckpt_path(&dir, 1, 1).exists());
        let stats = solve_all(&Rules::KENDALL5, &dir, &cfg).unwrap();
        assert_eq!(stats.len(), 2, "groups (1, 1) and (2, 1)");
        assert!(stats[0].resumed && !stats[1].resumed);
        assert!(!f32_ckpt_path(&dir, 1, 1).exists());
        drop(assert_solved(&dir, 3));
        let groups = meta::Meta::load(&dir).unwrap().groups;
        assert_eq!(groups["[(1, 1)]"].note.as_deref(), Some(meta::RESUMED));
        assert_eq!(groups["[(2, 1), (1, 2)]"].note, None);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_checkpoint_in_the_other_storage_is_not_discarded() {
        let dir = temp_dir("other_storage");
        let cfg = SolveConfig { tol: 1e-7, max_sum: 2, max_sweeps: 2, ..SolveConfig::DEFAULT };
        solve_all(&Rules::KENDALL5, &dir, &cfg).unwrap_err();
        let saved = std::fs::read(f32_ckpt_path(&dir, 1, 1)).unwrap();
        let compact = SolveConfig { compact_from_sum: 2, max_sweeps: 1000, ..cfg };
        let err = solve_all(&Rules::KENDALL5, &dir, &compact).unwrap_err().to_string();
        assert!(err.contains("L11.f32.ckpt is a checkpoint in float32 storage"), "{err}");
        assert_eq!(std::fs::read(f32_ckpt_path(&dir, 1, 1)).unwrap(), saved);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn one_solve_at_a_time_and_leftovers_removed() {
        let dir = temp_dir("lock");
        let cfg = SolveConfig { tol: 1e-7, max_sum: 2, ..SolveConfig::DEFAULT };
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("solve.lock"), "1234\n").unwrap();
        let err = solve_all(&Rules::KENDALL5, &dir, &cfg).unwrap_err().to_string();
        assert!(err.contains("solve.lock exists: another solve (process 1234)"), "{err}");
        assert!(!layer_path(&dir, 1, 1).exists());
        std::fs::remove_file(dir.join("solve.lock")).unwrap();
        // What interrupted solves leave: partly written files, 24-bit copies, and the
        // checkpoint of a group whose layers were written.
        solve_all(&Rules::KENDALL5, &dir, &cfg).unwrap();
        let leftovers = [tmp_path(&layer_path(&dir, 2, 1)), u24_path(&dir, 1, 1), tmp_path(&u24_ckpt_path(&dir, 2, 1))];
        for path in leftovers.iter().chain([&f32_ckpt_path(&dir, 1, 1)]) {
            std::fs::write(path, b"left over").unwrap();
        }
        solve_all(&Rules::KENDALL5, &dir, &cfg).unwrap();
        let mut names: Vec<String> =
            std::fs::read_dir(&dir).unwrap().map(|e| e.unwrap().file_name().into_string().unwrap()).collect();
        names.sort();
        assert_eq!(names, ["L11.f32", "meta.json"]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_failed_checkpoint_does_not_stop_the_solve() {
        let rules = Rules::KENDALL5;
        let missing = temp_dir("no_checkpoints").join("missing");
        let combos = Combos::build();
        let solve = |cfg: &SolveConfig| {
            let mut layers = Layers::new();
            alloc_or_resume(&missing, &mut layers, &[(1, 1)], false).unwrap();
            solve_group(&rules, &missing, cfg, &layers, &combos, &[(1, 1)])
        };
        // A checkpoint after every sweep, each failing: the directory does not exist.
        let cfg = SolveConfig { tol: 1e-7, checkpoint_secs: 0.0, ..SolveConfig::DEFAULT };
        let st = solve(&cfg).unwrap();
        assert!(st.sweeps > 2 && st.final_max_delta <= 1e-7, "{st:?}");
        // The checkpoint of a solve that gives up must be saved.
        let err = solve(&SolveConfig { max_sweeps: 1, ..cfg }).unwrap_err();
        assert!(err.to_string().contains("L11.f32.ckpt"), "{err}");
    }

    #[test]
    fn rejects_other_rulesets() {
        let rules = Rules { protection: false, ..Rules::KENDALL5 };
        assert!(solve_all(&rules, &temp_dir("rules"), &SolveConfig::DEFAULT).is_err());
    }
}
