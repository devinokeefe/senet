//! `senet` command-line tool: solver, analysis, matches, move dumps and the web server.

mod artifacts;
mod http;
mod server;

use clap::{Args, Parser, Subcommand};
use senet_core::board::{Bits, Pos, Rules};
use senet_core::bots::{BotContext, SPECS};
use senet_core::db::{Db, layer_path};
use senet_core::eval::move_value;
use senet_core::game::{analyze_quality, run_match};
use senet_core::index::{MAX_PIECES, all_layers, layer_size, random_position};
use senet_core::movegen::{MoveList, Outcome, gen_moves};
use senet_core::rng::Rng;
use senet_core::solver::meta::{self, Meta};
use senet_core::solver::{self, Layers, SolveConfig};
use serde::Serialize;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::sync::LazyLock;
use std::time::Instant;

/// `senet --version`: the version, and how this build was compiled (senet_core::build_info),
/// which says on which CPUs it runs.
static LONG_VERSION: LazyLock<String> = LazyLock::new(|| {
    let info = senet_core::build_info();
    let features: Vec<&str> =
        info["cpu_features"].as_array().into_iter().flatten().filter_map(|f| f.as_str()).collect();
    let optimized = if info["optimized"] == true { "optimized" } else { "debug" };
    format!(
        "{}\n{optimized} build for {}; CPU features required beyond the baseline: {}",
        info["version"].as_str().unwrap_or("?"),
        info["target"].as_str().unwrap_or("?"),
        if features.is_empty() { "none".to_string() } else { features.join(", ") }
    )
});

#[derive(Parser)]
#[command(name = "senet", version, about = "Perfect-play Senet engine (Modern Kendall rules, 5 pieces)")]
#[command(long_version = LONG_VERSION.as_str(), after_help = format!("Bot specs: {SPECS}"))]
struct Cli {
    /// Worker threads [default: one per logical core]
    #[arg(long, global = true)]
    threads: Option<usize>,
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Solve the game into a database directory (resumes from checkpoints)
    Solve(SolveArgs),
    /// Audit a database: the Bellman residual |T(V) - V| of every layer
    Check(CheckArgs),
    /// Show a position's value and the value of every move for each throw
    Value(ValueArgs),
    /// Play a match between two bots
    Match(MatchArgs),
    /// Measure how much win probability a bot gives away per decision
    Quality(QualityArgs),
    /// Write random positions and their legal moves as JSON lines (for differential tests)
    DumpMoves(DumpArgs),
    /// Write training records for the network (u32 me, u32 opp, f32 value; see docs/FORMATS.md)
    GenData(artifacts::GenDataArgs),
    /// Write the integrity manifest of an existing database or training-data file
    Manifest(artifacts::ManifestArgs),
    /// Check files against their manifest: every size and SHA-256
    Verify(artifacts::VerifyArgs),
    /// Serve the web app and its JSON API on 127.0.0.1
    Serve(server::ServeArgs),
}

#[derive(Args)]
struct SolveArgs {
    /// Database directory
    #[arg(long)]
    db: PathBuf,
    /// Solve the groups with at most this many pieces in total
    #[arg(long, default_value_t = SolveConfig::DEFAULT.max_sum, value_parser = piece_total)]
    max_sum: usize,
    /// Convergence tolerance: the largest change a final sweep may make
    #[arg(long, default_value_t = SolveConfig::DEFAULT.tol, value_parser = non_negative)]
    tol: f64,
    /// Give up on a group after this many sweeps (a checkpoint is saved)
    #[arg(long, default_value_t = SolveConfig::DEFAULT.max_sweeps)]
    max_sweeps: u32,
    /// Passes over each total-pip level per sweep
    #[arg(long, default_value_t = SolveConfig::DEFAULT.inner_iters)]
    inner: u32,
    /// Use 24-bit storage for groups with at least this many pieces in total
    #[arg(long, default_value_t = SolveConfig::DEFAULT.compact_from_sum)]
    compact_from: usize,
    /// Seconds between checkpoints
    #[arg(long, default_value_t = SolveConfig::DEFAULT.checkpoint_secs, value_parser = non_negative)]
    ckpt_secs: f64,
}

#[derive(Args)]
struct CheckArgs {
    /// Database directory
    #[arg(long)]
    db: PathBuf,
    /// Random positions per layer (whole layers if they are smaller)
    #[arg(long, default_value_t = 200_000, value_parser = clap::value_parser!(u64).range(1..))]
    samples: u64,
    /// Check the layers with at most this many pieces in total
    #[arg(long, default_value_t = 2 * MAX_PIECES, value_parser = piece_total)]
    max_sum: usize,
    /// Fail if any residual exceeds this
    #[arg(long, default_value_t = 1e-6, value_parser = non_negative)]
    max_residual: f64,
    /// Seed of the sample of positions
    #[arg(long, default_value_t = 12345)]
    seed: u64,
}

#[derive(Args)]
struct ValueArgs {
    /// Database directory
    #[arg(long)]
    db: PathBuf,
    /// Squares of the player to throw, e.g. 1,3,5
    #[arg(long)]
    me: Squares,
    /// Squares of the opponent
    #[arg(long)]
    opp: Squares,
}

#[derive(Args)]
struct BotFiles {
    /// Database directory (for `perfect`)
    #[arg(long)]
    db: Option<PathBuf>,
    /// Network file (for `net`)
    #[arg(long)]
    net: Option<PathBuf>,
}

#[derive(Args)]
struct MatchArgs {
    /// Bot A
    #[arg(long)]
    a: String,
    /// Bot B
    #[arg(long)]
    b: String,
    /// Pairs of games (same dice, colours swapped)
    #[arg(long, default_value_t = 2000, value_parser = clap::value_parser!(u64).range(1..))]
    pairs: u64,
    /// Seed of the throws (and of the random bot's choices)
    #[arg(long, default_value_t = 1)]
    seed: u64,
    #[command(flatten)]
    files: BotFiles,
    /// Also print the result as JSON
    #[arg(long)]
    json: bool,
}

#[derive(Args)]
struct QualityArgs {
    /// The bot to analyse
    #[arg(long)]
    bot: String,
    /// Its opponent
    #[arg(long, default_value = "perfect")]
    vs: String,
    /// Games to play (the bot is White in every other one)
    #[arg(long, default_value_t = 2000, value_parser = clap::value_parser!(u64).range(1..))]
    games: u64,
    /// Seed of the throws (and of the random bot's choices)
    #[arg(long, default_value_t = 7)]
    seed: u64,
    /// Database directory (the judge)
    #[arg(long)]
    db: PathBuf,
    /// Network file (for `net` bots)
    #[arg(long)]
    net: Option<PathBuf>,
    /// Also print the result as JSON
    #[arg(long)]
    json: bool,
}

#[derive(Args)]
struct DumpArgs {
    /// Output file (JSON lines)
    #[arg(long)]
    out: PathBuf,
    /// Number of positions
    #[arg(long, default_value_t = 100_000)]
    n: u64,
    /// Seed of the positions and throws
    #[arg(long, default_value_t = 1)]
    seed: u64,
}

/// A finite number >= 0 (a NaN would fail every comparison it is used in).
fn non_negative(s: &str) -> Result<f64, String> {
    match s.parse::<f64>() {
        Ok(x) if x.is_finite() && x >= 0.0 => Ok(x),
        _ => Err(format!("expected a finite number >= 0, not '{s}'")),
    }
}

/// A number of pieces in total that some layer has: 2 (one each) to 2 * MAX_PIECES.
fn piece_total(s: &str) -> Result<usize, String> {
    match s.parse::<usize>() {
        Ok(n) if (2..=2 * MAX_PIECES).contains(&n) => Ok(n),
        _ => Err(format!("expected 2..={}, not '{s}'", 2 * MAX_PIECES)),
    }
}

/// A comma-separated list of squares (empty for none).
#[derive(Clone)]
struct Squares(Vec<u32>);

impl FromStr for Squares {
    type Err = String;

    fn from_str(s: &str) -> Result<Squares, String> {
        if s.is_empty() {
            return Ok(Squares(vec![]));
        }
        s.split(',')
            .map(|x| x.trim().parse().map_err(|_| format!("bad square '{x}'")))
            .collect::<Result<_, _>>()
            .map(Squares)
    }
}

fn main() {
    // A build for another machine's CPU (.cargo/native.toml) would stop at an illegal instruction.
    let missing = senet_core::missing_cpu_features();
    if !missing.is_empty() {
        eprintln!("error: this build needs CPU features that this CPU lacks: {}", missing.join(", "));
        eprintln!("build it here with `cargo build --release`");
        std::process::exit(1);
    }
    let cli = Cli::parse();
    if let Err(e) = run(cli) {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

fn run(cli: Cli) -> Result<(), String> {
    if let Some(n) = cli.threads {
        rayon::ThreadPoolBuilder::new().num_threads(n).build_global().map_err(|e| e.to_string())?;
    }
    match cli.command {
        Command::Solve(a) => cmd_solve(a),
        Command::Check(a) => cmd_check(a),
        Command::Value(a) => cmd_value(a),
        Command::Match(a) => cmd_match(a),
        Command::Quality(a) => cmd_quality(a),
        Command::DumpMoves(a) => cmd_dump_moves(a),
        Command::GenData(a) => artifacts::cmd_gen_data(a),
        Command::Manifest(a) => artifacts::cmd_manifest(a),
        Command::Verify(a) => artifacts::cmd_verify(a),
        Command::Serve(a) => server::serve(a),
    }
}

fn create(path: &Path) -> Result<BufWriter<File>, String> {
    Ok(BufWriter::with_capacity(1 << 22, File::create(path).map_err(|e| format!("{}: {e}", path.display()))?))
}

/// Whether every layer with at most `max_sum` pieces in total is in `dir` and recorded in
/// its meta.json: `senet solve` then has nothing to do, and changes nothing.
fn solved_and_recorded(dir: &Path, max_sum: usize) -> Result<bool, String> {
    let meta = Meta::load(dir).map_err(|e| e.to_string())?;
    let recorded = |layer| meta.groups.values().any(|r| r.layers.contains(&layer));
    Ok(all_layers().filter(|&(w, b)| w + b <= max_sum).all(|(w, b)| layer_path(dir, w, b).exists() && recorded((w, b))))
}

fn cmd_solve(a: SolveArgs) -> Result<(), String> {
    let cfg = SolveConfig {
        tol: a.tol,
        max_sweeps: a.max_sweeps,
        inner_iters: a.inner,
        compact_from_sum: a.compact_from,
        checkpoint_secs: a.ckpt_secs,
        max_sum: a.max_sum,
        verbose: true,
    };
    if solved_and_recorded(&a.db, a.max_sum)? {
        eprintln!("{}: the layers of at most {} pieces are solved and recorded already", a.db.display(), a.max_sum);
        return artifacts::check_database_manifest(&a.db);
    }
    eprintln!("solving into {} with {} threads", a.db.display(), rayon::current_num_threads());
    artifacts::retire_database_manifest(&a.db)?;
    solver::solve_all(&Rules::KENDALL5, &a.db, &cfg).map_err(|e| e.to_string())?;
    let db = Db::open(&a.db).map_err(|e| format!("{}: {e}", a.db.display()))?;
    if let Some(v) = db.lookup(Rules::KENDALL5.start()) {
        eprintln!("V(start) = P(White wins) = {v:.7}");
    }
    if db.is_complete() {
        drop(db);
        eprintln!("hashing the database for its manifest");
        let path = artifacts::write_database_manifest(&a.db, &[])?;
        eprintln!("wrote {}", path.display());
    }
    Ok(())
}

/// What the database's meta.json in `dir` records, in one line.
fn meta_summary(dir: &Path) -> Result<String, String> {
    if !dir.join(meta::FILE).exists() {
        return Ok("no meta.json".into());
    }
    let meta = Meta::load(dir).map_err(|e| e.to_string())?;
    let reconstructed = meta.groups.values().filter(|r| r.note.is_some()).count();
    let unknown = meta.groups.values().filter(|r| r.sweeps.is_none()).count();
    Ok(format!(
        "meta.json: {} groups recorded, {reconstructed} reconstructed, {unknown} without solve statistics",
        meta.groups.len()
    ))
}

fn cmd_check(a: CheckArgs) -> Result<(), String> {
    if !a.db.is_dir() {
        return Err(format!("{}: no such directory", a.db.display()));
    }
    let layers = Layers::open(&a.db, a.max_sum).map_err(|e| format!("opening {}: {e}", a.db.display()))?;
    println!("{}", meta_summary(&a.db)?);
    let mut worst: f64 = 0.0;
    for (w, b) in all_layers().filter(|&(w, b)| w + b <= a.max_sum) {
        let r =
            solver::residual_check(&Rules::KENDALL5, &layers, w, b, a.samples, a.seed).map_err(|e| e.to_string())?;
        let n = a.samples.min(layer_size(w, b));
        println!("L{w}{b}: max |T(V) - V| over {n} positions = {r:.3e}");
        worst = worst.max(r);
    }
    println!("worst residual: {worst:.3e}");
    if worst > a.max_residual {
        return Err(format!("residual {worst:.3e} exceeds {:.1e}", a.max_residual));
    }
    Ok(())
}

fn cmd_value(a: ValueArgs) -> Result<(), String> {
    let rules = Rules::KENDALL5;
    let pos = Pos::from_squares(&a.me.0, &a.opp.0)?;
    if pos.is_over() {
        return Err("the game is already over".into());
    }
    let db = Db::open(&a.db).map_err(|e| format!("opening database {}: {e}", a.db.display()))?;
    let v = db.lookup(pos).ok_or("position not covered by the database")?;
    println!("position {pos:?}: P(mover wins) = {v:.6}");
    let mut ml = MoveList::new();
    for t in 1..=5u8 {
        gen_moves(&rules, pos, t, &mut ml);
        if ml.is_empty() {
            println!("  throw {t}: no legal move (turn passes)");
            continue;
        }
        let items: Vec<String> = ml
            .iter()
            .map(|m| format!("{}->{} {} {:.5}", m.from, m.to, m.kind.name(), move_value(&rules, &db, m, t, 0)))
            .collect();
        println!("  throw {t}: {}", items.join(" | "));
    }
    Ok(())
}

fn cmd_match(a: MatchArgs) -> Result<(), String> {
    let ctx = BotContext::load(a.files.db.as_deref(), a.files.net.as_deref())?;
    let t0 = Instant::now();
    let r = run_match(&Rules::KENDALL5, &a.a, &a.b, &ctx, a.pairs, a.seed).map_err(|e| e.to_string())?;
    println!(
        "{} vs {}: {} games, {} wins {:.2}% ± {:.2}% (95% CI)  (as White {}/{}), avg {:.1} throws/game  [{:.1}s]",
        r.bot_a,
        r.bot_b,
        r.games,
        r.bot_a,
        100.0 * r.a_win_rate,
        100.0 * r.ci95,
        r.a_white_wins,
        r.a_white_games,
        r.avg_throws,
        t0.elapsed().as_secs_f64()
    );
    if a.json {
        println!("{}", serde_json::to_string(&r).map_err(|e| e.to_string())?);
    }
    Ok(())
}

fn cmd_quality(a: QualityArgs) -> Result<(), String> {
    let ctx = BotContext::load(Some(&a.db), a.net.as_deref())?;
    let q = analyze_quality(&Rules::KENDALL5, &a.bot, &a.vs, &ctx, a.games, a.seed).map_err(|e| e.to_string())?;
    println!(
        "{}: {} games, {} decisions, error rate {:.2}%, avg loss {:.5} per decision, {:.4} per game, max {:.4}",
        q.bot,
        q.games,
        q.decisions,
        100.0 * q.errors as f64 / q.decisions.max(1) as f64,
        q.avg_loss_per_decision,
        q.avg_loss_per_game,
        q.max_loss
    );
    if a.json {
        println!("{}", serde_json::to_string(&q).map_err(|e| e.to_string())?);
    }
    Ok(())
}

/// One line of the move dump (docs/FORMATS.md).
#[derive(Serialize)]
struct DumpRecord {
    me: Vec<u32>,
    opp: Vec<u32>,
    t: u8,
    moves: Vec<DumpMove>,
}

#[derive(Serialize)]
struct DumpMove {
    from: u8,
    to: u8,
    kind: &'static str,
    dir: &'static str,
    me: Vec<u32>,
    opp: Vec<u32>,
}

/// Random positions for differential testing: half uniform over all layers, half taken
/// from a random game (more realistic structure), each with a random throw.
fn cmd_dump_moves(a: DumpArgs) -> Result<(), String> {
    let rules = Rules::KENDALL5;
    let mut rng = Rng::new(a.seed);
    let mut f = create(&a.out)?;
    let mut ml = MoveList::new();
    let mut game_pos = rules.start();
    let squares = |mask: u32| Bits(mask).collect::<Vec<u32>>();
    for k in 0..a.n {
        let pos = if k % 2 == 0 {
            random_position(&mut rng)
        } else {
            // Advance the random game by one throw (restarting it once it is won).
            let t = rng.throw();
            gen_moves(&rules, game_pos, t, &mut ml);
            game_pos = if ml.is_empty() {
                game_pos.flip()
            } else {
                match ml[rng.below(ml.len() as u64) as usize].outcome(&rules, t) {
                    Outcome::Won => rules.start(),
                    Outcome::ThrowAgain(next) | Outcome::OpponentThrows(next) => next,
                }
            };
            game_pos
        };
        let t = 1 + rng.below(5) as u8;
        gen_moves(&rules, pos, t, &mut ml);
        let record = DumpRecord {
            me: squares(pos.me),
            opp: squares(pos.opp),
            t,
            moves: ml
                .iter()
                .map(|m| DumpMove {
                    from: m.from,
                    to: m.to,
                    kind: m.kind.name(),
                    dir: m.direction_name(),
                    me: squares(m.after.me),
                    opp: squares(m.after.opp),
                })
                .collect(),
        };
        let line = serde_json::to_string(&record).map_err(|e| e.to_string())?;
        writeln!(f, "{line}").map_err(|e| e.to_string())?;
    }
    f.flush().map_err(|e| e.to_string())?;
    eprintln!("wrote {} positions to {}", a.n, a.out.display());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::CommandFactory;

    #[test]
    fn cli_definition_is_valid() {
        Cli::command().debug_assert();
    }

    #[test]
    fn squares_parse() {
        assert_eq!(Squares::from_str("1, 3,5").unwrap().0, [1, 3, 5]);
        assert!(Squares::from_str("").unwrap().0.is_empty());
        assert!(Squares::from_str("1,x").is_err());
    }

    #[test]
    fn value_refuses_a_finished_game() {
        let args = |me: &str, opp: &str| ValueArgs {
            db: PathBuf::from("no/such/db"),
            me: Squares::from_str(me).unwrap(),
            opp: Squares::from_str(opp).unwrap(),
        };
        assert_eq!(cmd_value(args("", "1")).unwrap_err(), "the game is already over");
        assert_eq!(cmd_value(args("1", "")).unwrap_err(), "the game is already over");
        assert!(cmd_value(args("1", "2")).unwrap_err().contains("no/such/db"));
    }

    #[test]
    fn options_that_would_play_nothing_are_refused() {
        let parse = |args: &[&str]| Cli::try_parse_from(args.iter().copied());
        assert!(parse(&["senet", "match", "--a", "greedy", "--b", "random", "--pairs", "0"]).is_err());
        assert!(parse(&["senet", "quality", "--bot", "greedy", "--db", "db", "--games", "0"]).is_err());
        assert!(parse(&["senet", "quality", "--bot", "greedy", "--db", "db", "--games", "1"]).is_ok());
    }

    #[test]
    fn check_refuses_a_missing_database_before_reporting_on_it() {
        let a = CheckArgs { db: PathBuf::from("no/such/db"), samples: 1, max_sum: 2, max_residual: 1e-6, seed: 1 };
        assert!(cmd_check(a).unwrap_err().contains("no such directory"));
        let dir = std::env::temp_dir().join(format!("senet_cli_empty_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        assert_eq!(meta_summary(&dir).unwrap(), "no meta.json");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_solve_with_nothing_to_solve_leaves_the_database_and_its_manifest() {
        use senet_core::manifest;
        let dir = std::env::temp_dir().join(format!("senet_cli_solve_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let solve = |max_sum| {
            cmd_solve(SolveArgs {
                db: dir.clone(),
                max_sum,
                tol: 1e-7,
                max_sweeps: 1000,
                inner: 1,
                compact_from: 2 * MAX_PIECES,
                ckpt_secs: 1800.0,
            })
        };
        assert!(!solved_and_recorded(&dir, 3).unwrap());
        solve(3).unwrap();
        assert!(solved_and_recorded(&dir, 3).unwrap() && !solved_and_recorded(&dir, 4).unwrap());
        assert_eq!(
            meta_summary(&dir).unwrap(),
            "meta.json: 2 groups recorded, 0 reconstructed, 0 without solve statistics"
        );

        // A manifest, as a solve of the whole game writes it, then a layer is spoiled.
        let path = artifacts::write_database_manifest(&dir, &[]).unwrap();
        let written = std::fs::read(&path).unwrap();
        let layer = layer_path(&dir, 2, 1);
        let good = std::fs::read(&layer).unwrap();
        let mut spoiled = good.clone();
        spoiled[0] ^= 1;
        std::fs::write(&layer, &spoiled).unwrap();
        // A rerun with nothing to solve does not vouch for it anew.
        solve(3).unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), written);
        assert!(manifest::verify(&dir).unwrap_err().to_string().contains("L21.f32: SHA-256"));
        // A file of the wrong size fails it.
        std::fs::write(&layer, &good[..8]).unwrap();
        assert!(solve(3).unwrap_err().contains("L21.f32: 8 bytes"));
        std::fs::write(&layer, &good).unwrap();
        // A layer solved without a record is recorded: that changes meta.json, so the
        // manifest is retired (and with the database incomplete, none is written).
        let mut meta = Meta::load(&dir).unwrap();
        meta.groups.remove("[(2, 1), (1, 2)]").unwrap();
        std::fs::write(dir.join(meta::FILE), serde_json::to_string(&meta).unwrap()).unwrap();
        assert!(!solved_and_recorded(&dir, 3).unwrap());
        solve(3).unwrap();
        assert!(!path.exists());
        assert_eq!(
            meta_summary(&dir).unwrap(),
            "meta.json: 2 groups recorded, 1 reconstructed, 1 without solve statistics"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn check_options_that_would_check_nothing_are_refused() {
        let parse = |args: &[&str]| Cli::try_parse_from([["senet", "check", "--db", "db"].as_slice(), args].concat());
        assert!(parse(&[]).is_ok());
        for bad in [["--max-residual", "NaN"], ["--max-residual", "-1"], ["--samples", "0"], ["--max-sum", "1"]] {
            assert!(parse(&bad).is_err(), "{bad:?}");
        }
    }
}
