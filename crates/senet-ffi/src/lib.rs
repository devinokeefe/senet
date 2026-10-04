//! C ABI used by the Python package (`python/senet`). Positions are passed as two `u32`
//! square masks (bit `i` = square `i`) from the point of view of the player about to
//! throw.
//!
//! Conventions:
//! * Integer results are non-negative on success; on failure they are `INVALID` (-1: a
//!   bad argument, such as an invalid position or an empty board, a finished game where
//!   moves are asked for, an invalid throw or bot spec, no games to play, a NULL or
//!   misaligned pointer, a non-UTF-8 string or a closed handle), `UNAVAILABLE` (-2: a
//!   database or network the call needs is not loaded or does not cover the position;
//!   perfect play and the quality analysis need the complete database) or, for
//!   `senet_bot_choose`, `NO_MOVE`.
//! * Value results (`f64`) are probabilities in [0, 1], or -1.0 / -2.0 with the same
//!   meanings as `INVALID` / `UNAVAILABLE`.
//! * Every failure stores a message in a per-thread text slot, and `senet_match`,
//!   `senet_quality` and `senet_build_info` store their JSON result there;
//!   `senet_last_text` reads it.
//! * A call that fails writes none of its outputs.
//! * Panics are caught and reported as `INVALID` failures instead of unwinding into C.
//!
//! Handles (`*mut Ctx`) own an optional database and an optional network until they are
//! closed. Any number of threads may use a handle at once, and close it while others use
//! it; only freeing it must wait until no other call can use it.

use rayon::prelude::*;
use senet_core::board::{Pos, Rules};
use senet_core::bots::{BotContext, BotError, make_bot, search_for};
use senet_core::eval::{Evaluator, heuristic};
use senet_core::game::{analyze_quality, run_match};
use senet_core::index::{MAX_PIECES, index_of, layer_size, position_of};
use senet_core::movegen::{MAX_MOVES, MoveList, gen_moves};
use senet_core::net::{N_INPUTS, dense_features};
use std::cell::RefCell;
use std::ffi::{CStr, c_char};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::Path;
use std::sync::{Arc, Mutex, PoisonError};

const RULES: Rules = Rules::KENDALL5;
const ABI_VERSION: u32 = 6;

pub const INVALID: i32 = -1;
pub const UNAVAILABLE: i32 = -2;
pub const NO_MOVE: i32 = -3;

#[repr(C)]
pub struct FfiMove {
    pub from: u8,
    pub to: u8,
    /// 0 step, 1 swap, 2 off, 3 water
    pub kind: u8,
    pub back: u8,
    pub me_after: u32,
    pub opp_after: u32,
}

// The layout senet.h declares for senet_move.
const _: () = assert!(size_of::<FfiMove>() == 12 && align_of::<FfiMove>() == 4);
const _: () = assert!(std::mem::offset_of!(FfiMove, me_after) == 4 && std::mem::offset_of!(FfiMove, opp_after) == 8);

/// An engine handle: the resources the bots and value functions use, until it is
/// closed. Each call holds its own reference to them, so closing a handle never pulls
/// them away from a call in progress: they are released when the last such call ends.
pub struct Ctx {
    bots: Mutex<Option<Arc<BotContext>>>,
}

thread_local! {
    static LAST_TEXT: RefCell<String> = const { RefCell::new(String::new()) };
}

fn set_text(text: String) {
    LAST_TEXT.with(|t| *t.borrow_mut() = text);
}

/// A failure: the status code to return and the message to store.
struct Failure(i32, String);

fn invalid(msg: impl Into<String>) -> Failure {
    Failure(INVALID, msg.into())
}

fn unavailable(msg: impl Into<String>) -> Failure {
    Failure(UNAVAILABLE, msg.into())
}

impl From<BotError> for Failure {
    fn from(e: BotError) -> Failure {
        match e {
            BotError::Invalid(msg) => invalid(msg),
            BotError::Unavailable(msg) => unavailable(msg),
        }
    }
}

/// Runs `f`, turning a failure or a panic into a stored message and an error value.
fn guard<T>(f: impl FnOnce() -> Result<T, Failure>, on_error: impl FnOnce(i32) -> T) -> T {
    let result = catch_unwind(AssertUnwindSafe(f)).unwrap_or_else(|panic| {
        let msg =
            panic.downcast_ref::<&str>().map(|s| s.to_string()).or_else(|| panic.downcast_ref::<String>().cloned());
        Err(invalid(format!("internal error: {}", msg.unwrap_or_else(|| "panic".into()))))
    });
    result.unwrap_or_else(|Failure(code, msg)| {
        set_text(msg);
        on_error(code)
    })
}

fn value_status(code: i32) -> f64 {
    code as f64
}

/// A valid position: a game in progress or one that a side has just won (not an empty board).
fn position(me: u32, opp: u32) -> Result<Pos, Failure> {
    let pos = Pos::new(me, opp);
    if pos.is_valid() && (me | opp) != 0 {
        Ok(pos)
    } else {
        Err(invalid(format!("invalid position (me {me:#x}, opp {opp:#x})")))
    }
}

/// A valid position in which there are moves to play.
fn game_in_progress(me: u32, opp: u32) -> Result<Pos, Failure> {
    let pos = position(me, opp)?;
    if pos.is_over() { Err(invalid("the game is already over")) } else { Ok(pos) }
}

fn throw(t: u8) -> Result<u8, Failure> {
    if (1..=5).contains(&t) { Ok(t) } else { Err(invalid(format!("throw must be 1..5, not {t}"))) }
}

/// # Safety
/// `p` must be NULL or point to a NUL-terminated string that outlives `'a`.
unsafe fn string<'a>(p: *const c_char, what: &str) -> Result<&'a str, Failure> {
    if p.is_null() {
        return Err(invalid(format!("{what} is NULL")));
    }
    // SAFETY: non-NULL and NUL-terminated by the caller's contract.
    unsafe { CStr::from_ptr(p) }.to_str().map_err(|_| invalid(format!("{what} is not UTF-8")))
}

/// The resources of a handle, which stay alive for the caller even if another thread
/// closes the handle meanwhile.
///
/// # Safety
/// `ctx` must be NULL or a handle from `senet_ctx_new` that has not been freed.
unsafe fn context(ctx: *const Ctx) -> Result<Arc<BotContext>, Failure> {
    // SAFETY: NULL or a live handle by the caller's contract.
    let ctx = unsafe { ctx.as_ref() }.ok_or_else(|| invalid("context is NULL"))?;
    let bots = ctx.bots.lock().unwrap_or_else(PoisonError::into_inner).clone();
    let bots = bots.ok_or_else(|| invalid("the engine is closed"))?;
    #[cfg(test)]
    tests::gate::pass(ctx);
    Ok(bots)
}

/// Fails if `p` is NULL or not aligned for a `T`.
fn aligned<T>(p: *const T, what: &str) -> Result<(), Failure> {
    if p.is_null() {
        Err(invalid(format!("{what} is NULL")))
    } else if !p.is_aligned() {
        Err(invalid(format!("{what} is not aligned for its type")))
    } else {
        Ok(())
    }
}

/// Fails if `n` values of `T` would take more than `isize::MAX` bytes, which no array can.
fn fits<T>(n: usize, what: &str) -> Result<(), Failure> {
    match n.checked_mul(size_of::<T>()) {
        Some(bytes) if bytes <= isize::MAX as usize => Ok(()),
        _ => Err(invalid(format!("{what}: {n} elements are more than any array holds"))),
    }
}

/// # Safety
/// Unless `n` is 0, `p` must be NULL, misaligned or valid for reading `n` values of `T`.
unsafe fn slice<'a, T>(p: *const T, n: usize, what: &str) -> Result<&'a [T], Failure> {
    if n == 0 {
        return Ok(&[]);
    }
    aligned(p, what)?;
    fits::<T>(n, what)?;
    // SAFETY: aligned, non-NULL and readable for `n` values (at most isize::MAX bytes) by
    // the caller's contract.
    Ok(unsafe { std::slice::from_raw_parts(p, n) })
}

/// # Safety
/// Unless `n` is 0, `p` must be NULL, misaligned or valid for writing `n` values of `T`,
/// and not aliased by any other argument.
unsafe fn slice_mut<'a, T>(p: *mut T, n: usize, what: &str) -> Result<&'a mut [T], Failure> {
    if n == 0 {
        return Ok(&mut []);
    }
    aligned(p, what)?;
    fits::<T>(n, what)?;
    // SAFETY: aligned, non-NULL, writable for `n` values (at most isize::MAX bytes) and
    // unaliased by the caller's contract.
    Ok(unsafe { std::slice::from_raw_parts_mut(p, n) })
}

/// Copies the calling thread's text slot (the message of its last failure, or the JSON
/// of its last successful `senet_match`, `senet_quality` or `senet_build_info`) into
/// `out` as a NUL-terminated string, cut to at most `cap - 1` bytes, never inside a
/// UTF-8 character. Returns the buffer size needed for all of it (its length + 1).
///
/// # Safety
/// `out` must be NULL (to query the size) or valid for writing `cap` bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn senet_last_text(out: *mut c_char, cap: usize) -> usize {
    LAST_TEXT.with(|t| {
        let text = t.borrow();
        if !out.is_null() && cap > 0 {
            let mut n = text.len().min(cap - 1);
            while !text.is_char_boundary(n) {
                n -= 1;
            }
            // SAFETY: `out` is writable for `cap > n` bytes by the caller's contract.
            unsafe {
                std::ptr::copy_nonoverlapping(text.as_ptr(), out.cast::<u8>(), n);
                out.add(n).write(0);
            }
        }
        text.len() + 1
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn senet_abi_version() -> u32 {
    ABI_VERSION
}

/// How the library was built (see `senet_core::build_info`): its version, target and the
/// CPU features it requires. Stores it as JSON (see `senet_last_text`) and returns its length.
#[unsafe(no_mangle)]
pub extern "C" fn senet_build_info() -> i64 {
    guard(|| store_json(serde_json::to_string(&senet_core::build_info())), i64::from)
}

/// Upper bound on the number of legal moves for one throw (the `cap` that always fits).
#[unsafe(no_mangle)]
pub extern "C" fn senet_max_moves() -> u32 {
    MAX_MOVES as u32
}

/// Length of the network's feature vector.
#[unsafe(no_mangle)]
pub extern "C" fn senet_n_features() -> u32 {
    N_INPUTS as u32
}

/// Writes White's view of the opening position. Returns 0.
///
/// # Safety
/// `me` and `opp` must each be NULL, misaligned or valid for writing a `u32`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn senet_start(me: *mut u32, opp: *mut u32) -> i32 {
    guard(
        || {
            aligned(me, "me")?;
            aligned(opp, "opp")?;
            let start = RULES.start();
            // SAFETY: aligned, non-NULL and writable by the caller's contract.
            unsafe { (me.write(start.me), opp.write(start.opp)) };
            Ok(0)
        },
        std::convert::identity,
    )
}

/// Legal moves for throw `t` in a game in progress: writes the first `cap` of them to
/// `out` and returns how many there are (at most `senet_max_moves()`).
///
/// # Safety
/// Unless `cap` is 0, `out` must be NULL, misaligned or valid for writing `cap` moves.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn senet_gen_moves(me: u32, opp: u32, t: u8, out: *mut FfiMove, cap: usize) -> i32 {
    guard(
        || {
            let (pos, t) = (game_in_progress(me, opp)?, throw(t)?);
            // SAFETY: forwarded caller contract.
            let out = unsafe { slice_mut(out, cap, "out")? };
            let mut ml = MoveList::new();
            gen_moves(&RULES, pos, t, &mut ml);
            for (slot, m) in out.iter_mut().zip(ml.iter()) {
                *slot = FfiMove {
                    from: m.from,
                    to: m.to,
                    kind: m.kind as u8,
                    back: m.back as u8,
                    me_after: m.after.me,
                    opp_after: m.after.opp,
                };
            }
            Ok(ml.len() as i32)
        },
        std::convert::identity,
    )
}

/// Writes the database layer `(w, b)` and index of a position with 1..=5 pieces per
/// side. Returns 0.
///
/// # Safety
/// `w`, `b` and `idx` must each be NULL, misaligned or valid for writing their type.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn senet_index_of(me: u32, opp: u32, w: *mut u32, b: *mut u32, idx: *mut u64) -> i32 {
    guard(
        || {
            let pos = position(me, opp)?;
            if pos.is_over() {
                return Err(invalid("a finished game has no index"));
            }
            aligned(w, "w")?;
            aligned(b, "b")?;
            aligned(idx, "idx")?;
            let (lw, lb, i) = index_of(pos);
            // SAFETY: aligned, non-NULL and writable by the caller's contract.
            unsafe { (w.write(lw as u32), b.write(lb as u32), idx.write(i)) };
            Ok(0)
        },
        std::convert::identity,
    )
}

fn layer(w: u32, b: u32) -> Result<(usize, usize), Failure> {
    let range = 1..=MAX_PIECES as u32;
    if range.contains(&w) && range.contains(&b) {
        Ok((w as usize, b as usize))
    } else {
        Err(invalid(format!("no layer ({w}, {b}); layers have 1..={MAX_PIECES} pieces per side")))
    }
}

/// Writes the position with index `idx` in layer `(w, b)`. Returns 0.
///
/// # Safety
/// `me` and `opp` must each be NULL, misaligned or valid for writing a `u32`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn senet_position_of(w: u32, b: u32, idx: u64, me: *mut u32, opp: *mut u32) -> i32 {
    guard(
        || {
            let (w, b) = layer(w, b)?;
            if idx >= layer_size(w, b) {
                return Err(invalid(format!("index {idx} is past the end of layer ({w}, {b})")));
            }
            aligned(me, "me")?;
            aligned(opp, "opp")?;
            let pos = position_of(w, b, idx);
            // SAFETY: aligned, non-NULL and writable by the caller's contract.
            unsafe { (me.write(pos.me), opp.write(pos.opp)) };
            Ok(0)
        },
        std::convert::identity,
    )
}

/// Number of positions in layer `(w, b)`, or 0 if there is no such layer.
#[unsafe(no_mangle)]
pub extern "C" fn senet_layer_size(w: u32, b: u32) -> u64 {
    layer(w, b).map(|(w, b)| layer_size(w, b)).unwrap_or(0)
}

/// The hand-crafted heuristic's estimate of P(player to throw wins).
#[unsafe(no_mangle)]
pub extern "C" fn senet_heuristic(me: u32, opp: u32) -> f64 {
    guard(|| Ok(heuristic(position(me, opp)?)), value_status)
}

/// Writes the network's `senet_n_features()` input features for a position. Returns 0.
///
/// # Safety
/// `out` must be NULL, misaligned or valid for writing `senet_n_features()` floats.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn senet_features(me: u32, opp: u32, out: *mut f32) -> i32 {
    guard(
        || {
            let features = dense_features(position(me, opp)?);
            // SAFETY: forwarded caller contract.
            unsafe { slice_mut(out, N_INPUTS, "out")? }.copy_from_slice(&features);
            Ok(0)
        },
        std::convert::identity,
    )
}

/// Creates a handle owning the database in `db_dir` and the network in `net_path`, either
/// of which may be NULL. Returns NULL on failure (see `senet_last_text`).
///
/// # Safety
/// `db_dir` and `net_path` must each be NULL or a NUL-terminated string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn senet_ctx_new(db_dir: *const c_char, net_path: *const c_char) -> *mut Ctx {
    guard(
        || {
            let missing = senet_core::missing_cpu_features();
            if !missing.is_empty() {
                let missing = missing.join(", ");
                return Err(unavailable(format!("this build needs CPU features that this CPU lacks: {missing}")));
            }
            // SAFETY: forwarded caller contract.
            let path = |p: *const c_char, what| unsafe { (!p.is_null()).then(|| string(p, what)).transpose() };
            let (db, net) = (path(db_dir, "db_dir")?, path(net_path, "net_path")?);
            let bots = BotContext::load(db.map(Path::new), net.map(Path::new)).map_err(invalid)?;
            Ok(Box::into_raw(Box::new(Ctx { bots: Mutex::new(Some(Arc::new(bots))) })))
        },
        |_| std::ptr::null_mut(),
    )
}

/// Closes a handle: its database and network are released once the calls using them
/// have finished, and later calls fail with `INVALID`. Closing it again does nothing.
///
/// # Safety
/// `ctx` must be NULL or a handle from `senet_ctx_new` that has not been freed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn senet_ctx_close(ctx: *const Ctx) {
    // SAFETY: NULL or a live handle by the caller's contract.
    if let Some(ctx) = unsafe { ctx.as_ref() } {
        let bots = ctx.bots.lock().unwrap_or_else(PoisonError::into_inner).take();
        drop(bots); // after the lock is released: unmapping a database takes a moment
    }
}

/// Frees a handle, closing it first.
///
/// # Safety
/// `ctx` must be NULL or a handle from `senet_ctx_new` that has not been freed, and no
/// other call may be using it or use it afterwards.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn senet_ctx_free(ctx: *mut Ctx) {
    if !ctx.is_null() {
        // SAFETY: a live, unshared handle created by `Box::into_raw`.
        drop(unsafe { Box::from_raw(ctx) });
    }
}

/// Perfect-play value of a position from the database.
///
/// # Safety
/// `ctx` must be NULL or a live handle.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn senet_db_value(ctx: *const Ctx, me: u32, opp: u32) -> f64 {
    guard(
        || {
            // SAFETY: forwarded caller contract.
            let bots = unsafe { context(ctx)? };
            let pos = position(me, opp)?;
            let db = bots.db.as_ref().ok_or_else(|| unavailable("no database loaded"))?;
            db.lookup(pos).ok_or_else(|| unavailable("position not in the database"))
        },
        value_status,
    )
}

/// Perfect-play values of `n` positions (in parallel), each -1 if the position is
/// invalid or -2 if the database does not cover it. Returns 0.
///
/// # Safety
/// `ctx` must be NULL or a live handle; unless `n` is 0, `me` and `opp` must each be
/// NULL, misaligned or valid for reading `n` masks and `out` NULL, misaligned or valid
/// for writing `n` floats.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn senet_db_values(
    ctx: *const Ctx,
    me: *const u32,
    opp: *const u32,
    n: usize,
    out: *mut f32,
) -> i32 {
    guard(
        || {
            // SAFETY: forwarded caller contract.
            let (bots, me, opp, out) =
                unsafe { (context(ctx)?, slice(me, n, "me")?, slice(opp, n, "opp")?, slice_mut(out, n, "out")?) };
            let db = bots.db.as_ref().ok_or_else(|| unavailable("no database loaded"))?;
            out.par_iter_mut().zip(me.par_iter().zip(opp)).for_each(|(x, (&me, &opp))| {
                *x = match position(me, opp) {
                    Err(_) => INVALID as f32,
                    Ok(pos) => db.lookup(pos).map_or(UNAVAILABLE as f32, |v| v as f32),
                }
            });
            Ok(0)
        },
        std::convert::identity,
    )
}

/// The network's estimate of P(player to throw wins).
///
/// # Safety
/// `ctx` must be NULL or a live handle.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn senet_net_value(ctx: *const Ctx, me: u32, opp: u32) -> f64 {
    guard(
        || {
            // SAFETY: forwarded caller contract.
            let bots = unsafe { context(ctx)? };
            let pos = position(me, opp)?;
            Ok(bots.net.as_ref().ok_or_else(|| unavailable("no network loaded"))?.value(pos))
        },
        value_status,
    )
}

/// The mover's win probability after each legal move for throw `t` in a game in
/// progress, as judged by the bot `spec` (with its search depth). Writes the first `cap`
/// values to `out` and returns how many legal moves there are.
///
/// # Safety
/// `ctx` must be NULL or a live handle, `spec` NULL or a NUL-terminated string, and,
/// unless `cap` is 0, `out` NULL, misaligned or valid for writing `cap` doubles.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn senet_move_values(
    ctx: *const Ctx,
    spec: *const c_char,
    me: u32,
    opp: u32,
    t: u8,
    out: *mut f64,
    cap: usize,
) -> i32 {
    guard(
        || {
            // SAFETY: forwarded caller contract.
            let (bots, spec, out) = unsafe { (context(ctx)?, string(spec, "spec")?, slice_mut(out, cap, "out")?) };
            let (pos, t) = (game_in_progress(me, opp)?, throw(t)?);
            let search =
                search_for(spec, &bots)?.ok_or_else(|| invalid(format!("'{spec}' does not evaluate moves")))?;
            let mut ml = MoveList::new();
            gen_moves(&RULES, pos, t, &mut ml);
            for (slot, v) in out.iter_mut().zip(search.move_values(&RULES, t, &ml)) {
                *slot = v;
            }
            Ok(ml.len() as i32)
        },
        std::convert::identity,
    )
}

/// Index (into the legal moves) of the move the bot `spec` plays for throw `t` in a game
/// in progress, or `NO_MOVE` if there is no legal move.
///
/// # Safety
/// `ctx` must be NULL or a live handle and `spec` NULL or a NUL-terminated string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn senet_bot_choose(
    ctx: *const Ctx,
    spec: *const c_char,
    me: u32,
    opp: u32,
    t: u8,
    seed: u64,
) -> i32 {
    guard(
        || {
            // SAFETY: forwarded caller contract.
            let (bots, spec) = unsafe { (context(ctx)?, string(spec, "spec")?) };
            let (pos, t) = (game_in_progress(me, opp)?, throw(t)?);
            let mut bot = make_bot(spec, &bots, seed)?;
            let mut ml = MoveList::new();
            gen_moves(&RULES, pos, t, &mut ml);
            if ml.is_empty() {
                return Err(Failure(NO_MOVE, "no legal move".into()));
            }
            Ok(bot.choose(&RULES, pos, t, &ml) as i32)
        },
        std::convert::identity,
    )
}

/// Stores serialized JSON in the text slot and returns its length.
fn store_json(json: serde_json::Result<String>) -> Result<i64, Failure> {
    let json = json.map_err(|e| invalid(e.to_string()))?;
    let len = json.len() as i64;
    set_text(json);
    Ok(len)
}

/// Plays `pairs` pairs of games between bots `a` and `b` (same dice, colours swapped).
/// Stores the result as JSON (see `senet_last_text`) and returns its length.
///
/// # Safety
/// `ctx` must be NULL or a live handle; `a` and `b` NULL or NUL-terminated strings.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn senet_match(
    ctx: *const Ctx,
    a: *const c_char,
    b: *const c_char,
    pairs: u64,
    seed: u64,
) -> i64 {
    guard(
        || {
            // SAFETY: forwarded caller contract.
            let (bots, a, b) = unsafe { (context(ctx)?, string(a, "a")?, string(b, "b")?) };
            store_json(serde_json::to_string(&run_match(&RULES, a, b, &bots, pairs, seed)?))
        },
        i64::from,
    )
}

/// How much win probability bot `spec` gives away per decision compared with perfect
/// play, over `games` (at least 1) games against `vs`. Stores the result as JSON (see
/// `senet_last_text`) and returns its length.
///
/// # Safety
/// `ctx` must be NULL or a live handle; `spec` and `vs` NULL or NUL-terminated strings.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn senet_quality(
    ctx: *const Ctx,
    spec: *const c_char,
    vs: *const c_char,
    games: u64,
    seed: u64,
) -> i64 {
    guard(
        || {
            // SAFETY: forwarded caller contract.
            let (bots, spec, vs) = unsafe { (context(ctx)?, string(spec, "spec")?, string(vs, "vs")?) };
            if games == 0 {
                return Err(invalid("a quality analysis needs at least one game"));
            }
            store_json(serde_json::to_string(&analyze_quality(&RULES, spec, vs, &bots, games, seed)?))
        },
        i64::from,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use senet_core::rng::Rng;
    use std::ptr::{null, null_mut};

    /// Lets a test pause a call on a handle once it holds the handle's resources.
    pub(super) mod gate {
        use super::Ctx;
        use std::sync::{Arc, Barrier, Mutex};

        static ARMED: Mutex<Option<(usize, Arc<Barrier>)>> = Mutex::new(None);

        /// The next call on `ctx` to take its resources waits on the barrier twice: when
        /// it holds them, and before it goes on.
        pub fn arm(ctx: *const Ctx) -> Arc<Barrier> {
            let barrier = Arc::new(Barrier::new(2));
            *ARMED.lock().unwrap() = Some((ctx as usize, barrier.clone()));
            barrier
        }

        pub fn pass(ctx: &Ctx) {
            let mut armed = ARMED.lock().unwrap();
            if armed.as_ref().is_some_and(|(armed_ctx, _)| *armed_ctx == ctx as *const Ctx as usize) {
                let (_, barrier) = armed.take().unwrap();
                drop(armed);
                barrier.wait();
                barrier.wait();
            }
        }
    }

    fn last_text() -> String {
        let mut buf = vec![0 as c_char; 256];
        let n = unsafe { senet_last_text(buf.as_mut_ptr(), buf.len()) };
        assert!(n <= buf.len());
        unsafe { CStr::from_ptr(buf.as_ptr()) }.to_str().unwrap().to_string()
    }

    #[test]
    fn invalid_arguments_are_reported() {
        let (mut me, mut opp) = (0u32, 0u32);
        assert_eq!(unsafe { senet_start(&mut me, &mut opp) }, 0);
        let n = senet_core::movegen::legal_moves(&RULES, Pos::new(me, opp), 2).len() as i32;
        assert_eq!(unsafe { senet_gen_moves(me, opp, 2, null_mut(), 0) }, n);
        assert_eq!(unsafe { senet_gen_moves(me, opp, 2, null_mut(), 1) }, INVALID);
        assert_eq!(unsafe { senet_gen_moves(me, opp, 6, null_mut(), 0) }, INVALID);
        assert!(last_text().contains("throw"));
        assert_eq!(unsafe { senet_gen_moves(1 << 27, opp, 1, null_mut(), 0) }, INVALID);
        assert_eq!(senet_heuristic(me, me), -1.0);
        // An empty board is no position; a side with no piece left has won.
        assert_eq!(senet_heuristic(0, 0), -1.0);
        assert_eq!(unsafe { senet_features(0, 0, [0f32; N_INPUTS].as_mut_ptr()) }, INVALID);
        assert_eq!(senet_heuristic(0, 1 << 5), 1.0);
        assert_eq!(unsafe { senet_position_of(0, 1, 0, &mut me, &mut opp) }, INVALID);
        assert_eq!(unsafe { senet_position_of(1, 1, 812, &mut me, &mut opp) }, INVALID);
        assert_eq!(senet_layer_size(6, 1), 0);
        assert_eq!(unsafe { senet_db_value(null(), 2, 4) }, -1.0);
        assert!(last_text().contains("NULL"));
        // A finished game has no moves (the opponent has borne off every piece).
        assert_eq!(unsafe { senet_gen_moves(1 << 30, 0, 1, null_mut(), 0) }, INVALID);
        assert!(last_text().contains("the game is already over"));
    }

    /// Masks near the edge of validity: a valid position, possibly with a bit flipped, a
    /// square outside the board (0, 27 or 31) or one shared with the opponent, a side with
    /// no pieces, or random words.
    fn edge_masks(rng: &mut Rng) -> (u32, u32) {
        let (w, b) = (1 + rng.below(5) as usize, 1 + rng.below(5) as usize);
        let pos = position_of(w, b, rng.below(layer_size(w, b)));
        let (mut me, mut opp) = (pos.me, pos.opp);
        match rng.below(7) {
            0 | 1 => {}
            2 => me ^= 1 << rng.below(32),
            3 => opp |= 1 << [0, 27, 31][rng.below(3) as usize],
            4 => me |= opp & opp.wrapping_neg(),
            5 if rng.below(2) == 0 => me = 0,
            5 => opp = 0,
            _ => (me, opp) = (rng.next_u32(), rng.next_u32()),
        }
        (me, opp)
    }

    #[test]
    fn random_scalar_arguments_are_answered_or_refused() {
        let blank = || FfiMove { from: 0, to: 0, kind: 0, back: 0, me_after: 0, opp_after: 0 };
        let mut out: Vec<FfiMove> = (0..MAX_MOVES).map(|_| blank()).collect();
        let mut rng = Rng::new(9);
        for _ in 0..20_000 {
            let (me, opp) = edge_masks(&mut rng);
            let pos = Pos::new(me, opp);
            let valid = pos.is_valid() && (me | opp) != 0;
            let playing = valid && !pos.is_over();
            let h = senet_heuristic(me, opp);
            assert!(if valid { (0.0..=1.0).contains(&h) } else { h == INVALID as f64 }, "{me:#x} {opp:#x}: {h}");
            let mut features = [f32::NAN; N_INPUTS];
            let status = unsafe { senet_features(me, opp, features.as_mut_ptr()) };
            assert_eq!(status == 0, valid);
            assert!(!valid || features.iter().all(|x| x.is_finite()));

            let t = [0, 1, 2, 3, 4, 5, 6, 255][rng.below(8) as usize];
            let n = unsafe { senet_gen_moves(me, opp, t, out.as_mut_ptr(), out.len()) };
            if playing && (1..=5).contains(&t) {
                assert!((0..=MAX_MOVES as i32).contains(&n));
                for m in &out[..n as usize] {
                    assert!(Pos::new(m.me_after, m.opp_after).is_valid() && m.kind <= 3 && m.back <= 1);
                }
            } else {
                assert_eq!(n, INVALID, "{me:#x} {opp:#x} throw {t}");
            }

            // Ranking a position in progress and unranking the result give it back.
            let (mut w, mut b, mut idx) = (0, 0, 0);
            let status = unsafe { senet_index_of(me, opp, &mut w, &mut b, &mut idx) };
            if playing {
                assert_eq!(status, 0);
                assert!(idx < senet_layer_size(w, b));
                let (mut me2, mut opp2) = (0, 0);
                assert_eq!(unsafe { senet_position_of(w, b, idx, &mut me2, &mut opp2) }, 0);
                assert_eq!((me2, opp2), (me, opp));
            } else {
                assert_eq!(status, INVALID);
            }
        }
    }

    #[test]
    fn layer_and_index_boundaries() {
        let pieces = [0, 1, 2, 5, 6, u32::MAX];
        for (w, b) in pieces.iter().flat_map(|&w| pieces.map(|b| (w, b))) {
            let size = senet_layer_size(w, b);
            let exists = (1..=5).contains(&w) && (1..=5).contains(&b);
            assert_eq!(size > 0, exists, "layer ({w}, {b})");
            for idx in [0, size.saturating_sub(1), size, u64::MAX] {
                let (mut me, mut opp) = (0, 0);
                let status = unsafe { senet_position_of(w, b, idx, &mut me, &mut opp) };
                if exists && idx < size {
                    assert_eq!(status, 0);
                    let (mut w2, mut b2, mut idx2) = (0, 0, 0);
                    assert_eq!(unsafe { senet_index_of(me, opp, &mut w2, &mut b2, &mut idx2) }, 0);
                    assert_eq!((w2, b2, idx2), (w, b, idx));
                } else {
                    assert_eq!(status, INVALID, "layer ({w}, {b}) index {idx}");
                }
            }
        }
    }

    #[test]
    fn the_text_is_cut_to_any_buffer() {
        assert_eq!(senet_heuristic(1, 1), INVALID as f64);
        let text = last_text();
        assert!(text.starts_with("invalid position"));
        let need = unsafe { senet_last_text(null_mut(), 0) };
        assert_eq!(need, text.len() + 1);
        assert_eq!(unsafe { senet_last_text(null_mut(), 8) }, need);
        for cap in 0..need + 2 {
            let mut buf = vec![b'#' as c_char; cap + 1]; // one byte more, which must stay untouched
            assert_eq!(unsafe { senet_last_text(buf.as_mut_ptr(), cap) }, need);
            if cap > 0 {
                let n = text.len().min(cap - 1);
                let written: Vec<u8> = buf[..n].iter().map(|&c| c as u8).collect();
                assert_eq!(written, text.as_bytes()[..n]);
                assert_eq!(buf[n], 0);
            }
            assert_eq!(buf[cap], b'#' as c_char);
        }
    }

    #[test]
    fn the_text_is_never_cut_inside_a_character() {
        let ctx = unsafe { senet_ctx_new(null(), null()) };
        assert_eq!(unsafe { senet_bot_choose(ctx, c"Ré".as_ptr(), 2, 4, 1, 0) }, INVALID);
        unsafe { senet_ctx_free(ctx) };
        let text = last_text();
        let at = text.find('é').unwrap();
        for cap in [at + 1, at + 2, at + 3] {
            let mut buf = vec![1 as c_char; cap];
            unsafe { senet_last_text(buf.as_mut_ptr(), cap) };
            let cut = unsafe { CStr::from_ptr(buf.as_ptr()) }.to_str().expect("UTF-8");
            assert_eq!(cut, &text[..if cap == at + 3 { at + 2 } else { at }]);
        }
    }

    #[test]
    fn a_failed_call_writes_no_output() {
        let mut slots = [7u32; 4];
        let misaligned = unsafe { slots.as_mut_ptr().cast::<u8>().add(1) }.cast::<u32>();
        let (mut me, mut w, mut b) = (7u32, 7u32, 7u32);
        assert_eq!(unsafe { senet_start(&mut me, misaligned) }, INVALID);
        assert_eq!(unsafe { senet_position_of(1, 1, 0, &mut me, misaligned) }, INVALID);
        assert_eq!(me, 7);
        let mut idx = [7u64; 2];
        let misaligned_idx = unsafe { idx.as_mut_ptr().cast::<u8>().add(4) }.cast::<u64>();
        assert_eq!(unsafe { senet_index_of(2, 4, &mut w, &mut b, misaligned_idx) }, INVALID);
        assert!(last_text().contains("idx is not aligned"));
        assert_eq!((w, b), (7, 7));
        assert_eq!(idx, [7, 7]);
    }

    #[test]
    fn arrays_larger_than_memory_are_refused() {
        let ctx = unsafe { senet_ctx_new(null(), null()) };
        let (masks, mut out) = ([2u32; 1], [0f32; 1]);
        let n = usize::MAX / 2;
        assert_eq!(unsafe { senet_db_values(ctx, masks.as_ptr(), masks.as_ptr(), n, out.as_mut_ptr()) }, INVALID);
        assert!(last_text().contains("more than any array holds"), "{}", last_text());
        let mut values = [0f64; 1];
        let n = usize::MAX / 8 + 1;
        assert_eq!(unsafe { senet_move_values(ctx, c"greedy".as_ptr(), 2, 4, 1, values.as_mut_ptr(), n) }, INVALID);
        unsafe { senet_ctx_free(ctx) };
    }

    #[test]
    fn misaligned_pointers_are_refused() {
        let ctx = unsafe { senet_ctx_new(null(), null()) };
        let masks = [0u32; 3];
        let misaligned = unsafe { masks.as_ptr().cast::<u8>().add(1) }.cast::<u32>();
        let mut out = [0f32; 2];
        assert_eq!(unsafe { senet_db_values(ctx, misaligned, masks.as_ptr(), 2, out.as_mut_ptr()) }, INVALID);
        assert!(last_text().contains("me is not aligned"));
        let (mut me, mut opp) = (0u32, 0u32);
        assert_eq!(unsafe { senet_start(misaligned.cast_mut(), &mut opp) }, INVALID);
        assert_eq!(unsafe { senet_start(&mut me, &mut opp) }, 0);
        unsafe { senet_ctx_free(ctx) };
    }

    #[test]
    fn a_handle_closed_while_in_use() {
        let ctx = unsafe { senet_ctx_new(null(), null()) };
        assert!(!ctx.is_null());
        let handle = ctx as usize; // raw pointers are not Send
        let run =
            move |pairs| unsafe { senet_match(handle as *const Ctx, c"greedy".as_ptr(), c"random".as_ptr(), pairs, 1) };
        let gate = gate::arm(ctx);
        let running = std::thread::spawn(move || run(200));
        gate.wait(); // the match holds the handle's resources
        unsafe { senet_ctx_close(ctx) };
        unsafe { senet_ctx_close(ctx) };
        gate.wait(); // and now plays on
        assert!(running.join().unwrap() > 0, "the match in progress outlives the close and finishes");
        assert_eq!(run(1), INVALID as i64);
        assert!(last_text().contains("closed"));
        unsafe { senet_ctx_free(ctx) };
    }

    #[test]
    fn a_context_without_resources() {
        let ctx = unsafe { senet_ctx_new(null(), null()) };
        assert!(!ctx.is_null());
        let (me, opp) = (1 << 30, 1 << 1);
        assert_eq!(unsafe { senet_db_value(ctx, me, opp) }, -2.0);
        assert_eq!(unsafe { senet_db_values(ctx, null(), null(), 0, null_mut()) }, UNAVAILABLE);
        assert_eq!(unsafe { senet_net_value(ctx, me, opp) }, -2.0);
        let mut values = [0f64; MAX_MOVES];
        let greedy = c"greedy".as_ptr();
        assert_eq!(unsafe { senet_move_values(ctx, greedy, me, opp, 1, values.as_mut_ptr(), MAX_MOVES) }, 1);
        assert_eq!(values[0], 1.0, "bearing off the last piece wins");
        assert_eq!(unsafe { senet_move_values(ctx, c"random".as_ptr(), me, opp, 1, values.as_mut_ptr(), 8) }, INVALID);
        assert_eq!(unsafe { senet_bot_choose(ctx, c"nope".as_ptr(), me, opp, 1, 0) }, INVALID);
        assert!(last_text().contains("unknown bot"));
        assert_eq!(unsafe { senet_bot_choose(ctx, c"perfect".as_ptr(), me, opp, 1, 0) }, UNAVAILABLE);
        assert_eq!(unsafe { senet_bot_choose(ctx, greedy, me, opp, 2, 0) }, NO_MOVE);
        let n = unsafe { senet_match(ctx, greedy, c"random".as_ptr(), 5, 1) };
        assert!(n > 0 && last_text().starts_with("{\"bot_a\":\"greedy\""));
        assert_eq!(unsafe { senet_match(ctx, greedy, c"net".as_ptr(), 1, 1) }, UNAVAILABLE as i64);
        assert_eq!(unsafe { senet_quality(ctx, greedy, greedy, 1, 1) }, UNAVAILABLE as i64);
        assert_eq!(unsafe { senet_quality(ctx, greedy, greedy, 0, 1) }, INVALID as i64);
        assert!(last_text().contains("at least one game"));
        assert_eq!(unsafe { senet_match(ctx, greedy, greedy, 0, 1) }, INVALID as i64);
        unsafe { senet_ctx_free(ctx) };
        assert!(unsafe { senet_ctx_new(c"no/such/db".as_ptr(), null()) }.is_null());
        assert!(last_text().contains("no/such/db"));
    }
}
