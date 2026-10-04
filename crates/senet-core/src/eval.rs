//! Position evaluators and expectimax search.
//!
//! An evaluator estimates V(pos) = P(player to throw wins). Decisions are made by
//! valuing the position each legal move leads to (`move_value`), optionally through a
//! depth-limited expectimax over future throws.

use crate::board::{BEAUTY, Bits, MAX_PIECES, Pos, Rules, THROW_PROBS};
use crate::db::Db;
use crate::movegen::{Move, MoveList, Outcome, between, blockade_mask, gen_moves, protected_mask, run_starts};
use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};

pub trait Evaluator: Send + Sync {
    /// Estimated probability that the player about to throw wins.
    fn value(&self, pos: Pos) -> f64;
}

impl Evaluator for Db {
    /// The solved value. The database's users only evaluate positions it covers: perfect
    /// play requires the complete database, and `Db::open` only accepts a partial one
    /// that holds the successors of every position it covers. Anything else falls back to
    /// the heuristic.
    fn value(&self, pos: Pos) -> f64 {
        self.lookup(pos).unwrap_or_else(|| heuristic(pos))
    }
}

/// Hand-crafted evaluation: a logistic function of race progress, safety and blocking.
/// This is the "traditional" baseline the solved database is compared against.
pub struct Heuristic;

impl Evaluator for Heuristic {
    fn value(&self, pos: Pos) -> f64 {
        heuristic(pos)
    }
}

/// Feature score for one side (higher is better for that side).
fn side_score(me: u32, opp: u32) -> f64 {
    let borne_off = MAX_PIECES as f64 - me.count_ones() as f64;
    // Race: distance remaining, with borne-off pieces at distance 0.
    let remaining: f64 = Bits(me)
        .map(|s| match s {
            30 => 1.9, // needs a 1: about 4 throws on average, but no square to travel
            29 => 2.3,
            28 => 2.6,
            _ => 31.0 - s as f64,
        })
        .sum();
    let protected = protected_mask(&Rules::KENDALL5, me);
    // Blocking: own runs of three are walls, pairs are safe.
    let walls = run_starts(me);
    -remaining + 1.5 * borne_off - 2.0 * threat(me, opp)
        + 1.5 * walls.count_ones() as f64
        + 0.4 * protected.count_ones() as f64
}

/// Safety: the unprotected pieces of `me` that an enemy piece 1..5 squares behind could
/// swap, weighted by the throw that does it. No move passes the Beauty (26) or a blockade
/// of `me`, so only a piece on 26 reaches 28..=30.
fn threat(me: u32, opp: u32) -> f64 {
    let blockade = blockade_mask(&Rules::KENDALL5, me);
    Bits(me & !protected_mask(&Rules::KENDALL5, me))
        .flat_map(|s| {
            (1..s.min(6)).filter(move |&d| {
                let a = s - d;
                opp & (1 << a) != 0 && (s <= BEAUTY || a == BEAUTY) && between(a, s) & blockade == 0
            })
        })
        .map(|d| THROW_PROBS[d as usize] * d as f64)
        .sum()
}

/// Heuristic estimate of P(mover wins).
pub fn heuristic(pos: Pos) -> f64 {
    if let Some(v) = pos.terminal_value() {
        return v;
    }
    // The mover is about to throw: worth roughly one average throw (~2.4 squares).
    let s = side_score(pos.me, pos.opp) - side_score(pos.opp, pos.me) + 2.4;
    1.0 / (1.0 + (-s / 9.0).exp())
}

/// Value for the mover of playing `m` with throw `t`: the position it leads to, valued
/// by `eval` through an expectimax search `depth` throws deep (0 = `eval` directly).
/// Panics if `depth` is 16 or more.
pub fn move_value(rules: &Rules, eval: &dyn Evaluator, m: &Move, t: u8, depth: u32) -> f64 {
    Searcher::new(rules, eval, depth).move_value(m, t)
}

/// Expectimax value of `pos` (player about to throw) searching `depth` throws ahead.
/// Panics if `depth` is 16 or more.
pub fn expectimax(rules: &Rules, eval: &dyn Evaluator, pos: Pos, depth: u32) -> f64 {
    Searcher::new(rules, eval, depth).value(pos)
}

/// An expectimax search `depth` throws deep that remembers the value of every position
/// it searches, at each depth, for all the moves of one decision: the same positions
/// recur, such as after two extra throws used in either order.
pub(crate) struct Searcher<'a> {
    rules: &'a Rules,
    eval: &'a dyn Evaluator,
    depth: u32,
    /// Values by `key(pos, depth)`.
    seen: HashMap<u64, f64, BuildHasherDefault<KeyHasher>>,
}

impl<'a> Searcher<'a> {
    pub(crate) fn new(rules: &'a Rules, eval: &'a dyn Evaluator, depth: u32) -> Searcher<'a> {
        assert!(depth < 16, "a search {depth} throws deep would never finish");
        Searcher { rules, eval, depth, seen: HashMap::default() }
    }

    /// The value for the mover of playing `m` with throw `t`.
    pub(crate) fn move_value(&mut self, m: &Move, t: u8) -> f64 {
        self.move_value_at(m, t, self.depth)
    }

    /// The value of `pos` for the player about to throw.
    pub(crate) fn value(&mut self, pos: Pos) -> f64 {
        self.search(pos, self.depth)
    }

    fn move_value_at(&mut self, m: &Move, t: u8, depth: u32) -> f64 {
        match m.outcome(self.rules, t) {
            Outcome::Won => 1.0,
            Outcome::ThrowAgain(pos) => self.search(pos, depth),
            Outcome::OpponentThrows(pos) => 1.0 - self.search(pos, depth),
        }
    }

    fn search(&mut self, pos: Pos, depth: u32) -> f64 {
        if let Some(v) = pos.terminal_value() {
            return v;
        }
        if self.depth == 0 {
            return self.eval.value(pos); // no search: nothing recurs
        }
        // Squares 1..=30 take 30 bits per side, the depth (below 16) the top 4.
        let key = u64::from(pos.me >> 1) | (u64::from(pos.opp >> 1) << 30) | (u64::from(depth) << 60);
        if let Some(&v) = self.seen.get(&key) {
            return v;
        }
        let v = if depth == 0 {
            self.eval.value(pos)
        } else {
            let mut ml = MoveList::new();
            let mut v = 0.0;
            for t in 1..=5u8 {
                gen_moves(self.rules, pos, t, &mut ml);
                let q = if ml.is_empty() {
                    1.0 - self.search(pos.flip(), depth - 1)
                } else {
                    ml.iter().map(|m| self.move_value_at(m, t, depth - 1)).fold(0.0, f64::max)
                };
                v += THROW_PROBS[t as usize] * q;
            }
            v
        };
        self.seen.insert(key, v);
        v
    }
}

/// The hash of the `Searcher`'s keys: a bit mixer, much cheaper than the standard SipHash.
#[derive(Default)]
struct KeyHasher(u64);

impl Hasher for KeyHasher {
    fn finish(&self) -> u64 {
        self.0
    }

    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.write_u64(u64::from(b));
        }
    }

    fn write_u64(&mut self, x: u64) {
        let mut state = self.0 ^ x;
        self.0 = crate::rng::splitmix64(&mut state);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn heuristic_is_a_sensible_probability() {
        let r = Rules::KENDALL5;
        let start = r.start();
        let v = heuristic(start);
        assert!(v > 0.5 && v < 0.6, "the player to throw is a slight favourite: {v}");
        let ahead = Pos::from_squares(&[25, 26, 28], &[1, 2, 3, 4, 5]).unwrap();
        assert!(heuristic(ahead) > 0.9);
        assert!(heuristic(ahead.flip()) < 0.1);
        assert_eq!(heuristic(Pos::from_squares(&[], &[1]).unwrap()), 1.0);
    }

    #[test]
    fn threats_follow_the_rules() {
        let threat = |me: &[u32], opp: &[u32]| {
            let pos = Pos::from_squares(me, opp).unwrap();
            threat(pos.me, pos.opp)
        };
        // From 26, a 2, 3 or 4 swaps a piece on 28, 29 or 30; nothing else reaches them.
        assert_eq!(threat(&[28], &[26]), THROW_PROBS[2] * 2.0);
        assert_eq!(threat(&[30], &[26]), THROW_PROBS[4] * 4.0);
        assert_eq!(threat(&[28], &[25]), 0.0);
        assert_eq!(threat(&[30], &[29]), 0.0);
        assert_eq!(threat(&[26], &[25]), THROW_PROBS[1]);
        // A blockade between attacker and target stops the attack; a protected piece is safe.
        assert_eq!(threat(&[14], &[9]), THROW_PROBS[5] * 5.0);
        assert_eq!(threat(&[10, 11, 12, 14], &[9]), 0.0);
        assert_eq!(threat(&[10, 11], &[9]), 0.0);
    }

    #[test]
    fn search_of_depth_zero_is_the_evaluator() {
        let r = Rules::KENDALL5;
        let pos = r.start();
        assert_eq!(expectimax(&r, &Heuristic, pos, 0), heuristic(pos));
        let v = expectimax(&r, &Heuristic, pos, 2);
        assert!((0.0..=1.0).contains(&v));
        let m = crate::movegen::legal_moves(&r, pos, 2)[0];
        assert_eq!(move_value(&r, &Heuristic, &m, 2, 0), 1.0 - heuristic(m.after.flip()));
    }

    /// Expectimax without remembering positions.
    fn plain_expectimax(r: &Rules, pos: Pos, depth: u32) -> f64 {
        if let Some(v) = pos.terminal_value() {
            return v;
        }
        if depth == 0 {
            return heuristic(pos);
        }
        let value = |m: &Move, t: u8| match m.outcome(r, t) {
            Outcome::Won => 1.0,
            Outcome::ThrowAgain(next) => plain_expectimax(r, next, depth - 1),
            Outcome::OpponentThrows(next) => 1.0 - plain_expectimax(r, next, depth - 1),
        };
        (1..=5u8)
            .map(|t| {
                let moves = crate::movegen::legal_moves(r, pos, t);
                let q = if moves.is_empty() {
                    1.0 - plain_expectimax(r, pos.flip(), depth - 1)
                } else {
                    moves.iter().map(|m| value(m, t)).fold(0.0, f64::max)
                };
                THROW_PROBS[t as usize] * q
            })
            .sum()
    }

    #[test]
    fn remembered_positions_leave_the_values_unchanged() {
        let r = Rules::KENDALL5;
        let mut rng = crate::rng::Rng::new(9);
        for _ in 0..20 {
            let pos = crate::index::random_position(&mut rng);
            let mut search = Searcher::new(&r, &Heuristic, 2);
            for t in 1..=5 {
                for m in crate::movegen::legal_moves(&r, pos, t) {
                    let plain = match m.outcome(&r, t) {
                        Outcome::Won => 1.0,
                        Outcome::ThrowAgain(next) => plain_expectimax(&r, next, 2),
                        Outcome::OpponentThrows(next) => 1.0 - plain_expectimax(&r, next, 2),
                    };
                    assert_eq!(search.move_value(&m, t), plain, "{pos:?} t={t}");
                }
            }
            assert_eq!(expectimax(&r, &Heuristic, pos, 2), plain_expectimax(&r, pos, 2));
        }
    }
}
