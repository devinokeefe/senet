//! Legal move generation (see RULES.md, "Legal moves for a throw").

use crate::board::{BEAUTY, MAX_PIECES, OFF, Pos, REBIRTH, Rules, WATER};
use std::ops::Deref;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Kind {
    /// Move to an empty square.
    Step = 0,
    /// Swap places with an unprotected enemy piece.
    Swap = 1,
    /// Bear the piece off the board.
    Off = 2,
    /// Landed on the House of Water and was sent back to 15 (or below).
    Water = 3,
}

impl Kind {
    pub fn name(self) -> &'static str {
        match self {
            Kind::Step => "move",
            Kind::Swap => "swap",
            Kind::Off => "off",
            Kind::Water => "water",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Move {
    pub from: u8,
    /// Final resting square (31 = borne off; for `Water`, the square sent back to).
    pub to: u8,
    pub kind: Kind,
    /// A backward move (only allowed when no forward move exists).
    pub back: bool,
    /// Position after the move, still from the mover's point of view.
    pub after: Pos,
}

/// What happens after a move (RULES.md, "Turn structure").
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Outcome {
    /// The mover bore off their last piece.
    Won,
    /// The throw grants another throw: the mover throws again in this position.
    ThrowAgain(Pos),
    /// The turn passes: the opponent throws in this position (their point of view).
    OpponentThrows(Pos),
}

impl Move {
    #[inline(always)]
    pub fn outcome(&self, rules: &Rules, t: u8) -> Outcome {
        if self.after.me == 0 {
            Outcome::Won
        } else if rules.extra_throw(t) {
            Outcome::ThrowAgain(self.after)
        } else {
            Outcome::OpponentThrows(self.after.flip())
        }
    }

    /// "fwd" or "back", as used in the JSON formats.
    pub fn direction_name(&self) -> &'static str {
        if self.back { "back" } else { "fwd" }
    }
}

/// Upper bound on the number of legal moves for one throw, the capacity of a `MoveList`.
/// A throw has at most one move per piece (`MAX_PIECES`). The bound is 8 all the same: the
/// C ABI exports it (`senet_max_moves`) for callers to size their buffers by.
pub const MAX_MOVES: usize = 8;
const _: () = assert!(MAX_PIECES <= MAX_MOVES);

/// A fixed-capacity list of moves; dereferences to `[Move]`.
#[derive(Clone, Copy)]
pub struct MoveList {
    len: usize,
    moves: [Move; MAX_MOVES],
}

impl MoveList {
    const PLACEHOLDER: Move = Move { from: 0, to: 0, kind: Kind::Step, back: false, after: Pos { me: 0, opp: 0 } };

    #[inline(always)]
    pub fn new() -> MoveList {
        MoveList { len: 0, moves: [Self::PLACEHOLDER; MAX_MOVES] }
    }

    #[inline(always)]
    fn push(&mut self, m: Move) {
        self.moves[self.len] = m;
        self.len += 1;
    }
}

impl Default for MoveList {
    fn default() -> Self {
        MoveList::new()
    }
}

impl Deref for MoveList {
    type Target = [Move];

    #[inline(always)]
    fn deref(&self) -> &[Move] {
        &self.moves[..self.len]
    }
}

/// Bits strictly between squares `lo` and `hi` (requires `lo < hi <= 31`).
#[inline(always)]
pub(crate) fn between(lo: u32, hi: u32) -> u32 {
    let below_hi = ((1u64 << hi) - 1) as u32;
    let upto_lo = ((1u64 << (lo + 1)) - 1) as u32;
    below_hi & !upto_lo
}

/// Squares occupied by enemy pieces that cannot be swapped.
#[inline(always)]
pub fn protected_mask(rules: &Rules, opp: u32) -> u32 {
    if rules.protection { opp & ((opp << 1) | (opp >> 1)) } else { 0 }
}

/// The lowest square of every three consecutive squares in `mask` (a run of four
/// pieces has two).
#[inline(always)]
pub fn run_starts(mask: u32) -> u32 {
    mask & (mask >> 1) & (mask >> 2)
}

/// Squares belonging to an enemy run of three or more.
#[inline(always)]
pub fn blockade_mask(rules: &Rules, opp: u32) -> u32 {
    if rules.blockades {
        let run = run_starts(opp);
        run | (run << 1) | (run << 2)
    } else {
        0
    }
}

/// Where a drowned piece goes: 15, or the highest empty square below it. One exists:
/// besides the drowned piece, at most `2 * MAX_PIECES - 1` pieces occupy squares 1..=15.
#[inline(always)]
fn water_target(occupied: u32) -> u32 {
    const _: () = assert!(2 * MAX_PIECES - 1 < REBIRTH as usize);
    let mut s = REBIRTH;
    while occupied & (1 << s) != 0 {
        s -= 1;
    }
    s
}

/// Resolves the piece on `a` landing on board square `d` (path already checked).
/// Returns `None` if `d` holds an own piece or a protected enemy piece.
#[inline(always)]
fn land(pos: Pos, prot: u32, a: u32, d: u32, back: bool) -> Option<Move> {
    let abit = 1u32 << a;
    let dbit = 1u32 << d;
    if pos.me & dbit != 0 {
        return None;
    }
    let (kind, opp) = if pos.opp & dbit != 0 {
        if prot & dbit != 0 {
            return None;
        }
        (Kind::Swap, pos.opp ^ dbit ^ abit)
    } else {
        (Kind::Step, pos.opp)
    };
    Some(Move { from: a as u8, to: d as u8, kind, back, after: Pos { me: pos.me ^ abit ^ dbit, opp } })
}

#[inline(always)]
fn water_move(pos: Pos, a: u32, back: bool) -> Move {
    let rest = pos.me & !(1 << a);
    let s = water_target(rest | pos.opp);
    Move { from: a as u8, to: s as u8, kind: Kind::Water, back, after: Pos { me: rest | (1 << s), opp: pos.opp } }
}

#[inline(always)]
fn off_move(pos: Pos, a: u32) -> Move {
    Move {
        from: a as u8,
        to: OFF as u8,
        kind: Kind::Off,
        back: false,
        after: Pos { me: pos.me & !(1 << a), opp: pos.opp },
    }
}

/// Generates all legal moves for throw `t` (1..=5) into `out`, sorted by origin square.
/// An empty list means the turn is forfeited. `pos` must be a game in progress: once
/// either side has borne off every piece, nobody moves.
#[inline]
pub fn gen_moves(rules: &Rules, pos: Pos, t: u8, out: &mut MoveList) {
    debug_assert!((1..=5).contains(&t));
    debug_assert!(!pos.is_over(), "{pos:?} is a finished game");
    out.len = 0;
    let t = t as u32;
    let prot = protected_mask(rules, pos.opp);
    let block = blockade_mask(rules, pos.opp);

    // Forward moves.
    let mut bits = pos.me;
    while bits != 0 {
        let a = bits.trailing_zeros();
        bits &= bits - 1;
        if a >= 28 && rules.final_immobile {
            if a + t == OFF {
                out.push(off_move(pos, a));
            }
            continue;
        }
        let d = a + t;
        if (a < BEAUTY && d > BEAUTY) || d > OFF {
            continue;
        }
        if between(a, d) & block != 0 {
            continue;
        }
        if d == OFF {
            out.push(off_move(pos, a));
        } else if d == WATER {
            out.push(water_move(pos, a, false));
        } else if let Some(m) = land(pos, prot, a, d, false) {
            out.push(m);
        }
    }
    if out.len > 0 {
        return;
    }

    // Backward moves, only when no forward move exists.
    let mut bits = pos.me;
    while bits != 0 {
        let a = bits.trailing_zeros();
        bits &= bits - 1;
        if a > BEAUTY && rules.final_immobile {
            continue;
        }
        if a <= t {
            continue;
        }
        let d = a - t;
        if between(d, a) & block != 0 {
            continue;
        }
        if d == WATER {
            out.push(water_move(pos, a, true));
        } else if let Some(m) = land(pos, prot, a, d, true) {
            out.push(m);
        }
    }
}

/// Convenience wrapper returning a `Vec`.
pub fn legal_moves(rules: &Rules, pos: Pos, t: u8) -> Vec<Move> {
    let mut ml = MoveList::new();
    gen_moves(rules, pos, t, &mut ml);
    ml.to_vec()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p(me: &[u32], opp: &[u32]) -> Pos {
        Pos::from_squares(me, opp).unwrap()
    }

    fn moves_with(rules: &Rules, pos: Pos, t: u8) -> Vec<(u8, u8, Kind, bool)> {
        legal_moves(rules, pos, t).iter().map(|m| (m.from, m.to, m.kind, m.back)).collect()
    }

    fn moves(pos: Pos, t: u8) -> Vec<(u8, u8, Kind, bool)> {
        moves_with(&Rules::KENDALL5, pos, t)
    }

    #[test]
    fn simple_step_and_own_piece() {
        assert_eq!(moves(p(&[3, 5], &[20]), 2), vec![(5, 7, Kind::Step, false)]);
    }

    #[test]
    fn swap_lone_enemy_and_protection() {
        let m = legal_moves(&Rules::KENDALL5, p(&[3], &[6]), 3);
        assert_eq!(m.len(), 1);
        assert_eq!(m[0].kind, Kind::Swap);
        assert_eq!(m[0].after, p(&[6], &[3]));
        // Protected by a neighbour on either side (and no backward move from 3 with a 3).
        assert_eq!(moves(p(&[3], &[6, 7]), 3), vec![]);
        assert_eq!(moves(p(&[3], &[5, 6]), 3), vec![]);
    }

    #[test]
    fn blockade_blocks_passing_two_run_does_not() {
        // Enemy run 5,6,7 blocks 4 -> 8.
        let fwd = moves(p(&[4, 20], &[5, 6, 7]), 4);
        assert_eq!(fwd, vec![(20, 24, Kind::Step, false)]);
        // A 2-run can be jumped.
        assert_eq!(moves(p(&[4], &[5, 6]), 3), vec![(4, 7, Kind::Step, false)]);
    }

    #[test]
    fn beauty_must_land_exactly() {
        assert_eq!(moves(p(&[24], &[1]), 3), vec![(24, 21, Kind::Step, true)]);
        assert_eq!(moves(p(&[24], &[1]), 2), vec![(24, 26, Kind::Step, false)]);
    }

    #[test]
    fn water_goes_to_rebirth_or_below() {
        assert_eq!(moves(p(&[26], &[1]), 1), vec![(26, 15, Kind::Water, false)]);
        assert_eq!(moves(p(&[26], &[15]), 1), vec![(26, 14, Kind::Water, false)]);
        assert_eq!(moves(p(&[26, 14], &[15]), 1)[1], (26, 13, Kind::Water, false));
    }

    #[test]
    fn bearing_off() {
        assert_eq!(moves(p(&[26], &[1]), 5), vec![(26, 31, Kind::Off, false)]);
        assert_eq!(moves(p(&[26], &[28, 29, 30]), 5), vec![(26, 21, Kind::Step, true)]);
        assert_eq!(moves(p(&[28], &[1]), 3), vec![(28, 31, Kind::Off, false)]);
        assert_eq!(moves(p(&[29], &[1]), 2), vec![(29, 31, Kind::Off, false)]);
        assert_eq!(moves(p(&[30], &[1]), 1), vec![(30, 31, Kind::Off, false)]);
        // Final squares are immobile otherwise, even backward.
        assert_eq!(moves(p(&[28], &[1]), 1), vec![]);
        assert_eq!(moves(p(&[30], &[1]), 2), vec![]);
    }

    #[test]
    fn backward_swap_sends_enemy_forward() {
        let m = legal_moves(&Rules::KENDALL5, p(&[26], &[24]), 2);
        // 26+2 = 28 is empty: forward move exists, so no backward.
        assert_eq!(m.len(), 1);
        assert_eq!(m[0].to, 28);
        let m = legal_moves(&Rules::KENDALL5, p(&[25], &[23]), 2);
        assert_eq!(m.len(), 1);
        assert!(m[0].back && m[0].kind == Kind::Swap);
        assert_eq!(m[0].after, p(&[23], &[25]));
    }

    #[test]
    fn start_position() {
        let s = Rules::KENDALL5.start();
        assert_eq!(s, p(&[1, 3, 5, 7, 9], &[2, 4, 6, 8, 10]));
    }

    #[test]
    fn outcome_follows_turn_structure() {
        let r = Rules::KENDALL5;
        let last = legal_moves(&r, p(&[30], &[1]), 1)[0];
        assert_eq!(last.outcome(&r, 1), Outcome::Won);
        let m = legal_moves(&r, p(&[3], &[20]), 1)[0];
        assert_eq!(m.outcome(&r, 1), Outcome::ThrowAgain(p(&[4], &[20])));
        let m = legal_moves(&r, p(&[3], &[20]), 2)[0];
        assert_eq!(m.outcome(&r, 2), Outcome::OpponentThrows(p(&[20], &[5])));
    }

    #[test]
    fn rule_switches() {
        let k = Rules::KENDALL5;
        // Without protection a guarded piece can be swapped.
        let open = Rules { protection: false, ..k };
        assert_eq!(moves_with(&open, p(&[3], &[6, 7]), 3), vec![(3, 6, Kind::Swap, false)]);
        // Without blockades a run of three can be passed.
        let open = Rules { blockades: false, ..k };
        assert_eq!(moves_with(&open, p(&[4], &[5, 6, 7]), 4), vec![(4, 8, Kind::Step, false)]);
        // Without immobile final squares, 28 moves like any square past the Beauty.
        let open = Rules { final_immobile: false, ..k };
        assert_eq!(moves_with(&open, p(&[28], &[1]), 1), vec![(28, 29, Kind::Step, false)]);
        assert_eq!(moves_with(&open, p(&[28], &[1]), 3), vec![(28, 31, Kind::Off, false)]);
    }
}
