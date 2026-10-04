//! Board geometry, positions, rules configuration and throw distribution.
//!
//! Squares are numbered 1..=30 and stored as bits of a `u32` (bit `i` = square `i`).
//! "Square 31" denotes a piece that has been borne off. Positions are always stored
//! from the point of view of the player about to throw (`me`) versus the opponent (`opp`).

use std::fmt;

pub const BEAUTY: u32 = 26;
pub const WATER: u32 = 27;
pub const REBIRTH: u32 = 15;
pub const OFF: u32 = 31;

/// Pieces per side; positions are indexed in layers `(w, b)` with `1 <= w, b <= MAX_PIECES`
/// pieces left on the board (crate::index).
pub const MAX_PIECES: usize = 5;

/// Squares 1..=30 except 27 (which is always empty).
pub const USABLE: u32 = 0x7FFF_FFFE & !(1 << WATER);

/// Throw probabilities for throws 1..=5 (index 0 unused).
pub const THROW_PROBS: [f64; 6] = [0.0, 4.0 / 16.0, 6.0 / 16.0, 4.0 / 16.0, 1.0 / 16.0, 1.0 / 16.0];

/// Rule switches. `Rules::KENDALL5` is the normative ruleset (see RULES.md); the
/// solver, the database, the heuristic and the network are only defined for it. The
/// switches exist so that individual rules can be studied in isolation (see the tests).
/// Every ruleset has `MAX_PIECES` pieces per player.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Rules {
    /// Pieces on 28/29/30 may only bear off with the exact throw (3/2/1).
    pub final_immobile: bool,
    /// An enemy piece with an enemy neighbour cannot be swapped.
    pub protection: bool,
    /// Runs of three or more enemy pieces cannot be passed.
    pub blockades: bool,
    /// Bitmask of throws that grant another throw (bit `t` for throw `t`).
    pub extra_throws: u8,
}

impl Rules {
    pub const KENDALL5: Rules =
        Rules { final_immobile: true, protection: true, blockades: true, extra_throws: (1 << 1) | (1 << 4) | (1 << 5) };

    /// Whether throw `t` lets the same player throw again.
    #[inline(always)]
    pub fn extra_throw(&self, t: u8) -> bool {
        self.extra_throws & (1 << t) != 0
    }

    /// White's view of the opening position: White on the odd squares, Black on the even
    /// squares of the first `2 * MAX_PIECES` squares. White moves first.
    pub fn start(&self) -> Pos {
        let mut me = 0;
        let mut opp = 0;
        for i in 0..MAX_PIECES as u32 {
            me |= 1 << (2 * i + 1);
            opp |= 1 << (2 * i + 2);
        }
        Pos { me, opp }
    }
}

impl Default for Rules {
    fn default() -> Self {
        Rules::KENDALL5
    }
}

/// A position from the point of view of the player about to throw.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Pos {
    pub me: u32,
    pub opp: u32,
}

impl Pos {
    #[inline(always)]
    pub fn new(me: u32, opp: u32) -> Pos {
        Pos { me, opp }
    }

    /// Builds a position from square lists, rejecting anything `is_valid` would reject
    /// as well as squares outside 1..=30 and repeated squares.
    pub fn from_squares(me: &[u32], opp: &[u32]) -> Result<Pos, String> {
        let side = |squares: &[u32]| -> Result<u32, String> {
            let mut mask = 0u32;
            for &s in squares {
                let bit = square_bit(s).ok_or_else(|| format!("invalid square {s} (squares are 1..=30, never 27)"))?;
                if mask & bit != 0 {
                    return Err(format!("square {s} listed twice"));
                }
                mask |= bit;
            }
            Ok(mask)
        };
        let pos = Pos { me: side(me)?, opp: side(opp)? };
        if pos.me & pos.opp != 0 {
            return Err("both sides occupy the same square".into());
        }
        if !pos.is_valid() {
            return Err(format!("at most {MAX_PIECES} pieces per side"));
        }
        Ok(pos)
    }

    /// The same board seen by the other player.
    #[inline(always)]
    pub fn flip(self) -> Pos {
        Pos { me: self.opp, opp: self.me }
    }

    /// Checks that the masks only use squares 1..=30 (never 27), do not overlap and do
    /// not exceed `MAX_PIECES` per side.
    #[inline]
    pub fn is_valid(&self) -> bool {
        self.me & !USABLE == 0
            && self.opp & !USABLE == 0
            && self.me & self.opp == 0
            && self.me.count_ones() <= MAX_PIECES as u32
            && self.opp.count_ones() <= MAX_PIECES as u32
    }

    /// True once either side has borne off every piece.
    #[inline(always)]
    pub fn is_over(&self) -> bool {
        self.me == 0 || self.opp == 0
    }

    /// The value of a finished game for the player to throw: 1 if they have borne off
    /// every piece, 0 if the opponent has, `None` while the game is in progress.
    #[inline(always)]
    pub fn terminal_value(&self) -> Option<f64> {
        if self.me == 0 {
            Some(1.0)
        } else if self.opp == 0 {
            Some(0.0)
        } else {
            None
        }
    }
}

impl fmt::Debug for Pos {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let me: Vec<u32> = Bits(self.me).collect();
        let opp: Vec<u32> = Bits(self.opp).collect();
        write!(f, "Pos{{me: {me:?}, opp: {opp:?}}}")
    }
}

/// The bit for square `s`, or `None` if no piece can stand there.
#[inline]
pub fn square_bit(s: u32) -> Option<u32> {
    (s < 32).then(|| 1u32 << s).filter(|bit| bit & USABLE != 0)
}

/// Iterator over the set bits of a mask, lowest first.
#[derive(Clone, Copy)]
pub struct Bits(pub u32);

impl Iterator for Bits {
    type Item = u32;

    #[inline(always)]
    fn next(&mut self) -> Option<u32> {
        if self.0 == 0 {
            None
        } else {
            let i = self.0.trailing_zeros();
            self.0 &= self.0 - 1;
            Some(i)
        }
    }
}

/// Converts four stick faces (bit set = light side up) into a throw value.
#[inline]
pub fn throw_from_sticks(sticks: u8) -> u8 {
    match (sticks & 0xF).count_ones() {
        0 => 5,
        n => n as u8,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_squares_validates() {
        assert_eq!(Pos::from_squares(&[1, 30], &[2]), Ok(Pos::new((1 << 1) | (1 << 30), 1 << 2)));
        assert!(Pos::from_squares(&[27], &[1]).is_err());
        assert!(Pos::from_squares(&[0], &[1]).is_err());
        assert!(Pos::from_squares(&[31], &[1]).is_err());
        assert!(Pos::from_squares(&[40], &[1]).is_err(), "must not wrap to another square");
        assert!(Pos::from_squares(&[3, 3], &[1]).is_err());
        assert!(Pos::from_squares(&[3], &[3]).is_err());
        assert!(Pos::from_squares(&[1, 2, 3, 4, 5, 6], &[7]).is_err());
    }

    #[test]
    fn sticks() {
        assert_eq!(throw_from_sticks(0b0000), 5);
        assert_eq!(throw_from_sticks(0b0101), 2);
        assert_eq!(throw_from_sticks(0b1111), 4);
    }
}
