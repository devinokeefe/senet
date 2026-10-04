//! Perfect indexing of positions into layers (see docs/FORMATS.md).
//!
//! Only 29 squares can hold a piece (27 is always empty). A position with `w` mover
//! pieces and `b` opponent pieces on the board lives in layer `(w, b)` of size
//! `C(29, w) * C(29 - w, b)`; its index combines the colex rank of the mover's squares
//! with the colex rank of the opponent's squares relative to the squares left free.

pub use crate::board::MAX_PIECES;
use crate::board::Pos;
use crate::rng::Rng;

/// Squares that can hold a piece (1..=30 without 27).
pub const N_USABLE: u32 = 29;
/// All 29 compact squares.
pub const ALL29: u32 = (1 << N_USABLE) - 1;

const fn make_binom() -> [[u64; 8]; 32] {
    let mut t = [[0u64; 8]; 32];
    let mut n = 0;
    while n < 32 {
        t[n][0] = 1;
        if n > 0 {
            let mut k = 1;
            while k < 8 {
                t[n][k] = t[n - 1][k - 1] + t[n - 1][k];
                k += 1;
            }
        }
        n += 1;
    }
    t
}

/// `BINOM[n][k] = C(n, k)` for `n < 32`, `k < 8`.
pub static BINOM: [[u64; 8]; 32] = make_binom();

/// Square mask (bits 1..=30) -> compact 29-bit mask.
#[inline(always)]
pub fn compact(mask: u32) -> u32 {
    ((mask >> 1) & 0x03FF_FFFF) | ((mask >> 2) & (0x7 << 26))
}

/// Compact 29-bit mask -> square mask.
#[inline(always)]
pub fn expand(c: u32) -> u32 {
    ((c & 0x03FF_FFFF) << 1) | ((c >> 26) << 28)
}

/// Parallel bit deposit: the low bits of `x` spread over the set bits of `mask`, lowest
/// first.
#[inline(always)]
pub fn pdep(x: u32, mask: u32) -> u32 {
    #[cfg(all(target_arch = "x86_64", target_feature = "bmi2"))]
    // SAFETY: the target has BMI2 (checked at compile time).
    unsafe {
        std::arch::x86_64::_pdep_u32(x, mask)
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "bmi2")))]
    pdep_portable(x, mask)
}

/// `pdep` without BMI2 (always compiled, so that the tests check it on every machine).
#[cfg_attr(all(target_arch = "x86_64", target_feature = "bmi2"), allow(dead_code))]
#[inline(always)]
fn pdep_portable(x: u32, mask: u32) -> u32 {
    let (mut res, mut m, mut bit) = (0u32, mask, 1u32);
    while m != 0 {
        if x & bit != 0 {
            res |= m & m.wrapping_neg();
        }
        m &= m - 1;
        bit <<= 1;
    }
    res
}

/// Colexicographic rank of a k-subset given as a bitmask.
#[inline(always)]
pub fn colex_rank(mut c: u32) -> u64 {
    let mut r = 0;
    let mut k = 1;
    while c != 0 {
        r += BINOM[c.trailing_zeros() as usize][k];
        k += 1;
        c &= c - 1;
    }
    r
}

/// Colexicographic rank of the squares of `c` among the squares not in `taken` (which must
/// not overlap `c`), numbered from 0: a square's number among them is its own number less
/// the number of taken squares below it.
#[inline(always)]
pub fn colex_rank_among(mut c: u32, taken: u32) -> u64 {
    let mut r = 0;
    let mut k = 1;
    while c != 0 {
        let below = (c & c.wrapping_neg()) - 1;
        r += BINOM[(c.trailing_zeros() - (taken & below).count_ones()) as usize][k];
        k += 1;
        c &= c - 1;
    }
    r
}

/// Inverse of `colex_rank` for k-subsets of an n-element universe.
pub fn colex_unrank(mut r: u64, k: usize, n: u32) -> u32 {
    let mut c = 0u32;
    let mut hi = n as usize;
    for kk in (1..=k).rev() {
        // Largest x < hi with C(x, kk) <= r.
        let mut x = hi - 1;
        while BINOM[x][kk] > r {
            x -= 1;
        }
        c |= 1 << x;
        r -= BINOM[x][kk];
        hi = x;
    }
    c
}

/// Number of positions with `w` mover pieces and `b` opponent pieces on the board.
#[inline(always)]
pub fn layer_size(w: usize, b: usize) -> u64 {
    BINOM[N_USABLE as usize][w] * BINOM[N_USABLE as usize - w][b]
}

/// `(w, b, index)` of a valid position with 1..=`MAX_PIECES` pieces on each side.
#[inline(always)]
pub fn index_of(pos: Pos) -> (usize, usize, u64) {
    let mc = compact(pos.me);
    let oc = compact(pos.opp);
    let w = mc.count_ones() as usize;
    let b = oc.count_ones() as usize;
    debug_assert!((1..=MAX_PIECES).contains(&w) && (1..=MAX_PIECES).contains(&b), "{pos:?}");
    (w, b, index_compact(mc, oc, w, b))
}

/// Index of a position given compact masks and their popcounts.
#[inline(always)]
pub fn index_compact(mc: u32, oc: u32, w: usize, b: usize) -> u64 {
    colex_rank(mc) * BINOM[N_USABLE as usize - w][b] + colex_rank_among(oc, mc)
}

/// Inverse of `index_of` (requires `index < layer_size(w, b)`).
pub fn position_of(w: usize, b: usize, index: u64) -> Pos {
    debug_assert!(index < layer_size(w, b), "index {index} out of range for layer ({w},{b})");
    let k = BINOM[N_USABLE as usize - w][b];
    let mc = colex_unrank(index / k, w, N_USABLE);
    let rel = colex_unrank(index % k, b, N_USABLE - w as u32);
    let oc = pdep(rel, !mc & ALL29);
    Pos { me: expand(mc), opp: expand(oc) }
}

/// Every layer `(w, b)` with `1 <= w, b <= MAX_PIECES`.
pub fn all_layers() -> impl Iterator<Item = (usize, usize)> {
    (1..=MAX_PIECES).flat_map(|w| (1..=MAX_PIECES).map(move |b| (w, b)))
}

/// Total number of positions in all layers (the size of the solved game).
pub fn total_positions() -> u64 {
    all_layers().map(|(w, b)| layer_size(w, b)).sum()
}

/// A random position: a uniformly random layer, then a uniformly random position in it.
pub fn random_position(rng: &mut Rng) -> Pos {
    let w = 1 + rng.below(MAX_PIECES as u64) as usize;
    let b = 1 + rng.below(MAX_PIECES as u64) as usize;
    position_of(w, b, rng.below(layer_size(w, b)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::{Bits, USABLE};

    #[test]
    fn compact_roundtrip() {
        assert_eq!(compact(USABLE), ALL29);
        assert_eq!(expand(ALL29), USABLE);
        for s in [1u32, 15, 26, 28, 29, 30] {
            assert_eq!(expand(compact(1 << s)), 1 << s);
        }
        assert_eq!(compact(1 << 28), 1 << 26);
    }

    /// The bits of `x` selected by `mask`, packed into the low bits (the inverse of `pdep`).
    fn pext(x: u32, mask: u32) -> u32 {
        Bits(mask).enumerate().map(|(k, s)| ((x >> s) & 1) << k).sum()
    }

    #[test]
    fn bit_deposit() {
        assert_eq!(pdep_portable(0b1011, 0b1111_0000), 0b1011_0000);
        assert_eq!(pdep_portable(0b0101, 0b1010_1010), 0b0010_0010);
        let mut rng = Rng::new(11);
        for _ in 0..100_000 {
            let (x, mask) = (rng.next_u32(), rng.next_u32());
            assert_eq!(pdep_portable(pext(x, mask), mask), x & mask);
            // The hardware version, when this build uses it.
            assert_eq!(pdep(x, mask), pdep_portable(x, mask));
        }
    }

    #[test]
    fn rank_among_free_squares() {
        let mut rng = Rng::new(13);
        for _ in 0..1_000_000 {
            let pos = random_position(&mut rng);
            let (mc, oc) = (compact(pos.me), compact(pos.opp));
            assert_eq!(colex_rank_among(oc, mc), colex_rank(pext(oc, !mc & ALL29)), "{pos:?}");
        }
    }

    #[test]
    fn layer_sizes() {
        assert_eq!(layer_size(5, 5), 118_755 * 42_504);
        assert_eq!(layer_size(1, 1), 29 * 28);
        assert_eq!(total_positions(), 8_560_690_670);
    }

    #[test]
    fn index_roundtrip_random() {
        let mut rng = Rng::new(7);
        for _ in 0..200_000 {
            let w = 1 + rng.below(5) as usize;
            let b = 1 + rng.below(5) as usize;
            let idx = rng.below(layer_size(w, b));
            let pos = position_of(w, b, idx);
            assert_eq!(pos.me.count_ones() as usize, w);
            assert_eq!(pos.opp.count_ones() as usize, b);
            assert!(pos.is_valid());
            assert_eq!(index_of(pos), (w, b, idx));
        }
    }

    #[test]
    fn index_is_bijective_small_layer() {
        let n = layer_size(2, 1);
        let mut seen = vec![false; n as usize];
        for i in 0..n {
            let p = position_of(2, 1, i);
            let (_, _, j) = index_of(p);
            assert!(!seen[j as usize]);
            seen[j as usize] = true;
        }
    }
}
