//! Small, fast, seedable PRNG (xoshiro256++), so results are reproducible without
//! external dependencies.

#[derive(Clone, Debug)]
pub struct Rng {
    s: [u64; 4],
}

/// The SplitMix64 generator: advances `x` and returns a well-mixed function of it.
pub(crate) fn splitmix64(x: &mut u64) -> u64 {
    *x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

impl Rng {
    pub fn new(seed: u64) -> Rng {
        let mut x = seed;
        Rng { s: [splitmix64(&mut x), splitmix64(&mut x), splitmix64(&mut x), splitmix64(&mut x)] }
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let s = &mut self.s;
        let result = (s[0].wrapping_add(s[3])).rotate_left(23).wrapping_add(s[0]);
        let t = s[1] << 17;
        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];
        s[2] ^= t;
        s[3] = s[3].rotate_left(45);
        result
    }

    #[inline]
    pub fn next_u32(&mut self) -> u32 {
        (self.next_u64() >> 32) as u32
    }

    /// Integer in `0..n` (n > 0), uniform up to a bias of at most n / 2^64
    /// (multiply-shift reduction, no rejection loop).
    #[inline]
    pub fn below(&mut self, n: u64) -> u64 {
        ((self.next_u64() as u128 * n as u128) >> 64) as u64
    }

    /// Uniform float in [0, 1).
    #[inline]
    pub fn f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// Throws the four sticks (each face up with probability 1/2) and returns the value.
    #[inline]
    pub fn throw(&mut self) -> u8 {
        crate::board::throw_from_sticks((self.next_u64() >> 60) as u8)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn throw_distribution() {
        let mut r = Rng::new(1);
        let mut c = [0u32; 6];
        let n = 1_600_000;
        for _ in 0..n {
            c[r.throw() as usize] += 1;
        }
        for (t, (&count, &p)) in c.iter().zip(&crate::board::THROW_PROBS).enumerate().skip(1) {
            let expected = p * n as f64;
            assert!((count as f64 - expected).abs() < 5.0 * expected.sqrt(), "throw {t}: {count} vs {expected}");
        }
    }
}
