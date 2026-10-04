//! SHA-256, for the integrity manifests of databases, training data and networks (see
//! `manifest`), computed by the `sha2` crate with the CPU's SHA instructions where it has
//! them. Manifests hash their files in parallel.

use sha2::Digest;

/// An incremental SHA-256 hash: `update` with the data in any pieces, then `finish`.
#[derive(Clone, Default)]
pub struct Sha256(sha2::Sha256);

impl Sha256 {
    pub fn new() -> Sha256 {
        Sha256(sha2::Sha256::new())
    }

    pub fn update(&mut self, data: &[u8]) {
        self.0.update(data);
    }

    pub fn finish(self) -> [u8; 32] {
        self.0.finalize().into()
    }
}

/// The digest as 64 lowercase hex digits, as `sha256sum` prints it.
pub fn hex(digest: &[u8; 32]) -> String {
    digest.iter().map(|b| format!("{b:02x}")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Rng;

    fn sha(data: &[u8]) -> String {
        let mut h = Sha256::new();
        h.update(data);
        hex(&h.finish())
    }

    #[test]
    fn standard_test_vectors() {
        // FIPS 180-4 examples and the empty message.
        assert_eq!(sha(b""), "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
        assert_eq!(sha(b"abc"), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
        assert_eq!(
            sha(b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"),
            "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
        );
        assert_eq!(sha(&[b'a'; 1_000_000]), "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0");
    }

    #[test]
    fn the_pieces_do_not_matter() {
        // Every length around the block and padding boundaries, fed whole and in random pieces.
        let mut rng = Rng::new(5);
        let data: Vec<u8> = (0..300).map(|_| rng.below(256) as u8).collect();
        for len in 0..data.len() {
            let whole = sha(&data[..len]);
            let mut h = Sha256::new();
            let mut at = 0;
            while at < len {
                let n = (1 + rng.below(70) as usize).min(len - at);
                h.update(&data[at..at + n]);
                at += n;
            }
            assert_eq!(hex(&h.finish()), whole, "length {len}");
        }
    }
}
