//! Read-only access to the solved perfect-play database (memory-mapped).
//!
//! The database is one file per layer `(w, b)`, `L{w}{b}.f32`: little-endian `f32`
//! values in index order (see docs/FORMATS.md).

use crate::board::{MAX_PIECES, Pos};
use crate::index::{all_layers, index_of, layer_size};
use crate::invalid_data;
use memmap2::Mmap;
use std::fs::File;
use std::io;
use std::path::{Path, PathBuf};

/// Path of a file belonging to layer `(w, b)`: `L{w}{b}.{ext}` in `dir`.
pub(crate) fn layer_file(dir: &Path, w: usize, b: usize, ext: &str) -> PathBuf {
    dir.join(format!("L{w}{b}.{ext}"))
}

/// Path of the database file for layer `(w, b)`.
pub fn layer_path(dir: &Path, w: usize, b: usize) -> PathBuf {
    layer_file(dir, w, b, "f32")
}

/// Memory-maps a layer file and checks it holds exactly one `f32` per position.
pub(crate) fn map_layer(path: &Path, w: usize, b: usize) -> io::Result<Mmap> {
    let file = File::open(path).map_err(|e| io::Error::new(e.kind(), format!("{}: {e}", path.display())))?;
    // SAFETY: database files are written once (atomically, by the solver) and are not
    // modified afterwards.
    let map = unsafe { Mmap::map(&file)? };
    let expected = layer_size(w, b) * 4;
    if map.len() as u64 != expected {
        return Err(invalid_data(format!("{}: {} bytes, expected {expected}", path.display(), map.len())));
    }
    Ok(map)
}

pub struct Db {
    dir: PathBuf,
    /// `maps[w][b]` for 1 <= w, b <= MAX_PIECES.
    maps: [[Option<Mmap>; MAX_PIECES + 1]; MAX_PIECES + 1],
}

/// The other layers that moves from layer `(w, b)` lead to: the mover may bear a piece
/// off, and either side may throw next. (Bearing off the last piece ends the game.)
fn successor_layers(w: usize, b: usize) -> impl Iterator<Item = (usize, usize)> {
    [(b, w), (w - 1, b), (b, w - 1)].into_iter().filter(|&(x, y)| x > 0 && y > 0)
}

impl Db {
    /// Opens every layer file present in `dir`. Fails if none is present, a file has the
    /// wrong size, or moves from a present layer lead into a missing one. The positions a
    /// database covers are thus closed under moves: valuing the moves of a covered position
    /// (as hints and searches do) never leaves the database.
    pub fn open(dir: &Path) -> io::Result<Db> {
        let mut maps: [[Option<Mmap>; MAX_PIECES + 1]; MAX_PIECES + 1] = Default::default();
        for (w, b) in all_layers() {
            let path = layer_path(dir, w, b);
            if path.exists() {
                maps[w][b] = Some(map_layer(&path, w, b)?);
            }
        }
        if maps.iter().flatten().all(Option::is_none) {
            return Err(io::Error::new(io::ErrorKind::NotFound, format!("no layer files in {}", dir.display())));
        }
        for (w, b) in all_layers().filter(|&(w, b)| maps[w][b].is_some()) {
            if let Some((x, y)) = successor_layers(w, b).find(|&(x, y)| maps[x][y].is_none()) {
                return Err(invalid_data(format!(
                    "L{w}{b}.f32 needs L{x}{y}.f32, where moves from its positions lead"
                )));
            }
        }
        Ok(Db { dir: dir.to_path_buf(), maps })
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// True if every layer of the 5-piece game is present.
    pub fn is_complete(&self) -> bool {
        all_layers().all(|(w, b)| self.maps[w][b].is_some())
    }

    /// Probability that the player to throw wins with perfect play. `None` if the
    /// position is invalid (see `Pos::is_valid`) or its layer is not in the database.
    #[inline]
    pub fn lookup(&self, pos: Pos) -> Option<f64> {
        if !pos.is_valid() {
            return None;
        }
        if let Some(v) = pos.terminal_value() {
            return Some(v);
        }
        let (w, b, i) = index_of(pos);
        let map = self.maps[w][b].as_ref()?;
        let off = i as usize * 4;
        let bytes: [u8; 4] = map[off..off + 4].try_into().expect("4-byte slice");
        Some(f32::from_le_bytes(bytes) as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::{Rules, WATER};
    use crate::index::position_of;
    use crate::movegen::{MoveList, Outcome, gen_moves};
    use crate::solver::{SolveConfig, solve_all};

    #[test]
    fn open_and_query_a_small_database() {
        let dir = std::env::temp_dir().join(format!("senet_db_test_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        solve_all(&Rules::KENDALL5, &dir, &SolveConfig { max_sum: 3, ..SolveConfig::DEFAULT }).unwrap();

        let db = Db::open(&dir).unwrap();
        assert!(!db.is_complete());
        let pos = Pos::from_squares(&[30], &[1]).unwrap();
        let v = db.lookup(pos).unwrap();
        assert!(v > 0.8 && v < 1.0, "V = {v}");
        assert_eq!(db.lookup(Pos::from_squares(&[], &[1]).unwrap()), Some(1.0));
        assert_eq!(db.lookup(Pos::from_squares(&[1], &[]).unwrap()), Some(0.0));
        // Layer (2, 2) was not solved; invalid positions are rejected, never indexed.
        assert_eq!(db.lookup(Pos::from_squares(&[1, 2], &[3, 4]).unwrap()), None);
        assert_eq!(db.lookup(Pos::new(0xFF << 1, 1 << 20)), None);
        assert_eq!(db.lookup(Pos::new(1 << 3, 1 << 3)), None);
        assert_eq!(db.lookup(Pos::new(1 << WATER, 1 << 3)), None);

        assert!(Db::open(&dir.join("missing")).is_err());
        drop(db);
        // Moves from layer (2, 1) lead into (1, 2): without it, (2, 1) cannot be valued.
        std::fs::remove_file(layer_path(&dir, 1, 2)).unwrap();
        let err = Db::open(&dir).err().unwrap().to_string();
        assert!(err.contains("L21.f32 needs L12.f32"), "{err}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn successor_layers_hold_every_move() {
        let r = Rules::KENDALL5;
        let mut moves = MoveList::default();
        for (w, b) in all_layers() {
            let n = layer_size(w, b);
            for i in (0..n).step_by((n / 1000).max(1) as usize) {
                let pos = position_of(w, b, i);
                for t in 1..=5 {
                    gen_moves(&r, pos, t, &mut moves);
                    for m in moves.iter() {
                        let (Outcome::ThrowAgain(next) | Outcome::OpponentThrows(next)) = m.outcome(&r, t) else {
                            continue;
                        };
                        let layer = (next.me.count_ones() as usize, next.opp.count_ones() as usize);
                        assert!(layer == (w, b) || successor_layers(w, b).any(|l| l == layer), "{pos:?} t={t}");
                    }
                }
            }
        }
    }
}
