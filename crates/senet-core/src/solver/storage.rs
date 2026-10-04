//! Value storage for the solver: layers being solved (in memory, updated by many threads
//! at once) and solved layers (memory-mapped files), plus the solver's own files.
//!
//! Files in the database directory, all little-endian (see docs/FORMATS.md):
//! - `L{w}{b}.f32`       a solved layer, one `f32` per position: the database itself
//! - `L{w}{b}.f32.ckpt`  checkpoint of a layer being solved with `f32` storage
//! - `L{w}{b}.u24.ckpt`  checkpoint of a layer being solved with 24-bit storage
//! - `L{w}{b}.u24`       24-bit copy of a solved layer, input to a compact group
//! - `*.tmp`             a file being written; renamed into place once complete
//! - `solve.lock`        present while a solve uses the directory

use crate::atomic::write_atomically;
use crate::board::MAX_PIECES;
use crate::db::{layer_file, layer_path, map_layer};
use crate::index::all_layers;
use crate::invalid_data;
use memmap2::Mmap;
use rayon::prelude::*;
use std::fs::{File, OpenOptions};
use std::io::{self, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU8, AtomicU32, Ordering::Relaxed};

/// 24-bit fixed point stores `round(v * (2^24 - 1))`: the same absolute resolution as
/// `f32` on [0.5, 1).
const SCALE24: f64 = 16_777_215.0;

/// Positions converted per chunk when streaming a layer to or from disk.
const CHUNK: usize = 1 << 24;

#[inline(always)]
pub(crate) fn encode24(v: f64) -> [u8; 3] {
    let q = (v.clamp(0.0, 1.0) * SCALE24).round() as u32;
    [q as u8, (q >> 8) as u8, (q >> 16) as u8]
}

#[inline(always)]
pub(crate) fn decode24(b: [u8; 3]) -> f64 {
    (b[0] as u32 | (b[1] as u32) << 8 | (b[2] as u32) << 16) as f64 / SCALE24
}

#[inline(always)]
fn array_at<const N: usize>(bytes: &[u8], offset: usize) -> [u8; N] {
    bytes[offset..offset + N].try_into().expect("slice of length N")
}

/// Values of one layer, indexed as in `crate::index`.
pub(crate) enum LayerData {
    /// Being solved: `f32` bit patterns, updated in place by many threads.
    Active(Vec<AtomicU32>),
    /// Being solved, compact: 24-bit fixed point, 3 bytes per position (for the 5v5
    /// group, so that it fits in memory). Each byte is atomic but a value is not: a read
    /// racing with a write can see bytes of two different values, far from both when a
    /// carry crosses a byte. Such a torn read only perturbs the update that uses it. A
    /// large perturbation moves that stored value by more than the tolerance, so that
    /// sweep is not accepted as the last one; one that goes unnoticed leaves the value
    /// within about twice the tolerance of the update without it. Finished databases are
    /// audited independently by `residual_check`.
    Active24(Vec<AtomicU8>),
    /// Solved: a memory-mapped `.f32` file.
    F32Map(Mmap),
    /// Solved: a memory-mapped `.u24` file.
    U24Map(Mmap),
}

impl LayerData {
    /// A layer being solved with every value set to `init`, in 24-bit storage if `compact`.
    pub(crate) fn new_active(n: u64, init: f64, compact: bool) -> io::Result<LayerData> {
        let n = n as usize;
        Ok(if compact {
            let bytes = encode24(init);
            let mut v = try_alloc::<AtomicU8>(3 * n)?;
            v.par_extend((0..3 * n).into_par_iter().map(|k| AtomicU8::new(bytes[k % 3])));
            LayerData::Active24(v)
        } else {
            let bits = (init as f32).to_bits();
            let mut v = try_alloc::<AtomicU32>(n)?;
            v.par_extend((0..n).into_par_iter().map(|_| AtomicU32::new(bits)));
            LayerData::Active(v)
        })
    }

    /// Reads a checkpoint written by `save_checkpoint` for a layer of `n` positions. Every
    /// value must be a probability (see `check_probabilities`); the 24-bit encoding holds
    /// nothing else.
    pub(crate) fn load_checkpoint(path: &Path, n: u64, compact: bool) -> io::Result<LayerData> {
        let n = n as usize;
        let expected = if compact { 3 * n } else { 4 * n };
        let len = std::fs::metadata(path)?.len();
        if len != expected as u64 {
            return Err(invalid_data(format!("{}: {len} bytes, expected {expected}", path.display())));
        }
        // Reads are whole chunks, so the file needs no buffering of its own.
        let mut file = File::open(path)?;
        let mut buf = vec![0u8; 4 * CHUNK];
        if compact {
            let mut v = try_alloc::<AtomicU8>(expected)?;
            while v.len() < expected {
                let k = (expected - v.len()).min(buf.len());
                file.read_exact(&mut buf[..k])?;
                v.extend(buf[..k].iter().map(|&x| AtomicU8::new(x)));
            }
            Ok(LayerData::Active24(v))
        } else {
            let mut v = try_alloc::<AtomicU32>(n)?;
            while v.len() < n {
                let k = 4 * (n - v.len()).min(CHUNK);
                file.read_exact(&mut buf[..k])?;
                v.extend(buf[..k].chunks_exact(4).map(|c| AtomicU32::new(u32::from_le_bytes(array_at(c, 0)))));
            }
            let data = LayerData::Active(v);
            check_probabilities(&data, n as u64, path)?;
            Ok(data)
        }
    }

    /// The value of position `i` (exactly as stored).
    #[inline(always)]
    pub(crate) fn get(&self, i: u64) -> f64 {
        let i = i as usize;
        match self {
            LayerData::Active(v) => f32::from_bits(v[i].load(Relaxed)) as f64,
            LayerData::Active24(v) => {
                let c = &v[3 * i..3 * i + 3];
                decode24([c[0].load(Relaxed), c[1].load(Relaxed), c[2].load(Relaxed)])
            }
            LayerData::F32Map(m) => f32::from_le_bytes(array_at(m, 4 * i)) as f64,
            LayerData::U24Map(m) => decode24(array_at(m, 3 * i)),
        }
    }

    /// Hints the CPU to fetch position `i` into cache ahead of `get`.
    #[inline(always)]
    pub(crate) fn prefetch(&self, i: u64) {
        #[cfg(target_arch = "x86_64")]
        {
            use std::arch::x86_64::{_MM_HINT_T0, _mm_prefetch};
            let i = i as usize;
            let p: *const i8 = match self {
                LayerData::Active(v) => v.as_ptr().wrapping_add(i).cast(),
                LayerData::Active24(v) => v.as_ptr().wrapping_add(3 * i).cast(),
                LayerData::F32Map(m) => m.as_ptr().wrapping_add(4 * i).cast(),
                LayerData::U24Map(m) => m.as_ptr().wrapping_add(3 * i).cast(),
            };
            // SAFETY: a prefetch is only a hint: it never faults and has no effect on
            // program state, whatever the address.
            unsafe { _mm_prefetch(p, _MM_HINT_T0) };
        }
    }

    /// Stores a new value for position `i` of a layer being solved; returns the absolute
    /// change of the stored value. Uses a plain load and store rather than an atomic
    /// swap: only the thread updating position `i` ever writes to it.
    #[inline(always)]
    pub(crate) fn store(&self, i: u64, v: f64) -> f64 {
        let i = i as usize;
        match self {
            LayerData::Active(a) => {
                let v = v as f32;
                let old = f32::from_bits(a[i].load(Relaxed));
                a[i].store(v.to_bits(), Relaxed);
                (v - old).abs() as f64
            }
            LayerData::Active24(a) => {
                let cells = &a[3 * i..3 * i + 3];
                let old = [cells[0].load(Relaxed), cells[1].load(Relaxed), cells[2].load(Relaxed)];
                let new = encode24(v);
                for (cell, byte) in cells.iter().zip(new) {
                    cell.store(byte, Relaxed);
                }
                (decode24(new) - decode24(old)).abs()
            }
            LayerData::F32Map(_) | LayerData::U24Map(_) => unreachable!("layer is not being solved"),
        }
    }

    fn bytes(&self) -> usize {
        match self {
            LayerData::Active(v) => 4 * v.len(),
            LayerData::Active24(v) => v.len(),
            LayerData::F32Map(m) | LayerData::U24Map(m) => m.len(),
        }
    }

    /// Writes all values as a `.f32` file (the database format).
    pub(crate) fn write_f32(&self, path: &Path) -> io::Result<()> {
        let n = self.len();
        write_atomically(path, |f| {
            let mut buf = Vec::with_capacity(4 * CHUNK);
            for start in (0..n).step_by(CHUNK) {
                let end = (start + CHUNK).min(n);
                buf.clear();
                buf.par_extend(
                    (start..end).into_par_iter().flat_map_iter(|i| (self.get(i as u64) as f32).to_le_bytes()),
                );
                f.write_all(&buf)?;
            }
            Ok(())
        })
    }

    /// Writes a layer being solved as a checkpoint for `load_checkpoint`. No thread may
    /// write to the layer meanwhile. An existing checkpoint is overwritten in place, so
    /// that the solve needs no room for a second copy (15 GB for the 5v5 layer). A crash
    /// meanwhile leaves values from two different sweeps, each still a probability: a
    /// starting point as good as any, since the parallel sweeps already update the values
    /// in no fixed order.
    pub(crate) fn save_checkpoint(&self, path: &Path) -> io::Result<()> {
        let write = |f: &mut BufWriter<File>| {
            let mut buf = Vec::with_capacity(4 * CHUNK);
            match self {
                LayerData::Active(v) => {
                    for chunk in v.chunks(CHUNK) {
                        buf.clear();
                        buf.extend(chunk.iter().flat_map(|a| a.load(Relaxed).to_le_bytes()));
                        f.write_all(&buf)?;
                    }
                }
                LayerData::Active24(v) => {
                    for chunk in v.chunks(4 * CHUNK) {
                        buf.clear();
                        buf.extend(chunk.iter().map(|a| a.load(Relaxed)));
                        f.write_all(&buf)?;
                    }
                }
                LayerData::F32Map(_) | LayerData::U24Map(_) => unreachable!("layer is not being solved"),
            }
            Ok(())
        };
        if !std::fs::metadata(path).is_ok_and(|m| m.len() == self.bytes() as u64) {
            return write_atomically(path, write);
        }
        let in_place = OpenOptions::new().write(true).open(path).and_then(|file| {
            let mut f = BufWriter::with_capacity(1 << 22, file);
            write(&mut f)?;
            f.into_inner().map_err(io::IntoInnerError::into_error)?.sync_all()
        });
        in_place.map_err(|e| io::Error::new(e.kind(), format!("writing {}: {e}", path.display())))
    }

    /// Writes all values as a `.u24` file and maps it.
    pub(crate) fn to_u24_map(&self, path: &Path) -> io::Result<LayerData> {
        let n = self.len();
        write_atomically(path, |f| {
            let mut buf = Vec::with_capacity(3 * CHUNK);
            for start in (0..n).step_by(CHUNK) {
                let end = (start + CHUNK).min(n);
                buf.clear();
                buf.par_extend((start..end).into_par_iter().flat_map_iter(|i| encode24(self.get(i as u64))));
                f.write_all(&buf)?;
            }
            Ok(())
        })?;
        let file = File::open(path)?;
        // SAFETY: the solver owns this file and does not modify it while it is mapped.
        let map = unsafe { Mmap::map(&file)? };
        if map.len() != 3 * n {
            return Err(invalid_data(format!("{}: {} bytes, expected {}", path.display(), map.len(), 3 * n)));
        }
        Ok(LayerData::U24Map(map))
    }

    fn len(&self) -> usize {
        match self {
            LayerData::Active(v) => v.len(),
            LayerData::Active24(v) => v.len() / 3,
            LayerData::F32Map(m) => m.len() / 4,
            LayerData::U24Map(m) => m.len() / 3,
        }
    }
}

/// The layers resident while solving or checking, indexed by `[w][b]`.
pub struct Layers {
    l: [[Option<LayerData>; MAX_PIECES + 1]; MAX_PIECES + 1],
}

impl Layers {
    pub(crate) fn new() -> Layers {
        Layers { l: Default::default() }
    }

    /// Memory-maps every solved layer of `dir` with at most `max_sum` pieces in total.
    pub fn open(dir: &Path, max_sum: usize) -> io::Result<Layers> {
        let mut layers = Layers::new();
        for (w, b) in all_layers().filter(|&(w, b)| w + b <= max_sum) {
            layers.set(w, b, LayerData::F32Map(map_layer(&layer_path(dir, w, b), w, b)?));
        }
        Ok(layers)
    }

    /// Whether layer `(w, b)` is resident (false for `w` or `b` out of range).
    pub fn has(&self, w: usize, b: usize) -> bool {
        self.l.get(w).and_then(|row| row.get(b)).is_some_and(Option::is_some)
    }

    /// Layer `(w, b)`, which must be resident.
    #[inline(always)]
    pub(crate) fn layer(&self, w: usize, b: usize) -> &LayerData {
        self.l[w][b].as_ref().expect("layer is resident")
    }

    pub(crate) fn set(&mut self, w: usize, b: usize, data: LayerData) {
        self.l[w][b] = Some(data);
    }

    pub(crate) fn take(&mut self, w: usize, b: usize) -> Option<LayerData> {
        self.l[w][b].take()
    }

    pub(crate) fn resident_bytes(&self) -> usize {
        self.l.iter().flatten().flatten().map(LayerData::bytes).sum()
    }
}

/// Fails unless the first `n` values of `data`, read from `path`, are probabilities. The
/// solver's updates keep probabilities probabilities, but they take the best move with a
/// `max` that skips NaN, so values read from files are checked before a solve uses them.
pub(crate) fn check_probabilities(data: &LayerData, n: u64, path: &Path) -> io::Result<()> {
    match (0..n).into_par_iter().find_first(|&i| !(0.0..=1.0).contains(&data.get(i))) {
        Some(i) => {
            Err(invalid_data(format!("{}: value {} at index {i} is not a probability", path.display(), data.get(i))))
        }
        None => Ok(()),
    }
}

/// A solve's exclusive use of a database directory: `solve.lock` in it, created when taken
/// and removed when dropped. A solve that is killed leaves it behind.
pub(crate) struct SolveLock(PathBuf);

impl SolveLock {
    pub(crate) fn take(dir: &Path) -> io::Result<SolveLock> {
        let path = dir.join("solve.lock");
        match OpenOptions::new().write(true).create_new(true).open(&path) {
            Ok(mut file) => {
                let lock = SolveLock(path);
                writeln!(file, "{}", std::process::id())?;
                Ok(lock)
            }
            Err(e) if e.kind() == io::ErrorKind::AlreadyExists => {
                let pid = std::fs::read_to_string(&path).unwrap_or_default();
                Err(io::Error::new(
                    e.kind(),
                    format!(
                        "{} exists: another solve (process {}) is using {}. If it is no longer running, \
                         delete the file and solve again",
                        path.display(),
                        pid.trim(),
                        dir.display()
                    ),
                ))
            }
            Err(e) => Err(io::Error::new(e.kind(), format!("{}: {e}", path.display()))),
        }
    }
}

impl Drop for SolveLock {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

pub(crate) fn f32_ckpt_path(dir: &Path, w: usize, b: usize) -> PathBuf {
    layer_file(dir, w, b, "f32.ckpt")
}

pub(crate) fn u24_ckpt_path(dir: &Path, w: usize, b: usize) -> PathBuf {
    layer_file(dir, w, b, "u24.ckpt")
}

pub(crate) fn u24_path(dir: &Path, w: usize, b: usize) -> PathBuf {
    layer_file(dir, w, b, "u24")
}

fn try_alloc<T>(n: usize) -> io::Result<Vec<T>> {
    let mut v = Vec::new();
    v.try_reserve_exact(n).map_err(|e| {
        let gb = (n * std::mem::size_of::<T>()) as f64 / 1e9;
        io::Error::new(io::ErrorKind::OutOfMemory, format!("cannot allocate {gb:.1} GB: {e}"))
    })?;
    Ok(v)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encode24_round_trip() {
        for v in [0.0, 1.0, 0.5, 0.25, 1.0 / 3.0, 0.999_999_9] {
            assert!((decode24(encode24(v)) - v).abs() <= 0.5 / SCALE24, "{v}");
        }
        assert_eq!(encode24(-0.1), encode24(0.0));
        assert_eq!(encode24(1.1), encode24(1.0));
        assert_eq!(decode24([0xFF, 0xFF, 0xFF]), 1.0);
    }

    #[test]
    fn store_get_and_files() {
        let dir = std::env::temp_dir().join(format!("senet_storage_test_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let n = 1000u64;
        for compact in [false, true] {
            let d = LayerData::new_active(n, 0.5, compact).unwrap();
            assert!((d.get(7) - 0.5).abs() <= 1e-7);
            for i in 0..n {
                d.store(i, i as f64 / n as f64);
            }
            let change = d.store(3, 0.75);
            assert!((change - (0.75 - 0.003)).abs() < 1e-6, "{change}");

            // A new checkpoint is written whole; an existing one of the same size is
            // overwritten in place, one of another size replaced.
            let ck = dir.join("ckpt");
            std::fs::write(&ck, [7u8; 10]).unwrap();
            d.save_checkpoint(&ck).unwrap();
            d.store(5, 0.25);
            d.save_checkpoint(&ck).unwrap();
            assert!(!crate::atomic::tmp_path(&ck).exists());
            let back = LayerData::load_checkpoint(&ck, n, compact).unwrap();
            assert!(LayerData::load_checkpoint(&ck, n + 1, compact).is_err());
            let f32_path = dir.join("L11.f32");
            d.write_f32(&f32_path).unwrap();
            let file = File::open(&f32_path).unwrap();
            let mapped = LayerData::F32Map(unsafe { Mmap::map(&file).unwrap() });
            let u24 = d.to_u24_map(&dir.join("L11.u24")).unwrap();
            for i in 0..n {
                let v = d.get(i);
                assert_eq!(back.get(i), v);
                assert_eq!(mapped.get(i), v as f32 as f64);
                assert!((u24.get(i) - v).abs() <= 1e-7);
                let expected = match i {
                    3 => 0.75,
                    5 => 0.25,
                    _ => i as f64 / n as f64,
                };
                if compact {
                    assert!((v - expected).abs() <= 0.5 / SCALE24, "{i}: {v}");
                } else {
                    assert_eq!(v, expected as f32 as f64, "{i}");
                }
            }
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
}
