//! Files published whole: written under a temporary name, flushed to the disk, then
//! renamed into place. A reader finds the previous complete file or the new complete one,
//! never a partial one. A crash can leave the temporary file behind, which the next
//! attempt overwrites. On Unix the directory is flushed after the rename, so that the
//! rename survives a power loss; Windows has no portable way to do this.

use std::fs::File;
use std::io::{self, BufWriter};
use std::path::{Path, PathBuf};

/// Where `write_atomically` writes `path` before renaming it: `path` with `.tmp` appended.
pub fn tmp_path(path: &Path) -> PathBuf {
    let mut tmp = path.as_os_str().to_owned();
    tmp.push(".tmp");
    PathBuf::from(tmp)
}

/// Writes `path` through `tmp_path(path)`: `write` fills the file, which is then flushed to
/// the disk and renamed over `path`. On failure the temporary file is removed and `path` is
/// left as it was. (On Windows a file that is open or memory-mapped cannot be replaced.)
pub fn write_atomically(path: &Path, write: impl FnOnce(&mut BufWriter<File>) -> io::Result<()>) -> io::Result<()> {
    let tmp = tmp_path(path);
    let result = File::create(&tmp).and_then(|file| {
        let mut f = BufWriter::with_capacity(1 << 22, file);
        write(&mut f)?;
        // The file is closed at the end of this statement, before the rename.
        f.into_inner().map_err(io::IntoInnerError::into_error)?.sync_all()?;
        std::fs::rename(&tmp, path)?;
        #[cfg(unix)]
        File::open(path.parent().filter(|p| !p.as_os_str().is_empty()).unwrap_or(Path::new(".")))?.sync_all()?;
        Ok(())
    });
    result.map_err(|e| {
        let _ = std::fs::remove_file(&tmp);
        io::Error::new(e.kind(), format!("writing {}: {e}", path.display()))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn temp_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("senet_atomic_{name}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn a_complete_write_replaces_the_file() {
        let dir = temp_dir("replace");
        let path = dir.join("data");
        for round in 0..20u8 {
            // Repeated replacement, as checkpoints are replaced during a long solve.
            write_atomically(&path, |f| f.write_all(&[round; 1000])).unwrap();
            assert_eq!(std::fs::read(&path).unwrap(), [round; 1000]);
            assert!(!tmp_path(&path).exists());
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_failed_write_leaves_the_previous_file() {
        let dir = temp_dir("fail");
        let path = dir.join("data");
        let fail = |f: &mut BufWriter<File>| {
            f.write_all(&[7; 5000])?;
            Err(io::Error::other("interrupted"))
        };
        // No previous file: none afterwards.
        let e = write_atomically(&path, fail).unwrap_err();
        assert!(e.to_string().contains("interrupted") && e.to_string().contains("data"), "{e}");
        assert!(!path.exists() && !tmp_path(&path).exists());
        // A previous file is kept whole.
        write_atomically(&path, |f| f.write_all(b"old")).unwrap();
        write_atomically(&path, fail).unwrap_err();
        assert_eq!(std::fs::read(&path).unwrap(), b"old");
        assert!(!tmp_path(&path).exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_stale_temporary_file_is_overwritten() {
        // As a crash between writing and renaming leaves it.
        let dir = temp_dir("stale");
        let path = dir.join("data");
        std::fs::write(tmp_path(&path), [9; 100_000]).unwrap();
        write_atomically(&path, |f| f.write_all(b"new")).unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), b"new");
        assert!(!tmp_path(&path).exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_rename_that_fails_leaves_the_destination() {
        // A directory where the file should go: the rename fails on every platform.
        let dir = temp_dir("rename");
        let path = dir.join("taken");
        std::fs::create_dir(&path).unwrap();
        std::fs::write(path.join("inside"), b"kept").unwrap();
        assert!(write_atomically(&path, |f| f.write_all(b"new")).is_err());
        assert_eq!(std::fs::read(path.join("inside")).unwrap(), b"kept");
        assert!(!tmp_path(&path).exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_mapped_destination_is_replaced_whole_or_not_at_all() {
        // Windows refuses to replace a file that is memory-mapped; other systems replace it
        // while the mapping keeps the old contents. Either way no reader sees a mix.
        let dir = temp_dir("mapped");
        let path = dir.join("layer");
        std::fs::write(&path, [1; 4096]).unwrap();
        let file = File::open(&path).unwrap();
        // SAFETY: the test owns the file; the only other writer is the replacement under test.
        let map = unsafe { memmap2::Mmap::map(&file).unwrap() };
        match write_atomically(&path, |f| f.write_all(&[2; 4096])) {
            Ok(()) => assert_eq!(std::fs::read(&path).unwrap(), [2; 4096]),
            Err(_) => assert_eq!(std::fs::read(&path).unwrap(), [1; 4096]),
        }
        assert_eq!(map[..], [1; 4096]);
        assert!(!tmp_path(&path).exists());
        drop((map, file));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
