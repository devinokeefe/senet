//! Integrity manifests (docs/FORMATS.md): what a database, a training-data file or a
//! network consists of, how it was made, and the size and SHA-256 of each of its files.
//! Writing one hashes every file. `verify` checks them all again, on request rather than
//! on every use: the database is 34 GB.
//!
//! The manifest of a directory is `MANIFEST.json` in it; that of a file `name.ext` is
//! `name.manifest.json` beside it, which may also cover other files of the same name
//! (a network's `.bin` and `.pt`).

use crate::atomic::write_atomically;
use crate::sha256::{Sha256, hex};
use crate::{invalid_data, invalid_input};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::fs::File;
use std::io::{self, Read, Write};
use std::path::{Component, Path, PathBuf};

/// The version of the manifest format this code reads and writes.
pub const VERSION: u32 = 1;
/// The name of a directory's manifest.
pub const DIR_MANIFEST: &str = "MANIFEST.json";

/// The kinds of artifacts, with the format of their files.
pub const DATABASE: (&str, &str) = ("database", "senet-db/1");
pub const TRAINING_DATA: (&str, &str) = ("training-data", "senet-records/1");

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct FileEntry {
    /// The file's name, in the manifest's directory.
    pub name: String,
    pub bytes: u64,
    /// SHA-256, 64 lowercase hex digits.
    pub sha256: String,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Manifest {
    /// The format version, `VERSION`.
    pub manifest: u32,
    /// "database", "training-data" or "network".
    pub kind: String,
    /// The format of the files and its version, such as "senet-db/1".
    pub format: String,
    /// The ruleset, "kendall5".
    pub rules: String,
    /// When the manifest was written (UTC, ISO 8601).
    pub created_utc: String,
    /// What wrote it: the command, the program's version and build, the source commit.
    pub created_by: Value,
    /// The settings the artifact was made with (null if they were not recorded).
    #[serde(default)]
    pub settings: Value,
    /// The identities of the artifacts it was made from (null if none or not recorded).
    #[serde(default)]
    pub inputs: Value,
    /// Facts about the contents: record counts, positions, values.
    #[serde(default)]
    pub contents: Value,
    /// Remarks a reader should know, such as what was recorded after the fact.
    #[serde(default)]
    pub notes: Vec<String>,
    pub files: Vec<FileEntry>,
}

/// The manifest that covers `path`: `path` itself if it is a `.json` file, `MANIFEST.json`
/// in it if it is a directory, else `name.manifest.json` beside it.
pub fn manifest_path(path: &Path) -> PathBuf {
    if path.is_dir() {
        return path.join(DIR_MANIFEST);
    }
    if path.extension().is_some_and(|e| e == "json") {
        return path.to_path_buf();
    }
    let stem = path.file_stem().unwrap_or_default().to_string_lossy();
    path.with_file_name(format!("{stem}.manifest.json"))
}

/// Hashes a stream, returning its length and SHA-256.
pub fn hash_reader(mut r: impl Read) -> io::Result<(u64, String)> {
    let mut h = Sha256::new();
    let mut buf = vec![0u8; 1 << 22];
    let mut total = 0u64;
    loop {
        let n = match r.read(&mut buf) {
            Ok(0) => break,
            Ok(n) => n,
            Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(e),
        };
        h.update(&buf[..n]);
        total += n as u64;
    }
    Ok((total, hex(&h.finish())))
}

/// The entries of files `names` in `dir`, hashed in parallel.
pub fn hash_files(dir: &Path, names: &[String]) -> io::Result<Vec<FileEntry>> {
    names
        .par_iter()
        .map(|name| {
            let path = dir.join(name);
            let file = File::open(&path).map_err(|e| io::Error::new(e.kind(), format!("{}: {e}", path.display())))?;
            let (bytes, sha256) = hash_reader(file)?;
            Ok(FileEntry { name: name.clone(), bytes, sha256 })
        })
        .collect()
}

/// A writer that hashes what it writes, for files hashed as they are made.
pub struct HashingWriter<W> {
    inner: W,
    hash: Sha256,
    bytes: u64,
}

impl<W: Write> HashingWriter<W> {
    pub fn new(inner: W) -> Self {
        HashingWriter { inner, hash: Sha256::new(), bytes: 0 }
    }

    /// The number of bytes written and their SHA-256.
    pub fn digest(&self) -> (u64, String) {
        (self.bytes, hex(&self.hash.clone().finish()))
    }
}

impl<W: Write> Write for HashingWriter<W> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let n = self.inner.write(buf)?;
        self.hash.update(&buf[..n]);
        self.bytes += n as u64;
        Ok(n)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.inner.flush()
    }
}

/// Writes manifest `m` to `path`, atomically.
pub fn write(path: &Path, m: &Manifest) -> io::Result<()> {
    let text = serde_json::to_string_pretty(m).map_err(io::Error::other)?;
    write_atomically(path, |f| f.write_all(format!("{text}\n").as_bytes()))
}

/// Reads the manifest at `path`, which must be of format version `VERSION`.
pub fn read(path: &Path) -> io::Result<Manifest> {
    let text =
        std::fs::read_to_string(path).map_err(|e| io::Error::new(e.kind(), format!("{}: {e}", path.display())))?;
    let version: Value =
        serde_json::from_str(&text).map_err(|e| invalid_data(format!("{}: not JSON: {e}", path.display())))?;
    if version.get("manifest") != Some(&Value::from(VERSION)) {
        return Err(invalid_data(format!("{}: not a manifest of format version {VERSION}", path.display())));
    }
    serde_json::from_str(&text).map_err(|e| invalid_data(format!("{}: {e}", path.display())))
}

/// The names of the database's layer files in `dir`: `L{w}{b}.f32`.
fn layer_files(dir: &Path) -> io::Result<Vec<String>> {
    let mut names = Vec::new();
    for entry in std::fs::read_dir(dir)? {
        let name = entry?.file_name().to_string_lossy().into_owned();
        if name.starts_with('L') && name.ends_with(".f32") {
            names.push(name);
        }
    }
    names.sort();
    Ok(names)
}

/// The files of the database in `dir` that its manifest lists: the layers and `meta.json`.
pub fn database_files(dir: &Path) -> io::Result<Vec<String>> {
    let mut names = layer_files(dir)?;
    if dir.join("meta.json").exists() {
        names.push("meta.json".into());
    }
    Ok(names)
}

/// Whether `name` is a plain file name, which a manifest's directory holds itself: one
/// path component, not hidden, without the `:` of a Windows drive (`C:x` is relative to
/// the current directory of drive C) or stream. The Python check (senet.manifest) agrees.
fn is_plain_file_name(name: &str) -> bool {
    let mut components = Path::new(name).components();
    matches!(components.next(), Some(Component::Normal(n)) if n == name)
        && components.next().is_none()
        && !name.starts_with('.')
        && !name.contains(['/', '\\', ':'])
}

/// Checks every file the manifest at `path` (see `manifest_path`) lists: present, of the
/// recorded size and SHA-256. A database's directory must also hold no layer file the
/// manifest leaves out. Returns the manifest, or an error that lists every problem.
pub fn verify(path: &Path) -> io::Result<Manifest> {
    check(path, true)
}

/// `verify` without hashing: quick, and enough to find missing and truncated files.
pub fn check_sizes(path: &Path) -> io::Result<Manifest> {
    check(path, false)
}

fn check(path: &Path, hashes: bool) -> io::Result<Manifest> {
    let manifest = manifest_path(path);
    let m = match read(&manifest) {
        Err(e) if e.kind() == io::ErrorKind::NotFound && !path.exists() => {
            return Err(io::Error::new(e.kind(), format!("{}: no such file or directory", path.display())));
        }
        m => m?,
    };
    let path = manifest;
    let dir = path.parent().unwrap_or(Path::new("."));
    let mut problems: Vec<String> = m
        .files
        .par_iter()
        .filter_map(|want| {
            if !is_plain_file_name(&want.name) {
                return Some(format!("{}: not a plain file name", want.name));
            }
            let file = dir.join(&want.name);
            match std::fs::metadata(&file) {
                Err(e) => return Some(format!("{}: {e}", want.name)),
                Ok(meta) if !meta.is_file() => return Some(format!("{}: not a file", want.name)),
                Ok(meta) if meta.len() != want.bytes => {
                    return Some(format!("{}: {} bytes, expected {}", want.name, meta.len(), want.bytes));
                }
                Ok(_) if !hashes => return None,
                Ok(_) => {}
            }
            match File::open(&file).and_then(hash_reader) {
                Err(e) => Some(format!("{}: {e}", want.name)),
                Ok((_, sha)) if sha != want.sha256 => {
                    Some(format!("{}: SHA-256 {sha}, expected {}", want.name, want.sha256))
                }
                Ok(_) => None,
            }
        })
        .collect();
    if m.kind == DATABASE.0 {
        for name in layer_files(dir)? {
            if !m.files.iter().any(|f| f.name == name) {
                problems.push(format!("{name}: not in the manifest"));
            }
        }
    }
    if problems.is_empty() {
        Ok(m)
    } else {
        problems.sort();
        Err(invalid_data(format!("{} does not match:\n  {}", path.display(), problems.join("\n  "))))
    }
}

/// The current time as UTC ISO 8601, to the second.
pub fn utc_now() -> String {
    let secs = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs());
    utc_iso(secs)
}

/// A Unix time as UTC ISO 8601 (the proleptic Gregorian calendar).
fn utc_iso(secs: u64) -> String {
    let (days, rem) = (secs / 86_400, secs % 86_400);
    // Civil date from days since 1970-01-01 (H. Hinnant's algorithm, for days >= 0).
    let z = days + 719_468;
    let era = z / 146_097;
    let doe = z % 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = yoe + era * 400 + u64::from(month <= 2);
    format!("{year:04}-{month:02}-{day:02}T{:02}:{:02}:{:02}+00:00", rem / 3600, rem / 60 % 60, rem % 60)
}

/// What a manifest written by the `senet` program records as its maker: the command line,
/// the source commit, if `git` knows it, and the engine's version and build (see
/// `crate::build_info`). Manifests written from Python (senet.provenance) record the same.
pub fn created_by_this_program() -> Value {
    let mut args = std::env::args_os();
    // The program by its name: its full path would only record where it was built.
    let program = args.next().and_then(|p| Some(Path::new(&p).file_stem()?.to_string_lossy().into_owned()));
    let command: Vec<String> = std::iter::once(program.unwrap_or_else(|| "senet".into()))
        .chain(args.map(|a| shell_quote(&a.to_string_lossy())))
        .collect();
    serde_json::json!({
        "command": command.join(" "),
        "commit": git_commit(),
        "engine": crate::build_info(),
    })
}

/// `arg` as a POSIX shell reads it back: as it is if it is plain, else in single quotes (as
/// Python's shlex.quote does).
fn shell_quote(arg: &str) -> String {
    let plain = !arg.is_empty() && arg.chars().all(|c| c.is_ascii_alphanumeric() || "@%+=:,./_-".contains(c));
    if plain { arg.to_string() } else { format!("'{}'", arg.replace('\'', r#"'"'"'"#)) }
}

/// `path` as manifests record it, with `/` between its components on every system.
fn portable(path: &Path) -> String {
    let path = path.to_string_lossy();
    if cfg!(windows) { path.replace('\\', "/") } else { path.into_owned() }
}

/// The checked-out commit of the current directory's repository, with "-dirty" if the work
/// tree has changes; None if `git` is missing or this is not a repository.
fn git_commit() -> Option<String> {
    let run = |args: &[&str]| {
        let out = std::process::Command::new("git").args(args).output().ok()?;
        out.status.success().then(|| String::from_utf8_lossy(&out.stdout).trim().to_string())
    };
    let head = run(&["rev-parse", "HEAD"])?;
    let dirty = !run(&["status", "--porcelain"])?.is_empty();
    Some(if dirty { format!("{head}-dirty") } else { head })
}

/// A new manifest of `kind` (`DATABASE` or `TRAINING_DATA`) made by this program, with
/// `files` and nothing else recorded yet.
pub fn new(kind: (&str, &str), files: Vec<FileEntry>) -> Manifest {
    Manifest {
        manifest: VERSION,
        kind: kind.0.into(),
        format: kind.1.into(),
        rules: "kendall5".into(),
        created_utc: utc_now(),
        created_by: created_by_this_program(),
        settings: Value::Null,
        inputs: Value::Null,
        contents: Value::Null,
        notes: Vec::new(),
        files,
    }
}

/// The identity of the artifact at `path` for another's manifest to record as its input:
/// its path and the SHA-256 of its manifest, or null for this if it has none.
pub fn identity(path: &Path) -> io::Result<Value> {
    let m = manifest_path(path);
    let sha = if m.exists() { Some(File::open(&m).and_then(hash_reader)?.1) } else { None };
    Ok(serde_json::json!({ "path": portable(path), "manifest_sha256": sha }))
}

/// The name of file `path` as a manifest beside it lists it.
pub fn file_name(path: &Path) -> io::Result<String> {
    path.file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .ok_or_else(|| invalid_input(format!("{}: not a file", path.display())))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("senet_manifest_{name}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn dates() {
        assert_eq!(utc_iso(0), "1970-01-01T00:00:00+00:00");
        assert_eq!(utc_iso(951_782_400), "2000-02-29T00:00:00+00:00");
        assert_eq!(utc_iso(1_791_076_545), "2026-10-04T01:15:45+00:00");
        assert_eq!(utc_iso(4_107_542_399), "2100-02-28T23:59:59+00:00");
    }

    #[test]
    fn paths() {
        let dir = temp_dir("paths");
        assert_eq!(manifest_path(&dir), dir.join("MANIFEST.json"));
        assert_eq!(manifest_path(Path::new("runs/train.bin")), Path::new("runs/train.manifest.json"));
        assert_eq!(manifest_path(Path::new("m/net.manifest.json")), Path::new("m/net.manifest.json"));
        let _ = std::fs::remove_dir_all(&dir);
        if cfg!(windows) {
            assert_eq!(portable(Path::new(r"db\kendall5")), "db/kendall5");
        }
        assert_eq!(portable(Path::new("db/kendall5")), "db/kendall5");
    }

    #[test]
    fn commands_are_recorded_as_a_shell_reads_them() {
        let quoted: Vec<String> = ["--seed", "1", "a b", "it's", "", "x=y,z/w.bin", "$HOME", "naïve"]
            .iter()
            .map(|a| shell_quote(a))
            .collect();
        let expected = ["--seed", "1", "'a b'", r#"'it'"'"'s'"#, "''", "x=y,z/w.bin", "'$HOME'", "'naïve'"];
        assert_eq!(quoted, expected);
    }

    #[test]
    fn hashing_writer_agrees_with_hashing_the_file() {
        let dir = temp_dir("writer");
        let data: Vec<u8> = (0..100_000u32).map(|i| (i * 7 % 251) as u8).collect();
        let mut w = HashingWriter::new(Vec::new());
        for piece in data.chunks(999) {
            w.write_all(piece).unwrap();
        }
        std::fs::write(dir.join("f"), &data).unwrap();
        let entry = hash_files(&dir, &["f".into()]).unwrap().remove(0);
        assert_eq!(w.digest(), (entry.bytes, entry.sha256));
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A small database-like directory with a manifest; returns its path.
    fn database(name: &str) -> PathBuf {
        let dir = temp_dir(name);
        std::fs::write(dir.join("L11.f32"), [1u8; 64]).unwrap();
        std::fs::write(dir.join("L12.f32"), [2u8; 128]).unwrap();
        std::fs::write(dir.join("meta.json"), b"{}").unwrap();
        let files = hash_files(&dir, &database_files(&dir).unwrap()).unwrap();
        write(&dir.join(DIR_MANIFEST), &new(DATABASE, files)).unwrap();
        dir
    }

    #[test]
    fn an_intact_database_verifies() {
        let dir = database("intact");
        let m = verify(&dir).unwrap();
        assert_eq!(m.files.iter().map(|f| f.name.as_str()).collect::<Vec<_>>(), ["L11.f32", "L12.f32", "meta.json"]);
        assert_eq!(m.kind, "database");
        check_sizes(&dir).unwrap();
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn modified_truncated_missing_and_extra_files_are_found() {
        type Spoil = fn(&Path);
        let cases: [(&str, Spoil); 5] = [
            ("SHA-256", |d| std::fs::write(d.join("L11.f32"), [9u8; 64]).unwrap()),
            ("bytes", |d| std::fs::write(d.join("L12.f32"), [2u8; 120]).unwrap()),
            ("meta.json", |d| std::fs::remove_file(d.join("meta.json")).unwrap()),
            ("not in the manifest", |d| std::fs::write(d.join("L21.f32"), [3u8; 128]).unwrap()),
            ("format version", |d| {
                let text = std::fs::read_to_string(d.join(DIR_MANIFEST)).unwrap();
                std::fs::write(d.join(DIR_MANIFEST), text.replace("\"manifest\": 1", "\"manifest\": 2")).unwrap()
            }),
        ];
        for (i, (problem, spoil)) in cases.into_iter().enumerate() {
            let dir = database(&format!("spoiled{i}"));
            spoil(&dir);
            let e = verify(&dir).unwrap_err().to_string();
            assert!(e.contains(problem), "{problem}: {e}");
            let _ = std::fs::remove_dir_all(&dir);
        }
    }

    #[test]
    fn size_checks_find_truncation_without_hashing() {
        let dir = database("sizes");
        std::fs::write(dir.join("L11.f32"), [9u8; 64]).unwrap(); // same size: only hashing finds it
        check_sizes(&dir).unwrap();
        std::fs::write(dir.join("L11.f32"), [1u8; 60]).unwrap();
        std::fs::remove_file(dir.join("meta.json")).unwrap();
        let e = check_sizes(&dir).unwrap_err().to_string();
        assert!(e.contains("L11.f32: 60 bytes, expected 64") && e.contains("meta.json: "), "{e}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn names_with_directories_are_refused() {
        let bad = ["../L11.f32", "sub/L11.f32", r"sub\L11.f32", "C:L11.f32", r"C:\L11.f32", "/L11.f32", "L11.f32:x"];
        for name in bad.into_iter().chain(["", ".", "..", ".hidden"]) {
            assert!(!is_plain_file_name(name), "{name:?}");
        }
        assert!(is_plain_file_name("L11.f32") && is_plain_file_name("senet_net.manifest.json"));
        let dir = database("names");
        let mut m = read(&dir.join(DIR_MANIFEST)).unwrap();
        m.files[0].name = "C:L11.f32".into();
        write(&dir.join(DIR_MANIFEST), &m).unwrap();
        for check in [verify, check_sizes] {
            assert!(check(&dir).unwrap_err().to_string().contains("C:L11.f32: not a plain file name"));
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_missing_directory_is_reported_as_such() {
        let dir = temp_dir("missing").join("typo");
        let e = verify(&dir).unwrap_err();
        assert_eq!(e.kind(), io::ErrorKind::NotFound);
        assert_eq!(e.to_string(), format!("{}: no such file or directory", dir.display()));
        // An existing file without a manifest: the manifest is what is missing.
        let file = dir.with_file_name("train.bin");
        std::fs::write(&file, b"data").unwrap();
        assert!(check_sizes(&file).unwrap_err().to_string().contains("train.manifest.json"));
        let _ = std::fs::remove_dir_all(file.parent().unwrap());
    }
}
