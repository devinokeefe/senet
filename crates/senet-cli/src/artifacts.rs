//! The files `senet` publishes for others to trust: the training data for the network,
//! and the integrity manifests (senet_core::manifest) of the data and of the database,
//! which `senet verify` checks.

use clap::Args;
use rayon::prelude::*;
use senet_core::atomic::write_atomically;
use senet_core::board::{Pos, Rules};
use senet_core::bots::{Bot, BotContext, EpsilonBot, make_bot};
use senet_core::db::Db;
use senet_core::game::{Decision, play_game};
use senet_core::index::{MAX_PIECES, layer_size, position_of};
use senet_core::manifest::{self, FileEntry, HashingWriter, Manifest};
use senet_core::rng::Rng;
use serde_json::json;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

/// Bytes per training record: `u32` mover mask, `u32` opponent mask, `f32` value.
const RECORD_BYTES: u64 = 12;
/// Self-play games, and uniformly random positions, generated at once: they bound the
/// memory `gen-data` takes (about 50 MB of records) whatever the size of the data.
const GAMES_PER_CHUNK: u64 = 4096;
const UNIFORM_PER_CHUNK: u64 = 1 << 22;

/// The note of a manifest written after the fact.
const AFTER_THE_FACT: &str = "Written after the fact by `senet manifest`: it vouches for the files as they were then, not for how they were made.";

#[derive(Args)]
pub struct GenDataArgs {
    /// Database directory (must be complete)
    #[arg(long)]
    db: PathBuf,
    /// Output file (its manifest is written beside it, as NAME.manifest.json)
    #[arg(long)]
    out: PathBuf,
    /// Self-play games to take positions from
    #[arg(long, default_value_t = 200_000)]
    games: u64,
    /// Average exploration rate of the self-play
    #[arg(long, default_value_t = 0.2, value_parser = crate::non_negative)]
    eps: f64,
    /// Uniformly random positions to add
    #[arg(long, default_value_t = 5_000_000)]
    uniform: u64,
    /// Seed of the games and of the random positions
    #[arg(long, default_value_t = 1)]
    seed: u64,
}

#[derive(Args)]
pub struct ManifestArgs {
    /// Database directory to write the manifest of (MANIFEST.json in it)
    #[arg(long, required_unless_present = "data", conflicts_with = "data")]
    db: Option<PathBuf>,
    /// Training-data file to write the manifest of (NAME.manifest.json beside it)
    #[arg(long)]
    data: Option<PathBuf>,
    /// A remark to record in the manifest, such as what is known of how the files were made
    #[arg(long)]
    note: Vec<String>,
}

#[derive(Args)]
pub struct VerifyArgs {
    /// A manifest, or the database directory or the file it covers
    path: PathBuf,
    /// Compare sizes only, without hashing: quick, and finds missing and truncated files
    #[arg(long)]
    sizes_only: bool,
}

fn io_error(path: &Path) -> impl Fn(io::Error) -> String + '_ {
    move |e| format!("{}: {e}", path.display())
}

/// Removes `path` if it exists.
fn remove_if_present(path: &Path) -> Result<(), String> {
    match std::fs::remove_file(path) {
        Err(e) if e.kind() != io::ErrorKind::NotFound => Err(io_error(path)(e)),
        _ => Ok(()),
    }
}

/// The manifest of the database in `dir`, hashing every file of it.
pub fn database_manifest(dir: &Path) -> Result<Manifest, String> {
    let db = Db::open(dir).map_err(io_error(dir))?;
    let names = manifest::database_files(dir).map_err(io_error(dir))?;
    let files = manifest::hash_files(dir, &names).map_err(|e| e.to_string())?;
    let layers = || files.iter().filter(|f| f.name.ends_with(".f32"));
    let contents = json!({
        "complete": db.is_complete(),
        "layers": layers().count(),
        "positions": layers().map(|f| f.bytes / 4).sum::<u64>(),
        "start_value_white": db.lookup(Rules::KENDALL5.start()),
    });
    let mut m = manifest::new(manifest::DATABASE, files);
    m.contents = contents;
    m.notes.push("How each group of layers was solved: meta.json.".into());
    Ok(m)
}

/// Writes the manifest of the database `dir` (MANIFEST.json); returns its path.
pub fn write_database_manifest(dir: &Path, notes: &[String]) -> Result<PathBuf, String> {
    let mut m = database_manifest(dir)?;
    m.notes.extend_from_slice(notes);
    let path = dir.join(manifest::DIR_MANIFEST);
    manifest::write(&path, &m).map_err(|e| e.to_string())?;
    Ok(path)
}

/// Before `senet solve` changes a database: its manifest would no longer hold.
pub fn retire_database_manifest(dir: &Path) -> Result<(), String> {
    remove_if_present(&dir.join(manifest::DIR_MANIFEST))
}

/// When `senet solve` leaves a database as it is: its manifest, if it has one, stays, and
/// the files must still have the sizes it records.
pub fn check_database_manifest(dir: &Path) -> Result<(), String> {
    let path = dir.join(manifest::DIR_MANIFEST);
    if !path.exists() {
        eprintln!("{} has no manifest; `senet manifest --db {}` writes one", dir.display(), dir.display());
        return Ok(());
    }
    manifest::check_sizes(dir).map_err(|e| format!("{e} (`senet verify` checks every file)"))?;
    eprintln!("{}: the files have the sizes it records (`senet verify` also checks their SHA-256)", path.display());
    Ok(())
}

/// The manifest of training-data file `file`, `NAME.manifest.json` beside it, if `file`
/// may have one: it is not a directory or a `.json` file, and that manifest is not one of
/// another file of the same name (such as `NAME.bin` beside `NAME.dat`).
fn data_manifest_path(file: &Path) -> Result<PathBuf, String> {
    if file.is_dir() {
        return Err(format!("{}: a directory, not a file", file.display()));
    }
    let path = manifest::manifest_path(file);
    if path == file {
        return Err(format!("{}: training data must not be a .json file", file.display()));
    }
    let name = manifest::file_name(file).map_err(|e| e.to_string())?;
    if let Ok(m) = manifest::read(&path) {
        let others: Vec<&str> = m.files.iter().map(|f| f.name.as_str()).filter(|n| *n != name).collect();
        if !others.is_empty() {
            return Err(format!(
                "{} is the manifest of {}: give {} another name",
                path.display(),
                others.join(", "),
                file.display()
            ));
        }
    }
    Ok(path)
}

pub fn cmd_manifest(a: ManifestArgs) -> Result<(), String> {
    let t0 = Instant::now();
    let notes: Vec<String> = std::iter::once(AFTER_THE_FACT.to_string()).chain(a.note).collect();
    let path = match (a.db, a.data) {
        (Some(dir), _) => write_database_manifest(&dir, &notes)?,
        (None, Some(file)) => {
            let path = data_manifest_path(&file)?;
            let bytes = std::fs::metadata(&file).map_err(io_error(&file))?.len();
            if bytes % RECORD_BYTES != 0 {
                return Err(format!("{}: {bytes} bytes is not a whole number of records", file.display()));
            }
            let name = manifest::file_name(&file).map_err(|e| e.to_string())?;
            let dir = file.parent().filter(|p| !p.as_os_str().is_empty()).unwrap_or(Path::new("."));
            let files = manifest::hash_files(dir, &[name]).map_err(|e| e.to_string())?;
            let mut m = manifest::new(manifest::TRAINING_DATA, files);
            m.contents = json!({ "records": bytes / RECORD_BYTES });
            m.notes = notes;
            manifest::write(&path, &m).map_err(|e| e.to_string())?;
            path
        }
        (None, None) => unreachable!("clap requires --db or --data"),
    };
    eprintln!("wrote {} ({:.1}s)", path.display(), t0.elapsed().as_secs_f64());
    Ok(())
}

pub fn cmd_verify(a: VerifyArgs) -> Result<(), String> {
    let t0 = Instant::now();
    let (m, what) = if a.sizes_only {
        (manifest::check_sizes(&a.path), "sizes")
    } else {
        (manifest::verify(&a.path), "sizes and SHA-256")
    };
    let m = m.map_err(|e| e.to_string())?;
    let bytes: u64 = m.files.iter().map(|f| f.bytes).sum();
    println!(
        "{}: {} files, {bytes} bytes, as recorded ({what}, {:.1}s)",
        manifest::manifest_path(&a.path).display(),
        m.files.len(),
        t0.elapsed().as_secs_f64()
    );
    Ok(())
}

/// The records of one game of epsilon-greedy perfect self-play: every position a move was
/// played in, with its value.
fn game_records(ctx: &BotContext, db: &Db, a: &GenDataArgs, g: u64) -> Vec<(Pos, f32)> {
    let s = a.seed ^ g.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    // Vary the exploration rate per game so data covers both clean and messy play.
    let eps = a.eps * (0.25 + 1.5 * Rng::new(s ^ 77).f64());
    let explorer = |k: u64| -> Box<dyn Bot> {
        let inner = make_bot("perfect", ctx, 0).expect("the database is complete");
        Box::new(EpsilonBot { inner, eps, rng: Rng::new(s ^ k) })
    };
    let mut seen = Vec::with_capacity(160);
    let mut observe = |d: &Decision| seen.extend(db.lookup(d.pos).map(|v| (d.pos, v as f32)));
    play_game(&Rules::KENDALL5, &mut [explorer(1), explorer(2)], &mut Rng::new(s), Some(&mut observe));
    seen
}

/// Random position `i`: a layer, then a position of it, each uniformly at random. The index
/// is drawn as `next_u64() % size` (biased by at most size / 2^64), as when the published
/// training data was made, so that `gen-data` with the same settings makes the same file.
fn uniform_record(db: &Db, a: &GenDataArgs, i: u64) -> (Pos, f32) {
    let mut rng = Rng::new(a.seed ^ 0xABCD ^ i.wrapping_mul(0xD1B5_4A32_D192_ED03));
    let w = 1 + rng.below(MAX_PIECES as u64) as usize;
    let b = 1 + rng.below(MAX_PIECES as u64) as usize;
    let pos = position_of(w, b, rng.next_u64() % layer_size(w, b));
    (pos, db.lookup(pos).expect("complete database") as f32)
}

/// A training record as the file holds it, little-endian.
fn record((pos, v): (Pos, f32)) -> [u8; RECORD_BYTES as usize] {
    let mut r = [0; RECORD_BYTES as usize];
    r[0..4].copy_from_slice(&pos.me.to_le_bytes());
    r[4..8].copy_from_slice(&pos.opp.to_le_bytes());
    r[8..12].copy_from_slice(&v.to_le_bytes());
    r
}

/// Training data for the distilled network: every position a move was played in during
/// games of epsilon-greedy perfect self-play (the positions that matter in real play),
/// then uniformly random positions (each layer equally likely) for coverage. Generated in
/// chunks, in order, so the file is the same as if made at once; it is published whole,
/// then its manifest (records, settings, SHA-256) beside it.
pub fn cmd_gen_data(a: GenDataArgs) -> Result<(), String> {
    // The output is checked before the work, which can take hours.
    let manifest_path = data_manifest_path(&a.out)?;
    if std::fs::metadata(&a.out).is_ok_and(|m| m.permissions().readonly()) {
        return Err(format!("{}: read-only", a.out.display()));
    }
    let ctx = BotContext::load(Some(&a.db), None)?;
    let db = ctx.complete_db().map_err(|e| e.to_string())?.as_ref();
    let t0 = Instant::now();
    // From now until the new manifest, no manifest vouches for whatever `out` holds.
    remove_if_present(&manifest_path)?;
    let mut from_games = 0u64;
    let mut digest = (0, String::new());
    write_atomically(&a.out, |f| {
        let mut w = HashingWriter::new(f);
        for start in (0..a.games).step_by(GAMES_PER_CHUNK as usize) {
            let games = start..(start + GAMES_PER_CHUNK).min(a.games);
            let records: Vec<_> = games
                .into_par_iter()
                .flat_map_iter(|g| game_records(&ctx, db, &a, g).into_iter().map(record))
                .collect();
            w.write_all(records.as_flattened())?;
            from_games += records.len() as u64;
        }
        for start in (0..a.uniform).step_by(UNIFORM_PER_CHUNK as usize) {
            let indices = start..(start + UNIFORM_PER_CHUNK).min(a.uniform);
            let records: Vec<_> = indices.into_par_iter().map(|i| record(uniform_record(db, &a, i))).collect();
            w.write_all(records.as_flattened())?;
        }
        digest = w.digest();
        Ok(())
    })
    .map_err(|e| e.to_string())?;
    let (bytes, sha256) = digest;
    let records = bytes / RECORD_BYTES;
    let name = manifest::file_name(&a.out).map_err(|e| e.to_string())?;
    let mut m = manifest::new(manifest::TRAINING_DATA, vec![FileEntry { name, bytes, sha256 }]);
    m.settings = json!({ "games": a.games, "eps": a.eps, "uniform": a.uniform, "seed": a.seed });
    m.inputs = json!({ "database": manifest::identity(&a.db).map_err(|e| e.to_string())? });
    m.contents = json!({ "records": records, "from_games": from_games, "uniform": a.uniform });
    manifest::write(&manifest_path, &m).map_err(|e| e.to_string())?;
    eprintln!(
        "wrote {records} records ({from_games} from {} games, {} uniform) to {} and its manifest {} in {:.1}s",
        a.games,
        a.uniform,
        a.out.display(),
        manifest_path.display(),
        t0.elapsed().as_secs_f64()
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("senet_cli_{name}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn records_are_little_endian() {
        let r = record((Pos::new(0x0403_0201, 0x0807_0605), 0.5));
        assert_eq!(r, [1, 2, 3, 4, 5, 6, 7, 8, 0, 0, 0, 0x3f]);
    }

    #[test]
    fn a_data_file_needs_a_manifest_of_its_own() {
        let dir = temp_dir("data_manifest");
        let data = dir.join("train.bin");
        std::fs::write(&data, [0u8; 24]).unwrap();
        let manifest = |file: &Path| cmd_manifest(ManifestArgs { db: None, data: Some(file.into()), note: vec![] });
        manifest(&data).unwrap();
        let path = dir.join("train.manifest.json");
        let written = std::fs::read(&path).unwrap();
        // Neither a manifest itself, nor another file of the same name, nor a directory.
        assert!(manifest(&path).unwrap_err().contains("must not be a .json file"));
        std::fs::write(dir.join("train.dat"), [0u8; 24]).unwrap();
        assert!(manifest(&dir.join("train.dat")).unwrap_err().contains("is the manifest of train.bin"));
        assert_eq!(std::fs::read(&path).unwrap(), written);
        std::fs::create_dir(dir.join("sub")).unwrap();
        std::fs::write(dir.join("sub").join(manifest::DIR_MANIFEST), b"kept").unwrap();
        assert!(data_manifest_path(&dir.join("sub")).unwrap_err().contains("a directory"));
        // gen-data refuses such outputs before it opens the database.
        let gen_data = |out: PathBuf| {
            cmd_gen_data(GenDataArgs { db: dir.join("no_db"), out, games: 1, eps: 0.2, uniform: 1, seed: 1 })
        };
        assert!(gen_data(dir.join("sub")).unwrap_err().contains("a directory"));
        assert_eq!(std::fs::read(dir.join("sub").join(manifest::DIR_MANIFEST)).unwrap(), b"kept");
        assert!(gen_data(dir.join("train.dat")).unwrap_err().contains("is the manifest of train.bin"));
        assert!(gen_data(dir.join("x.json")).unwrap_err().contains("must not be a .json file"));
        assert!(gen_data(dir.join("new.bin")).unwrap_err().contains("no_db"));
        // Sizes only: a change of the same size passes; the hashes find it.
        std::fs::write(&data, [1u8; 24]).unwrap();
        let verify = |sizes_only| cmd_verify(VerifyArgs { path: data.clone(), sizes_only });
        verify(true).unwrap();
        assert!(verify(false).unwrap_err().contains("SHA-256"));
        std::fs::write(&data, [1u8; 12]).unwrap();
        assert!(verify(true).unwrap_err().contains("12 bytes"));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
