//! The solver's record of a database, `meta.json` (docs/FORMATS.md): the layer sizes, how
//! each group was solved, and the value of the opening once its layer is solved.
//!
//! The solver writes a group's layer files first and its record second, so an
//! interruption between the two leaves a solved group without a record. A rerun finds the
//! group solved and records it as such, with its statistics as unknown: they are not
//! invented, and the layers are not presented as re-checked (`senet check` audits them).

use super::GroupStats;
use crate::atomic::write_atomically;
use crate::board::{MAX_PIECES, Rules};
use crate::db::{layer_path, map_layer};
use crate::index::{all_layers, index_of, layer_size};
use crate::invalid_data;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::io::{self, Write};
use std::path::Path;

pub const FILE: &str = "meta.json";

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct Meta {
    pub ruleset: String,
    /// `"L{w}{b}"` -> the number of positions of layer `(w, b)`.
    #[serde(default)]
    pub layer_sizes: BTreeMap<String, u64>,
    /// Keyed by the group's layers, as in `"[(5, 4), (4, 5)]"`.
    pub groups: BTreeMap<String, GroupRecord>,
    /// P(White wins) at the opening, once its layer is solved.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub start_value_white: Option<f64>,
}

/// How a group was solved. The statistics are null when they were not recorded.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct GroupRecord {
    pub layers: Vec<(usize, usize)>,
    pub states: u64,
    pub sweeps: Option<u32>,
    /// The largest change of the last sweep.
    pub final_max_delta: Option<f64>,
    pub seconds: Option<f64>,
    /// The tolerance the solve stopped at (recorded since meta.json had this field).
    #[serde(default)]
    pub tol: Option<f64>,
    /// Where the record comes from, if not from the solve itself, or what it leaves out.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub note: Option<String>,
}

/// The note on the record of a group that was found solved, without a record.
pub const NOT_RECORDED: &str = "found solved; the solve's statistics were not recorded";
/// The note on the record of a solve that continued from a checkpoint.
pub const RESUMED: &str = "resumed from a checkpoint: sweeps and seconds count only the run that finished the solve";

/// The key of a group's record.
pub fn group_key(layers: &[(usize, usize)]) -> String {
    format!("{layers:?}")
}

impl GroupRecord {
    pub fn solved(stats: &GroupStats, tol: f64) -> GroupRecord {
        GroupRecord {
            layers: stats.layers.clone(),
            states: stats.states,
            sweeps: Some(stats.sweeps),
            final_max_delta: Some(stats.final_max_delta),
            seconds: Some(stats.seconds),
            tol: Some(tol),
            note: stats.resumed.then(|| RESUMED.into()),
        }
    }

    pub fn not_recorded(layers: &[(usize, usize)]) -> GroupRecord {
        GroupRecord {
            layers: layers.to_vec(),
            states: layers.iter().map(|&(w, b)| layer_size(w, b)).sum(),
            sweeps: None,
            final_max_delta: None,
            seconds: None,
            tol: None,
            note: Some(NOT_RECORDED.into()),
        }
    }

    fn check(&self, key: &str) -> Result<(), String> {
        let valid = |&(w, b): &(usize, usize)| (1..=MAX_PIECES).contains(&w) && (1..=MAX_PIECES).contains(&b);
        if self.layers.is_empty() || !self.layers.iter().all(valid) {
            return Err(format!("group {key}: layers {:?} are not layers of the game", self.layers));
        }
        if group_key(&self.layers) != key {
            return Err(format!("group {key}: the record is of layers {:?}", self.layers));
        }
        let states: u64 = self.layers.iter().map(|&(w, b)| layer_size(w, b)).sum();
        if self.states != states {
            return Err(format!("group {key}: {} states, but its layers have {states}", self.states));
        }
        for (name, x) in [("final_max_delta", self.final_max_delta), ("seconds", self.seconds), ("tol", self.tol)] {
            if x.is_some_and(|x| !(x.is_finite() && x >= 0.0)) {
                return Err(format!("group {key}: {name} {x:?} is not a finite number >= 0"));
            }
        }
        Ok(())
    }
}

impl Meta {
    fn new() -> Meta {
        Meta {
            ruleset: "kendall5".into(),
            layer_sizes: BTreeMap::new(),
            groups: BTreeMap::new(),
            start_value_white: None,
        }
    }

    /// Reads and checks `dir/meta.json`; a new, empty record if there is none.
    pub fn load(dir: &Path) -> io::Result<Meta> {
        let path = dir.join(FILE);
        let text = match std::fs::read_to_string(&path) {
            Ok(text) => text,
            Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(Meta::new()),
            Err(e) => return Err(io::Error::new(e.kind(), format!("{}: {e}", path.display()))),
        };
        let bad = |msg: String| {
            invalid_data(format!(
                "{}: {msg}. Fix it, or move it away: a rerun of `senet solve` then records the solved \
                 groups again, with their statistics as unknown",
                path.display()
            ))
        };
        let meta: Meta = serde_json::from_str(&text).map_err(|e| bad(e.to_string()))?;
        if meta.ruleset != "kendall5" {
            return Err(bad(format!("ruleset {:?}, expected \"kendall5\"", meta.ruleset)));
        }
        for (key, record) in &meta.groups {
            record.check(key).map_err(bad)?;
        }
        if meta.start_value_white.is_some_and(|v| !(0.0..=1.0).contains(&v)) {
            return Err(bad(format!("start_value_white {:?} is not a probability", meta.start_value_white)));
        }
        Ok(meta)
    }

    /// Updates the derived fields and writes `dir/meta.json` atomically, if it changed.
    fn save(mut self, dir: &Path, before: &Meta) -> io::Result<()> {
        self.layer_sizes = all_layers().map(|(w, b)| (format!("L{w}{b}"), layer_size(w, b))).collect();
        let (w, b, i) = index_of(Rules::KENDALL5.start());
        let start_layer = layer_path(dir, w, b);
        if start_layer.exists() {
            let map = map_layer(&start_layer, w, b)?;
            let at = 4 * i as usize;
            self.start_value_white = Some(f32::from_le_bytes(map[at..at + 4].try_into().expect("4 bytes")) as f64);
        }
        if self == *before {
            return Ok(());
        }
        let text = serde_json::to_string_pretty(&self).map_err(io::Error::other)?;
        write_atomically(&dir.join(FILE), |f| f.write_all(format!("{text}\n").as_bytes()))
    }
}

/// Records group `record` in `dir/meta.json`, replacing any earlier record of it.
pub fn record(dir: &Path, record: GroupRecord) -> io::Result<()> {
    let before = Meta::load(dir)?;
    let mut meta = before.clone();
    meta.groups.insert(group_key(&record.layers), record);
    meta.save(dir, &before)
}

/// Records group `layers`, found solved, if `dir/meta.json` has no record of it.
pub fn record_if_missing(dir: &Path, layers: &[(usize, usize)]) -> io::Result<()> {
    let before = Meta::load(dir)?;
    if before.groups.contains_key(&group_key(layers)) {
        return Ok(());
    }
    let mut meta = before.clone();
    meta.groups.insert(group_key(layers), GroupRecord::not_recorded(layers));
    meta.save(dir, &before)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::{SolveConfig, solve_all};

    fn temp_dir(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("senet_meta_{name}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        dir
    }

    const CFG: SolveConfig = SolveConfig { tol: 1e-7, max_sum: 3, ..SolveConfig::DEFAULT };

    #[test]
    fn records_written_before_this_field_still_load() {
        // A record as the published database's meta.json holds it, without `tol`.
        let dir = temp_dir("old");
        std::fs::create_dir_all(&dir).unwrap();
        let old = r#"{"ruleset": "kendall5", "groups": {"[(1, 1)]": {"final_max_delta": 5.960464477539063e-8,
            "layers": [[1, 1]], "seconds": 0.01, "states": 812, "sweeps": 18}}, "layer_sizes": {}}"#;
        std::fs::write(dir.join(FILE), old).unwrap();
        let meta = Meta::load(&dir).unwrap();
        let record = &meta.groups["[(1, 1)]"];
        assert_eq!((record.sweeps, record.tol, record.note.as_deref()), (Some(18), None, None));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_solve_records_its_groups() {
        let dir = temp_dir("solve");
        solve_all(&Rules::KENDALL5, &dir, &CFG).unwrap();
        let meta = Meta::load(&dir).unwrap();
        assert_eq!(meta.groups.keys().collect::<Vec<_>>(), ["[(1, 1)]", "[(2, 1), (1, 2)]"]);
        for record in meta.groups.values() {
            assert!(record.sweeps.is_some() && record.final_max_delta.unwrap() <= 1e-7 && record.note.is_none());
            assert_eq!(record.tol, Some(1e-7));
        }
        assert_eq!(meta.layer_sizes.len(), 25);
        assert_eq!(meta.start_value_white, None, "the 5v5 layer is not solved");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_group_solved_without_a_record_is_recorded_as_such() {
        // As an interruption between writing a group's layers and its record leaves it.
        let dir = temp_dir("unrecorded");
        solve_all(&Rules::KENDALL5, &dir, &CFG).unwrap();
        let mut meta = Meta::load(&dir).unwrap();
        let solved = meta.groups.remove("[(2, 1), (1, 2)]").unwrap();
        std::fs::write(dir.join(FILE), serde_json::to_string(&meta).unwrap()).unwrap();
        let layer = std::fs::read(layer_path(&dir, 2, 1)).unwrap();

        assert!(solve_all(&Rules::KENDALL5, &dir, &CFG).unwrap().is_empty(), "nothing is solved again");
        assert_eq!(std::fs::read(layer_path(&dir, 2, 1)).unwrap(), layer);
        let record = Meta::load(&dir).unwrap().groups.remove("[(2, 1), (1, 2)]").unwrap();
        assert_eq!(record, GroupRecord::not_recorded(&[(2, 1), (1, 2)]));
        assert_eq!((record.states, record.sweeps, record.final_max_delta), (solved.states, None, None));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_malformed_record_stops_the_solve_with_a_way_out() {
        let dir = temp_dir("malformed");
        solve_all(&Rules::KENDALL5, &dir, &CFG).unwrap();
        let good = std::fs::read_to_string(dir.join(FILE)).unwrap();
        for (bad, problem) in [
            ("{\"ruleset\": \"kendall5\", \"groups\": []".to_string(), "line"),
            (good.replace("\"groups\": {", "\"groups\": ["), "expected"),
            (good.replace("\"states\": 812", "\"states\": 813"), "812"),
            (good.replace("\"ruleset\": \"kendall5\"", "\"ruleset\": \"kendall7\""), "ruleset"),
            (good.replacen("\"tol\": 1e-7", "\"tol\": -1.0", 1), "tol"),
        ] {
            std::fs::write(dir.join(FILE), &bad).unwrap();
            let e = solve_all(&Rules::KENDALL5, &dir, &CFG).unwrap_err().to_string();
            assert!(e.contains("meta.json") && e.contains(problem) && e.contains("move it away"), "{e}");
            assert_eq!(std::fs::read_to_string(dir.join(FILE)).unwrap(), bad, "left as it was");
        }
        // Moved away: the groups are recorded again, as found solved.
        std::fs::remove_file(dir.join(FILE)).unwrap();
        assert!(solve_all(&Rules::KENDALL5, &dir, &CFG).unwrap().is_empty());
        let meta = Meta::load(&dir).unwrap();
        assert_eq!(meta.groups.len(), 2);
        assert!(meta.groups.values().all(|r| r.note.as_deref() == Some(NOT_RECORDED)));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
