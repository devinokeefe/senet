//! Playing agents.
//!
//! Specs (used by the CLI, server and Python API):
//! * `random`         — uniformly random legal move
//! * `greedy`         — best move by the hand-crafted heuristic, 1 ply
//! * `expectimax:N`   — heuristic + expectimax over N future throws (N = 0..=4, default 2)
//! * `perfect`        — perfect play from the complete solved database
//! * `net` / `net:N`  — distilled neural network, optionally with N throws of search

use crate::board::{Pos, Rules};
use crate::db::Db;
use crate::eval::{Evaluator, Heuristic, Searcher};
use crate::movegen::Move;
use crate::net::Net;
use crate::rng::Rng;
use std::fmt;
use std::path::Path;
use std::sync::Arc;

/// The bot spec forms, for help texts and validation messages.
pub const SPECS: &str = "random, greedy, expectimax[:N], perfect, net[:N]";

/// Deepest search a spec may ask for: every throw of depth multiplies the work by the
/// 5 possible throws times the number of legal moves.
pub const MAX_SEARCH_DEPTH: u32 = 4;

/// Move values closer than this count as equal, so that a bot's choice does not depend
/// on floating-point noise.
pub const TIE_TOLERANCE: f64 = 1e-12;

/// Why a bot spec cannot be used.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BotError {
    /// The spec is malformed or names no bot.
    Invalid(String),
    /// The bot needs a resource that is not loaded: the complete database or a network.
    Unavailable(String),
}

impl fmt::Display for BotError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (BotError::Invalid(msg) | BotError::Unavailable(msg)) = self;
        f.write_str(msg)
    }
}

impl std::error::Error for BotError {}

pub trait Bot: Send {
    /// Picks one of `moves` (non-empty) for throw `t` in position `pos`.
    fn choose(&mut self, rules: &Rules, pos: Pos, t: u8, moves: &[Move]) -> usize;
}

pub struct RandomBot {
    rng: Rng,
}

impl RandomBot {
    pub fn new(seed: u64) -> RandomBot {
        RandomBot { rng: Rng::new(seed) }
    }
}

impl Bot for RandomBot {
    fn choose(&mut self, _: &Rules, _: Pos, _: u8, moves: &[Move]) -> usize {
        self.rng.below(moves.len() as u64) as usize
    }
}

/// An evaluator and how many throws to search ahead with it.
#[derive(Clone)]
pub struct Search {
    pub eval: Arc<dyn Evaluator>,
    pub depth: u32,
}

impl Search {
    /// Values (for the mover) of every move in `moves` for throw `t`.
    pub fn move_values(&self, rules: &Rules, t: u8, moves: &[Move]) -> Vec<f64> {
        let mut search = Searcher::new(rules, self.eval.as_ref(), self.depth);
        moves.iter().map(|m| search.move_value(m, t)).collect()
    }
}

/// Index of the highest of `values` (non-empty), the first one on ties within
/// `TIE_TOLERANCE`.
pub fn best_index(values: &[f64]) -> usize {
    (1..values.len()).fold(0, |best, i| if values[i] > values[best] + TIE_TOLERANCE { i } else { best })
}

/// Chooses the move with the highest searched value.
pub struct SearchBot {
    pub search: Search,
}

impl Bot for SearchBot {
    fn choose(&mut self, rules: &Rules, _: Pos, t: u8, moves: &[Move]) -> usize {
        if moves.len() == 1 {
            return 0;
        }
        best_index(&self.search.move_values(rules, t, moves))
    }
}

/// Plays a uniformly random move with probability `eps`, otherwise defers to `inner`.
/// Used to diversify self-play data.
pub struct EpsilonBot {
    pub inner: Box<dyn Bot>,
    pub eps: f64,
    pub rng: Rng,
}

impl Bot for EpsilonBot {
    fn choose(&mut self, rules: &Rules, pos: Pos, t: u8, moves: &[Move]) -> usize {
        if self.rng.f64() < self.eps {
            self.rng.below(moves.len() as u64) as usize
        } else {
            self.inner.choose(rules, pos, t, moves)
        }
    }
}

/// The resources bots can use: the solved database and the distilled network.
#[derive(Clone, Default)]
pub struct BotContext {
    pub db: Option<Arc<Db>>,
    pub net: Option<Arc<Net>>,
}

impl BotContext {
    /// Opens the database directory and loads the network file that are given.
    pub fn load(db: Option<&Path>, net: Option<&Path>) -> Result<BotContext, String> {
        let db =
            db.map(|dir| Db::open(dir).map_err(|e| format!("opening database {}: {e}", dir.display()))).transpose()?;
        let net = net
            .map(|file| Net::load(file).map_err(|e| format!("loading network {}: {e}", file.display())))
            .transpose()?;
        Ok(BotContext { db: db.map(Arc::new), net: net.map(Arc::new) })
    }

    /// The database, if it is loaded and complete, as perfect play requires.
    pub fn complete_db(&self) -> Result<&Arc<Db>, BotError> {
        self.db.as_ref().filter(|db| db.is_complete()).ok_or_else(|| {
            BotError::Unavailable("perfect play needs the complete solved database (db/kendall5)".into())
        })
    }
}

/// The evaluators a spec can name.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum EvalKind {
    Heuristic,
    Perfect,
    Net,
}

/// The evaluator and search depth a spec names, or `None` for `random`.
fn parse(spec: &str) -> Result<Option<(EvalKind, u32)>, BotError> {
    let (kind, arg) = match spec.split_once(':') {
        Some((kind, arg)) => (kind, Some(arg)),
        None => (spec, None),
    };
    let depth = |default: u32| -> Result<u32, BotError> {
        let Some(arg) = arg else { return Ok(default) };
        match arg.parse::<u32>() {
            Ok(d) if d <= MAX_SEARCH_DEPTH => Ok(d),
            Ok(d) => Err(BotError::Invalid(format!(
                "search depth {d} in '{spec}' is above the maximum of {MAX_SEARCH_DEPTH}"
            ))),
            Err(_) => Err(BotError::Invalid(format!("bad search depth in '{spec}'"))),
        }
    };
    let no_arg = || match arg {
        None => Ok(()),
        Some(_) => Err(BotError::Invalid(format!("'{kind}' takes no ':' argument (bots: {SPECS})"))),
    };
    Ok(match kind {
        "random" => {
            no_arg()?;
            None
        }
        "greedy" => {
            no_arg()?;
            Some((EvalKind::Heuristic, 0))
        }
        "expectimax" => Some((EvalKind::Heuristic, depth(2)?)),
        "perfect" => {
            no_arg()?;
            Some((EvalKind::Perfect, 0))
        }
        "net" => Some((EvalKind::Net, depth(0)?)),
        _ => return Err(BotError::Invalid(format!("unknown bot '{spec}' (bots: {SPECS})"))),
    })
}

/// The search a spec refers to, or `None` for `random`.
pub fn search_for(spec: &str, ctx: &BotContext) -> Result<Option<Search>, BotError> {
    let Some((kind, depth)) = parse(spec)? else { return Ok(None) };
    let eval: Arc<dyn Evaluator> = match kind {
        EvalKind::Heuristic => Arc::new(Heuristic),
        EvalKind::Perfect => ctx.complete_db()?.clone(),
        EvalKind::Net => ctx
            .net
            .clone()
            .ok_or_else(|| BotError::Unavailable("the neural bot needs a trained network file".into()))?,
    };
    Ok(Some(Search { eval, depth }))
}

/// Whether two specs name the same search, such as `expectimax` and `expectimax:2`, or
/// `greedy` and `expectimax:0`: the same deterministic bot. (`random` bots differ by seed.)
pub fn same_search(a: &str, b: &str) -> Result<bool, BotError> {
    let a = parse(a)?;
    Ok(a.is_some() && a == parse(b)?)
}

/// Builds the bot a spec describes; `seed` drives its random choices, if any.
pub fn make_bot(spec: &str, ctx: &BotContext, seed: u64) -> Result<Box<dyn Bot>, BotError> {
    Ok(match search_for(spec, ctx)? {
        None => Box::new(RandomBot::new(seed)),
        Some(search) => Box::new(SearchBot { search }),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn specs() {
        let ctx = BotContext::default();
        for ok in ["random", "greedy", "expectimax", "expectimax:0", "expectimax:4"] {
            assert!(make_bot(ok, &ctx, 1).is_ok(), "{ok}");
        }
        assert_eq!(search_for("expectimax", &ctx).unwrap().unwrap().depth, 2);
        for (a, b) in [("expectimax", "expectimax:2"), ("greedy", "expectimax:0"), ("net", "net:0")] {
            assert!(same_search(a, b).unwrap() && same_search(b, a).unwrap(), "{a} {b}");
        }
        for (a, b) in [("expectimax", "expectimax:1"), ("greedy", "net"), ("random", "random")] {
            assert!(!same_search(a, b).unwrap(), "{a} {b}");
        }
        for bad in ["expectimax:5", "expectimax:x", "random:1", "greedy:2", "perfect:1", "net:9", "nope", ""] {
            assert!(matches!(make_bot(bad, &ctx, 1), Err(BotError::Invalid(_))), "{bad}");
        }
        for missing in ["perfect", "net", "net:1"] {
            assert!(matches!(make_bot(missing, &ctx, 1), Err(BotError::Unavailable(_))), "{missing}");
        }
        assert!(BotContext::load(Some(Path::new("no/such/dir")), None).is_err());
    }

    #[test]
    fn perfect_play_needs_the_complete_database() {
        use crate::solver::{SolveConfig, solve_all};
        let dir = std::env::temp_dir().join(format!("senet_bots_test_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        solve_all(&Rules::KENDALL5, &dir, &SolveConfig { max_sum: 2, ..SolveConfig::DEFAULT }).unwrap();
        let ctx = BotContext::load(Some(&dir), None).unwrap();
        let err = make_bot("perfect", &ctx, 1).err().unwrap();
        assert!(matches!(&err, BotError::Unavailable(msg) if msg.contains("complete")), "{err}");
        drop(ctx);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn best_index_takes_the_first_of_near_ties() {
        assert_eq!(best_index(&[0.2]), 0);
        assert_eq!(best_index(&[0.2, 0.5, 0.4]), 1);
        assert_eq!(best_index(&[0.5, 0.5 + TIE_TOLERANCE / 2.0, 0.1]), 0);
    }

    #[test]
    fn greedy_prefers_bearing_off() {
        let r = Rules::KENDALL5;
        let pos = Pos::from_squares(&[10, 30], &[1, 2]).unwrap();
        let moves = crate::movegen::legal_moves(&r, pos, 1);
        assert_eq!(moves.len(), 2);
        let mut bot = make_bot("greedy", &BotContext::default(), 0).unwrap();
        let chosen = bot.choose(&r, pos, 1, &moves);
        assert_eq!(moves[chosen].kind, crate::movegen::Kind::Off);
    }
}
