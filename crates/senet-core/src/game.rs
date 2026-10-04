//! Game simulation, matches between bots, and decision-quality analysis against the
//! perfect-play database.

use crate::board::{Pos, Rules};
use crate::bots::{Bot, BotContext, BotError, make_bot, same_search};
use crate::eval::move_value;
use crate::movegen::{Move, MoveList, Outcome, gen_moves};
use crate::rng::Rng;
use rayon::prelude::*;
use serde::Serialize;

/// Safety valve against pathological endless games (never reached in practice).
const MAX_THROWS: u32 = 100_000;

/// A move that gives away more than this much win probability counts as an error: twice
/// the estimated precision of the database's values (docs/ACCURACY.md), below which the
/// database cannot tell the better of two moves for certain.
pub const ERROR_THRESHOLD: f64 = 2e-6;

/// A move played in a game, as shown to a `play_game` observer.
pub struct Decision<'a> {
    /// 0 = White, 1 = Black.
    pub side: u8,
    pub pos: Pos,
    pub t: u8,
    /// The legal moves (one if the move was forced) and the index of the one played.
    pub moves: &'a [Move],
    pub chosen: usize,
}

#[derive(Clone, Copy, Debug)]
pub struct GameResult {
    /// 0 = White won, 1 = Black won.
    pub winner: u8,
    pub throws: u32,
    /// Moves made, including forced ones (a throw without a legal move makes none).
    pub moves_played: u32,
}

/// Plays one game; `bots[0]` is White (moves first). `observer` sees every move played;
/// forced moves are played without asking the bot.
pub fn play_game(
    rules: &Rules,
    bots: &mut [Box<dyn Bot>; 2],
    dice: &mut Rng,
    mut observer: Option<&mut dyn FnMut(&Decision)>,
) -> GameResult {
    let mut pos = rules.start();
    let mut side = 0u8;
    let mut ml = MoveList::new();
    let mut moves_played = 0;
    for throws in 1..=MAX_THROWS {
        let t = dice.throw();
        gen_moves(rules, pos, t, &mut ml);
        if ml.is_empty() {
            pos = pos.flip();
            side ^= 1;
            continue;
        }
        let chosen = if ml.len() == 1 { 0 } else { bots[side as usize].choose(rules, pos, t, &ml) };
        if let Some(observer) = observer.as_mut() {
            observer(&Decision { side, pos, t, moves: &ml, chosen });
        }
        moves_played += 1;
        match ml[chosen].outcome(rules, t) {
            Outcome::Won => return GameResult { winner: side, throws, moves_played },
            Outcome::ThrowAgain(next) => pos = next,
            Outcome::OpponentThrows(next) => {
                pos = next;
                side ^= 1;
            }
        }
    }
    // Treat as a coin flip; never happens in practice.
    GameResult { winner: (dice.next_u32() & 1) as u8, throws: MAX_THROWS, moves_played }
}

#[derive(Clone, Debug, Serialize)]
pub struct MatchResult {
    pub bot_a: String,
    pub bot_b: String,
    pub games: u64,
    pub a_wins: u64,
    /// Games where A played White (moved first) and A's wins in them.
    pub a_white_games: u64,
    pub a_white_wins: u64,
    pub avg_throws: f64,
    /// A's win rate and the half-width of a 95% confidence interval around it (see `ci95`).
    pub a_win_rate: f64,
    pub ci95: f64,
}

/// Running totals of a match.
#[derive(Default)]
struct MatchTally {
    games: u64,
    a_wins: u64,
    a_white_games: u64,
    a_white_wins: u64,
    throws: u64,
    /// Sum over the pairs of A's wins in the pair (0, 1 or 2), squared.
    a_pair_wins_sq: u64,
}

impl MatchTally {
    fn merge(mut self, o: MatchTally) -> MatchTally {
        self.games += o.games;
        self.a_wins += o.a_wins;
        self.a_white_games += o.a_white_games;
        self.a_white_wins += o.a_white_wins;
        self.throws += o.throws;
        self.a_pair_wins_sq += o.a_pair_wins_sq;
        self
    }
}

/// z of a two-sided 95% interval.
const Z95: f64 = 1.96;
/// The fewest pairs, and the fewest games A must win and lose, for which `ci95` trusts the
/// normal approximation with the observed spread of the pair scores.
const LARGE_SAMPLE_PAIRS: u64 = 30;
const LARGE_SAMPLE_GAMES: u64 = 10;

/// Half-width of a 95% confidence interval for A's win rate, centred on the observed rate,
/// from `pairs` duplicate pairs in which A won `a_wins` games (`a_pair_wins_sq`: the sum
/// over the pairs of A's wins in the pair, squared). The two games of a pair share their
/// dice, so the pairs are the independent samples, each scoring 0, 1/2 or 1 for A.
/// * The same deterministic bot in both seats (`mirrored`) plays each pair's game twice,
///   once from each seat: A wins exactly half the games, without sampling error.
/// * A large sample (`LARGE_SAMPLE_PAIRS` pairs, `LARGE_SAMPLE_GAMES` games won and lost
///   by A, and pair scores that differ) uses the normal approximation with the observed
///   spread of the pair scores, which the pairing makes small.
/// * Otherwise that spread can mislead (two pairs that both score 0 show none), and the
///   Wilson score interval is used instead, with p(1 - p), the largest variance a score
///   of mean p can have; the half-width is that of the narrowest symmetric interval
///   containing it.
fn ci95(pairs: u64, a_wins: u64, a_pair_wins_sq: u64, mirrored: bool) -> f64 {
    if mirrored {
        return 0.0;
    }
    let n = pairs as f64;
    let mean = a_wins as f64 / (2.0 * n);
    // By Cauchy-Schwarz, n * sum(w^2) = (sum w)^2 exactly when every pair has the same w.
    let spread = u128::from(pairs) * u128::from(a_pair_wins_sq) != u128::from(a_wins).pow(2);
    let a_losses = 2 * pairs - a_wins;
    if pairs >= LARGE_SAMPLE_PAIRS && a_wins.min(a_losses) >= LARGE_SAMPLE_GAMES && spread {
        let var = (a_pair_wins_sq as f64 / 4.0 - n * mean * mean) / (n - 1.0);
        return Z95 * (var / n).sqrt();
    }
    let z2n = Z95 * Z95 / n;
    let centre = (mean + z2n / 2.0) / (1.0 + z2n);
    let half = Z95 / (1.0 + z2n) * (mean * (1.0 - mean) / n + z2n / (4.0 * n)).sqrt();
    (mean - (centre - half)).max(centre + half - mean)
}

/// Plays `pairs` pairs of games between A and B. Both games of a pair use the same dice
/// seed with colours swapped, which removes most of the luck from the comparison.
pub fn run_match(
    rules: &Rules,
    spec_a: &str,
    spec_b: &str,
    ctx: &BotContext,
    pairs: u64,
    seed: u64,
) -> Result<MatchResult, BotError> {
    // Validate the arguments once, up front: the per-game constructions below cannot fail.
    if pairs == 0 {
        return Err(BotError::Invalid("a match needs at least one pair of games".into()));
    }
    make_bot(spec_a, ctx, 0)?;
    make_bot(spec_b, ctx, 0)?;
    // A bot that searches is deterministic; only `random` uses its seed.
    let mirrored = same_search(spec_a, spec_b)?;
    let tally = (0..pairs)
        .into_par_iter()
        .map(|k| {
            let pair_seed = seed ^ (k.wrapping_mul(0x9E37_79B9_7F4A_7C15)).rotate_left(17);
            let mut tally = MatchTally::default();
            let mut a_pair_wins = 0;
            for a_is_white in [true, false] {
                let a = make_bot(spec_a, ctx, pair_seed ^ 0xA).expect("validated spec");
                let b = make_bot(spec_b, ctx, pair_seed ^ 0xB).expect("validated spec");
                let mut bots = if a_is_white { [a, b] } else { [b, a] };
                let g = play_game(rules, &mut bots, &mut Rng::new(pair_seed), None);
                let a_won = (g.winner == 0) == a_is_white;
                tally.games += 1;
                a_pair_wins += a_won as u64;
                tally.throws += g.throws as u64;
                if a_is_white {
                    tally.a_white_games += 1;
                    tally.a_white_wins += a_won as u64;
                }
            }
            tally.a_wins = a_pair_wins;
            tally.a_pair_wins_sq = a_pair_wins * a_pair_wins;
            tally
        })
        .reduce(MatchTally::default, MatchTally::merge);
    let games = tally.games as f64;
    let a_win_rate = tally.a_wins as f64 / games;
    Ok(MatchResult {
        bot_a: spec_a.to_string(),
        bot_b: spec_b.to_string(),
        games: tally.games,
        a_wins: tally.a_wins,
        a_white_games: tally.a_white_games,
        a_white_wins: tally.a_white_wins,
        avg_throws: tally.throws as f64 / games,
        a_win_rate,
        ci95: ci95(pairs, tally.a_wins, tally.a_pair_wins_sq, mirrored),
    })
}

#[derive(Clone, Debug, Serialize)]
pub struct Quality {
    pub bot: String,
    pub games: u64,
    /// Decisions with at least two legal moves.
    pub decisions: u64,
    /// Decisions where the chosen move gave away more than `ERROR_THRESHOLD`.
    pub errors: u64,
    /// Average win probability given away per decision / per game (vs perfect play).
    pub avg_loss_per_decision: f64,
    pub avg_loss_per_game: f64,
    pub max_loss: f64,
}

/// Running totals of a quality analysis.
#[derive(Default)]
struct QualityTally {
    games: u64,
    decisions: u64,
    errors: u64,
    total_loss: f64,
    max_loss: f64,
}

impl QualityTally {
    fn merge(mut self, o: QualityTally) -> QualityTally {
        self.games += o.games;
        self.decisions += o.decisions;
        self.errors += o.errors;
        self.total_loss += o.total_loss;
        self.max_loss = self.max_loss.max(o.max_loss);
        self
    }
}

/// Measures how much win probability `spec` gives away per decision compared with
/// perfect play (judged by the database in `ctx`, which must be complete), over `games`
/// games against `opponent`, playing White in every other game.
pub fn analyze_quality(
    rules: &Rules,
    spec: &str,
    opponent: &str,
    ctx: &BotContext,
    games: u64,
    seed: u64,
) -> Result<Quality, BotError> {
    make_bot(spec, ctx, 0)?;
    make_bot(opponent, ctx, 0)?;
    let db = ctx.complete_db()?.as_ref();
    let tallies: Vec<QualityTally> = (0..games)
        .into_par_iter()
        .map(|k| {
            let game_seed = seed ^ (k.wrapping_mul(0xD1B5_4A32_D192_ED03)).rotate_left(29);
            let my_side = (k % 2) as u8;
            let me = make_bot(spec, ctx, game_seed ^ 1).expect("validated spec");
            let op = make_bot(opponent, ctx, game_seed ^ 2).expect("validated spec");
            let mut bots = if my_side == 0 { [me, op] } else { [op, me] };
            let mut tally = QualityTally { games: 1, ..QualityTally::default() };
            let mut observe = |d: &Decision| {
                if d.side != my_side || d.moves.len() < 2 {
                    return;
                }
                let values: Vec<f64> = d.moves.iter().map(|m| move_value(rules, db, m, d.t, 0)).collect();
                let best = values.iter().copied().fold(0.0, f64::max);
                let loss = (best - values[d.chosen]).max(0.0);
                tally.decisions += 1;
                tally.errors += (loss > ERROR_THRESHOLD) as u64;
                tally.total_loss += loss;
                tally.max_loss = tally.max_loss.max(loss);
            };
            play_game(rules, &mut bots, &mut Rng::new(game_seed), Some(&mut observe));
            tally
        })
        .collect();
    // Summed in game order, so that the floating-point totals do not depend on how the
    // games were shared between threads.
    let tally = tallies.into_iter().fold(QualityTally::default(), QualityTally::merge);
    Ok(Quality {
        bot: spec.to_string(),
        games: tally.games,
        decisions: tally.decisions,
        errors: tally.errors,
        avg_loss_per_decision: tally.total_loss / tally.decisions.max(1) as f64,
        avg_loss_per_game: tally.total_loss / tally.games.max(1) as f64,
        max_loss: tally.max_loss,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_are_reproducible_and_colour_balanced() {
        let ctx = BotContext::default();
        let r = Rules::KENDALL5;
        let m = run_match(&r, "greedy", "random", &ctx, 200, 7).unwrap();
        assert_eq!((m.games, m.a_white_games), (400, 200));
        assert!(m.a_win_rate > 0.55, "greedy should beat random: {m:?}");
        let again = run_match(&r, "greedy", "random", &ctx, 200, 7).unwrap();
        assert_eq!((m.a_wins, m.a_white_wins), (again.a_wins, again.a_white_wins));
        assert!(m.ci95 > 0.0 && m.ci95 < 0.1, "{m:?}");
        // Identical deterministic bots replay the same game in both seats: A wins exactly
        // one game of every pair, so the result is certain.
        let mirror = run_match(&r, "greedy", "greedy", &ctx, 200, 7).unwrap();
        assert_eq!((mirror.a_win_rate, mirror.ci95), (0.5, 0.0));
        let mirror = run_match(&r, "greedy", "expectimax:0", &ctx, 20, 7).unwrap();
        assert_eq!((mirror.a_win_rate, mirror.ci95), (0.5, 0.0), "the same bot by another name");
        // Two random bots are not mirrored, nor is a small sample certain because its pairs
        // happen to score alike: random loses all four games against greedy here.
        let random = run_match(&r, "random", "random", &ctx, 200, 7).unwrap();
        assert!(random.ci95 > 0.05, "{random:?}");
        let small = run_match(&r, "random", "greedy", &ctx, 2, 4).unwrap();
        assert_eq!(small.a_wins, 0);
        assert!(small.ci95 > 0.5, "{small:?}");
        assert!(run_match(&r, "greedy", "nope", &ctx, 1, 7).is_err());
        assert!(matches!(run_match(&r, "greedy", "random", &ctx, 0, 7), Err(BotError::Invalid(_))));
        let quality = |spec: &str| analyze_quality(&r, spec, "random", &ctx, 1, 7).err().unwrap();
        assert!(matches!(quality("greedy"), BotError::Unavailable(_)), "needs the database");
        assert!(matches!(quality("nope"), BotError::Invalid(_)), "a bad spec is reported first");
    }

    #[test]
    fn confidence_intervals() {
        let close = |a: f64, b: f64| (a - b).abs() < 1e-12;
        // A large sample: 40 pairs scoring 1, 1/2, 1/2, 0 ten times each (wins 2, 1, 1, 0),
        // so the mean is 1/2 and the sample variance 5 / 39.
        assert!(close(ci95(40, 40, 60, false), Z95 * (5.0_f64 / 39.0 / 40.0).sqrt()));
        // Small samples, boundaries and pairs that all score alike use the Wilson interval
        // (here symmetric around 1/2); scores that all agree are not taken for certainty.
        let wilson_half = |n: f64| Z95 / (1.0 + Z95 * Z95 / n) * (0.25 / n + Z95 * Z95 / (4.0 * n * n)).sqrt();
        assert!(close(ci95(2, 2, 4, false), wilson_half(2.0)));
        assert!(close(ci95(40, 40, 40, false), wilson_half(40.0)), "every pair split");
        assert_eq!(ci95(40, 40, 40, true), 0.0, "a mirrored match is certain");
        // No wins in 2 pairs: the interval reaches up to z^2 / (n + z^2).
        assert!(close(ci95(2, 0, 0, false), Z95 * Z95 / (2.0 + Z95 * Z95)));
        assert!(close(ci95(2, 4, 8, false), ci95(2, 0, 0, false)), "all wins mirror all losses");
        // Fewer than 10 games lost out of 60: still Wilson, wider than the observed spread.
        let lopsided = ci95(30, 55, 105, false);
        assert!(lopsided > Z95 * ((105.0 / 4.0 - 30.0 * (55.0f64 / 60.0).powi(2)) / 29.0 / 30.0).sqrt());
        for (pairs, wins, sq) in [(1, 0, 0), (1, 1, 1), (1, 2, 4), (30, 59, 117), (1000, 1999, 3997)] {
            let half = ci95(pairs, wins, sq, false);
            assert!(half > 0.0 && half < 1.0, "{pairs} {wins} {sq}: {half}");
        }
    }

    #[test]
    fn observer_sees_every_move() {
        let r = Rules::KENDALL5;
        let mut bots = [
            make_bot("random", &BotContext::default(), 1).unwrap(),
            make_bot("greedy", &BotContext::default(), 2).unwrap(),
        ];
        let mut seen = 0;
        let mut observe = |d: &Decision| {
            assert!(d.chosen < d.moves.len());
            seen += 1;
        };
        let g = play_game(&r, &mut bots, &mut Rng::new(3), Some(&mut observe));
        assert!(g.winner <= 1 && g.moves_played <= g.throws);
        assert_eq!(seen, g.moves_played);
    }
}
