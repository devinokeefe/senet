"""What the solved game says about Senet strategy.

    python -m senet.insights --db db/kendall5 [--games 20000] [--out docs/INSIGHTS.md]

Writes docs/INSIGHTS.md: the first-player advantage, the best opening move for
every throw, and statistics from perfect self-play; and the raw results, with their
provenance, as docs/INSIGHTS.json (--json). The default paths are in the source checkout,
or in the current directory for an installed copy of the package. The self-play games run
on parallel threads, each drawing its throws from its own seeded generator: a rerun of the
same code on the same database gives the same numbers.
"""

from __future__ import annotations

import argparse
import math
import os
import random
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, NamedTuple

from . import EXTRA_THROWS, THROW_PROBS, Engine, Move, legal_moves, start, throw_sticks
from .provenance import home, provenance, write_report
from .provenance import render as render_provenance

COMEBACK_THRESHOLDS = (0.25, 0.10, 0.05, 0.01)
SELFPLAY_SEED = 2026

# Opening moves whose value is within this of the best are marked optimal (as the web
# app's hints mark the best moves).
OPTIMAL_TOLERANCE = 1e-9


def describe(m: Move) -> str:
    if m.kind == "off":
        return f"{m.frm}→off"
    if m.kind == "water":
        return f"{m.frm}→27 (drowns, back to {m.to})"
    return f"{m.frm}→{m.to}" + (" swap" if m.kind == "swap" else "") + (" (backward)" if m.back else "")


def opening_moves(eng: Engine) -> dict[int, list[tuple[str, float]]]:
    """For every throw, White's legal first moves with White's win probability after each, best first."""
    me, opp = start()
    table: dict[int, list[tuple[str, float]]] = {}
    for t in THROW_PROBS:
        moves = zip(legal_moves(me, opp, t), eng.move_values(me, opp, t), strict=True)
        table[t] = sorted(((describe(m), v) for m, v in moves), key=lambda mv: -mv[1])
    return table


class Game(NamedTuple):
    """One game of perfect self-play."""

    white_won: bool
    throws: int
    forfeits: int  # throws with no legal move
    choices: int  # decisions between two or more legal moves
    kinds: Counter[str]  # the moves played, by kind
    winner_low: float  # the lowest win probability the winner had during the game


def play(eng: Engine, rng: random.Random) -> Game:
    """A game of perfect self-play from the opening, with the throws drawn from `rng`."""
    me, opp = start()
    side = 0  # 0 = White, 1 = Black; (me, opp) is always the view of `side`
    thrown: list[tuple[int, int, int]] = []  # (me, opp, side) at every throw
    kinds: Counter[str] = Counter()
    forfeits = choices = 0
    while True:
        thrown.append((me, opp, side))
        t = throw_sticks(rng)
        moves = legal_moves(me, opp, t)
        if not moves:
            forfeits += 1
            me, opp, side = opp, me, 1 - side
            continue
        if len(moves) > 1:
            choices += 1
            choice = eng.choose(me, opp, t, "perfect")
            if choice is None:
                raise RuntimeError("the perfect bot returned no move although legal moves exist")
            m = moves[choice]
        else:
            m = moves[0]
        kinds[m.kind + ("-back" if m.back else "")] += 1
        if m.wins:
            break
        if t in EXTRA_THROWS:
            me, opp = m.me_after, m.opp_after
        else:
            me, opp, side = m.opp_after, m.me_after, 1 - side
    # The thrower's win probability at every throw, looked up in one call.
    values = eng.values([p[0] for p in thrown], [p[1] for p in thrown]).tolist()
    low = min(v if s == side else 1 - v for (_, _, s), v in zip(thrown, values, strict=True))
    return Game(side == 0, len(thrown), forfeits, choices, kinds, low)


def selfplay(eng: Engine, games: int, seed: int, workers: int | None = None) -> dict[str, Any]:
    """Statistics of `games` perfect-vs-perfect games, played on `workers` threads (by default
    one per CPU, at most 8: more would mostly wait for Python's global interpreter lock). Game
    g draws its throws from a generator seeded with `seed` and g, so the results do not depend
    on the threads."""
    with ThreadPoolExecutor(max_workers=workers or min(8, os.cpu_count() or 1)) as pool:
        played = list(pool.map(lambda g: play(eng, random.Random(seed << 32 | g)), range(games)))
    kinds: Counter[str] = Counter()
    for game in played:
        kinds.update(game.kinds)
    total_moves = sum(kinds.values())
    return {
        "games": games,
        "white_wins": sum(game.white_won for game in played),
        "avg_throws": sum(game.throws for game in played) / games,
        "kinds": {k: v / total_moves for k, v in kinds.items()},
        "forfeits_per_game": sum(game.forfeits for game in played) / games,
        "choices_per_game": sum(game.choices for game in played) / games,
        # The games whose winner was below each win probability at some point.
        "comeback_games": {thr: sum(game.winner_low < thr for game in played) for thr in COMEBACK_THRESHOLDS},
    }


def analyze(eng: Engine, games: int, seed: int = SELFPLAY_SEED) -> dict[str, Any]:
    """The value of the opening, the value of every first move, and `games` games of perfect self-play."""
    me, opp = start()
    return {
        "white_win_prob": eng.value(me, opp),
        "openings": opening_moves(eng),
        "selfplay": selfplay(eng, games, seed),
    }


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """The Wilson score interval (95% for the default z) of a proportion of k in n."""
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z / (1 + z * z / n) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, centre - half), min(1.0, centre + half)


def places(p: float) -> int:
    """Decimal places for fraction p as a percentage: one, or two significant figures below 1%."""
    return 1 if p >= 0.01 or p <= 0 else 1 - math.floor(math.log10(p * 100))


def share(k: int, n: int, of: str = "", decimals: int | None = None) -> str:
    """k of n as a percentage `of` something with its 95% interval: "2.5% of games (2.3–2.7%)"."""
    p, (lo, hi) = k / n, wilson(k, n)
    d = places(p) if decimals is None else decimals
    return f"{p * 100:.{d}f}%{of} ({lo * 100:.{d}f}–{hi * 100:.{d}f}%)"


def render(r: dict[str, Any], raw: str | None = None) -> str:
    """The markdown report for the results of `analyze` (with their provenance, if recorded),
    which links the raw results at `raw` if given. `r` may also be those results read back
    from their JSON, whose keys are strings."""
    v0, s = r["white_win_prob"], r["selfplay"]
    lead = f"{abs(2 * v0 - 1) * 100:.2f} points"
    lines = [
        "# What the solution says",
        "",
        "Ruleset: Modern Kendall, 5 pieces ([RULES.md](../RULES.md))."
        + (f" The raw results are in [{Path(raw).name}]({raw})." if raw else ""),
        "",
        "## First-player advantage",
        "",
        f"With perfect play, **White, who throws first, wins {v0:.4%}** of games, and Black wins {1 - v0:.4%}.",
        f"Moving first outweighs Black's head start (front piece on 10 vs 9): White leads by {lead}."
        if v0 >= 0.5
        else f"Black's head start (front piece on 10 vs 9) outweighs moving first: Black leads by {lead}.",
        "",
        "## The best opening move for every throw",
        "",
        "Win probability for White after each legal first move. ★ = optimal.",
        "",
        "| Throw (prob.) | Move | White wins |",
        "|---|---|---:|",
    ]
    for key, moves in r["openings"].items():
        t = int(key)
        best = max((v for _, v in moves), default=0.0)
        for i, (text, v) in enumerate(moves):
            star = " ★" if best - v < OPTIMAL_TOLERANCE else ""
            label = f"{t} ({THROW_PROBS[t] * 16:.0f}/16)" if i == 0 else ""
            lines.append(f"| {label} | {text}{star} | {v:.3%} |")
    n, k = s["games"], s["kinds"]
    comebacks = []
    for i, (key, wins) in enumerate(s["comeback_games"].items()):
        below, of = (f"below a {float(key):.0%} chance", " of games") if i == 0 else (f"below {float(key):.0%}", "")
        comebacks.append(f"never {below}" if wins == 0 else f"{below} in {share(wins, n, of)}")
    played = {
        "ordinary": k.get("move", 0),
        "swaps": k.get("swap", 0),
        "forced backward": k.get("swap-back", 0) + k.get("move-back", 0),
        "into the House of Water": k.get("water", 0),
        "bearing off": k.get("off", 0),
    }
    lines += [
        "",
        f"## Perfect self-play ({n:,} games)",
        "",
        "Ranges are 95% intervals.",
        "",
        f"* White won {share(s['white_wins'], n, decimals=2)}; the database's value is {v0:.2%}.",
        f"* A game lasts {s['avg_throws']:.0f} throws on average, with {s['choices_per_game']:.0f} real decisions "
        f"(two or more legal moves) and {s['forfeits_per_game']:.1f} forfeited throws.",
        "* Moves played: " + ", ".join(f"{p * 100:.{places(p)}f}% {what}" for what, p in played.items()) + ".",
        "* Comebacks: winners had been " + ", ".join(comebacks) + ".",
        "",
    ]
    if "provenance" in r:
        lines.append(render_provenance(r["provenance"]))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", type=Path, default=home() / "db" / "kendall5")
    ap.add_argument("--games", type=int, default=20000, help="perfect self-play games")
    ap.add_argument("--out", type=Path, default=home() / "docs" / "INSIGHTS.md")
    ap.add_argument("--json", type=Path, default=home() / "docs" / "INSIGHTS.json", help="raw results")
    args = ap.parse_args(argv)
    if args.games < 1:
        ap.error("--games must be at least 1")
    # Before the run: the provenance is that of the code and files the run starts with.
    origin = provenance(db=args.db, games=args.games, seed=SELFPLAY_SEED, optimal_tolerance=OPTIMAL_TOLERANCE)
    with Engine(db=args.db) as eng:
        results = analyze(eng, args.games)
    results["provenance"] = origin
    write_report(results, args.out, args.json, render)


if __name__ == "__main__":
    main()
