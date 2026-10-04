"""Benchmark suite: how far is each bot from perfect play, and how do they fare head-to-head?

    python -m senet.bench --db db/kendall5 [--net models/senet_net.bin] [--quick]

Writes docs/BENCHMARKS.md and the raw results, with their provenance (the commit, machine,
engine build, database and network, and the settings), as docs/BENCHMARKS.json. With --quick
(few games, for smoke tests) the defaults are runs/bench_quick.md and runs/bench_quick.json
instead; --out and --json override. The default paths are in the source checkout, or in the
current directory for an installed copy of the package. The results depend only on the seeds
and game counts (the engine adds up each game's results in game order), so a rerun of the
same build on the same database and network reproduces every number but the timings.

* Decision quality: for every decision a bot makes (with 2+ legal moves), the win
  probability it gives away compared with the best move, by the solved database's values.
* Matches: duplicate format — each pair of games replays the same stick throws with
  colours swapped, which cancels most of the luck.

Perfect self-play is in senet.insights (docs/INSIGHTS.md).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

from . import Engine, start
from .provenance import home, provenance, write_report
from .provenance import render as render_provenance

LABELS = {
    "random": "Random mover",
    "greedy": "Greedy heuristic (1 ply)",
    "expectimax:1": "Heuristic + expectimax, 1 throw",
    "expectimax:2": "Heuristic + expectimax, 2 throws",
    "expectimax:3": "Heuristic + expectimax, 3 throws",
    "net": "Distilled neural net (1 ply)",
    "net:1": "Distilled neural net + 1-throw search",
    "perfect": "Perfect play (solved database)",
}
SLOW = {"expectimax:3": 4}  # game counts are divided by this for slow bots
SEEDS = {"quality": 11, "matches": 23}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", type=Path, default=home() / "db" / "kendall5")
    ap.add_argument("--net", type=Path, help="SNN1 network file; adds the 'net' and 'net:1' bots")
    ap.add_argument("--quick", action="store_true", help="few games, for smoke tests")
    ap.add_argument(
        "--out", type=Path, help="markdown report (default docs/BENCHMARKS.md; --quick: runs/bench_quick.md)"
    )
    ap.add_argument(
        "--json", type=Path, help="raw results (default docs/BENCHMARKS.json; --quick: runs/bench_quick.json)"
    )
    args = ap.parse_args(argv)
    base = home()
    if args.out is None:
        args.out = (base / "runs" / "bench_quick.md") if args.quick else (base / "docs" / "BENCHMARKS.md")
    if args.json is None:
        args.json = (base / "runs" / "bench_quick.json") if args.quick else (base / "docs" / "BENCHMARKS.json")
    return args


def run(eng: Engine, bots: list[str], quality_games: int, pairs: int) -> dict[str, Any]:
    """The value of the opening, the decision quality of every bot, and perfect play against each of them."""
    results: dict[str, Any] = {"quality": {}, "matches": {}, "start": {}}
    me, opp = start()
    results["start"]["white_win_prob"] = eng.value(me, opp)

    for b in bots:
        t = time.perf_counter()
        q = eng.quality(b, vs="perfect", games=quality_games // SLOW.get(b, 1), seed=SEEDS["quality"])
        q["seconds"] = time.perf_counter() - t
        results["quality"][b] = q
        print(
            f"quality {b:14s} loss/decision {q['avg_loss_per_decision']:.5f}  "
            f"errors {q['errors'] / max(q['decisions'], 1):6.2%}  ({q['seconds']:.0f}s)",
            flush=True,
        )

    for b in bots:
        if b == "perfect":
            continue
        t = time.perf_counter()
        r = eng.match("perfect", b, pairs=pairs // SLOW.get(b, 1), seed=SEEDS["matches"])
        r["seconds"] = time.perf_counter() - t
        results["matches"][f"perfect vs {b}"] = r
        print(f"match perfect vs {b:14s} {r['a_win_rate']:.2%} ± {r['ci95']:.2%}  ({r['seconds']:.0f}s)", flush=True)
    return results


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    bots = ["random", "greedy", "expectimax:1", "expectimax:2", "expectimax:3"]
    if args.net:
        bots += ["net", "net:1"]
    bots.append("perfect")
    quality_games, pairs = (400, 200) if args.quick else (20000, 50000)
    # Before the run: the provenance is that of the code and files the run starts with.
    origin = provenance(
        db=args.db, net=args.net, quality_games=quality_games, pairs=pairs, seeds=SEEDS, divided_for_slow_bots=SLOW
    )
    with Engine(db=args.db, net=args.net) as eng:
        results = run(eng, bots, quality_games, pairs)
    results["provenance"] = origin
    write_report(results, args.out, args.json, render)


def render(r: dict[str, Any], raw: str | None = None) -> str:
    """The markdown report for the results of `run` (with their provenance, if recorded), which
    links the raw results at `raw` if given."""
    s = r["start"]
    lines = [
        "# Benchmarks",
        "",
        "Generated by `python -m senet.bench`"
        + (f"; the raw results are in [{Path(raw).name}]({raw})" if raw else "")
        + ". Ruleset: Modern Kendall, 5 pieces (RULES.md).",
        "",
        "## The opening, solved",
        "",
        f"With perfect play by both sides, **White (moving first) wins {s['white_win_prob']:.4%}** of games.",
        "Perfect self-play, which checks this value against actual play, is in docs/INSIGHTS.md.",
        "",
        "## Decision quality versus perfect play",
        "",
        "Win probability given away per decision (only decisions with 2+ legal moves count), measured",
        "exactly with the solved database over the listed number of games against the perfect bot,",
        "alternating colours. An error is a decision that gives away more than 0.0002% (2e-6), twice",
        "the database's estimated precision.",
        "",
        "| Bot | Games | Decisions | Error rate | Avg loss / decision | Avg loss / game | Worst single loss |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for b, q in r["quality"].items():
        rate = q["errors"] / max(q["decisions"], 1)
        lines.append(
            f"| {LABELS.get(b, b)} | {q['games']:,} | {q['decisions']:,} | {rate:.2%} "
            f"| {q['avg_loss_per_decision'] * 100:.3f}% | {q['avg_loss_per_game'] * 100:.2f}% "
            f"| {q['max_loss'] * 100:.1f}% |"
        )
    lines += [
        "",
        "## Head-to-head (duplicate matches)",
        "",
        "Both games of a pair replay the same stick throws with colours swapped, so the pairs are the",
        "independent samples: the 95% interval comes from the spread of Perfect's score per pair.",
        "",
        "| Match | Games | Pairs | Perfect's win rate | 95% CI |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, m in r["matches"].items():
        opp = name.split(" vs ", 1)[1]
        lines.append(
            f"| Perfect vs {LABELS.get(opp, opp)} | {m['games']:,} | {m['games'] // 2:,} | {m['a_win_rate']:.2%} "
            f"| ± {m['ci95']:.2%} |"
        )
    lines.append("")
    if "provenance" in r:
        lines.append(render_provenance(r["provenance"]))
    return "\n".join(lines)


if __name__ == "__main__":
    main()
