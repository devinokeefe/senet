"""The playing strength of the network's bots on a fixed corpus of held-out decisions.

The corpus (models/strength_positions.json) holds decisions (a position and a throw with
two or more legal moves) from games of a perfect player that explores (it plays a random
move a third of the time), whose positions are not in the network's training data (the
positions after their moves may be). Each decision has one or more categories:

    water      a move lands on the House of Water
    backward   every legal move goes backward
    blockade   the opponent has three pieces in a row
    bear-off   a move bears a piece off
    near-tie   the two best moves are at most NEAR_TIE apart, but not equal
    typical    a random sample of the decisions

and the win probability after each of its legal moves, from the solved database. A
bot's loss on a decision is the best move's win probability minus that of the move it
plays. The check measures the mean and the worst loss of the network's bots per category,
and fails if one is over its limit: that of the reference network (the committed one),
times MARGIN, plus a slack.

    python -m senet_train.strength [--net models/senet_net.bin] [--db db/kendall5]
    python -m senet_train.strength --rebase [--net models/senet_net.bin]
    python -m senet_train.strength --generate --db db/kendall5 [--exclude runs/train.bin]

--db first re-measures the corpus's values with the database. --rebase makes the
network the reference. --generate writes a new corpus, without the positions of the
training data given with --exclude, with the network as the reference. Needs the
senet_ffi library, not PyTorch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np

from senet import EXTRA_THROWS, Engine, Move, legal_moves, mask, squares, start, throw_sticks
from senet._lib import ROOT

CORPUS = ROOT / "models" / "strength_positions.json"
NET = ROOT / "models" / "senet_net.bin"
BOTS = ("net", "net:1")
CATEGORIES = ("water", "backward", "blockade", "bear-off", "near-tie", "typical")
NEAR_TIE = 1e-3  # about half the corpus's decisions are this close
# Generation: the exploring player, the share of decisions sampled as typical, the
# decisions collected per category (most positions of the backward decisions, about 93%,
# are in the training data) and the decisions kept per category.
EXPLORE = 1 / 3
TYPICAL_RATE = 0.02
CANDIDATES = 6000
KEEP = 300
MAX_GAMES = 100_000
SEED = 22
# A limit is the reference network's value times MARGIN, plus a slack: about twice its
# mean losses (some 1e-4 for net, 4e-5 for net:1) and 1.5 times its worst plus 0.5%.
MARGIN = 1.5
MEAN_SLACK = 5e-5
WORST_SLACK = 0.005
RECORD = np.dtype([("me", "<u4"), ("opp", "<u4"), ("value", "<f4")])  # of the training data (gen-data)


def has_blockade(m: int) -> bool:
    """Whether the pieces of mask `m` hold three squares in a row."""
    return m & (m >> 1) & (m >> 2) != 0


def categories(opp: int, moves: list[Move], values: list[float]) -> list[str]:
    """The categories (but `typical`) of a decision with these legal moves and database values."""
    found = []
    if any(m.kind == "water" for m in moves):
        found.append("water")
    if all(m.back for m in moves):
        found.append("backward")
    if has_blockade(opp):
        found.append("blockade")
    if any(m.kind == "off" for m in moves):
        found.append("bear-off")
    best, second = sorted(values, reverse=True)[:2]
    if 0 < best - second <= NEAR_TIE:
        found.append("near-tie")
    return found


def collect(eng: Engine, seed: int) -> list[dict[str, Any]]:
    """Up to CANDIDATES decisions per category, from games of the exploring perfect player."""
    rng = random.Random(seed)
    counts = dict.fromkeys(CATEGORIES, 0)
    seen: set[tuple[int, int, int]] = set()
    found = []
    for _ in range(MAX_GAMES):
        if min(counts.values()) >= CANDIDATES:
            break
        me, opp = start()
        while True:
            t = throw_sticks(rng)
            moves = legal_moves(me, opp, t)
            if not moves:
                me, opp = opp, me
                continue
            choice = 0
            if len(moves) > 1:
                values = eng.move_values(me, opp, t)
                cats = categories(opp, moves, values) + (["typical"] if rng.random() < TYPICAL_RATE else [])
                if any(counts[c] < CANDIDATES for c in cats) and (me, opp, t) not in seen:
                    seen.add((me, opp, t))
                    found.append({"me": squares(me), "opp": squares(opp), "t": t, "categories": cats, "values": values})
                    for c in cats:
                        counts[c] += 1
                choice = rng.randrange(len(moves)) if rng.random() < EXPLORE else int(np.argmax(values))
            m = moves[choice]
            if m.wins:
                break
            me, opp = (m.me_after, m.opp_after) if t in EXTRA_THROWS else (m.opp_after, m.me_after)
    return found


def positions_in(path: Path, keys: np.ndarray, chunk: int = 1 << 23) -> set[int]:
    """Those of the sorted position keys (me << 32 | opp) found in a training data file."""
    hits: set[int] = set()
    with path.open("rb") as f:
        while records := f.read(chunk * RECORD.itemsize):
            r = np.frombuffer(records, dtype=RECORD)
            found = r["me"].astype(np.uint64) << np.uint64(32) | r["opp"]
            at = np.minimum(np.searchsorted(keys, found), len(keys) - 1)
            hits.update(int(k) for k in found[keys[at] == found])
    return hits


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(1 << 24):
            h.update(block)
    return h.hexdigest()


def losses(eng: Engine, entries: list[dict[str, Any]], bot: str) -> np.ndarray:
    """The win probability that `bot` gives away on each decision."""
    out = []
    for e in entries:
        choice = eng.choose(mask(e["me"]), mask(e["opp"]), e["t"], bot=bot)
        assert choice is not None, "a decision has legal moves"
        out.append(max(e["values"]) - e["values"][choice])
    return np.array(out)


def measure(net: Path, entries: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, float]]]:
    """Per bot and category: the number of decisions and the mean and worst loss."""
    with Engine(net=net) as eng:
        result: dict[str, dict[str, dict[str, float]]] = {}
        for bot in BOTS:
            lost = losses(eng, entries, bot)
            result[bot] = {}
            for c in CATEGORIES:
                mine = lost[[c in e["categories"] for e in entries]]
                mean, worst = (float(mine.mean()), float(mine.max())) if len(mine) else (0.0, 0.0)
                result[bot][c] = {"n": len(mine), "mean": mean, "worst": worst}
    return result


def limit(reference: dict[str, float]) -> dict[str, float]:
    """The highest mean and worst loss that pass, from the reference network's."""
    return {"mean": reference["mean"] * MARGIN + MEAN_SLACK, "worst": reference["worst"] * MARGIN + WORST_SLACK}


def reference(net: Path, entries: list[dict[str, Any]]) -> dict[str, Any]:
    return {"file": net.name, "sha256": sha256(net), "stats": measure(net, entries)}


def remeasure(db: Path, entries: list[dict[str, Any]]) -> int:
    """How many decisions' legal moves or values differ from the database's now."""
    with Engine(db=db) as eng:
        return sum(
            len(legal_moves(mask(e["me"]), mask(e["opp"]), e["t"])) != len(e["values"])
            or eng.move_values(mask(e["me"]), mask(e["opp"]), e["t"]) != e["values"]
            for e in entries
        )


def generate(db: Path, net: Path, exclude: Path | None, out: Path) -> None:
    with Engine(db=db) as eng:
        found = collect(eng, SEED)
    excluded: dict[str, Any] | None = None
    if exclude is not None:
        keys = np.unique(np.array([mask(e["me"]) << 32 | mask(e["opp"]) for e in found], dtype=np.uint64))
        trained = positions_in(exclude, keys)
        print(f"{len(trained)} of {len(keys)} positions are in {exclude}: left out", flush=True)
        found = [e for e in found if mask(e["me"]) << 32 | mask(e["opp"]) not in trained]
        excluded = {"file": exclude.name, "bytes": exclude.stat().st_size, "sha256": sha256(exclude)}
    kept = dict.fromkeys(CATEGORIES, 0)
    entries = []
    for e in found:
        if any(kept[c] < KEEP for c in e["categories"]):
            entries.append(e)
            for c in e["categories"]:
                kept[c] += 1
    if min(kept.values()) < KEEP:
        raise SystemExit(f"too few decisions in some categories: {kept}")
    corpus = {
        "about": "python -m senet_train.strength: decisions with the database's win probability after each legal move",
        "seed": SEED,
        "near_tie": NEAR_TIE,
        "excluded_training_data": excluded,
        "reference": reference(net, entries),
        "decisions": entries,
    }
    write(out, corpus)
    print(f"wrote {len(entries)} decisions to {out}: {kept}")


def write(path: Path, corpus: dict[str, Any]) -> None:
    path.write_text(json.dumps(corpus, indent=1) + "\n", encoding="utf-8", newline="\n")


def check(net: Path, corpus: dict[str, Any]) -> bool:
    """Measures the network's bots and compares them with the limits."""
    stats = measure(net, corpus["decisions"])
    ok = True
    print(f"{'bot':6} {'category':9} {'n':>5}  {'mean loss':>9} (max)    {'worst loss':>10} (max)")
    for bot, by_category in stats.items():
        for c, s in by_category.items():
            most = limit(corpus["reference"]["stats"][bot][c])
            bad = s["mean"] > most["mean"] or s["worst"] > most["worst"]
            ok &= not bad
            print(
                f"{bot:6} {c:9} {s['n']:5}  {s['mean']:9.5f} ({most['mean']:.5f})  "
                f"{s['worst']:10.5f} ({most['worst']:.5f})" + ("  FAIL" if bad else "")
            )
    return ok


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--net", type=Path, default=NET, help="SNN1 network file")
    ap.add_argument("--db", type=Path, help="solved database: re-measure the corpus (with --generate: build it)")
    ap.add_argument("--corpus", type=Path, default=CORPUS)
    ap.add_argument("--rebase", action="store_true", help="make the network the reference")
    ap.add_argument("--generate", action="store_true", help="write a new corpus, with the network as the reference")
    ap.add_argument("--exclude", type=Path, help="with --generate: training data whose positions to leave out")
    args = ap.parse_args(argv)
    if args.generate:
        if args.db is None:
            ap.error("--generate needs --db")
        generate(args.db, args.net, args.exclude, args.corpus)
        return 0
    corpus = json.loads(args.corpus.read_text(encoding="utf-8"))
    if args.rebase:
        corpus["reference"] = reference(args.net, corpus["decisions"])
        write(args.corpus, corpus)
        print(f"{args.net} is the reference of {args.corpus}")
        return 0
    if args.db is not None:
        stale = remeasure(args.db, corpus["decisions"])
        print(f"database: {len(corpus['decisions'])} decisions re-measured, {stale} differ")
        if stale:
            return 1
    ok = check(args.net, corpus)
    print("OK" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
