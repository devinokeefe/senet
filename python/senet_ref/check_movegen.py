"""Cross-check a Rust move-generation dump (JSON Lines, docs/FORMATS.md) against rules.py.

Usage:  python -m senet_ref.check_movegen <dump.jsonl> [--show 10]

For every record the legal moves are recomputed with ``rules.legal_moves`` and compared
with the dump, including order (moves must be sorted by (from, to)).  Exit code 1 on any
mismatch or malformed record.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from typing import Any, TypeVar

from .rules import Move, legal_moves, validate_position

T = TypeVar("T")


def _field(obj: dict[str, Any], key: str, kind: type[T]) -> T:
    """``obj[key]``, which must be a JSON value of type ``kind`` (``int`` excludes true and
    false; no number is converted, so 3.75 is not read as 3)."""
    value = obj[key]
    if type(value) is not kind:
        raise ValueError(f"{key!r} is {value!r}, not a JSON {kind.__name__}")
    return value


def _squares(obj: dict[str, Any], key: str) -> list[int]:
    squares = _field(obj, key, list)
    if any(type(s) is not int for s in squares):
        raise ValueError(f"{key!r} is {squares!r}, not a list of squares")
    if squares != sorted(squares):
        raise ValueError(f"{key!r} is not sorted: {squares}")
    return squares


MOVE_KEYS = {"from", "to", "kind", "dir", "me", "opp"}


def _move(obj: Any) -> Move:
    """The move described by one entry of a record's ``moves`` list."""
    if type(obj) is not dict:
        raise ValueError(f"move {obj!r} is not a JSON object")
    if set(obj) != MOVE_KEYS:
        raise ValueError(f"move {obj!r} does not have exactly the keys {sorted(MOVE_KEYS)}")
    return Move(
        frm=_field(obj, "from", int),
        to=_field(obj, "to", int),
        kind=_field(obj, "kind", str),
        dir=_field(obj, "dir", str),
        me_after=tuple(_squares(obj, "me")),
        opp_after=tuple(_squares(obj, "opp")),
    )


def _fmt(m: Move) -> str:
    return f"{m.frm:>2}->{m.to:<2} {m.kind:<5} {m.dir:<4} me={list(m.me_after)} opp={list(m.opp_after)}"


def _describe(
    lineno: int, me: list[int], opp: list[int], t: int, rust: list[Move], ref: list[Move], problem: str
) -> str:
    lines = [f"line {lineno}: me={list(me)} opp={list(opp)} t={t}: {problem}"]
    rust_set, ref_set = set(rust), set(ref)
    lines.append("  python (reference):" + ("" if ref else " <no moves: forfeit>"))
    lines += [f"    {'  ' if m in rust_set else '- '}{_fmt(m)}" for m in ref]
    lines.append("  rust:" + ("" if rust else " <no moves: forfeit>"))
    lines += [f"    {'  ' if m in ref_set else '+ '}{_fmt(m)}" for m in rust]
    return "\n".join(lines)


def check_file(path: str, show: int = 10) -> int:
    records = moves_total = 0
    mismatches = 0
    reasons: Counter[str] = Counter()
    by_throw: Counter[int] = Counter()
    shown = 0

    def report(text: str) -> None:
        nonlocal shown
        if shown < show:
            print(text)
            shown += 1

    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            records += 1
            try:
                obj = json.loads(line)
                if type(obj) is not dict:
                    raise ValueError("not a JSON object")
                me, opp, t = _squares(obj, "me"), _squares(obj, "opp"), _field(obj, "t", int)
                rust = [_move(m) for m in _field(obj, "moves", list)]
                validate_position(me, opp)
                ref = legal_moves(me, opp, t)  # ValueError for a bad throw or a finished game
            except (ValueError, KeyError, json.JSONDecodeError) as e:
                mismatches += 1
                reasons["malformed record"] += 1
                report(f"line {lineno}: malformed record: {e}")
                continue

            moves_total += len(rust)
            if rust == ref:
                continue
            mismatches += 1
            by_throw[t] += 1
            if set(rust) == set(ref) and len(rust) == len(ref):
                problem = "same moves, wrong order (must be sorted by (from, to))"
            elif not ref:
                problem = "python says forfeit, rust lists moves"
            elif not rust:
                problem = "rust says forfeit, python lists moves"
            else:
                problem = "different move sets"
            reasons[problem] += 1
            report(_describe(lineno, me, opp, t, rust, ref, problem))

    print()
    print(f"records checked: {records:,}   moves in dump: {moves_total:,}")
    if mismatches:
        print(f"MISMATCHES: {mismatches:,} records" + (f" (first {min(show, mismatches)} shown above)" if show else ""))
        for reason, n in reasons.most_common():
            print(f"  {n:>8,}  {reason}")
        if by_throw:
            print("  by throw: " + ", ".join(f"t={t}: {n:,}" for t, n in sorted(by_throw.items())))
        return 1
    if records == 0:
        print("FAIL: dump is empty")
        return 1
    print("OK: every record matches the Python reference")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dump", help="JSON Lines file produced by `senet dump-moves`")
    ap.add_argument("--show", type=int, default=10, help="how many mismatches to print in detail")
    args = ap.parse_args(argv)
    return check_file(args.dump, show=args.show)


if __name__ == "__main__":
    sys.exit(main())
