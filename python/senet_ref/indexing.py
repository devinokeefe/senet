"""Position indexing exactly as specified in docs/FORMATS.md.

* usable squares U = [1..26, 28, 29, 30]; compact index c(s) = s-1 (s <= 26), s-2 (s >= 28)
* layer (w, b): w mover pieces, b opponent pieces on the board
* N(w, b) = C(29, w) * C(29 - w, b)
* index = me_rank * C(29 - w, b) + opp_rank, both colexicographic ranks; the opponent's
  compact squares are first renumbered among the 29 - w squares the mover leaves free.
"""

from __future__ import annotations

import random
from collections.abc import Iterable
from math import comb

from .rules import HOUSE_OF_BEAUTY, HOUSE_OF_WATER, NUM_PIECES, Position

NUM_USABLE = 29
USABLE_SQUARES = (*range(1, HOUSE_OF_BEAUTY + 1), 28, 29, 30)


def compact(s: int) -> int:
    """Compact index of square ``s`` (0..28)."""
    if 1 <= s <= HOUSE_OF_BEAUTY:
        return s - 1
    if s in (28, 29, 30):
        return s - 2
    raise ValueError(f"square {s} cannot hold a piece")


def square_of(c: int) -> int:
    """Inverse of ``compact``."""
    if 0 <= c <= 25:
        return c + 1
    if 26 <= c <= 28:
        return c + 2
    raise ValueError(f"invalid compact index {c}")


def layer_size(w: int, b: int) -> int:
    return comb(NUM_USABLE, w) * comb(NUM_USABLE - w, b)


def random_position(rng: random.Random) -> Position:
    """1..5 pieces per side (each count uniform) on distinct, uniformly chosen usable squares."""
    w, b = rng.randint(1, NUM_PIECES), rng.randint(1, NUM_PIECES)
    squares = rng.sample(USABLE_SQUARES, w + b)
    return tuple(sorted(squares[:w])), tuple(sorted(squares[w:]))


def _colex_rank(sorted_values: list[int]) -> int:
    """sum_i C(v_i, i + 1) for v_0 < v_1 < ... (math.comb is 0 when n < k)."""
    return sum(comb(v, i + 1) for i, v in enumerate(sorted_values))


def _colex_unrank(rank: int, k: int) -> list[int]:
    """Inverse of _colex_rank: the unique v_0 < ... < v_{k-1} with the given rank."""
    values = []
    for i in range(k, 0, -1):
        # Greedy (combinatorial number system): the largest c with C(c, i) <= rank.
        # C(i - 1, i) = 0 <= rank, so start there and climb.  The chosen values come
        # out strictly decreasing automatically.
        c = i - 1
        while comb(c + 1, i) <= rank:
            c += 1
        values.append(c)
        rank -= comb(c, i)
    if rank != 0:
        raise AssertionError("colex unrank did not consume the rank")
    return sorted(values)


def index_of(me: Iterable[int], opp: Iterable[int]) -> tuple[int, int, int]:
    """Return (w, b, index) of the position (mover's view) per docs/FORMATS.md."""
    me, opp = tuple(me), tuple(opp)
    m = sorted(compact(s) for s in me)
    o = sorted(compact(s) for s in opp)
    w, b = len(m), len(o)
    if len(set(m)) != w or len(set(o)) != b or set(m) & set(o):
        raise ValueError(f"overlapping squares in me={list(me)} opp={list(opp)}")
    if not (0 <= w <= NUM_PIECES and 0 <= b <= NUM_PIECES):
        raise ValueError(f"invalid piece counts w={w} b={b}")

    # Step 1: colex rank of the mover's compact squares.
    me_rank = _colex_rank(m)
    # Step 2: renumber each opponent square among the squares the mover leaves free.
    o_prime = sorted(oc - sum(1 for mc in m if mc < oc) for oc in o)
    opp_rank = _colex_rank(o_prime)
    # Step 3.
    return w, b, me_rank * comb(NUM_USABLE - w, b) + opp_rank


def position_of(w: int, b: int, index: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Inverse of ``index_of``: the (me, opp) position with this index in layer (w, b)."""
    if not (0 <= w <= NUM_PIECES and 0 <= b <= NUM_PIECES):
        raise ValueError(f"invalid layer ({w}, {b})")
    if not (0 <= index < layer_size(w, b)):
        raise ValueError(f"index {index} out of range for layer ({w}, {b})")
    me_rank, opp_rank = divmod(index, comb(NUM_USABLE - w, b))
    m = _colex_unrank(me_rank, w)
    o_prime = _colex_unrank(opp_rank, b)
    taken = set(m)
    free = [c for c in range(NUM_USABLE) if c not in taken]
    o = [free[op] for op in o_prime]
    me = tuple(sorted(square_of(c) for c in m))
    opp = tuple(sorted(square_of(c) for c in o))
    assert HOUSE_OF_WATER not in me and HOUSE_OF_WATER not in opp
    return me, opp
