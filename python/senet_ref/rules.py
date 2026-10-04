"""Independent reference implementation of the ``kendall5`` Senet rules.

Everything in this module follows RULES.md clause by clause and favours
clarity over speed.  It is deliberately written without looking at the Rust
engine so that the two can be differentially tested against each other.

Positions are always seen from the point of view of the player about to throw:
``me`` and ``opp`` are sorted tuples of occupied squares (1..30).  A borne-off
piece simply disappears from its tuple (it is "on square 31").
"""

from __future__ import annotations

import random
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import NamedTuple

# ---------------------------------------------------------------------------
# Board constants (RULES.md, "Board" and "Special squares")
# ---------------------------------------------------------------------------

NUM_SQUARES = 30
NUM_PIECES = 5
HOUSE_OF_REBIRTH = 15
HOUSE_OF_BEAUTY = 26
HOUSE_OF_WATER = 27
FINAL_SQUARES = (28, 29, 30)
OFF = 31  # "borne off" is treated as square 31

# ---------------------------------------------------------------------------
# Throwing sticks (RULES.md, "Throwing sticks" and "Turn structure")
# ---------------------------------------------------------------------------

THROW_PROBS = {1: 4 / 16, 2: 6 / 16, 3: 4 / 16, 4: 1 / 16, 5: 1 / 16}
EXTRA_THROW = {1, 4, 5}  # moving with one of these throws earns another throw
THROWS = (1, 2, 3, 4, 5)

Position = tuple[tuple[int, ...], tuple[int, ...]]


@dataclass(frozen=True)
class Move:
    """One legal move, mirroring an entry of the JSONL dump (docs/FORMATS.md).

    ``to`` is the square where the piece finally rests: 31 when borne off, and
    for a House of Water move the square it was sent back to (15 or lower).
    ``me_after`` / ``opp_after`` are still from the same mover's point of view.
    """

    frm: int
    to: int
    kind: str  # "move" | "swap" | "off" | "water"
    dir: str  # "fwd" | "back"
    me_after: tuple[int, ...]
    opp_after: tuple[int, ...]


# ---------------------------------------------------------------------------
# Position helpers
# ---------------------------------------------------------------------------


def _sorted(squares: Iterable[int]) -> tuple[int, ...]:
    return tuple(sorted(squares))


def start_position() -> Position:
    """White's view at the start of the game (White moves first)."""
    return (1, 3, 5, 7, 9), (2, 4, 6, 8, 10)


def flip(me: Iterable[int], opp: Iterable[int]) -> Position:
    """The same board seen from the other player's point of view."""
    return _sorted(opp), _sorted(me)


def validate_position(me: Iterable[int], opp: Iterable[int]) -> None:
    """Raise ValueError if (me, opp) cannot occur in a kendall5 game."""
    me, opp = list(me), list(opp)
    for side, squares in (("me", me), ("opp", opp)):
        if len(squares) > NUM_PIECES:
            raise ValueError(f"{side} has {len(squares)} pieces (max {NUM_PIECES})")
        if len(set(squares)) != len(squares):
            raise ValueError(f"{side} has two pieces on one square: {squares}")
        for s in squares:
            if not (1 <= s <= NUM_SQUARES):
                raise ValueError(f"{side} piece on invalid square {s}")
            if s == HOUSE_OF_WATER:
                raise ValueError(f"{side} piece on the House of Water (27), which is always empty")
    if set(me) & set(opp):
        raise ValueError(f"me and opp share squares {sorted(set(me) & set(opp))}")


# ---------------------------------------------------------------------------
# Definitions (RULES.md, "Definitions")
# ---------------------------------------------------------------------------


def is_protected(s: int, opp: set[int] | frozenset[int]) -> bool:
    """An enemy piece on ``s`` is protected if another enemy piece is on s-1 or s+1."""
    return (s - 1) in opp or (s + 1) in opp


def blockade_squares(opp: Iterable[int]) -> frozenset[int]:
    """All squares belonging to a run of three or more enemy pieces on consecutive squares."""
    opp = set(opp)
    result: set[int] = set()
    for s in opp:
        if (s - 1) in opp:
            continue  # not the first square of its run
        run = [s]
        while (run[-1] + 1) in opp:
            run.append(run[-1] + 1)
        if len(run) >= 3:
            result.update(run)
    return frozenset(result)


def water_square(occupied: set[int]) -> int:
    """Forward rule 6: square 15 if empty, else the highest empty square below 15."""
    for s in range(HOUSE_OF_REBIRTH, 0, -1):
        if s not in occupied:
            return s
    # At most 9 other pieces exist, so squares 1..15 can never all be full.
    raise AssertionError("no empty square at or below 15")


# ---------------------------------------------------------------------------
# Move generation (RULES.md, "Legal moves for a throw t")
# ---------------------------------------------------------------------------


def _bear_off(a: int, me: set[int], opp: set[int]) -> Move:
    return Move(a, OFF, "off", "fwd", _sorted(me - {a}), _sorted(opp))


def _land(a: int, d: int, direction: str, me: set[int], opp: set[int]) -> Move | None:
    """Rules 7-9 (shared by forward and backward moves) for a piece going a -> d."""
    # Rule 7: cannot land on a friendly piece.
    if d in me:
        return None
    # Rule 8: enemy piece on d -> illegal if protected, otherwise swap.
    if d in opp:
        if is_protected(d, opp):
            return None
        return Move(a, d, "swap", direction, _sorted((me - {a}) | {d}), _sorted((opp - {d}) | {a}))
    # Rule 9: empty square -> plain move.
    return Move(a, d, "move", direction, _sorted((me - {a}) | {d}), _sorted(opp))


def _forward_move(a: int, t: int, me: set[int], opp: set[int], blockade: frozenset[int]) -> Move | None:
    """The forward move of the piece on ``a`` with throw ``t``, or None if illegal."""
    # Forward rule 1: a piece on a final square can only bear off, with exactly 31 - a.
    if a in FINAL_SQUARES:
        if a + t == OFF:
            return _bear_off(a, me, opp)
        return None
    # Forward rule 2: destination.
    d = a + t
    # Forward rule 3 (House of Beauty): no move may jump past 26.
    if a < HOUSE_OF_BEAUTY and d > HOUSE_OF_BEAUTY:
        return None
    # Forward rule 4 (Blockade): no blockade square strictly between a and d.
    if any(s in blockade for s in range(a + 1, d)):
        return None
    # Forward rule 5: d = 31 (only from 26 with a 5) bears the piece off.
    if d == OFF:
        return _bear_off(a, me, opp)
    # Forward rule 6 (House of Water): 26 + 1 -> back to 15 (or the highest empty square below).
    if d == HOUSE_OF_WATER:
        rest = water_square((me - {a}) | opp)
        return Move(a, rest, "water", "fwd", _sorted((me - {a}) | {rest}), _sorted(opp))
    # Forward rules 7-9.
    return _land(a, d, "fwd", me, opp)


def _backward_move(a: int, t: int, me: set[int], opp: set[int], blockade: frozenset[int]) -> Move | None:
    """The backward move of the piece on ``a`` with throw ``t``, or None if illegal."""
    # Pieces on final squares (28-30) never move backward: only a <= 26 may.
    if a > HOUSE_OF_BEAUTY:
        return None
    # Backward rule 1: d = a - t, illegal if below square 1.
    d = a - t
    if d < 1:
        return None
    # Backward rule 2 (Blockade): no blockade square strictly between d and a.
    if any(s in blockade for s in range(d + 1, a)):
        return None
    # Backward rule 3: rules 7-9 (the swapped enemy goes forward to a).
    return _land(a, d, "back", me, opp)


def legal_moves(me: Iterable[int], opp: Iterable[int], t: int) -> list[Move]:
    """Every legal move for throw ``t`` in a game in progress, sorted by (frm, to).  Empty
    list = forfeit."""
    if t not in THROW_PROBS:
        raise ValueError(f"invalid throw {t}")
    me_t, opp_t = _sorted(me), _sorted(opp)
    # Turn rule 5: once a side has borne off its last piece, the game is over.
    if not me_t or not opp_t:
        raise ValueError(f"the game is already over: me={list(me_t)} opp={list(opp_t)}")
    me_set, opp_set = set(me_t), set(opp_t)
    blockade = blockade_squares(opp_set)

    moves = [m for a in me_t if (m := _forward_move(a, t, me_set, opp_set, blockade)) is not None]
    if not moves:
        # Backward moves are allowed only if no forward move exists for any piece.
        moves = [m for a in me_t if (m := _backward_move(a, t, me_set, opp_set, blockade)) is not None]

    moves.sort(key=lambda m: (m.frm, m.to))
    return moves


# ---------------------------------------------------------------------------
# Game simulation (RULES.md, "Turn structure")
# ---------------------------------------------------------------------------


def throw_sticks(rng: random.Random) -> int:
    """Four fair two-sided sticks; the number of light sides up, with 0 counting as 5."""
    light = sum(1 for _ in range(4) if rng.random() < 0.5)
    return 5 if light == 0 else light


Policy = Callable[[tuple[int, ...], tuple[int, ...], int, list[Move]], Move]


def random_policy(rng: random.Random) -> Policy:
    return lambda me, opp, t, moves: rng.choice(moves)


class GameResult(NamedTuple):
    winner: int  # 0 = the player who moved first in this game, 1 = the other player
    throws: int
    moves: int
    forfeits: int


def play_game(
    rng: random.Random,
    me: Iterable[int],
    opp: Iterable[int],
    policies: tuple[Policy, Policy],
    max_throws: int = 1_000_000,
) -> GameResult:
    """Play a game from (me, opp), which must be in progress, with player 0 to throw first.

    ``policies[p]`` chooses a move for player ``p`` given its own point of view.
    """
    me, opp = _sorted(me), _sorted(opp)
    player = 0
    moves_made = forfeits = 0
    for throw_no in range(1, max_throws + 1):
        validate_position(me, opp)
        t = throw_sticks(rng)
        moves = legal_moves(me, opp, t)
        if not moves:
            # Turn rule 4: no legal move -> the turn passes, even after 1/4/5.
            forfeits += 1
            me, opp = flip(me, opp)
            player ^= 1
            continue
        move = policies[player](me, opp, t, moves)
        if move not in moves:
            raise ValueError(f"policy returned an illegal move: {move}")
        moves_made += 1
        me, opp = move.me_after, move.opp_after
        # Turn rule 5: bearing off the last piece wins immediately.
        if not me:
            return GameResult(player, throw_no, moves_made, forfeits)
        # Turn rule 3: 1, 4 or 5 -> throw again; 2 or 3 -> the opponent's turn.
        if t not in EXTRA_THROW:
            me, opp = flip(me, opp)
            player ^= 1
    raise RuntimeError(f"game did not finish within {max_throws} throws")


def play_random_game(rng: random.Random | None = None) -> GameResult:
    """A full game from the start position with both sides playing uniformly random moves.

    ``winner`` is 0 for White (who moves first) and 1 for Black.
    """
    rng = rng if rng is not None else random.Random()
    me, opp = start_position()
    policy = random_policy(rng)
    return play_game(rng, me, opp, (policy, policy))
