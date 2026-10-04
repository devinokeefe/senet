"""Python API for the Senet engine (Modern Kendall rules, 5 pieces).

Positions are given from the point of view of the player about to throw, either as
lists of squares (1..30, never 27) or as bit masks (bit i = square i)::

    import senet
    eng = senet.Engine(db="db/kendall5")          # perfect-play database
    me, opp = senet.start()                        # White to throw
    eng.value(me, opp)                             # P(White wins) with perfect play
    for m, v in zip(senet.legal_moves(me, opp, 3), eng.move_values(me, opp, 3)):
        print(m, v)
    eng.match("perfect", "expectimax:2", pairs=2000)

Bot specs: "random", "greedy", "expectimax[:N]", "perfect", "net[:N]" (N = 0..4 throws
of search).

Errors: invalid arguments (positions, throws, counts, seeds, bot specs, a finished game
where moves are asked for) and calls to a closed engine raise ValueError. A resource that a
call needs but the engine lacks (a database or network that is not loaded, or an
incomplete database) raises RuntimeError. `Engine.value` raises KeyError for a position
that the loaded database does not cover, and `Engine(...)` raises OSError for a database
or network file it cannot load.
"""

from __future__ import annotations

import ctypes as C
import json
import operator
import os
import random
from collections.abc import Sequence
from types import TracebackType
from typing import Any, NamedTuple, NoReturn, TypeVar

import numpy as np
import numpy.typing as npt

from ._lib import INVALID, MAX_MOVES, N_FEATURES, NO_MOVE, UNAVAILABLE, FfiMove, last_text, lib

__all__ = [
    "EXTRA_THROWS",
    "THROW_PROBS",
    "Engine",
    "Move",
    "build_info",
    "features",
    "flip",
    "heuristic",
    "index_of",
    "layer_size",
    "legal_moves",
    "mask",
    "position_of",
    "squares",
    "start",
    "throw_sticks",
]

THROW_PROBS = {1: 4 / 16, 2: 6 / 16, 3: 4 / 16, 4: 1 / 16, 5: 1 / 16}
EXTRA_THROWS = (1, 4, 5)
KINDS = ("move", "swap", "off", "water")
_USABLE = sum(1 << s for s in range(1, 31) if s != 27)
_U32_MAX = (1 << 32) - 1

_Int = int | np.integer[Any]
Squares = Sequence[_Int] | npt.NDArray[np.integer[Any]] | _Int  # squares, or a mask
_Result = TypeVar("_Result", int, float)


def mask(x: Squares) -> int:
    """Bit mask of a list of squares; masks are checked and passed through."""
    if isinstance(x, (int, np.integer)):
        m = int(x)
        if m < 0 or m & ~_USABLE:
            raise ValueError(f"invalid square mask {m:#x}")
        return m
    m = 0
    for item in x:
        s = operator.index(item)  # a Python int: a narrow NumPy integer would overflow in `1 << s`
        if not 1 <= s <= 30 or s == 27:
            raise ValueError(f"invalid square {s}")
        if m >> s & 1:
            raise ValueError(f"square {s} listed twice")
        m |= 1 << s
    return m


def squares(m: int) -> list[int]:
    """The squares set in mask `m`, in increasing order."""
    return [i for i in range(1, 31) if m >> i & 1]


def flip(me: Squares, opp: Squares) -> tuple[int, int]:
    """The same board seen by the other player."""
    return mask(opp), mask(me)


def _check(result: _Result) -> _Result:
    """An engine result. A failure code (a negative integer or value) raises the engine's
    message: ValueError for INVALID (a bad argument), RuntimeError otherwise (UNAVAILABLE:
    a missing resource)."""
    if result < 0:
        raise (ValueError if result == INVALID else RuntimeError)(last_text())
    return result


def _uint(x: int, bits: int, what: str) -> int:
    """`x` checked to fit an unsigned C integer of `bits` bits (ctypes would wrap it silently)."""
    n = operator.index(x)
    if not 0 <= n < 1 << bits:
        raise ValueError(f"{what} must be in 0..{(1 << bits) - 1}, not {n}")
    return n


def _throw(t: int) -> int:
    """`t` checked to be a throw (1..5)."""
    n = operator.index(t)
    if n not in THROW_PROBS:
        raise ValueError(f"throw must be 1..5, not {n}")
    return n


def _mask_array(x: npt.ArrayLike, what: str) -> np.ndarray:
    """A 1-d array of masks as contiguous, aligned uint32 (as the engine reads it), rejecting
    values that the cast would wrap."""
    a = np.asarray(x)
    if a.ndim != 1 or (a.size and a.dtype.kind not in "iu"):
        raise ValueError(f"{what} must be a 1-d array of integer masks")
    if a.size and (a.min() < 0 or a.max() > _U32_MAX):
        raise ValueError(f"{what} has a mask outside 0..{_U32_MAX:#x}")
    return np.require(a, dtype=np.uint32, requirements=["C", "A"])


def _c_string(text: str | bytes, what: str) -> bytes:
    """`text` as the bytes of a C string, in which a NUL character would end it early."""
    data = text.encode() if isinstance(text, str) else text
    if b"\0" in data:
        raise ValueError(f"{what} contains a NUL character")
    return data


def _path(p: str | os.PathLike[str] | None, what: str) -> bytes | None:
    return None if p is None else _c_string(os.fsencode(p), what)


def _json_result(n: int) -> dict[str, Any]:
    """The JSON that a successful match, quality run or build_info stored in the text slot."""
    _check(n)
    result: dict[str, Any] = json.loads(last_text())
    return result


class Move(NamedTuple):
    frm: int
    to: int  # resting square; 31 = borne off
    kind: str  # move / swap / off / water
    back: bool
    me_after: int  # masks, still from the mover's point of view
    opp_after: int

    @property
    def wins(self) -> bool:
        return self.me_after == 0

    def __repr__(self) -> str:
        dest = "off" if self.kind == "off" else self.to
        return f"Move({self.frm}->{dest} {self.kind}{' back' if self.back else ''})"


def start() -> tuple[int, int]:
    """White's view of the opening position (White throws first)."""
    me, opp = C.c_uint32(), C.c_uint32()
    _check(lib.senet_start(C.byref(me), C.byref(opp)))
    return me.value, opp.value


def throw_sticks(rng: random.Random) -> int:
    """A throw of the four fair sticks: the number of light sides up, with none counting as 5."""
    light = sum(rng.random() < 0.5 for _ in range(4))
    return light or 5


def legal_moves(me: Squares, opp: Squares, t: int) -> list[Move]:
    """Legal moves for throw t (1..5), by origin square; empty if the turn passes."""
    buf = (FfiMove * MAX_MOVES)()
    n = _check(lib.senet_gen_moves(mask(me), mask(opp), _throw(t), buf, MAX_MOVES))
    return [Move(b.frm, b.to, KINDS[b.kind], bool(b.back), b.me_after, b.opp_after) for b in buf[:n]]


def heuristic(me: Squares, opp: Squares) -> float:
    """The hand-crafted heuristic's estimate of P(player to throw wins)."""
    return _check(lib.senet_heuristic(mask(me), mask(opp)))


def features(me: Squares, opp: Squares) -> np.ndarray:
    """The network's input features (see docs/FORMATS.md)."""
    out = np.zeros(N_FEATURES, dtype=np.float32)
    _check(lib.senet_features(mask(me), mask(opp), out.ctypes.data_as(C.POINTER(C.c_float))))
    return out


def index_of(me: Squares, opp: Squares) -> tuple[int, int, int]:
    """Database layer (w, b) and index of a position with 1..5 pieces per side."""
    w, b, i = C.c_uint32(), C.c_uint32(), C.c_uint64()
    _check(lib.senet_index_of(mask(me), mask(opp), C.byref(w), C.byref(b), C.byref(i)))
    return w.value, b.value, i.value


def position_of(w: int, b: int, idx: int) -> tuple[int, int]:
    """Inverse of index_of."""
    me, opp = C.c_uint32(), C.c_uint32()
    _check(
        lib.senet_position_of(_uint(w, 32, "w"), _uint(b, 32, "b"), _uint(idx, 64, "idx"), C.byref(me), C.byref(opp))
    )
    return me.value, opp.value


def layer_size(w: int, b: int) -> int:
    """Number of positions in layer (w, b); 0 if there is no such layer."""
    return lib.senet_layer_size(_uint(w, 32, "w"), _uint(b, 32, "b"))


def build_info() -> dict[str, Any]:
    """How the engine library was built: its version, target, whether it is optimized and the
    CPU features it requires (a CPU without one of them cannot run it)."""
    return _json_result(lib.senet_build_info())


class Engine:
    """Holds the (memory-mapped) perfect-play database and/or the distilled network.

    An engine can be shared between threads: its calls run in parallel (ctypes releases
    the GIL), and it can be closed while calls are in progress."""

    def __init__(self, db: str | os.PathLike[str] | None = None, net: str | os.PathLike[str] | None = None) -> None:
        self.has_db = db is not None
        self.has_net = net is not None
        paths = _path(db, "db"), _path(net, "net")
        self._ctx = lib.senet_ctx_new(*paths)
        if not self._ctx:
            raise OSError(last_text())

    def close(self) -> None:
        """Releases the database and network once the calls in progress have finished;
        later calls raise ValueError. Closing again does nothing."""
        if ctx := getattr(self, "_ctx", None):
            lib.senet_ctx_close(ctx)

    def __del__(self) -> None:
        # No call can be using the engine: each one holds a reference to it.
        if ctx := getattr(self, "_ctx", None):
            lib.senet_ctx_free(ctx)

    def __reduce__(self) -> NoReturn:
        # A copy would share the handle and free it a second time. copy.copy and
        # copy.deepcopy go through here too.
        raise TypeError("an Engine cannot be copied or pickled; open another with the same db and net")

    def __enter__(self) -> Engine:
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> None:
        self.close()

    # -- values ---------------------------------------------------------------
    def value(self, me: Squares, opp: Squares) -> float:
        """P(player to throw wins) under perfect play, from the database; KeyError if the loaded database
        does not cover the position."""
        v = lib.senet_db_value(self._ctx, mask(me), mask(opp))
        if v == UNAVAILABLE and self.has_db:
            raise KeyError(last_text())
        return _check(v)

    def values(self, me: npt.ArrayLike, opp: npt.ArrayLike) -> np.ndarray:
        """Perfect values for 1-d arrays of masks (in parallel); NaN where the database has none."""
        me_a, opp_a = _mask_array(me, "me"), _mask_array(opp, "opp")
        if me_a.shape != opp_a.shape:
            raise ValueError(f"me and opp have different lengths ({len(me_a)} and {len(opp_a)})")
        out = np.empty(len(me_a), dtype=np.float32)
        u32p = C.POINTER(C.c_uint32)
        n = lib.senet_db_values(
            self._ctx,
            me_a.ctypes.data_as(u32p),
            opp_a.ctypes.data_as(u32p),
            len(me_a),
            out.ctypes.data_as(C.POINTER(C.c_float)),
        )
        _check(n)
        if (bad := np.flatnonzero(out == INVALID)).size:
            raise ValueError(f"invalid position at index {bad[0]}")
        out[out == UNAVAILABLE] = np.nan
        return out

    def net_value(self, me: Squares, opp: Squares) -> float:
        """The distilled network's estimate of P(player to throw wins)."""
        return _check(lib.senet_net_value(self._ctx, mask(me), mask(opp)))

    def move_values(self, me: Squares, opp: Squares, t: int, bot: str = "perfect") -> list[float]:
        """Win probability for the mover after each legal move (same order as legal_moves),
        judged by `bot`'s evaluator and search depth."""
        out = (C.c_double * MAX_MOVES)()
        n = _check(
            lib.senet_move_values(self._ctx, _c_string(bot, "bot"), mask(me), mask(opp), _throw(t), out, MAX_MOVES)
        )
        return list(out[:n])

    def choose(self, me: Squares, opp: Squares, t: int, bot: str = "perfect", seed: int = 0) -> int | None:
        """Index (into legal_moves) of the move `bot` plays, or None if there is no move."""
        args = _c_string(bot, "bot"), mask(me), mask(opp), _throw(t), _uint(seed, 64, "seed")
        i = lib.senet_bot_choose(self._ctx, *args)
        return None if i == NO_MOVE else _check(i)

    # -- matches & analysis ---------------------------------------------------
    def match(self, a: str, b: str, pairs: int = 1000, seed: int = 1) -> dict[str, Any]:
        """Duplicate match between bots `a` and `b`: each pair of games replays the same throws
        with colours swapped. Returns the engine's summary (games, a_win_rate, ci95, ...)."""
        bots = _c_string(a, "bot a"), _c_string(b, "bot b")
        return _json_result(lib.senet_match(self._ctx, *bots, _uint(pairs, 64, "pairs"), _uint(seed, 64, "seed")))

    def quality(self, bot: str, vs: str = "perfect", games: int = 1000, seed: int = 7) -> dict[str, Any]:
        """Win probability `bot` gives away per decision compared with perfect play, measured
        with the (complete) database over `games` games against the opponent `vs`. Returns the
        engine's summary (decisions, errors, avg_loss_per_decision, ...)."""
        bots = _c_string(bot, "bot"), _c_string(vs, "vs")
        return _json_result(lib.senet_quality(self._ctx, *bots, _uint(games, 64, "games"), _uint(seed, 64, "seed")))
