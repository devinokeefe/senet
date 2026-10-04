"""Small float64 reference solver for the low-piece layers of kendall5 Senet.

V(me, opp) = probability that the player about to throw wins under optimal play.

    V(s)   = sum_t p_t * Q_t(s)
    Q_t(s) = max over legal moves (me', opp') of
                 1                    if me' is empty (last piece borne off)
                 V(me', opp')         if t in {1, 4, 5}   (mover throws again)
                 1 - V(opp', me')     otherwise           (turn passes)
    Q_t(s) = 1 - V(opp, me)           if there is no legal move (forfeit)

Each candidate R in Q_t (1, V or 1 - V of a successor) is affine in one table entry:
R = a + s * V[g] with (a, s) = (1, 0) for a win, (0, +1) for an extra throw, (1, -1)
for a pass.  Each state's successors are generated once with ``rules.legal_moves`` and
stored as flat numpy arrays, so a sweep is a gather + ``np.maximum.reduceat`` per throw.

Layers {(w, b), (b, w)} depend on each other and on layers with one piece fewer
(after a bear-off), so groups are solved in increasing w + b.  Within a group the
layers are updated one after the other (block Gauss-Seidel), each layer as a whole
from the current table (Jacobi inside a layer), until the max change < tol.

Usage:  python -m senet_ref.solver [--K 5] [--tol 1e-13] [--init 0.5]

The default K = 5 (every layer with up to 5 pieces in total, 3,917,900 positions) takes
about 10 minutes; K = 4 takes under a minute.  The tables are cached for
``python -m senet_ref.check_db`` in CACHE_DIR: python/senet_ref/_cache in the source
checkout, or senet_ref_cache in the current directory for an installed copy of the package.
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .indexing import index_of, layer_size, position_of
from .rules import EXTRA_THROW, NUM_PIECES, THROW_PROBS, THROWS, legal_moves

_HERE = Path(__file__).resolve().parent
CACHE_DIR = _HERE / "_cache" if (_HERE.parents[1] / "Cargo.toml").exists() else Path("senet_ref_cache")
DEFAULT_K = 5
DEFAULT_TOL = 1e-13
DEFAULT_INIT = 0.5

# Successor kinds -> (a, s) in R = a + s * V[g]
WIN, SAME_MOVER, OTHER_MOVER = 0, 1, 2
_A = np.array([1.0, 0.0, 1.0])
_S = np.array([0.0, 1.0, -1.0])

Layer = tuple[int, int]


def layers_up_to(K: int) -> list[Layer]:
    """All layers (w, b) with 1 <= w, b <= NUM_PIECES and w + b <= K, in increasing w + b."""
    return [(w, n - w) for n in range(2, K + 1) for w in range(1, n) if w <= NUM_PIECES and n - w <= NUM_PIECES]


def groups_up_to(K: int) -> list[list[Layer]]:
    """Mutually dependent groups {(w, b), (b, w)} in increasing w + b."""
    return [[(w, b), (b, w)] if w < b else [(w, b)] for w, b in layers_up_to(K) if w <= b]


@dataclass
class Transitions:
    """Flat successor lists of one layer, per throw."""

    size: int
    g: dict[int, np.ndarray]  # global table index of each successor
    a: dict[int, np.ndarray]
    s: dict[int, np.ndarray]
    starts: dict[int, np.ndarray]  # first successor of each state (every state has >= 1)

    @property
    def num_entries(self) -> int:
        return sum(len(v) for v in self.g.values())


def build_transitions(w: int, b: int, offsets: dict[Layer, int]) -> Transitions:
    """Enumerate every position of layer (w, b) and record its successors per throw."""
    n = layer_size(w, b)
    g: dict[int, list[int]] = {t: [] for t in THROWS}
    kind: dict[int, list[int]] = {t: [] for t in THROWS}
    starts: dict[int, list[int]] = {t: [] for t in THROWS}

    def add(t: int, k: int, layer: Layer | None, idx: int) -> None:
        g[t].append(0 if layer is None else offsets[layer] + idx)
        kind[t].append(k)

    for i in range(n):
        me, opp = position_of(w, b, i)
        for t in THROWS:
            starts[t].append(len(g[t]))
            moves = legal_moves(me, opp, t)
            if not moves:
                # Forfeit: the opponent throws next in the unchanged position.
                _, _, j = index_of(opp, me)
                add(t, OTHER_MOVER, (b, w), j)
                continue
            for m in moves:
                if not m.me_after:
                    add(t, WIN, None, 0)
                elif t in EXTRA_THROW:
                    w2, b2, j = index_of(m.me_after, m.opp_after)
                    add(t, SAME_MOVER, (w2, b2), j)
                else:
                    w2, b2, j = index_of(m.opp_after, m.me_after)
                    add(t, OTHER_MOVER, (w2, b2), j)

    tr = Transitions(n, {}, {}, {}, {})
    for t in THROWS:
        k = np.array(kind[t], dtype=np.int8)
        tr.g[t] = np.array(g[t], dtype=np.int64)
        tr.a[t] = _A[k]
        tr.s[t] = _S[k]
        tr.starts[t] = np.array(starts[t], dtype=np.int64)
    return tr


def bellman(V: np.ndarray, tr: Transitions) -> np.ndarray:
    """One application of the Bellman operator to every state of a layer."""
    out = np.zeros(tr.size)
    for t in THROWS:
        R = tr.a[t] + tr.s[t] * V[tr.g[t]]
        out += THROW_PROBS[t] * np.maximum.reduceat(R, tr.starts[t])
    return out


def solve(
    K: int = DEFAULT_K,
    tol: float = DEFAULT_TOL,
    init: float = DEFAULT_INIT,
    save: bool = True,
    verbose: bool = True,
    max_sweeps: int = 1_000_000,
    cache_dir: Path = CACHE_DIR,
) -> dict[Layer, np.ndarray]:
    """Solve every layer with w + b <= K and, with `save`, write each to cache_dir/L{w}{b}.npy.

    Returns {(w, b): float64 array indexed per FORMATS.md}.
    """
    log = print if verbose else (lambda *a, **k: None)
    layers = layers_up_to(K)
    offsets: dict[Layer, int] = {}
    total = 1  # slot 0 is a constant 0.0 that WIN entries point at (R = 1 + 0 * V[0])
    for layer in layers:
        offsets[layer] = total
        total += layer_size(*layer)
    V = np.full(total, np.nan)  # NaN marks "not solved yet"; reading one would poison the result
    V[0] = 0.0

    t_all = time.perf_counter()
    for group in groups_up_to(K):
        t0 = time.perf_counter()
        trs: dict[Layer, Transitions] = {}
        for layer in group:
            t1 = time.perf_counter()
            trs[layer] = build_transitions(*layer, offsets)
            log(
                f"  L{layer[0]}{layer[1]}: {trs[layer].size:>9,} states, "
                f"{trs[layer].num_entries:>10,} successor entries, built in {time.perf_counter() - t1:.1f}s"
            )
            off = offsets[layer]
            V[off : off + trs[layer].size] = init
        t_build = time.perf_counter() - t0

        t0 = time.perf_counter()
        sweeps = 0
        while True:
            sweeps += 1
            delta = 0.0
            for layer in group:
                off, n = offsets[layer], trs[layer].size
                new = bellman(V, trs[layer])
                delta = max(delta, float(np.max(np.abs(new - V[off : off + n]))))
                V[off : off + n] = new
            if delta < tol:
                break
            if sweeps >= max_sweeps:
                raise RuntimeError(f"group {group} did not converge in {max_sweeps} sweeps (delta={delta:.3e})")
        t_iter = time.perf_counter() - t0
        names = "+".join(f"L{w}{b}" for w, b in group)
        log(
            f"{names}: converged in {sweeps} sweeps (last max change {delta:.2e}); "
            f"build {t_build:.1f}s, iterate {t_iter:.2f}s"
        )

    tables = {layer: V[offsets[layer] : offsets[layer] + layer_size(*layer)].copy() for layer in layers}
    for layer, arr in tables.items():
        if not (np.all(np.isfinite(arr)) and arr.min() >= -1e-12 and arr.max() <= 1 + 1e-12):
            raise RuntimeError(f"layer {layer} has non-finite values or values outside [0, 1]")
    log(f"total solve time {time.perf_counter() - t_all:.1f}s")

    if save:
        cache_dir.mkdir(parents=True, exist_ok=True)
        for (w, b), arr in tables.items():
            np.save(cache_dir / f"L{w}{b}.npy", arr)
        log(f"saved {len(tables)} layers to {cache_dir}")
    return tables


def load_tables(cache_dir: Path = CACHE_DIR) -> dict[Layer, np.ndarray]:
    """Load every cached L{w}{b}.npy layer."""
    tables = {}
    for path in sorted(cache_dir.glob("L??.npy")):
        w, b = int(path.stem[1]), int(path.stem[2])
        tables[(w, b)] = np.load(path)
    return tables


def value(tables: dict[Layer, np.ndarray], me: Sequence[int], opp: Sequence[int]) -> float:
    """V(me, opp) looked up in solved tables (1.0 / 0.0 for finished games)."""
    if not me:
        return 1.0  # cannot happen for a player about to throw, but keep it total
    if not opp:
        return 0.0
    w, b, i = index_of(me, opp)
    return float(tables[(w, b)][i])


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--K",
        type=int,
        default=DEFAULT_K,
        choices=range(2, 2 * NUM_PIECES + 1),
        metavar="K",
        help=f"solve all layers with w + b <= K (default {DEFAULT_K})",
    )
    ap.add_argument("--tol", type=float, default=DEFAULT_TOL, help="convergence threshold on the max change")
    ap.add_argument("--init", type=float, default=DEFAULT_INIT, help="initial value of every unsolved state")
    ap.add_argument("--no-save", action="store_true", help="do not write the tables to the cache")
    args = ap.parse_args(argv)
    solve(K=args.K, tol=args.tol, init=args.init, save=not args.no_save)


if __name__ == "__main__":
    main()
