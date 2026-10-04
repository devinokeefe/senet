"""Compare a Rust perfect-play database against the Python reference solver.

Usage:  python -m senet_ref.check_db <db_dir> [--K 5] [--solve] [--threshold 2e-6] [--worst 5]

Every layer with w + b <= K is checked: the reference table L{w}{b}.npy in the solver's
cache (senet_ref.solver.CACHE_DIR, written by ``python -m senet_ref.solver --K K``) is
compared element-wise with the Rust file ``<db_dir>/L{w}{b}.f32`` (raw little-endian
float32, length N(w, b), entry ``index`` = V(me, opp), docs/FORMATS.md).  With --solve,
missing reference tables are solved first
(about 10 minutes for K = 5).  Passes iff every layer is present on both sides, has the right
length, is finite, and max |rust - python| <= threshold.  Exit code 1 on failure.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

from .indexing import layer_size, position_of
from .rules import NUM_PIECES
from .solver import CACHE_DIR, DEFAULT_K, layers_up_to, load_tables, solve

DEFAULT_THRESHOLD = 2e-6


def _threshold(x: float) -> float:
    """`x`, if it is a usable threshold: finite and >= 0 (NaN would fail every comparison)."""
    if not (math.isfinite(x) and x >= 0):
        raise ValueError(f"the threshold must be a finite number >= 0, not {x}")
    return x


def _threshold_arg(text: str) -> float:
    try:
        return _threshold(float(text))
    except ValueError as e:
        raise argparse.ArgumentTypeError(str(e)) from None


def check_db(
    db_dir: Path,
    K: int = DEFAULT_K,
    threshold: float = DEFAULT_THRESHOLD,
    worst: int = 5,
    cache_dir: Path = CACHE_DIR,
    solve_missing: bool = False,
) -> int:
    """Compare every layer with w + b <= K; returns the exit code (0 if all pass)."""
    _threshold(threshold)
    layers = layers_up_to(K)
    tables = load_tables(cache_dir)
    missing = [f"L{w}{b}" for w, b in layers if (w, b) not in tables]
    if missing and solve_missing:
        print(f"no reference for {', '.join(missing)}: solving every layer with w + b <= {K}", flush=True)
        tables.update(solve(K=K, cache_dir=cache_dir))
        print(flush=True)
    elif missing:
        print(
            f"FAIL: no Python reference for {', '.join(missing)} in {cache_dir}; "
            f"run `python -m senet_ref.solver --K {K}` first (or pass --solve)"
        )
        return 1

    ok = True
    print(f"{'layer':<6} {'size':>10} {'max |diff|':>12} {'mean |diff|':>12}  status")
    details = []
    for w, b in layers:
        ref = tables[(w, b)]
        name = f"L{w}{b}"
        n = layer_size(w, b)
        path = db_dir / f"{name}.f32"
        if len(ref) != n:
            print(f"{name:<6} {n:>10,} {'':>12} {'':>12}  FAIL (python cache has {len(ref)} entries)")
            ok = False
            continue
        if not path.exists():
            print(f"{name:<6} {n:>10,} {'':>12} {'':>12}  FAIL (missing {path})")
            ok = False
            continue
        if (size := path.stat().st_size) != 4 * n:
            print(f"{name:<6} {n:>10,} {'':>12} {'':>12}  FAIL (rust file has {size:,} bytes, not {4 * n:,})")
            ok = False
            continue
        rust = np.fromfile(path, dtype="<f4")
        rust64 = rust.astype(np.float64)
        bad = ~np.isfinite(rust64)
        diff = np.abs(rust64 - ref)
        diff[bad] = np.inf
        max_diff = float(diff.max())
        layer_ok = not bad.any() and max_diff <= threshold
        ok &= layer_ok
        status = "ok" if layer_ok else "FAIL"
        if bad.any():
            status += f" ({int(bad.sum()):,} non-finite rust values)"
        mean_diff = float(np.mean(diff[~bad])) if (~bad).any() else float("nan")
        print(f"{name:<6} {n:>10,} {max_diff:>12.3e} {mean_diff:>12.3e}  {status}")
        order = np.argsort(-diff, kind="stable")[:worst]
        details.append((name, w, b, order, rust64, ref, diff))

    if worst:
        print()
        for name, w, b, order, rust64, ref, diff in details:
            print(f"{name} worst positions:")
            for i in order:
                me, opp = position_of(w, b, int(i))
                print(
                    f"  index {int(i):>8}: me={list(me)} opp={list(opp)}  "
                    f"rust={rust64[i]:.9f} python={ref[i]:.9f} |diff|={diff[i]:.3e}"
                )

    print()
    positions = sum(layer_size(w, b) for w, b in layers)
    if ok:
        print(f"OK: all {len(layers)} layers with w + b <= {K} ({positions:,} positions) within {threshold:g}")
    else:
        print(f"FAIL: some layer with w + b <= {K} exceeds max |diff| {threshold:g} or is missing")
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("db_dir", type=Path, help="directory containing the Rust L{w}{b}.f32 files")
    ap.add_argument(
        "--K",
        type=int,
        default=DEFAULT_K,
        choices=range(2, 2 * NUM_PIECES + 1),
        metavar="K",
        help=f"check every layer with w + b <= K (default {DEFAULT_K})",
    )
    ap.add_argument("--solve", action="store_true", help="solve missing reference layers first instead of failing")
    ap.add_argument("--threshold", type=_threshold_arg, default=DEFAULT_THRESHOLD)
    ap.add_argument("--worst", type=int, default=5, help="worst positions to print per layer")
    args = ap.parse_args(argv)
    return check_db(args.db_dir, K=args.K, threshold=args.threshold, worst=args.worst, solve_missing=args.solve)


if __name__ == "__main__":
    sys.exit(main())
