"""How fast the engine answers: database lookups, network evaluations and bot decisions.

    python -m senet.latency --db db/kendall5 --net models/senet_net.bin [--n 5000]

Writes docs/LATENCY.md and the raw results, with their provenance (the commit, machine,
engine build, database and network), as docs/LATENCY.json. The default paths are in the
source checkout, or in the current directory for an installed copy of the package.

The decisions (a position and a throw with two or more legal moves) come from seeded games
between random players, which reach every stage of the game; lookups and network
evaluations use the distinct positions among them. Each operation is called from Python,
one call at a time on this thread, and timed call by call with a monotonic clock while the
garbage collector is paused, so each time includes the cost of a call from Python, which
the report measures with a call that does almost nothing. Engine.values looks all the
positions up in one call, in parallel: the report gives the median of BATCH_CALLS such
calls, after a first one that starts the engine's worker threads. For every other operation
it gives the median, the 95th percentile, the maximum and the calls per second.

The database is memory-mapped. The first pass over the positions reads each one's page from
the disk or from the operating system's file cache, whichever holds it, unless a lookup of a
nearby position has mapped it already; the second pass finds every page mapped. Neither is a
cold-disk measurement unless the file cache was emptied before the run, which this does not
do. Timings depend on the machine and on what else runs on it, so the report says how busy
the machine's CPUs were during the run; they are measurements to compare, not limits to
test against.
"""

from __future__ import annotations

import argparse
import ctypes
import gc
import os
import random
import sys
import time
from collections.abc import Callable, Sequence
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np

from . import EXTRA_THROWS, Engine, layer_size, legal_moves, start, throw_sticks
from .provenance import home, provenance, write_report
from .provenance import render as render_provenance

SEED = 31
BATCH_CALLS = 5  # timed calls of Engine.values
NET_BOTS = ("net", "net:1")
HEURISTIC_BOTS = ("greedy", "expectimax:1", "expectimax:2")
Decision = tuple[int, int, int]  # mover's mask, opponent's mask, throw
Position = tuple[int, int]  # mover's mask, opponent's mask


def decisions(n: int, seed: int = SEED) -> list[Decision]:
    """`n` decisions from games between random players, in the order they arise."""
    rng = random.Random(seed)
    found: list[Decision] = []
    while len(found) < n:
        me, opp = start()
        while len(found) < n:
            t = throw_sticks(rng)
            moves = legal_moves(me, opp, t)
            if not moves:
                me, opp = opp, me
                continue
            if len(moves) > 1:
                found.append((me, opp, t))
            m = rng.choice(moves)
            if m.wins:
                break
            me, opp = (m.me_after, m.opp_after) if t in EXTRA_THROWS else (m.opp_after, m.me_after)
    return found


def pieces(me: int, opp: int) -> int:
    """The number of pieces on the board."""
    return me.bit_count() + opp.bit_count()


def times(call: Callable[..., object], cases: Sequence[Decision | Position]) -> np.ndarray:
    """The time of `call(*case)` for each case, in microseconds, with the garbage collector
    paused (a collection would land in one call's time)."""
    us = np.empty(len(cases))
    clock = time.perf_counter
    collecting = gc.isenabled()
    gc.disable()
    try:
        for i, case in enumerate(cases):
            t0 = clock()
            call(*case)
            us[i] = (clock() - t0) * 1e6
    finally:
        if collecting:
            gc.enable()
    return us


def summary(us: np.ndarray) -> dict[str, float]:
    """The median, 95th percentile and maximum of times `us` (microseconds), and the calls
    per second one after another."""
    return {
        "calls": len(us),
        "median_us": float(np.median(us)),
        "p95_us": float(np.percentile(us, 95)),
        "max_us": float(us.max()),
        "per_second": float(len(us) / (us.sum() / 1e6)),
    }


def by_pieces(us: np.ndarray, cases: Sequence[Decision | Position]) -> dict[int, dict[str, float]]:
    """`summary` of times `us` for each number of pieces on the board."""
    counts = np.array([pieces(case[0], case[1]) for case in cases])
    return {int(k): summary(us[counts == k]) for k in np.unique(counts)}


def memory() -> dict[str, int | None]:
    """The process's peak private memory (what it allocated) and peak resident memory (with
    the pages of mapped files it touched), in bytes; None where they cannot be read."""
    if sys.platform == "win32":
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        kernel32, psapi = ctypes.WinDLL("kernel32"), ctypes.WinDLL("psapi")
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        c = Counters(cb=ctypes.sizeof(Counters))
        if not psapi.GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(c), c.cb):
            return {"peak_private_bytes": None, "peak_resident_bytes": None}
        return {"peak_private_bytes": int(c.PeakPagefileUsage), "peak_resident_bytes": int(c.PeakWorkingSetSize)}
    import resource

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {"peak_private_bytes": None, "peak_resident_bytes": peak if sys.platform == "darwin" else peak * 1024}


def cpu_times() -> tuple[float, float] | None:
    """The machine's CPU time so far, summed over its CPUs, in seconds: (busy, in all). None
    where it is not read: on systems other than Windows and Linux."""
    if sys.platform == "win32":
        from ctypes import wintypes

        idle, kernel, user = wintypes.FILETIME(), wintypes.FILETIME(), wintypes.FILETIME()
        if not ctypes.WinDLL("kernel32").GetSystemTimes(ctypes.byref(idle), ctypes.byref(kernel), ctypes.byref(user)):
            return None
        idle_s, kernel_s, user_s = ((f.dwHighDateTime << 32 | f.dwLowDateTime) / 1e7 for f in (idle, kernel, user))
        return kernel_s - idle_s + user_s, kernel_s + user_s  # kernel time includes idle time
    try:
        with open("/proc/stat", encoding="ascii") as f:
            # user, nice, system, idle, iowait, irq, softirq, steal
            ticks = [int(x) for x in f.readline().split()[1:9]]
    except OSError:
        return None
    hz = os.sysconf("SC_CLK_TCK")
    return (sum(ticks) - ticks[3] - ticks[4]) / hz, sum(ticks) / hz


def busy(before: tuple[float, float] | None, after: tuple[float, float] | None) -> float | None:
    """The fraction of the machine's CPU time that was busy between two `cpu_times`."""
    if before is None or after is None or after[1] <= before[1]:
        return None
    # The two differences round separately, so a fully busy interval can come out just above 1.
    return min(1.0, max(0.0, (after[0] - before[0]) / (after[1] - before[1])))


def measure(db: Path | None, net: Path | None, n: int) -> dict[str, Any]:
    """Every measurement the report gives, for the database `db` and network `net` (either
    may be None, which leaves its measurements out), on `n` decisions and the distinct
    positions among them."""
    cases = decisions(n)
    positions: list[Position] = list(dict.fromkeys((me, opp) for me, opp, _ in cases))
    cpu_before = cpu_times()
    operations: dict[str, Any] = {}
    # The cost of a call from Python: one into the engine that only computes a binomial.
    operations["call overhead"] = summary(times(lambda me, opp, t: layer_size(1, 1), cases))
    before = memory()  # Python and NumPy, before the engine
    t0 = time.perf_counter()
    eng = Engine(db=db, net=net)
    opened = time.perf_counter() - t0
    lookups: dict[str, Any] = {}
    with eng:
        if db is not None:
            # First, before anything else touches the database in this process.
            first, second = times(eng.value, positions), times(eng.value, positions)
            operations["database lookup, first pass"] = summary(first)
            operations["database lookup, second pass"] = summary(second)
            lookups = {"first_pass": by_pieces(first, positions), "second_pass": by_pieces(second, positions)}
            me = np.array([p[0] for p in positions], dtype=np.uint32)
            opp = np.array([p[1] for p in positions], dtype=np.uint32)
            eng.values(me, opp)  # starts the worker threads
            batches = []
            for _ in range(BATCH_CALLS):
                t0 = time.perf_counter()
                eng.values(me, opp)
                batches.append(time.perf_counter() - t0)
            operations["database lookups in one call (Engine.values)"] = {
                "calls": len(positions),
                "per_second": len(positions) / float(np.median(batches)),
            }
        if net is not None:
            operations["network evaluation"] = summary(times(eng.net_value, positions))
        bots = (*(("perfect",) if db is not None else ()), *(NET_BOTS if net is not None else ()), *HEURISTIC_BOTS)
        for bot in bots:
            operations[f"move choice: {bot}"] = summary(times(partial(eng.choose, bot=bot), cases))
    return {
        "decisions": n,
        "positions": len(positions),
        "open_seconds": opened,
        "operations": operations,
        "database_lookups_by_pieces": lookups,
        "memory": {"before_engine": before, "peak": memory()},
        # The whole machine's, this process's included.
        "machine_cpu_busy": busy(cpu_before, cpu_times()),
    }


def _us(x: float) -> str:
    return f"{x:,.0f} µs" if x >= 100 else f"{x:.1f} µs"


def render(r: dict[str, Any], raw: str | None = None) -> str:
    """The markdown report for the results of `measure` (with their provenance, if recorded),
    which links the raw results at `raw` if given."""
    lines = [
        "# Latency",
        "",
        "Generated by `python -m senet.latency` (python/senet/latency.py says how it measures)"
        + (f"; the raw results are in [{Path(raw).name}]({raw})" if raw else "")
        + ".",
        "",
        f"{r['decisions']:,} decisions (a position and a throw with two or more legal moves) from seeded",
        f"games between random players; lookups and network evaluations use their {r['positions']:,} distinct",
        "positions. Each operation was called from Python, one call at a time on one thread;",
        f"Engine.values looks its positions up in parallel (the median of {BATCH_CALLS} calls, after one",
        f"that started its threads). Opening the engine took {r['open_seconds'] * 1000:,.0f} ms.",
    ]
    if r["machine_cpu_busy"] is not None:
        lines += ["", f"Measured with the CPUs {r['machine_cpu_busy']:.0%} busy, which inflates the tail."]
    lines += [
        "",
        "| Operation | Median | 95th percentile | Maximum | Calls per second |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, s in r["operations"].items():
        timed = "median_us" in s  # not a batch
        cells = [_us(s["median_us"]), _us(s["p95_us"]), _us(s["max_us"])] if timed else ["", "", ""]
        lines.append(f"| {name} | {' | '.join(cells)} | {s['per_second']:,.0f} |")
    lookups = r["database_lookups_by_pieces"]
    if lookups:
        lines += [
            "",
            "The first pass reads a position's page of the memory-mapped database from the disk or",
            "the file cache, unless a lookup of a nearby position has mapped it; the second pass finds",
            "it mapped. By the number of pieces on the board:",
            "",
            "| Pieces | Positions | First pass, median | 95th percentile | Second pass, median |",
            "|---:|---:|---:|---:|---:|",
        ]
        for k, first in lookups["first_pass"].items():
            second = lookups["second_pass"][k]
            lines.append(
                f"| {k} | {first['calls']:,} | {_us(first['median_us'])} | {_us(first['p95_us'])} "
                f"| {_us(second['median_us'])} |"
            )
    peak, before = r["memory"]["peak"], r["memory"]["before_engine"]
    mib = {key: (peak[key] / 2**20, before[key] / 2**20) for key in peak if peak[key] is not None}
    facts = []
    if "peak_resident_bytes" in mib:
        facts.append(f"Peak: {mib['peak_resident_bytes'][0]:,.0f} MiB resident.")
    if "peak_private_bytes" in mib:
        committed, python = mib["peak_private_bytes"]
        facts.append(
            f"Of {committed:,.0f} MiB committed, {python:,.0f} MiB was Python and NumPy before the engine opened."
        )
    if facts:
        lines += ["", " ".join(facts)]
    lines.append("")
    if "provenance" in r:
        lines.append(render_provenance(r["provenance"]))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", type=Path, help="the solved database; adds lookups and the perfect bot")
    ap.add_argument("--net", type=Path, help="SNN1 network file; adds evaluations and the 'net' and 'net:1' bots")
    ap.add_argument(
        "--n", type=int, default=5000, help="decisions to time each operation on (lookups use their distinct positions)"
    )
    ap.add_argument("--out", type=Path, default=home() / "docs" / "LATENCY.md")
    ap.add_argument("--json", type=Path, default=home() / "docs" / "LATENCY.json")
    args = ap.parse_args(argv)
    if args.n < 1:
        ap.error("--n must be at least 1")
    origin = provenance(db=args.db, net=args.net, n=args.n, seed=SEED)
    results = measure(args.db, args.net, args.n)
    results["provenance"] = origin
    write_report(results, args.out, args.json, render)


if __name__ == "__main__":
    main()
