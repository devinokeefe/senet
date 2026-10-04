"""The latency report (senet.latency): its inputs, statistics and rendering. Nothing here
asserts how fast anything is: timings depend on the machine."""

from __future__ import annotations

import gc
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from conftest import DB, NET, needs_db, needs_net

import senet
from senet import latency


def test_decisions_are_real_choices_from_every_stage() -> None:
    cases = latency.decisions(3000)
    assert cases == latency.decisions(3000) and cases[:100] == latency.decisions(100)
    assert cases != latency.decisions(3000, seed=1)
    assert all(len(senet.legal_moves(me, opp, t)) > 1 for me, opp, t in cases)
    # A choice needs 3 pieces on the board at least: one each never gives two legal moves.
    assert {latency.pieces(me, opp) for me, opp, _ in cases} == set(range(3, 11))


def test_times_pause_the_garbage_collector() -> None:
    collecting: list[bool] = []
    us = latency.times(lambda me, opp: collecting.append(gc.isenabled()), [(1, 2), (3, 4)])
    assert us.shape == (2,) and collecting == [False, False] and gc.isenabled()
    with pytest.raises(ZeroDivisionError):
        latency.times(lambda me, opp: 1 / 0, [(1, 2)])
    assert gc.isenabled()


def test_summary() -> None:
    s = latency.summary(np.array([4.0, 1.0, 3.0, 2.0]))
    expected = {"calls": 4, "median_us": 2.5, "p95_us": 3.85, "max_us": 4.0, "per_second": 400_000.0}
    assert s == pytest.approx(expected)
    cases = [(senet.mask([1]), senet.mask([2]), 1), (senet.mask([1, 3]), senet.mask([2]), 2), (1 << 5, 1 << 6, 3)]
    groups = latency.by_pieces(np.array([1.0, 5.0, 3.0]), cases)
    assert list(groups) == [2, 3]
    assert groups[2]["calls"] == 2 and groups[2]["median_us"] == 2.0 and groups[3]["max_us"] == 5.0


def test_how_busy_the_machine_was() -> None:
    before = after = latency.cpu_times()
    if before is None:
        pytest.skip("this system's CPU times are not read")
    # The system's CPU times advance by timer ticks (15.6 ms on Windows).
    deadline = time.monotonic() + 5
    while after is not None and after[1] == before[1] and time.monotonic() < deadline:
        sum(range(10**5))
        after = latency.cpu_times()
    assert after is not None
    assert 0 <= before[0] <= before[1] and before[0] <= after[0] and before[1] < after[1]
    fraction = latency.busy(before, after)
    assert fraction is not None and 0 <= fraction <= 1
    assert latency.busy(None, after) is None and latency.busy(after, after) is None
    assert latency.busy((1.0, 10.0), (4.0, 14.0)) == pytest.approx(0.75)


def check_report(r: dict[str, Any], tmp_path: Path) -> str:
    md = latency.render(r, "LATENCY.json")
    # The report is made from the results as read back from their JSON, whose keys are strings.
    assert latency.render(json.loads(json.dumps(r)), "LATENCY.json") == md
    assert md.startswith("# Latency\n") and md.endswith("\n")
    assert "the raw results are in [LATENCY.json](LATENCY.json)." in md
    assert f"use their {r['positions']:,} distinct\npositions." in md
    assert ("Measured with the CPUs" in md) == (r["machine_cpu_busy"] is not None)
    assert "busy" not in latency.render({**r, "machine_cpu_busy": None})
    peak, before = r["memory"]["peak"], r["memory"]["before_engine"]
    if peak["peak_resident_bytes"] is not None:
        assert f"Peak: {peak['peak_resident_bytes'] / 2**20:,.0f} MiB resident." in md
    if peak["peak_private_bytes"] is not None:
        mib = peak["peak_private_bytes"] / 2**20, before["peak_private_bytes"] / 2**20
        assert f"Of {mib[0]:,.0f} MiB committed, {mib[1]:,.0f} MiB was Python and NumPy before the engine" in md
    return md


@needs_net
def test_measure_without_the_database(tmp_path: Path) -> None:
    r = latency.measure(None, NET, 20)
    positions = {(me, opp) for me, opp, _ in latency.decisions(20)}
    assert r["positions"] == len(positions) <= 20
    assert list(r["operations"]) == [
        "call overhead",
        "network evaluation",
        "move choice: net",
        "move choice: net:1",
        "move choice: greedy",
        "move choice: expectimax:1",
        "move choice: expectimax:2",
    ]
    ops = r["operations"]
    assert ops["network evaluation"]["calls"] == r["positions"]  # evaluations of the distinct positions
    assert all(ops[name]["calls"] == 20 for name in ops if name != "network evaluation")
    assert all(0 < s["median_us"] <= s["p95_us"] <= s["max_us"] for s in ops.values())
    assert r["database_lookups_by_pieces"] == {}
    before, peak = r["memory"]["before_engine"], r["memory"]["peak"]
    for key in ("peak_private_bytes", "peak_resident_bytes"):
        assert (before[key] is None) == (peak[key] is None) and (before[key] or 0) <= (peak[key] or 0)
    md = check_report(r, tmp_path)
    assert "| move choice: net:1 |" in md and "First lookup" not in md


@needs_db
def test_measure_with_the_database(tmp_path: Path) -> None:
    r = latency.measure(DB, None, 20)
    ops = r["operations"]
    assert "database lookup, first pass" in ops and "database lookup, second pass" in ops
    assert ops["database lookup, first pass"]["calls"] == r["positions"] and "move choice: perfect" in ops
    assert ops["database lookups in one call (Engine.values)"]["calls"] == r["positions"]
    first = r["database_lookups_by_pieces"]["first_pass"]
    assert sum(s["calls"] for s in first.values()) == r["positions"]
    md = check_report(r, tmp_path)
    assert "| Pieces | Positions | First pass, median | 95th percentile | Second pass, median |" in md


def test_main_writes_the_report_and_its_raw_results(tmp_path: Path) -> None:
    out, raw = tmp_path / "docs" / "LATENCY.md", tmp_path / "data" / "latency.json"
    latency.main(["--n", "5", "--out", str(out), "--json", str(raw)])
    r = json.loads(raw.read_text(encoding="utf-8"))
    assert r["provenance"]["settings"] == {"n": 5, "seed": latency.SEED}
    assert r["provenance"]["database"] is None and r["provenance"]["network"] is None
    md = out.read_text(encoding="utf-8")
    assert "[latency.json](../data/latency.json)" in md and "## Provenance" in md
    assert b"\r" not in out.read_bytes() + raw.read_bytes()
