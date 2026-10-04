"""The verdicts of the reference checkers (senet_ref.check_db and senet_ref.check_movegen) on
hand-made inputs, including the failures they exist to catch."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from senet_ref.check_db import check_db
from senet_ref.check_movegen import check_file
from senet_ref.solver import solve


def test_check_db(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    cache, db = tmp_path / "cache", tmp_path / "db"
    db.mkdir()
    layer = solve(K=2, verbose=False, cache_dir=cache)[(1, 1)]  # K = 2: only layer L11
    layer.astype("<f4").tofile(db / "L11.f32")
    assert check_db(db, K=2, cache_dir=cache) == 0

    # Every layer with w + b <= K is required, not just the cached ones.
    capsys.readouterr()
    assert check_db(db, K=3, cache_dir=cache) == 1
    assert "no Python reference for L12, L21" in capsys.readouterr().out

    # --solve computes missing reference layers instead.
    assert check_db(db, K=2, cache_dir=tmp_path / "empty", solve_missing=True) == 0
    assert (tmp_path / "empty" / "L11.npy").exists()

    wrong = layer.astype("<f4")
    wrong[7] += 1e-4
    wrong.tofile(db / "L11.f32")
    assert check_db(db, K=2, cache_dir=cache) == 1
    # A non-finite value fails under any threshold; a threshold must be finite and >= 0.
    wrong[7] = np.nan
    wrong.tofile(db / "L11.f32")
    assert check_db(db, K=2, cache_dir=cache, threshold=1e300) == 1
    for bad in (float("inf"), float("nan"), -1.0):
        with pytest.raises(ValueError, match="threshold"):
            check_db(db, K=2, cache_dir=cache, threshold=bad)
    layer[:-1].astype("<f4").tofile(db / "L11.f32")
    assert check_db(db, K=2, cache_dir=cache) == 1
    # A partial trailing value is an error too, not ignored.
    (db / "L11.f32").write_bytes(layer.astype("<f4").tobytes() + b"\0")
    capsys.readouterr()
    assert check_db(db, K=2, cache_dir=cache) == 1
    assert "rust file has 3,249 bytes, not 3,248" in capsys.readouterr().out
    (db / "L11.f32").unlink()
    assert check_db(db, K=2, cache_dir=cache) == 1


# The example record of docs/FORMATS.md.
RECORD: dict[str, Any] = {
    "me": [3, 9, 26],
    "opp": [4, 5, 28],
    "t": 2,
    "moves": [
        {"from": 9, "to": 11, "kind": "move", "dir": "fwd", "me": [3, 11, 26], "opp": [4, 5, 28]},
        {"from": 26, "to": 28, "kind": "swap", "dir": "fwd", "me": [3, 9, 28], "opp": [4, 5, 26]},
    ],
}


def test_check_movegen(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    dump = tmp_path / "moves.jsonl"
    dump.write_text(json.dumps(RECORD) + "\n", encoding="utf-8")
    assert check_file(str(dump)) == 0

    missing_move = {**RECORD, "moves": RECORD["moves"][:1]}
    unsorted = {**RECORD, "moves": RECORD["moves"][::-1]}
    lines = [json.dumps(r) for r in (RECORD, missing_move, unsorted)] + ["{not json"]
    dump.write_text("\n".join(lines) + "\n", encoding="utf-8")
    capsys.readouterr()
    assert check_file(str(dump)) == 1
    out = capsys.readouterr().out
    assert "MISMATCHES: 3 records" in out
    for reason in ("different move sets", "wrong order", "malformed record"):
        assert reason in out

    # Values are not converted: a fractional or boolean number is malformed, as are a bad
    # throw and a finished game.
    move = RECORD["moves"][0]
    malformed: list[Any] = [
        {**RECORD, "t": 2.75},
        {**RECORD, "t": True},
        {**RECORD, "t": 6},
        {**RECORD, "me": [3.5, 9, 26]},
        {**RECORD, "moves": [{**move, "from": 9.25}, RECORD["moves"][1]]},
        {**RECORD, "moves": [{**move, "kind": 0}, RECORD["moves"][1]]},
        {**RECORD, "moves": [{**move, "extra": 1}, RECORD["moves"][1]]},
        {**RECORD, "opp": [], "moves": []},
        [RECORD],
    ]
    dump.write_text("\n".join(json.dumps(r) for r in malformed) + "\n", encoding="utf-8")
    capsys.readouterr()
    assert check_file(str(dump), show=0) == 1
    assert f"{len(malformed)}  malformed record" in capsys.readouterr().out

    dump.write_text("", encoding="utf-8")
    assert check_file(str(dump)) == 1
