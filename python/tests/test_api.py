"""Tests for the Python API (Rust engine via ctypes), cross-checked against the
independent pure-Python reference implementation."""

from __future__ import annotations

import copy
import json
import pickle
import random
import shutil
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from conftest import DB, NET, needs_db, needs_net

import senet
from senet import insights
from senet._lib import ROOT
from senet_ref import indexing as ref_index
from senet_ref import rules as ref


def test_start() -> None:
    me, opp = senet.start()
    assert senet.squares(me) == [1, 3, 5, 7, 9]
    assert senet.squares(opp) == [2, 4, 6, 8, 10]


def test_flip() -> None:
    me, opp = senet.start()
    assert senet.flip(me, opp) == (opp, me)
    assert senet.flip([30, 1], [5]) == (senet.mask([5]), senet.mask([1, 30]))
    with pytest.raises(ValueError):
        senet.flip([1], [27])


def test_finding_the_engine_library(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from senet import _lib

    monkeypatch.delenv("SENET_FFI", raising=False)
    release = ROOT / "target" / "release" / _lib._NAME
    if release.exists():
        assert _lib._find() == release
    else:  # the tests use another build, through $SENET_FFI
        with pytest.raises(OSError, match="not found; build it with"):
            _lib._find()
    monkeypatch.setenv("SENET_FFI", str(tmp_path / "none.dll"))
    with pytest.raises(OSError, match=r"SENET_FFI=.*none\.dll: no such file"):
        _lib._find()
    monkeypatch.setenv("SENET_FFI", str(_lib._PATH))
    assert _lib._find() == _lib._PATH
    # A relative path is made absolute: the system would search its library path for a bare name.
    (tmp_path / _lib._NAME).touch()
    monkeypatch.chdir(tmp_path)
    for relative in (_lib._NAME, f"./{_lib._NAME}"):
        monkeypatch.setenv("SENET_FFI", relative)
        assert _lib._find() == (tmp_path / _lib._NAME).resolve()

    monkeypatch.delenv("SENET_FFI")
    (tmp_path / "Cargo.toml").touch()  # a source checkout without the build
    monkeypatch.setattr(_lib, "ROOT", tmp_path)
    with pytest.raises(OSError, match="not found; build it with: cargo build --release -p senet-ffi"):
        _lib._find()
    (tmp_path / "Cargo.toml").unlink()  # an installed copy of the package
    with pytest.raises(OSError, match=r"is not in its source checkout.*set SENET_FFI to its path"):
        _lib._find()


def test_build_info() -> None:
    info = senet.build_info()
    assert info.keys() == {"version", "target", "optimized", "cpu_features"}
    cargo = (ROOT / "Cargo.toml").read_text(encoding="utf-8")
    assert f'\nversion = "{info["version"]}"\n' in cargo
    assert isinstance(info["optimized"], bool)  # $SENET_FFI may name a debug build
    known = ["popcnt", "sse4.2", "avx", "avx2", "fma", "bmi1", "bmi2", "lzcnt", "avx512f"]
    assert info["cpu_features"] == [f for f in known if f in info["cpu_features"]]


@pytest.mark.parametrize("dtype", [np.int8, np.uint8, np.int16, np.uint16, np.int32, np.uint32, np.int64, np.uint64])
def test_numpy_squares(dtype: type[np.integer]) -> None:
    # Squares as narrow NumPy integers: `1 << s` must not wrap in their width.
    assert senet.mask([dtype(10), dtype(30)]) == senet.mask([10, 30]) == 1 << 10 | 1 << 30
    assert senet.mask(np.array([10, 30], dtype=dtype)) == 1 << 10 | 1 << 30
    assert senet.legal_moves([dtype(30)], [1], 1) == senet.legal_moves([30], [1], 1) != []
    with pytest.raises(TypeError):
        senet.mask([10.0])  # type: ignore[list-item]


def test_moves_match_reference() -> None:
    rng = random.Random(5)
    for _ in range(20000):
        me, opp = ref_index.random_position(rng)
        t = rng.randint(1, 5)
        got = [
            (
                m.frm,
                m.to,
                m.kind,
                "back" if m.back else "fwd",
                tuple(senet.squares(m.me_after)),
                tuple(senet.squares(m.opp_after)),
            )
            for m in senet.legal_moves(list(me), list(opp), t)
        ]
        exp = [(m.frm, m.to, m.kind, m.dir, tuple(m.me_after), tuple(m.opp_after)) for m in ref.legal_moves(me, opp, t)]
        assert got == exp, (me, opp, t)


def test_indexing_matches_reference() -> None:
    rng = random.Random(9)
    for _ in range(5000):
        me, opp = ref_index.random_position(rng)
        w, b, i = senet.index_of(list(me), list(opp))
        assert (w, b, i) == ref_index.index_of(me, opp)
        m2, o2 = senet.position_of(w, b, i)
        assert (senet.squares(m2), senet.squares(o2)) == (list(me), list(opp))


def test_features_layout() -> None:
    f = senet.features([1, 26, 30], [28])
    assert f.shape == (72,)
    assert f[0] == 1 and f[25] == 1 and f[28] == 1  # squares 1, 26, 30 -> compact 0, 25, 28
    assert f[29 + 26] == 1  # opponent on 28 -> compact 26
    assert f[58 + 2] == 1 and f[64 + 4] == 1  # 2 and 4 pieces borne off
    assert abs(f[70] - (30 + 5 + 1) / 100) < 1e-6
    assert abs(f[71] - 3 / 100) < 1e-6


@pytest.mark.parametrize(
    "call",
    [
        lambda: senet.mask([27]),
        lambda: senet.mask([0]),
        lambda: senet.mask([3, 3]),
        lambda: senet.mask(1 << 31),
        lambda: senet.mask(-1),
        lambda: senet.legal_moves([1], [2], 6),
        lambda: senet.legal_moves([1], [2], 0),
        lambda: senet.legal_moves([1], [2], 257),  # a C uint8 would wrap it to a throw of 1
        lambda: senet.legal_moves([1], [1], 2),
        lambda: senet.legal_moves([1, 2, 3, 4, 5, 6], [7], 2),
        lambda: senet.legal_moves([30], [], 1),  # the opponent has already won
        lambda: senet.index_of([], [1]),
        lambda: senet.position_of(6, 1, 0),
        lambda: senet.position_of(1, 1, senet.layer_size(1, 1)),
        lambda: senet.position_of(1 + (1 << 32), 1, 0),  # a C uint32 would wrap it to layer (1, 1)
        lambda: senet.position_of(1, 1, -1),
        lambda: senet.layer_size(-1, 1),
        lambda: senet.heuristic([1], [1]),
    ],
)
def test_invalid_arguments_raise_value_error(call: Callable[[], object]) -> None:
    with pytest.raises(ValueError):
        call()


@pytest.mark.parametrize(
    "call",
    [
        lambda eng: eng.choose([1], [10], 257, bot="greedy"),
        lambda eng: eng.move_values([1], [10], 0, bot="greedy"),
        lambda eng: eng.choose([1], [10], 1, bot="greedy", seed=-1),
        lambda eng: eng.choose([1], [10], 1, bot="greedy", seed=1 << 64),
        lambda eng: eng.match("greedy", "random", pairs=-1),  # a C uint64 would wrap it to 2**64 - 1
        lambda eng: eng.match("greedy", "random", pairs=0),  # a match of no games
        lambda eng: eng.match("greedy", "random", pairs=1, seed=-1),
        lambda eng: eng.quality("greedy", games=-1),
        lambda eng: eng.values(np.array([senet.mask([30])]), np.array([1 << 33])),  # uint32 would wrap it to 0
        lambda eng: eng.values([-1], [2]),
        lambda eng: eng.values([2.0], [4.0]),
        lambda eng: eng.values([[2]], [[4]]),
        lambda eng: eng.values((m for m in [2]), [4]),
        lambda eng: eng.values([2, 8], [4]),
        lambda eng: eng.choose([1], [10], 1, bot="greedy\0unknown"),  # C would read "greedy"
        lambda eng: eng.match("greedy", "random\0", pairs=1),
        lambda eng: eng.move_values([30], [], 1, bot="greedy"),  # a finished game
        lambda eng: eng.choose([30], [], 1, bot="greedy"),
    ],
)
def test_engine_rejects_invalid_arguments_before_the_call(call: Callable[[senet.Engine], object]) -> None:
    # These would reach the engine as different (wrapped) numbers, so they must fail in Python.
    with senet.Engine() as eng, pytest.raises(ValueError):
        call(eng)


def test_bad_bot_specs_raise_value_error() -> None:
    with senet.Engine() as eng:
        for bad in ("nope", "expectimax:9", "random:1"):
            with pytest.raises(ValueError):
                eng.choose([1, 3], [2, 4], 3, bot=bad)
            with pytest.raises(ValueError):
                eng.move_values([1, 3], [2, 4], 3, bot=bad)
            with pytest.raises(ValueError):
                eng.match("greedy", bad, pairs=1)
            # A bad spec is reported before the missing database.
            with pytest.raises(ValueError):
                eng.quality(bad, games=1)
            with pytest.raises(ValueError):
                eng.quality("greedy", bad, games=1)
        with pytest.raises(ValueError, match="does not evaluate"):
            eng.move_values([30], [1], 1, bot="random")


def test_engine_without_resources() -> None:
    with senet.Engine() as eng:
        assert not eng.has_db and not eng.has_net
        me, opp = [30], [1]
        with pytest.raises(RuntimeError, match="database"):
            eng.value(me, opp)
        with pytest.raises(RuntimeError, match="database"):
            eng.values([senet.mask(me)], [senet.mask(opp)])
        with pytest.raises(RuntimeError, match="network"):
            eng.net_value(me, opp)
        assert eng.move_values(me, opp, 1, bot="greedy") == [1.0]
        assert eng.choose(me, opp, 1, bot="greedy") == 0
        assert eng.choose(me, opp, 2, bot="greedy") is None  # 30 bears off only with a 1
        # Bots that need the database or the network the engine does not have.
        with pytest.raises(RuntimeError, match="database"):
            eng.choose([1, 3], [2, 4], 3, bot="perfect")
        with pytest.raises(RuntimeError, match="database"):
            eng.move_values([1, 3], [2, 4], 3, bot="perfect")
        with pytest.raises(RuntimeError, match="network"):
            eng.choose([1, 3], [2, 4], 3, bot="net:1")
        with pytest.raises(RuntimeError, match="database"):
            eng.match("perfect", "greedy", pairs=1)
        with pytest.raises(RuntimeError, match="database"):
            eng.quality("greedy", "greedy", games=1)
        # An unaligned array (a uint32 view at an odd address) is copied for the engine.
        unaligned = np.zeros(9, dtype=np.uint8)[1:].view(np.uint32)
        assert not unaligned.flags.aligned
        with pytest.raises(RuntimeError, match="database"):
            eng.values(unaligned, unaligned)
    with pytest.raises(ValueError, match="closed"):
        eng.choose(me, opp, 1, bot="greedy")


def test_an_engine_cannot_be_copied() -> None:
    # A copy would share the engine's handle and free it a second time.
    with senet.Engine() as eng:
        for duplicate in (copy.copy, copy.deepcopy, pickle.dumps):
            with pytest.raises(TypeError, match="cannot be copied or pickled"):
                duplicate(eng)
        assert eng.choose([30], [1], 1, bot="greedy") == 0


def test_close_during_a_call() -> None:
    # close() does not wait for the calls in progress, which keep what they use until they
    # end. Python cannot pause a native call (the Rust test a_handle_closed_while_in_use
    # does, deterministically), so the close comes after a growing delay until it meets a
    # match in progress; a match that starts after the close fails, as it should.
    for delay in [0.0] + [0.001 * 2**k for k in range(10)]:
        eng = senet.Engine()
        outcome: list[tuple[dict[str, Any] | ValueError, float]] = []

        def play(eng: senet.Engine = eng, outcome: list[Any] = outcome) -> None:
            try:
                result: dict[str, Any] | ValueError = eng.match("greedy", "random", pairs=5000)
            except ValueError as e:
                result = e
            outcome.append((result, time.perf_counter()))

        match = threading.Thread(target=play)
        match.start()
        time.sleep(delay)
        eng.close()
        closed = time.perf_counter()
        eng.close()  # closing again does nothing
        match.join()
        ((result, finished),) = outcome
        with pytest.raises(ValueError, match="closed"):
            eng.choose([30], [1], 1, bot="greedy")
        if isinstance(result, ValueError):
            assert "closed" in str(result)  # the match started after the close
        elif finished > closed:
            assert result["games"] == 10_000
            return
    pytest.fail("no close met a match in progress")


def test_engine_resource_paths() -> None:
    for db in ("no_such_dir", Path("no_such_dir")):
        with pytest.raises(OSError, match="no_such_dir"):
            senet.Engine(db=db)
    with pytest.raises(OSError, match="no_such_file"):
        senet.Engine(net=Path("no_such_file.bin"))
    with pytest.raises(ValueError, match="NUL"):
        senet.Engine(db="no_such_dir\0suffix")  # C would open "no_such_dir"


@needs_db
def test_db_values_and_moves() -> None:
    with senet.Engine(db=DB) as eng:
        assert eng.has_db and not eng.has_net
        v = eng.value([30], [1])
        assert 0.9 < v < 1.0
        vals = eng.values(np.array([senet.mask([30]), senet.mask([1, 2])], dtype=np.uint32), [senet.mask([1]), 0])
        assert abs(vals[0] - v) < 1e-6
        assert vals[1] == 0.0  # the opponent has borne off every piece
        assert eng.values([], []).shape == (0,)
        unaligned = np.zeros(9, dtype=np.uint8)[1:].view(np.uint32)
        unaligned[:] = senet.mask([30]), senet.mask([1, 2])
        assert np.array_equal(eng.values(unaligned, [senet.mask([1]), 0]), vals)
        with pytest.raises(ValueError, match="index 0"):
            eng.values([senet.mask([30])], [senet.mask([30])])
        # Move values are consistent with the definition (no extra throw on 2).
        me, opp = senet.mask([20, 24]), senet.mask([22, 25])
        moves = senet.legal_moves(me, opp, 2)
        mv = eng.move_values(me, opp, 2)
        for m, x in zip(moves, mv, strict=True):
            expect = 1.0 if m.wins else 1.0 - eng.value(m.opp_after, m.me_after)
            assert abs(x - expect) < 1e-6
        i = eng.choose(me, opp, 2, bot="perfect")
        assert i is not None and mv[i] == max(mv)
        q = eng.quality("greedy", "random", games=20)
        assert q["games"] == 20 and q["decisions"] > 0
        with pytest.raises(ValueError):
            eng.quality("nope", games=1)


@needs_db
def test_partial_database(tmp_path: Path) -> None:
    # The layers with at most three pieces in total, copied from the complete database.
    for name in ("L11", "L12", "L21"):
        shutil.copyfile(DB / f"{name}.f32", tmp_path / f"{name}.f32")
    with senet.Engine(db=tmp_path) as eng:
        assert 0.9 < eng.value([30], [1]) < 1.0
        with pytest.raises(KeyError):
            eng.value([1, 2], [3, 4])
        assert np.isnan(eng.values([senet.mask([1, 2])], [senet.mask([3, 4])])[0])
        with pytest.raises(RuntimeError, match="complete"):
            eng.choose([30], [1], 1, bot="perfect")
        with pytest.raises(RuntimeError, match="complete"):
            eng.quality("greedy", "random", games=1)
    # Moves from layer (2, 1) lead into (1, 2): a database without it is refused.
    (tmp_path / "L12.f32").unlink()
    with pytest.raises(OSError, match=r"L21\.f32 needs L12\.f32"):
        senet.Engine(db=tmp_path)


@needs_db
def test_insights_analysis() -> None:
    with senet.Engine(db=DB) as eng:
        r = insights.analyze(eng, games=3)
    assert abs(r["white_win_prob"] - 0.5025) < 1e-3
    assert list(r["openings"]) == [1, 2, 3, 4, 5]
    assert [text for text, _ in r["openings"][2]] == ["9→11"]  # the only legal move for a 2
    for moves in r["openings"].values():
        assert [v for _, v in moves] == sorted((v for _, v in moves), reverse=True)
    assert r["selfplay"]["games"] == 3 and sum(r["selfplay"]["kinds"].values()) == pytest.approx(1.0)
    assert insights.render(r).count("★") >= 5
    # The report is made from the results as read back from their JSON, whose keys are strings.
    assert insights.render(json.loads(json.dumps(r))) == insights.render(r)


@needs_db
def test_selfplay_does_not_depend_on_the_threads() -> None:
    with senet.Engine(db=DB) as eng:
        one = insights.selfplay(eng, 8, seed=5, workers=1)
        assert insights.selfplay(eng, 8, seed=5, workers=4) == one
        assert insights.selfplay(eng, 8, seed=6, workers=4) != one
    assert list(one["comeback_games"].values()) == sorted(one["comeback_games"].values(), reverse=True)


@needs_net
def test_net_value() -> None:
    with senet.Engine(net=NET) as eng:
        assert eng.has_net and not eng.has_db
        me, opp = senet.start()
        assert abs(eng.net_value(me, opp) - 0.5025) < 0.01


def test_match_runs() -> None:
    with senet.Engine() as eng:
        r = eng.match("greedy", "random", pairs=200, seed=3)
    assert r["games"] == 400
    assert r["a_win_rate"] > 0.6  # the heuristic should crush random play
