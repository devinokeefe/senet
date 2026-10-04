"""The distillation script's checks of its inputs and its training schedule
(senet_train.distill; skipped without PyTorch)."""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytest.importorskip("torch", reason="needs PyTorch")

import torch

from senet import manifest
from senet._lib import ROOT
from senet.provenance import created_by
from senet_ref.indexing import random_position
from senet_train import check_net, distill, snn1
from senet_train.distill import (
    REC,
    Net,
    Records,
    Split,
    evaluate,
    export_snn1,
    featurize,
    keys,
    load_records,
    load_snn1,
    publish,
    train,
    widths,
)

NET = ROOT / "models" / "senet_net.bin"


def mask(squares: tuple[int, ...]) -> int:
    return sum(1 << s for s in squares)


def records(n: int, seed: int = 0) -> np.ndarray:
    """`n` records of random positions, each with the value 0.5."""
    rng = random.Random(seed)
    positions = [random_position(rng) for _ in range(n)]
    recs = np.zeros(n, dtype=REC)
    recs["me"] = [mask(me) for me, _ in positions]
    recs["opp"] = [mask(opp) for _, opp in positions]
    recs["v"] = 0.5
    return recs


def data_file(path: Path, recs: np.ndarray) -> Path:
    """`recs` written as training data, with its manifest."""
    recs.tofile(path)
    manifest.write(path, manifest.new(manifest.TRAINING_DATA, [manifest.file_entry(path)], created_by()))
    return path


def recurring(n: int, seed: int) -> np.ndarray:
    """`n` records, most of them of a few positions that recur (as the opening's do in games)."""
    rng = np.random.default_rng(seed)
    pool = records(40, seed)
    recs = records(n, seed + 1)
    common = rng.random(n) < 0.8
    recs[common] = pool[rng.integers(0, len(pool), int(common.sum()))]
    recs["v"] = rng.random(n, dtype=np.float32)
    return recs


def dataset(recs: np.ndarray) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.from_numpy(recs["me"].astype(np.int32)),
        torch.from_numpy(recs["opp"].astype(np.int32)),
        torch.from_numpy(recs["v"].copy()),
    )


def test_load_records(tmp_path: Path) -> None:
    path = tmp_path / "train.bin"
    good = records(100)
    good.tofile(path)
    assert np.array_equal(load_records(path), good)

    path.write_bytes(good.tobytes() + b"\0")
    with pytest.raises(ValueError, match="1,201 bytes is not a whole number of 12-byte records"):
        load_records(path)

    corruptions: list[tuple[str, Any]] = [
        ("me", 1 << 27),  # the House of Water is always empty
        ("me", 1),  # square 0 does not exist
        ("me", 1 << 31),
        ("opp", 0),  # a finished game
        ("opp", mask((1, 2, 3, 4, 5, 6))),  # six pieces
        ("opp", good["me"][42]),  # both sides on the same squares
        ("v", np.nan),
        ("v", 1.5),
        ("v", -0.25),
    ]
    for field, value in corruptions:
        bad = good.copy()
        bad[field][42] = value
        bad.tofile(path)
        with pytest.raises(ValueError, match="record 42 is not a game in progress"):
            load_records(path)


@pytest.mark.parametrize("steps", [3, 4, 20, 39, 40, 41])
def test_short_training_schedules(steps: int) -> None:
    # OneCycleLR with a 5% warm-up divides by zero at 20 steps; the warm-up takes at least 2.
    torch.manual_seed(0)
    data = dataset(records(steps))
    model = Net([4])
    train(model, data, {"val": data}, epochs=1, batch=1, lr=1e-3, wd=0.0)


def test_too_few_training_steps() -> None:
    data = dataset(records(2))
    with pytest.raises(ValueError, match="2 steps; at least 3"):
        train(Net([4]), data, {"val": data}, epochs=1, batch=1, lr=1e-3, wd=0.0)
    with pytest.raises(ValueError, match="at least 1"):
        train(Net([4]), data, {"val": data}, epochs=0, batch=1, lr=1e-3, wd=0.0)


def test_a_network_predicting_nan_fails_evaluation() -> None:
    data = dataset(records(10))
    model = Net([4])
    with torch.no_grad():
        model.linears[0].bias.fill_(float("nan"))
    with pytest.raises(ArithmeticError, match="NaN"):
        evaluate(model, data)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16])
def test_export_converts_to_float32(tmp_path: Path, dtype: torch.dtype) -> None:
    torch.manual_seed(0)
    model = Net([4]).to(dtype)
    path = tmp_path / "net.bin"
    export_snn1(model, path)
    for layer, (w, b) in zip(model.linears, snn1.read(path.read_bytes()), strict=True):
        assert np.array_equal(w, layer.weight.detach().float().numpy())
        assert np.array_equal(b, layer.bias.detach().float().numpy())
    # A float64 value beyond float32's range would become infinite.
    with torch.no_grad():
        model.linears[0].bias[0] = 1e300 if dtype == torch.float64 else float("inf")
    with pytest.raises(ValueError, match="not finite"):
        export_snn1(model, path)


def test_networks_are_float32_whatever_the_default_dtype(tmp_path: Path) -> None:
    path = tmp_path / "net.bin"
    export_snn1(Net([4]), path)
    data = dataset(records(5))
    default = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        for model in (Net([4]), load_snn1(path)):
            assert all(p.dtype == torch.float32 for p in model.parameters())
            assert model(featurize(data[0], data[1])).dtype == torch.float32
    finally:
        torch.set_default_dtype(default)


def test_hidden_widths() -> None:
    assert widths("512,256,128") == [512, 256, 128]
    assert widths("4096") == [4096]
    for bad, error in (("4097", "1..4096 outputs"), (",".join(["8"] * 16), "17 layers"), ("8,0", "8 -> 0"), ("x", "x")):
        with pytest.raises(argparse.ArgumentTypeError, match=error):
            widths(bad)


def test_the_browser_check_can_require_node(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(check_net.shutil, "which", lambda name: None)
    assert check_net.check_js(NET, [], np.zeros(0), require_node=False)
    assert "skipping the browser engine check" in capsys.readouterr().out
    assert not check_net.check_js(NET, [], np.zeros(0), require_node=True)
    assert "node not found: FAIL" in capsys.readouterr().out


def test_training_data_needs_its_manifest(tmp_path: Path) -> None:
    path = tmp_path / "train.bin"
    records(20).tofile(path)
    with pytest.raises(manifest.ManifestError, match="no manifest"):
        Records([path])
    data_file(path, records(20))
    path.write_bytes(records(20, seed=1).tobytes())  # the same size, other records
    with pytest.raises(manifest.ManifestError, match="SHA-256"):
        Records([path])
    net = tmp_path / "net.bin"
    export_snn1(Net([4]), net)
    manifest.write(net, manifest.new(manifest.NETWORK, [manifest.file_entry(net)], created_by()))
    with pytest.raises(manifest.ManifestError, match="a network manifest, not training data"):
        Records([net])


@pytest.mark.parametrize("chunk", [7, 64, 1 << 24])
def test_records_of_several_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, chunk: int) -> None:
    monkeypatch.setattr(distill, "CHUNK", chunk)
    parts = [recurring(n, seed) for seed, n in enumerate((100, 1, 57))]
    recs = Records([data_file(tmp_path / f"part{i}.bin", r) for i, r in enumerate(parts)])
    everything = np.concatenate(parts)
    assert len(recs) == len(everything)
    assert np.array_equal(np.concatenate([c for _, c in recs.chunks()]), everything)
    assert [start for start, _ in recs.chunks()][:2] == [0, min(chunk, 100)]
    idx = np.random.default_rng(0).permutation(len(everything))[:90]
    assert np.array_equal(recs.take(idx), everything[idx])


@pytest.mark.parametrize("chunk", [5, 64, 1 << 24])
def test_the_split_is_the_one_documented(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, chunk: int) -> None:
    # The definition (SPLIT_DEFINITION), computed the straightforward way, which takes all the
    # records in memory at once: the network was trained on this split.
    monkeypatch.setattr(distill, "CHUNK", chunk)
    parts = [recurring(n, seed) for seed, n in enumerate((400, 3, 250))]
    everything = np.concatenate(parts)
    recs = Records([data_file(tmp_path / f"p{i}.bin", r) for i, r in enumerate(parts)])
    for seed, val_frac in ((0, 0.02), (5, 0.25)):
        split = Split(recs, val_frac, seed)
        perm = np.random.default_rng(seed).permutation(len(everything))
        n_val = int(len(everything) * val_frac)
        val, training = everything[perm[:n_val]], everything[perm[n_val:]]
        assert np.array_equal(split.val, val)
        me, opp, v = split.training_data(torch.device("cpu"))
        assert np.array_equal(me.numpy(), training["me"].astype(np.int32))
        assert np.array_equal(opp.numpy(), training["opp"].astype(np.int32))
        assert np.array_equal(v.numpy(), training["v"])
        unseen, distinct = split.positions()
        assert np.array_equal(unseen, ~np.isin(keys(val), keys(training)))
        assert unseen.any() and not unseen.all()
        assert distinct == {
            "training": len(np.unique(keys(training))),
            "validation": len(np.unique(keys(val))),
            "unseen": len(np.unique(keys(val[unseen]))),
            "all": len(np.unique(keys(everything))),
        }


def test_a_split_needs_records_on_both_sides(tmp_path: Path) -> None:
    recs = Records([data_file(tmp_path / "train.bin", records(10))])
    for val_frac in (0.05, 1.0):
        with pytest.raises(ValueError, match="empty training or validation split of 10 records"):
            Split(recs, val_frac, 0)


def test_publish_writes_the_files_then_their_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    out = tmp_path / "net"
    record: dict[str, Any] = {"by": created_by(), "settings": {"seed": 1}, "inputs": None, "contents": {}, "notes": []}
    model = Net([4])
    path = publish(out, model, record)
    m = manifest.verify(path)
    assert (m["kind"], m["settings"]) == (manifest.NETWORK[0], {"seed": 1})
    assert [f["name"] for f in m["files"]] == ["net.bin", "net.pt"]
    assert out.with_suffix(".bin").read_bytes() == distill.snn1_bytes(model)
    assert torch.load(out.with_suffix(".pt"))["layers.0.weight"].shape == (4, snn1.N_INPUTS)

    def fail(path: Path, data: bytes) -> None:
        raise OSError("disk full")

    # A publication that fails part way leaves no manifest to vouch for what is there.
    monkeypatch.setattr(manifest, "write_atomically", fail)
    with pytest.raises(OSError, match="disk full"):
        publish(out, Net([8]), record)
    assert not path.exists()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["net.bin", "net.pt"]


def test_evaluate_and_record(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    data = data_file(tmp_path / "train.bin", recurring(500, 3))
    net = tmp_path / "net.bin"
    export_snn1(Net([4]), net)
    distill.main(["--data", str(data), "--evaluate", str(net)])
    out = capsys.readouterr().out
    assert "500 records" in out and "10 validation records" in out and "validation records: MAE" in out
    assert not manifest.manifest_path(net).exists()

    net.with_suffix(".pt").write_bytes(b"a state dict, which is not published")
    distill.main(["--data", str(data), "--evaluate", str(net), "--record", "--note", "a remark"])
    m = manifest.verify(net)
    assert [f["name"] for f in m["files"]] == ["net.bin"]
    assert m["settings"] is None
    assert m["notes"][0].startswith("Recorded after the fact") and m["notes"][1] == "a remark"
    assert m["inputs"] == {"training_data": [manifest.identity(data)]}
    split = m["contents"]["split"]
    assert (split["records"], split["validation_records"], split["val_frac"], split["seed"]) == (500, 10, 0.02, 0)
    assert split["data_sha256"] == [manifest.file_sha256(data)]
    assert split["distinct_positions"]["all"] == len(np.unique(keys(recurring(500, 3))))
    assert (split["numpy"], split["definition"]) == (np.__version__, distill.SPLIT_DEFINITION)
    assert m["contents"]["errors"]["validation records"]["records"] == 10
    assert m["contents"]["architecture"] == [snn1.N_INPUTS, 4, 1]
    assert m["contents"]["features"] == distill.FEATURES
    assert m["created_by"]["environment"]["torch"] == torch.__version__


def test_evaluate_keeps_to_the_networks_split(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    data = data_file(tmp_path / "train.bin", recurring(500, 3))
    other = data_file(tmp_path / "other.bin", recurring(500, 4))
    net = tmp_path / "net.bin"
    export_snn1(Net([4]), net)
    distill.main(["--data", str(data), "--evaluate", str(net), "--record", "--seed", "7", "--val-frac", "0.1"])
    split = manifest.read(net)["contents"]["split"]
    assert (split["seed"], split["val_frac"]) == (7, 0.1)
    capsys.readouterr()

    # The seed and val_frac default to the network's; other ones, or other data, need --force.
    distill.main(["--data", str(data), "--evaluate", str(net)])
    assert "50 validation records" in capsys.readouterr().out
    for args, problem in (
        (["--data", str(other)], "it was trained on other data"),
        (["--data", str(data), "--seed", "0"], "its split has seed 7, not --seed 0"),
        (["--data", str(data), "--val-frac", "0.02"], "its split has val_frac 0.1, not --val-frac 0.02"),
    ):
        with pytest.raises(SystemExit):
            distill.main([*args, "--evaluate", str(net)])
        assert f"{problem} (--force to evaluate it anyway)" in capsys.readouterr().err
        distill.main([*args, "--evaluate", str(net), "--force"])
        assert f"{problem}; evaluating it anyway (--force)" in capsys.readouterr().out

    # A manifest of another network is not used.
    export_snn1(Net([8]), net)
    with pytest.raises(SystemExit):
        distill.main(["--data", str(data), "--evaluate", str(net)])
    assert "not the manifest of" in capsys.readouterr().err


def test_output_names() -> None:
    runs = Path("runs")
    for out, name in (("lr0.001", "lr0.001"), ("net_v1.2", "net_v1.2"), ("net", "net"), ("net.bin", "net")):
        assert distill.outputs(runs / out) == (runs / f"{name}.bin", runs / f"{name}.pt")


def test_train_and_publish(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    data = data_file(tmp_path / "train.bin", recurring(300, 4))
    out = tmp_path / "models" / "small_lr0.002"
    args = ["--data", str(data), "--out", str(out), "--hidden", "8,4", "--epochs", "2", "--batch", "64", "--seed", "3"]
    distill.main(args)
    net = tmp_path / "models" / "small_lr0.002.bin"
    m = manifest.verify(net)
    assert [f["name"] for f in m["files"]] == ["small_lr0.002.bin", "small_lr0.002.pt"]
    settings = {"hidden": [8, 4], "epochs": 2, "batch": 64, "lr": 2e-3, "wd": 1e-5, "val_frac": 0.02, "seed": 3}
    assert m["settings"] == settings
    assert m["contents"]["architecture"] == [snn1.N_INPUTS, 8, 4, 1]
    assert m["contents"]["split"]["validation_records"] == 6
    assert len(m["contents"]["epoch_seconds"]) == 2
    model = load_snn1(net)
    assert distill.parameters(model) == m["contents"]["parameters"]
    assert "epoch  2:" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("args", "error"),
    [
        (["--out", "net", "--record"], "--record, --note and --force go with --evaluate"),
        (["--out", "net", "--note", "x"], "--record, --note and --force go with --evaluate"),
        (["--out", "net", "--force"], "--record, --note and --force go with --evaluate"),
        (["--evaluate", "net.bin", "--note", "x"], "--note goes with --record"),
        (["--out", "net", "--evaluate", "net.bin"], "give either --out"),
        ([], "give either --out"),
    ],
)
def test_conflicting_options(capsys: pytest.CaptureFixture[str], args: list[str], error: str) -> None:
    with pytest.raises(SystemExit):
        distill.main(["--data", "train.bin", *args])
    assert error in capsys.readouterr().err
