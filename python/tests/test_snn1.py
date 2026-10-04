"""The SNN1 network format: its reader and writer (senet_train.snn1), the portable build's
conversion (tools/build_portable.py) and, when PyTorch is installed, the training code's
export and import (senet_train.distill); and the committed network's record of where it
comes from (models/senet_net.manifest.json, models/training/)."""

from __future__ import annotations

import base64
import importlib.util
import json
import struct
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
from conftest import NET, needs_net

from senet import manifest
from senet_train import snn1 as fmt

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def build_portable() -> ModuleType:
    """tools/build_portable.py is a script, not part of a package: load it from its path."""
    spec = importlib.util.spec_from_file_location("build_portable", ROOT / "tools" / "build_portable.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def snn1(shapes: list[tuple[int, int]], extra: bytes = b"", bias: float = -1) -> bytes:
    """An SNN1 file whose weights are 0, 1, 2, ... and whose biases are `bias`, -2, -3, ...
    per layer, written byte by byte rather than with the module under test."""
    data = b"SNN1" + struct.pack("<I", len(shapes))
    for n_in, n_out in shapes:
        data += struct.pack("<II", n_in, n_out)
        data += np.arange(n_in * n_out, dtype="<f4").tobytes()
        data += np.array([bias, *range(-2, -n_out - 1, -1)], dtype="<f4").tobytes()
    return data + extra


def test_net_to_js(tmp_path: Path, build_portable: ModuleType) -> None:
    path = tmp_path / "net.bin"
    path.write_bytes(snn1([(72, 3), (3, 1)]))
    js = build_portable.net_to_js(path)
    prefix = "window.SENET_NET = "
    assert js.startswith(prefix) and js.endswith(";")
    net = json.loads(js[len(prefix) : -1])
    assert net["format"] == "senet-mlp-v1"
    assert [(layer["in"], layer["out"]) for layer in net["layers"]] == [(72, 3), (3, 1)]

    def floats(b64: str) -> list[float]:
        return np.frombuffer(base64.b64decode(b64), dtype="<f4").tolist()

    assert floats(net["layers"][0]["w"]) == list(range(72 * 3))
    assert floats(net["layers"][0]["b"]) == [-1, -2, -3]
    assert floats(net["layers"][1]["w"]) == [0, 1, 2]
    assert floats(net["layers"][1]["b"]) == [-1]


GOOD = snn1([(72, 3), (3, 1)])
WIDEST = [(72, fmt.MAX_WIDTH), (fmt.MAX_WIDTH, 1)]
DEEPEST = [(72, 1)] + [(1, 1)] * (fmt.MAX_LAYERS - 1)


def test_read_and_write() -> None:
    layers = fmt.read(GOOD)
    assert [(w.shape, b.shape) for w, b in layers] == [((3, 72), (3,)), ((1, 3), (1,))]
    assert layers[0][0].ravel().tolist() == list(range(72 * 3)) and layers[0][1].tolist() == [-1, -2, -3]
    assert fmt.write(layers) == GOOD
    # The format's limits are reachable.
    for shapes in (WIDEST, DEEPEST):
        assert fmt.write(fmt.read(snn1(shapes))) == snn1(shapes)


@pytest.mark.parametrize(
    ("data", "error"),
    [
        pytest.param(b"", "truncated", id="empty"),
        pytest.param(b"SNN2" + GOOD[4:], "not an SNN1", id="bad-magic"),
        pytest.param(GOOD[:6], "truncated", id="truncated-count"),
        pytest.param(GOOD[:20], "truncated", id="truncated-weights"),
        pytest.param(GOOD[:-1], "truncated", id="truncated-last-bias"),
        pytest.param(GOOD + b"\0", "1 unexpected bytes", id="trailing-byte"),
        pytest.param(snn1([]), "0 layers", id="no-layers"),
        pytest.param(snn1([*DEEPEST, (1, 1)]), "17 layers", id="too-deep"),
        pytest.param(snn1([(71, 1)]), "layer 0 is 71 -> 1; expected 72 inputs", id="wrong-inputs"),
        pytest.param(snn1([(72, 3)]), "3 outputs", id="wrong-outputs"),
        pytest.param(snn1([(72, 3), (4, 1)]), "layer 1 is 4 -> 1; expected 3 inputs", id="unchained"),
        pytest.param(snn1([(72, 4097), (4097, 1)]), "1..4096 outputs", id="too-wide"),
        pytest.param(snn1([(72, 3), (3, 0)]), "1..4096 outputs", id="no-outputs"),
        pytest.param(
            snn1([(72, 3), (3, 1)], bias=float("nan")), "layer 0 has a parameter that is not finite", id="nan"
        ),
        pytest.param(snn1([(72, 3), (3, 1)], bias=float("inf")), "not finite", id="inf"),
    ],
)
def test_malformed_files_are_rejected(tmp_path: Path, build_portable: ModuleType, data: bytes, error: str) -> None:
    with pytest.raises(ValueError, match=error):
        fmt.read(data)
    path = tmp_path / "net.bin"
    path.write_bytes(data)
    with pytest.raises(ValueError, match=error):
        build_portable.net_to_js(path)


def test_write_rejects_what_read_would() -> None:
    w0, b0 = np.zeros((3, 72)), np.zeros(3)
    w1, b1 = np.zeros((1, 3)), np.zeros(1)
    with pytest.raises(ValueError, match="0 layers"):
        fmt.write([])
    with pytest.raises(ValueError, match="biases of shape"):
        fmt.write([(w0, np.zeros(4)), (w1, b1)])
    with pytest.raises(ValueError, match="layer 1 is 4 -> 1"):
        fmt.write([(w0, b0), (np.zeros((1, 4)), b1)])
    with pytest.raises(ValueError, match="layer 1 has a parameter that is not finite"), np.errstate(over="ignore"):
        fmt.write([(w0, b0), (np.full((1, 3), 1e300), b1)])  # inf as float32


def test_training_round_trip(tmp_path: Path, build_portable: ModuleType) -> None:
    torch = pytest.importorskip("torch", reason="needs PyTorch")
    from senet_train.distill import Net, export_snn1, load_snn1

    torch.manual_seed(0)
    model = Net([8, 4])
    path = tmp_path / "net.bin"
    export_snn1(model, path)
    assert [w.shape for w, _ in fmt.read(path.read_bytes())] == [(8, 72), (4, 8), (1, 4)]
    loaded = load_snn1(path)
    for (name, a), (name2, b) in zip(model.state_dict().items(), loaded.state_dict().items(), strict=True):
        assert name == name2 and torch.equal(a, b)

    path.write_bytes(path.read_bytes()[:-4])
    with pytest.raises(ValueError, match="truncated"):
        load_snn1(path)
    with torch.no_grad():
        model.linears[1].bias[0] = float("nan")
    with pytest.raises(ValueError, match="not finite"):
        export_snn1(model, path)


@needs_net
def test_the_committed_network_records_where_it_comes_from() -> None:
    m = manifest.verify(NET)
    contents = m["contents"]
    layers = fmt.read(NET.read_bytes())
    assert contents["architecture"] == [fmt.N_INPUTS, *(len(b) for _, b in layers)]
    assert contents["parameters"] == sum(w.size + b.size for w, b in layers)
    assert contents["features"]["name"] == "senet-features/1"
    # The training data is not in the repository; its manifest and the training log are.
    training = ROOT / "models" / "training"
    data = manifest.read(training / "train.manifest.json")
    assert m["inputs"]["training_data"] == [
        {"path": "runs/train.bin", "manifest_sha256": manifest.file_sha256(training / "train.manifest.json")}
    ]
    assert contents["split"]["data_sha256"] == [f["sha256"] for f in data["files"]]
    assert contents["split"]["records"] == data["contents"]["records"]
    assert (training / "distill.log").exists() and any("models/training/distill.log" in note for note in m["notes"])
