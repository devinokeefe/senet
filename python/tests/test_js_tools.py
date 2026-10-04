"""The browser engine's network loader (web/portable/senet-engine.js) and the JS checkers
(tools/check_js_rules.mjs, tools/check_js_net.mjs, tools/check_portable.mjs) on hand-made
inputs, including those they must refuse, and mutated network files, which the Rust, Python
and browser loaders must accept or refuse alike. Skipped without node."""

from __future__ import annotations

import base64
import json
import shutil
import struct
import subprocess
import sys
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from conftest import NET, needs_net

import senet
from senet._lib import ROOT
from senet_train import snn1

NODE = shutil.which("node")
pytestmark = pytest.mark.skipif(NODE is None, reason="needs Node.js")


def node(*args: str | Path, stdin: str = "") -> subprocess.CompletedProcess[str]:
    assert NODE is not None
    return subprocess.run(
        [NODE, *map(str, args)], cwd=ROOT, input=stdin, capture_output=True, text=True, encoding="utf-8", check=False
    )


def layer(n_in: int, n_out: int, value: float = 0.0, n_weights: int | None = None) -> dict[str, Any]:
    """A layer in the portable build's JSON form, every parameter `value`."""

    def b64(n: int) -> str:
        return base64.b64encode(np.full(n, value, dtype="<f4").tobytes()).decode()

    return {"in": n_in, "out": n_out, "w": b64(n_in * n_out if n_weights is None else n_weights), "b": b64(n_out)}


def test_load_net() -> None:
    good = [layer(72, 3), layer(3, 1)]
    nets: list[tuple[dict[str, Any], str | None]] = [
        ({"format": "senet-mlp-v1", "layers": good}, None),
        ({"format": "senet-mlp-v1", "layers": [layer(72, 4096), layer(4096, 1)]}, None),
        ({"format": "senet-mlp-v2", "layers": good}, "network format senet-mlp-v2"),
        ({"format": "senet-mlp-v1", "layers": []}, "0 layers"),
        ({"format": "senet-mlp-v1", "layers": [layer(72, 1)] + [layer(1, 1)] * 16}, "17 layers"),
        ({"format": "senet-mlp-v1", "layers": [layer(72, 4097), layer(4097, 1)]}, "1..4096 outputs"),
        ({"format": "senet-mlp-v1", "layers": [layer(71, 1)]}, "layer 0 is 71 -> 1; expected 72 inputs"),
        ({"format": "senet-mlp-v1", "layers": [layer(72, 3), layer(4, 1)]}, "layer 1 is 4 -> 1; expected 3"),
        ({"format": "senet-mlp-v1", "layers": [layer(72, 3)]}, "3 outputs"),
        ({"format": "senet-mlp-v1", "layers": [layer(72, 3, n_weights=144), layer(3, 1)]}, "144 weights"),
        ({"format": "senet-mlp-v1", "layers": [{**layer(72, 3), "b": "AAAAAAA="}, layer(3, 1)]}, "not a whole number"),
        ({"format": "senet-mlp-v1", "layers": [layer(72, 3), layer(3, 1, float("nan"))]}, "layer 1 has a parameter"),
        ({"format": "senet-mlp-v1", "layers": [layer(72, 3, float("inf")), layer(3, 1)]}, "not finite"),
    ]
    script = """
        const E = require("./web/portable/senet-engine.js");
        const nets = JSON.parse(require("node:fs").readFileSync(0, "utf8"));
        const error = (net) => { try { E.loadNet(net); return null; } catch (e) { return e.message; } };
        console.log(JSON.stringify(nets.map(error)));
    """
    result = node("-e", script, stdin=json.dumps([net for net, _ in nets]))
    assert result.returncode == 0, result.stderr
    for (_, expected), error in zip(nets, json.loads(result.stdout), strict=True):
        assert (error is None) if expected is None else (error is not None and expected in error), (expected, error)


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


def test_check_js_rules(tmp_path: Path) -> None:
    dump = tmp_path / "moves.jsonl"

    def check(*records: Any) -> int:
        dump.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
        return node("tools/check_js_rules.mjs", dump).returncode

    assert check(RECORD) == 0
    # The order of an object's keys does not matter; the order of the moves does.
    assert check(json.loads(json.dumps(RECORD, sort_keys=True))) == 0
    assert check({**RECORD, "moves": RECORD["moves"][::-1]}) == 1
    assert check() == 1  # an empty dump checks nothing
    assert check({**RECORD, "moves": RECORD["moves"][:1]}) == 1
    for bad in (
        {**RECORD, "t": 0},
        {**RECORD, "t": 2.5},
        {**RECORD, "me": [3, 9, 27]},
        {**RECORD, "me": [3, 3, 26]},
        {**RECORD, "opp": [3, 5, 28]},
        {**RECORD, "opp": [], "moves": []},
        {"me": [3, 9, 26], "opp": [4, 5, 28], "t": 2},
        {**RECORD, "moves": [{k: v for k, v in RECORD["moves"][0].items() if k != "dir"}, RECORD["moves"][1]]},
        {**RECORD, "moves": [{**RECORD["moves"][0], "extra": 1}, RECORD["moves"][1]]},
        [RECORD],
    ):
        assert check(RECORD, bad) == 1, bad


@needs_net
def test_check_js_net(tmp_path: Path) -> None:
    cases = tmp_path / "cases.json"
    me, opp = [1, 3, 5, 7, 9], [2, 4, 6, 8, 10]
    with senet.Engine(net=NET) as eng:
        want = eng.net_value(me, opp)

    def check(*rows: Any) -> int:
        cases.write_text(json.dumps(list(rows)), encoding="utf-8")
        return node("tools/check_js_net.mjs", NET, cases).returncode

    m, o = senet.mask(me), senet.mask(opp)
    assert check([m, o, want]) == 0
    assert check() == 1  # no cases
    assert check([m, o, want + 0.01]) == 1
    for bad in (
        [m, o, None],  # would count as 0
        [m, o, 1.5],
        [m, 0, 0.0],  # a finished game
        [m, m, want],
        [m | 1 << 27, o, want],
        [m, o],
    ):
        assert check([m, o, want], bad) == 1, bad


def test_check_portable(tmp_path: Path) -> None:
    # A small network keeps the games quick.
    rng = np.random.default_rng(0)
    nets = []
    for i in range(2):
        net = tmp_path / f"net{i}.bin"
        net.write_bytes(
            snn1.write([(rng.normal(0, 0.1, (8, 72)), np.zeros(8)), (rng.normal(0, 0.1, (1, 8)), np.zeros(1))])
        )
        nets.append(net)
    page = tmp_path / "page.html"
    build: list[str | Path] = [sys.executable, "tools/build_portable.py", "--net", nets[0], "--out", page]
    assert subprocess.run(build, cwd=ROOT, capture_output=True, check=False).returncode == 0
    assert b"\r" not in page.read_bytes()  # LF line ends on every platform, so the same bytes
    html = page.read_text(encoding="utf-8")

    def check(text: str, net: Path = nets[0]) -> subprocess.CompletedProcess[str]:
        page.write_text(text, encoding="utf-8")
        return node("tools/check_portable.mjs", page, net)

    result = check(html)
    assert result.returncode == 0, result.stdout
    for bad, net, error in (
        (html, nets[1], "embeds"),
        (
            html.replace("</body>", '<script>fetch("/api/info").catch(() => {});</script></body>'),
            nets[0],
            "asks no server",
        ),
        (html.replace("</body>", '<script src="extra.js"></script></body>'), nets[0], "all inline"),
        (html.replace("</head>", '<link rel="icon" href="icon.png"></head>'), nets[0], "links nothing"),
        (html.replace("</body>", "<script>console.warn('oops');</script></body>"), nets[0], "logs nothing"),
        (html.replace("</body>", "<script>throw new Error('broken');</script></body>"), nets[0], "broken"),
    ):
        result = check(bad, net)
        assert result.returncode == 1 and error in result.stdout, (error, result.stdout)


def snn1_mutants(n: int, seed: int) -> list[bytes]:
    """`n` SNN1 files, each a small valid network with one random change: a header field set
    to a boundary value, a word to a special float, a flipped bit, or bytes cut off, added
    or cut out."""
    rng = np.random.default_rng(seed)
    fields = [0, 1, 2, 3, 16, 17, 71, 72, 73, 4096, 4097, 2**32 - 1]
    floats = [np.nan, np.inf, -np.inf, np.finfo(np.float32).max, -0.0, 1e-45]
    bases = []
    for widths in ([1], [3, 1], [4, 2, 1]):
        shapes = list(pairwise([snn1.N_INPUTS, *widths]))
        layers = [(rng.normal(0, 0.1, (b, a)), rng.normal(0, 0.1, b)) for a, b in shapes]
        offsets = [4]  # of the header fields: the layer count, then each layer's input and output counts
        at = 8
        for a, b in shapes:
            offsets += [at, at + 4]
            at += 8 + 4 * (a * b + b)
        bases.append((snn1.write(layers), offsets))
    mutants = []
    for _ in range(n):
        base, offsets = bases[rng.integers(len(bases))]
        data = bytearray(base)
        at = int(rng.integers(len(data)))
        match rng.integers(6):
            case 0:
                field = offsets[rng.integers(len(offsets))]
                data[field : field + 4] = struct.pack("<I", fields[rng.integers(len(fields))])
            case 1:
                word = at // 4 * 4
                data[word : word + 4] = struct.pack("<f", floats[rng.integers(len(floats))])
            case 2:
                data[at] ^= 1 << int(rng.integers(8))
            case 3:
                del data[at:]
            case 4:
                data += rng.integers(256, size=1 + rng.integers(8)).astype(np.uint8).tobytes()
            case _:
                del data[at : at + 1 + rng.integers(8)]
        mutants.append(bytes(data))
    return mutants


def test_snn1_files_are_accepted_alike_by_rust_python_and_js(tmp_path: Path) -> None:
    paths = []
    for i, data in enumerate(snn1_mutants(400, seed=5)):
        paths.append(tmp_path / f"net{i}.bin")
        paths[-1].write_bytes(data)

    def python_accepts(path: Path) -> bool:
        try:
            snn1.read(path.read_bytes())
        except ValueError:
            return False
        return True

    def rust_accepts(path: Path) -> bool:
        try:
            senet.Engine(net=path).close()
        except OSError:
            return False
        return True

    script = """
        import { readFileSync } from "node:fs";
        import { createRequire } from "node:module";
        import { readSnn1 } from "./tools/snn1.mjs";
        const E = createRequire(import.meta.url)("./web/portable/senet-engine.js");
        const accepts = (path) => { try { E.loadNet(readSnn1(path)); return true; } catch { return false; } };
        console.log(JSON.stringify(JSON.parse(readFileSync(0, "utf8")).map(accepts)));
    """
    result = node("--input-type=module", "-e", script, stdin=json.dumps([str(p) for p in paths]))
    assert result.returncode == 0, result.stderr
    js = json.loads(result.stdout)
    python = [python_accepts(p) for p in paths]
    rust = [rust_accepts(p) for p in paths]
    disagreements = [(p.name, a, b, c) for p, a, b, c in zip(paths, python, rust, js, strict=True) if not a == b == c]
    assert not disagreements, f"(file, python, rust, js): {disagreements[:5]}"
    assert 50 < sum(python) < 350, sum(python)  # both verdicts are well exercised
