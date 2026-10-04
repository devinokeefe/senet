"""Parity checks between the PyTorch training code, the Rust engine and the browser engine.

    python -m senet_train.check_net [--net models/senet_net.bin] [--n 20000] [--require-node]

1. The PyTorch featurizer matches senet_core::net::features on random positions.
2. With --net: the network is rebuilt in PyTorch from the SNN1 file itself, and its output
   logits are compared with the Rust forward pass's and with the browser engine's
   (tools/check_js_net.mjs; skipped if node is not installed, unless --require-node).
   Logits, not probabilities, which sigmoid flattens where they are near 0 or 1; the Rust
   engine's logit is recovered, in float64, from the probability it returns.

Needs PyTorch and the senet_ffi library, but neither the database nor the .pt checkpoint.
Exits with status 1 if a check fails.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

import senet
from senet._lib import ROOT
from senet_ref.indexing import random_position
from senet_train.distill import featurize, load_snn1

FEATURE_TOL = 1e-6
# Float32 forward passes that add in different orders differ by up to about 2e-6 in the logit
# (as much as PyTorch's float32 and float64 passes do); tools/check_js_net.mjs has the same.
NET_TOL = 5e-6


def random_positions(n: int, seed: int) -> list[tuple[int, int]]:
    """`n` random positions as masks: a uniformly random layer, then a uniformly random
    position within it."""
    rng = random.Random(seed)
    return [(senet.mask(me), senet.mask(opp)) for me, opp in (random_position(rng) for _ in range(n))]


def within(label: str, diff: float, tol: float) -> bool:
    ok = diff < tol
    print(f"{label} = {diff:.2e}" + ("" if ok else f"  FAIL (tolerance {tol:g})"), flush=True)
    return ok


def logit(p: np.ndarray) -> np.ndarray:
    """The logits of probabilities, in float64."""
    p = np.asarray(p, dtype=np.float64)
    return np.log(p) - np.log1p(-p)


def sigmoid(x: np.ndarray) -> np.ndarray:
    """The probabilities of logits, in float64."""
    return 1 / (1 + np.exp(-np.asarray(x, dtype=np.float64)))


def check_js(net: Path, pos: list[tuple[int, int]], expected: np.ndarray, require_node: bool) -> bool:
    """Run the browser engine's forward pass on `pos` under node and compare its logits with
    those of the probabilities `expected`. Without node, the check is skipped, or fails if
    `require_node`."""
    if not shutil.which("node"):
        print("node not found: " + ("FAIL" if require_node else "skipping the browser engine check"))
        return not require_node
    with tempfile.TemporaryDirectory() as tmp:
        cases = Path(tmp) / "cases.json"
        cases.write_text(json.dumps([[a, b, float(p)] for (a, b), p in zip(pos, expected, strict=True)]))
        script = ROOT / "tools" / "check_js_net.mjs"
        return subprocess.run(["node", str(script), str(net), str(cases)], check=False).returncode == 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--net", type=Path, help="SNN1 network file, e.g. models/senet_net.bin")
    ap.add_argument("--n", type=int, default=20000, help="number of random positions")
    ap.add_argument("--require-node", action="store_true", help="fail, rather than skip, without node")
    args = ap.parse_args(argv)
    model = None
    if args.net:
        try:
            model = load_snn1(args.net)
        except (OSError, ValueError) as e:
            ap.error(str(e))

    pos = random_positions(args.n, seed=11)
    me = torch.tensor([p[0] for p in pos], dtype=torch.int64)
    opp = torch.tensor([p[1] for p in pos], dtype=torch.int64)
    rust_features = np.stack([senet.features(a, b) for a, b in pos])
    torch_features = featurize(me, opp).numpy()
    diff = float(np.abs(rust_features - torch_features).max())
    ok = within(f"features: {args.n} positions, max |rust - torch|", diff, FEATURE_TOL)

    if model is not None:
        model.eval()
        with torch.no_grad():
            l_torch = model(featurize(me, opp)).numpy().astype(np.float64)
        with senet.Engine(net=args.net) as eng:
            p_rust = np.array([eng.net_value(a, b) for a, b in pos])
        diff = float(np.abs(logit(p_rust) - l_torch).max())
        ok &= within(f"network: {args.n} positions, max |rust - torch| logit", diff, NET_TOL)
        ok &= check_js(args.net, pos, sigmoid(l_torch), args.require_node)

    print("OK" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
