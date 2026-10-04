"""Distil the perfect-play database into a small neural network.

Input: records written by `senet gen-data` (u32 mover mask, u32 opponent mask, f32 win
probability from the database), each file with its manifest, which is verified first.
Output: `<out>.bin`, the network in SNN1 format that both the Rust engine (`--net
models/senet_net.bin`) and the portable browser build (tools/build_portable.py) load, the
PyTorch state dict `<out>.pt`, and last their manifest `<out>.manifest.json`: the files'
SHA-256, the data, settings, architecture, input features, split, environment, final errors
and epoch times. Until the manifest is written the outputs are not complete; an older
manifest is removed first.

    python -m senet_train.distill --data runs/train.bin --out runs/senet_net

A share of the records is held out for validation: the first `--val-frac` of the records
in the order of `numpy.random.default_rng(seed).permutation`, so the split depends only on
the data, the seed and NumPy's generator. Positions recur across records (the opening's in
every game), so most validation records are at positions that training records share. The
errors are reported on every validation record, and on the validation records at "unseen
positions", which occur in no training record: with a small `--val-frac` nearly all of
those occur once in the data, so they stand for its rare positions, not for the positions
of play. `--evaluate NET` reports the errors of an existing network (`--record` also writes
them, with what is known of it, into its manifest). If the network has a manifest, its
split's seed and val_frac are the defaults, and other training data or settings are refused
unless --force:

    python -m senet_train.distill --data runs/train.bin --evaluate models/senet_net.bin

The records are memory-mapped and read in chunks: besides them (in the file cache), the
process holds the permutation (4 bytes per record), the training records' distinct
positions (8 bytes each), the validation records and one chunk, and the training data where
the network is trained (on the GPU if there is one). Training is seeded, but GPU arithmetic
is not bit-for-bit deterministic: a rerun gives a network with about the same errors, not
the same parameters.
"""

from __future__ import annotations

import argparse
import io
import math
import platform
import time
from collections.abc import Iterator
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from senet import manifest
from senet.provenance import created_by
from senet_ref.indexing import USABLE_SQUARES
from senet_ref.rules import NUM_PIECES, NUM_SQUARES, OFF

from . import snn1

REC = np.dtype([("me", "<u4"), ("opp", "<u4"), ("v", "<f4")])
USABLE = np.uint32(sum(1 << s for s in USABLE_SQUARES))
CHUNK = 1 << 24  # records read at once, which bounds the memory each pass takes
EVAL_CHUNK = 65536
KEEP = [s - 1 for s in USABLE_SQUARES]  # the columns of the usable squares among squares 1..30
SEED, VAL_FRAC = 0, 0.02
# The input features, as a network's manifest records them: senet_core::net::features, and
# docs/FORMATS.md ("Input features").
FEATURES = {
    "name": "senet-features/1",
    "layout": [
        "0-28: 1 if the mover has a piece on square 1-26, 28-30 (a column each, in order)",
        "29-57: the same for the opponent",
        "58-63: one-hot number of the mover's pieces borne off, 0-5",
        "64-69: the same for the opponent",
        "70: the mover's remaining distance, the sum of 31 - s over its squares s, divided by 100",
        "71: the same for the opponent",
    ],
}


def featurize(me: torch.Tensor, opp: torch.Tensor) -> torch.Tensor:
    """The input features (FEATURES) of positions given as masks."""
    sq = torch.arange(1, NUM_SQUARES + 1, device=me.device)
    mb = ((me[:, None] >> sq) & 1).float()
    ob = ((opp[:, None] >> sq) & 1).float()
    me_occ, opp_occ = mb[:, KEEP], ob[:, KEEP]
    me_off = (NUM_PIECES - me_occ.sum(1)).clamp(0, NUM_PIECES).long()
    opp_off = (NUM_PIECES - opp_occ.sum(1)).clamp(0, NUM_PIECES).long()
    dist = (OFF - sq).float()
    me_d = (mb * dist).sum(1, keepdim=True) / 100.0
    opp_d = (ob * dist).sum(1, keepdim=True) / 100.0
    return torch.cat(
        [
            me_occ,
            opp_occ,
            nn.functional.one_hot(me_off, NUM_PIECES + 1).float(),
            nn.functional.one_hot(opp_off, NUM_PIECES + 1).float(),
            me_d,
            opp_d,
        ],
        dim=1,
    )


class Net(nn.Module):
    """MLP with ReLU hidden layers; the output is the logit of P(mover wins). Its parameters
    are float32, like the features and the SNN1 format, whatever torch's default dtype."""

    def __init__(self, hidden: list[int]):
        super().__init__()
        dims = [snn1.N_INPUTS, *hidden, 1]
        self.linears = [nn.Linear(a, b, dtype=torch.float32) for a, b in pairwise(dims)]
        self.layers = nn.ModuleList(self.linears)  # registers them, as layers.0.weight, ...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.linears[:-1]:
            x = torch.relu(layer(x))
        return self.linears[-1](x).squeeze(1)


def snn1_bytes(model: Net) -> bytes:
    """The network, of any floating-point dtype, as an SNN1 file of float32 parameters;
    ValueError if a parameter is not finite as a float32."""

    def float32(t: torch.Tensor) -> np.ndarray:
        return t.detach().cpu().float().numpy()  # NumPy has no bfloat16

    return snn1.write([(float32(layer.weight), float32(layer.bias)) for layer in model.linears])


def export_snn1(model: Net, path: Path) -> None:
    """Writes the network as an SNN1 file (see `snn1_bytes`)."""
    Path(path).write_bytes(snn1_bytes(model))


def load_snn1(path: Path) -> Net:
    """Rebuild the network from an SNN1 file."""
    try:
        layers = snn1.read(Path(path).read_bytes())
    except ValueError as e:
        raise ValueError(f"{path}: {e}") from None
    model = Net([len(b) for _, b in layers[:-1]])
    with torch.no_grad():
        for layer, (w, b) in zip(model.linears, layers, strict=True):
            layer.weight.copy_(torch.from_numpy(w.astype(np.float32)))
            layer.bias.copy_(torch.from_numpy(b.astype(np.float32)))
    return model


Dataset = tuple[torch.Tensor, torch.Tensor, torch.Tensor]  # mover masks, opponent masks, database values


def load_records(path: str | Path) -> np.ndarray:
    """The records of a `senet gen-data` file, memory-mapped; ValueError unless it holds whole
    records, each of a game in progress (1..NUM_PIECES pieces per side on distinct usable
    squares) and a probability."""
    size = Path(path).stat().st_size
    if size % REC.itemsize:
        raise ValueError(f"{path}: {size:,} bytes is not a whole number of {REC.itemsize}-byte records")
    if size == 0:
        return np.zeros(0, dtype=REC)
    recs = np.memmap(path, dtype=REC, mode="r")
    for start in range(0, len(recs), CHUNK):
        chunk = np.asarray(recs[start : start + CHUNK])
        me, opp, v = chunk["me"], chunk["opp"], chunk["v"]
        ok = ((me | opp) & ~USABLE == 0) & (me & opp == 0) & (v >= 0) & (v <= 1)  # NaN fails v >= 0
        for side in (me, opp):
            pieces = np.bitwise_count(side)
            ok &= (pieces >= 1) & (pieces <= NUM_PIECES)
        if not ok.all():
            i = int(np.argmin(ok))
            raise ValueError(
                f"{path}: record {start + i:,} is not a game in progress with a probability: "
                f"me={me[i]:#010x} opp={opp[i]:#010x} v={v[i]}"
            )
    return recs


class Records:
    """The records of one or more `senet gen-data` files, in order, as one sequence; each file's
    manifest is verified (every byte hashed) before its records are checked and used."""

    def __init__(self, paths: list[str] | list[Path]):
        self.paths = [Path(p) for p in paths]
        self.sha256: list[str] = []  # of the files, from their verified manifests
        for p in self.paths:
            m = manifest.verify(p)
            if m.get("kind") != manifest.TRAINING_DATA[0]:
                raise manifest.ManifestError(
                    f"{manifest.manifest_path(p)}: a {m.get('kind')} manifest, not training data"
                )
            self.sha256 += [f["sha256"] for f in m["files"]]
        self.files = [load_records(p) for p in self.paths]
        self.starts = np.cumsum([0] + [len(f) for f in self.files])

    def __len__(self) -> int:
        return int(self.starts[-1])

    def take(self, idx: np.ndarray) -> np.ndarray:
        """The records at `idx` (indices into the sequence), in that order."""
        out = np.empty(len(idx), dtype=REC)
        which = np.searchsorted(self.starts, idx, side="right") - 1
        for k, f in enumerate(self.files):
            sel = which == k
            if sel.any():
                out[sel] = f[idx[sel] - self.starts[k]]
        return out

    def chunks(self) -> Iterator[tuple[int, np.ndarray]]:
        """The records in order, a chunk at a time, with the index of each chunk's first record."""
        for k, f in enumerate(self.files):
            for start in range(0, len(f), CHUNK):
                yield int(self.starts[k]) + start, np.asarray(f[start : start + CHUNK])


class Split:
    """The records split into training and validation records (see the module docstring)."""

    def __init__(self, records: Records, val_frac: float, seed: int):
        n = len(records)
        n_val = int(n * val_frac)
        if not 0 < n_val < n:
            raise ValueError(f"--val-frac {val_frac} leaves an empty training or validation split of {n:,} records")
        # Shuffling an arange is what permutation(n) does, with the same draws for any dtype:
        # this is its order, in half the memory while the indices fit 32 bits.
        perm = np.arange(n, dtype=np.uint32 if n <= 1 << 32 else np.int64)
        np.random.default_rng(seed).shuffle(perm)
        self.records = records
        self.val_idx = perm[:n_val]
        self.train_idx = perm[n_val:]
        self.val = records.take(self.val_idx)

    def positions(self) -> tuple[np.ndarray, dict[str, int]]:
        """Mask of the validation records whose position occurs in no training record, and the
        number of distinct positions among the training records, the validation records, those
        unseen ones and all the records."""
        is_val = np.zeros(len(self.records), dtype=bool)
        is_val[self.val_idx] = True
        trained = np.zeros(0, dtype=np.uint64)
        for start, chunk in self.records.chunks():
            trained = union(trained, np.unique(keys(chunk[~is_val[start : start + len(chunk)]])))
        val_keys = keys(self.val)
        unseen = trained[np.minimum(np.searchsorted(trained, val_keys), len(trained) - 1)] != val_keys
        n_unseen = len(np.unique(val_keys[unseen]))
        distinct = {
            "training": len(trained),
            "validation": len(np.unique(val_keys)),
            "unseen": n_unseen,
            "all": len(trained) + n_unseen,
        }
        return unseen, distinct

    def training_data(self, dev: torch.device) -> Dataset:
        """The training records on `dev`, in split order, copied a chunk at a time."""
        n = len(self.train_idx)
        me = torch.empty(n, dtype=torch.int32, device=dev)
        opp = torch.empty(n, dtype=torch.int32, device=dev)
        v = torch.empty(n, dtype=torch.float32, device=dev)
        for start in range(0, n, CHUNK):
            part = slice(start, start + CHUNK)
            r = self.records.take(self.train_idx[part])
            for dst, src in ((me, r["me"].astype(np.int32)), (opp, r["opp"].astype(np.int32)), (v, r["v"])):
                dst[part] = torch.from_numpy(np.ascontiguousarray(src)).to(dev)
        return me, opp, v


def keys(r: np.ndarray) -> np.ndarray:
    """One 64-bit key per record's position."""
    return (r["me"].astype(np.uint64) << np.uint64(32)) | r["opp"].astype(np.uint64)


def union(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """The distinct values of two sorted arrays of distinct values, sorted."""
    c = np.concatenate([a, b])
    c.sort(kind="stable")  # a merge of the two sorted runs
    return c[np.concatenate([[True], c[1:] != c[:-1]])]


def to_device(r: np.ndarray, dev: torch.device) -> Dataset:
    # Masks use bits 1..30, so int32 holds them; this halves GPU memory for the dataset.
    return (
        torch.from_numpy(r["me"].astype(np.int32)).to(dev),
        torch.from_numpy(r["opp"].astype(np.int32)).to(dev),
        torch.from_numpy(r["v"].astype(np.float32)).to(dev),
    )


def evaluate(model: Net, data: Dataset) -> tuple[float, float, float]:
    """Mean, RMS and max absolute error of the predicted win probability on `data`;
    ArithmeticError if the network predicts NaN."""
    me, opp, v = data
    model.eval()
    errs = []
    with torch.no_grad():
        for i in range(0, len(v), EVAL_CHUNK):
            chunk = slice(i, i + EVAL_CHUNK)
            p = torch.sigmoid(model(featurize(me[chunk], opp[chunk])))
            errs.append((p - v[chunk]).abs())
    model.train()
    e = torch.cat(errs)
    if e.isnan().any():
        raise ArithmeticError("the network predicts NaN")
    return e.mean().item(), e.pow(2).mean().sqrt().item(), e.max().item()


def errors(model: Net, vals: dict[str, Dataset]) -> dict[str, dict[str, float]]:
    """The errors (see `evaluate`) on each named validation set."""
    out = {}
    for name, data in vals.items():
        mae, rmse, mx = evaluate(model, data)
        out[name] = {"records": len(data[2]), "mae": mae, "rmse": rmse, "max": mx}
    return out


def report(errs: dict[str, dict[str, float]]) -> str:
    """`errors` on one line."""
    return " | ".join(
        f"{name}: MAE {e['mae']:.5f}  RMSE {e['rmse']:.5f}  max {e['max']:.4f}" for name, e in errs.items()
    )


def training_steps(n: int, batch: int, epochs: int) -> int:
    """The optimizer steps of `epochs` passes over `n` records in batches of `batch`; ValueError
    if they are fewer than the 3 that the learning-rate schedule needs."""
    if batch < 1 or epochs < 1:
        raise ValueError("the batch size and the number of epochs must be at least 1")
    steps = math.ceil(n / batch) * epochs
    if steps < 3:
        raise ValueError(
            f"{epochs} epoch(s) of {n:,} records in batches of {batch:,} are {steps} steps; at least 3 are needed"
        )
    return steps


def train(
    model: Net, data: Dataset, vals: dict[str, Dataset], epochs: int, batch: int, lr: float, wd: float
) -> list[float]:
    """Minimise the cross-entropy against the database's values (AdamW, one-cycle learning rate),
    printing the errors on the validation sets after every epoch. Returns each epoch's seconds,
    its validation included."""
    me, opp, v = data
    steps = training_steps(len(v), batch, epochs)
    steps_per_epoch = steps // epochs
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    # The learning rate warms up over 5% of the steps, but at least 2: OneCycleLR divides by
    # the number of warm-up steps minus 1, and by the number of steps after them.
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=max(0.05, 2 / steps))
    bce = nn.BCEWithLogitsLoss()
    seconds = []
    for ep in range(epochs):
        t0 = time.perf_counter()
        order = torch.randperm(len(v), device=v.device)
        for k in range(steps_per_epoch):
            idx = order[k * batch : (k + 1) * batch]
            loss = bce(model(featurize(me[idx], opp[idx])), v[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
        errs = report(errors(model, vals))  # waits for the GPU
        seconds.append(round(time.perf_counter() - t0, 1))
        print(f"epoch {ep + 1:2d}: {errs}  [{seconds[-1]:.0f}s, {sum(seconds):.0f}s in all]", flush=True)
    return seconds


def widths(text: str) -> list[int]:
    """Comma-separated hidden layer widths, which must form a network that SNN1 allows."""
    try:
        hidden = [int(h) for h in text.split(",")]
        snn1.check_shapes(list(pairwise([snn1.N_INPUTS, *hidden, 1])))
    except ValueError as e:
        raise argparse.ArgumentTypeError(str(e)) from None
    return hidden


def environment(dev: torch.device) -> dict[str, Any]:
    """The software and device that trained or evaluated a network."""
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "device": torch.cuda.get_device_name(dev) if dev.type == "cuda" else platform.processor() or "cpu",
    }


SPLIT_DEFINITION = (
    "validation records: the first val_frac of the records in the order of "
    "numpy.random.default_rng(seed).permutation (of the NumPy version given); training records: the others; "
    "unseen positions: the validation records whose (me, opp) occurs in no training record, which are positions "
    "rare in the data; "
    "distinct_positions: the number of different (me, opp) among the training records, the validation records, "
    "the unseen ones and all the records"
)


def outputs(out: Path) -> tuple[Path, Path]:
    """The network and state dict files of `--out`, a path without extension (or ending in .bin)."""
    base = out.name.removesuffix(".bin")
    return out.with_name(base + ".bin"), out.with_name(base + ".pt")


def network_manifest(files: list[Path], record: dict[str, Any]) -> dict[str, Any]:
    """The manifest of a network's `files`, with `record` (by, settings, inputs, contents, notes)."""
    m = manifest.new(manifest.NETWORK, [manifest.file_entry(f) for f in files], record["by"])
    m.update({k: record[k] for k in ("settings", "inputs", "contents", "notes")})
    return m


def publish(out: Path, model: Net, record: dict[str, Any]) -> Path:
    """Writes the network and its state dict (see `outputs`), each whole, then their manifest
    with `record`. An older manifest is removed first, so none vouches for outputs that are
    being replaced. Returns the manifest's path."""
    bin_path, pt_path = outputs(out)
    manifest.manifest_path(bin_path).unlink(missing_ok=True)
    manifest.write_atomically(bin_path, snn1_bytes(model))
    state = io.BytesIO()
    torch.save(model.state_dict(), state)
    manifest.write_atomically(pt_path, state.getvalue())
    return manifest.write(bin_path, network_manifest([bin_path, pt_path], record))


def recorded(net: Path, data: Records, args: argparse.Namespace) -> tuple[dict[str, Any] | None, list[str]]:
    """The manifest of the network to evaluate (None if it has none) and how the evaluation asked
    for differs from what it records: other training data, split or features. Fills in --seed
    and --val-frac from its split where they are not given."""
    if not manifest.manifest_path(net).exists():
        return None, []
    m = manifest.read(net)
    # Only the network itself: a state dict that its manifest also lists may be missing.
    if [f.get("sha256") for f in m["files"] if f["name"] == net.name] != [manifest.file_sha256(net)]:
        raise manifest.ManifestError(f"{manifest.manifest_path(net)}: not the manifest of {net}")
    contents = m.get("contents") or {}
    split = contents.get("split") or {}
    problems = []
    for key, option in (("seed", "--seed"), ("val_frac", "--val-frac")):
        given = getattr(args, key)
        if given is None:
            setattr(args, key, split.get(key))
        elif key in split and given != split[key]:
            problems.append(f"its split has {key} {split[key]}, not {option} {given}")
    if "data_sha256" in split:
        same_data = split["data_sha256"] == data.sha256
    else:
        trained_on = (m.get("inputs") or {}).get("training_data") or []
        same_data = [i.get("manifest_sha256") for i in trained_on] == [
            manifest.identity(p)["manifest_sha256"] for p in data.paths
        ]
    if not same_data:
        problems.append("it was trained on other data")
    features = contents.get("features")
    if features is not None and features.get("name") != FEATURES["name"]:
        problems.append(f"its input features are {features.get('name')}, not {FEATURES['name']}")
    return m, problems


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, nargs="+", help="training record files from `senet gen-data`")
    ap.add_argument("--out", type=Path, help="train, and write the network here (path without extension: .bin, .pt)")
    ap.add_argument(
        "--evaluate", type=Path, metavar="NET", help="instead of training, report the errors of this SNN1 network"
    )
    ap.add_argument(
        "--record",
        action="store_true",
        help="with --evaluate: write the network's manifest, with the errors, as a record made after the fact",
    )
    ap.add_argument("--note", action="append", default=[], help="with --record: a remark for the manifest")
    ap.add_argument(
        "--force",
        action="store_true",
        help="with --evaluate: evaluate on other data or another split than the network's manifest records",
    )
    ap.add_argument("--hidden", type=widths, default="512,256,128", help="hidden layer widths, comma-separated")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=16384)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--wd", type=float, default=1e-5)
    ap.add_argument("--val-frac", type=float, help=f"default {VAL_FRAC}, or with --evaluate the network's")
    ap.add_argument("--seed", type=int, help=f"default {SEED}, or with --evaluate the network's")
    return ap


def main(argv: list[str] | None = None) -> None:
    ap = parser()
    args = ap.parse_args(argv)
    if (args.out is None) == (args.evaluate is None):
        ap.error("give either --out (to train) or --evaluate")
    if (args.record or args.note or args.force) and not args.evaluate:
        ap.error("--record, --note and --force go with --evaluate")
    if args.note and not args.record:
        ap.error("--note goes with --record")

    try:
        records = Records(args.data)
        old, problems = recorded(args.evaluate, records, args) if args.evaluate else (None, [])
        if problems and not args.force:
            raise ValueError(f"{args.evaluate}: {'; '.join(problems)} (--force to evaluate it anyway)")
        args.seed = SEED if args.seed is None else args.seed
        args.val_frac = VAL_FRAC if args.val_frac is None else args.val_frac
        split = Split(records, args.val_frac, args.seed)
        if args.out:
            training_steps(len(split.train_idx), args.batch, args.epochs)
    except ValueError as e:  # including manifest.ManifestError
        ap.error(str(e))
    for problem in problems:
        print(f"{args.evaluate}: {problem}; evaluating it anyway (--force)")
    unseen, distinct = split.positions()
    torch.manual_seed(args.seed)
    # With CUDA_VISIBLE_DEVICES empty, is_available() can be true with no device to use.
    dev = torch.device("cuda" if torch.cuda.is_available() and torch.cuda.device_count() else "cpu")
    n_val = len(split.val)
    print(
        f"{len(records):,} records on {dev}: {n_val:,} validation records, {unseen.sum():,} ({unseen.mean():.1%}) of "
        f"them at unseen positions; distinct positions: {distinct['training']:,} in training, "
        f"{distinct['validation']:,} in validation, {distinct['unseen']:,} unseen, {distinct['all']:,} in all"
    )
    vals = {"validation records": to_device(split.val, dev)}
    if unseen.any():
        vals["unseen positions"] = to_device(split.val[unseen], dev)
    record: dict[str, Any] = {
        "by": created_by(environment=environment(dev)),
        "inputs": {"training_data": [manifest.identity(p) for p in records.paths]},
    }
    split_facts = {
        "records": len(records),
        "data_sha256": records.sha256,
        "validation_records": n_val,
        "unseen_records": int(unseen.sum()),
        "distinct_positions": distinct,
        "val_frac": args.val_frac,
        "seed": args.seed,
        "numpy": np.__version__,
        "definition": SPLIT_DEFINITION,
    }
    if args.evaluate:
        evaluate_network(args, vals, record, split_facts, old, dev)
    else:
        train_network(args, split, vals, record, split_facts, dev)


def evaluate_network(
    args: argparse.Namespace,
    vals: dict[str, Dataset],
    record: dict[str, Any],
    split_facts: dict[str, Any],
    old: dict[str, Any] | None,
    dev: torch.device,
) -> None:
    """Reports the errors of the network `--evaluate`, and with --record writes its manifest. The
    training settings are kept from its manifest, if it has one."""
    model = load_snn1(args.evaluate).to(dev)
    errs = errors(model, vals)
    print(report(errs))
    if not args.record:
        return
    record["settings"] = old.get("settings") if old else None
    record["contents"] = {
        "architecture": architecture(model),
        "parameters": parameters(model),
        "features": FEATURES,
        "split": split_facts,
        "errors": errs,
    }
    first = (
        "Recorded after the fact by `distill --evaluate --record`: the errors are measured now, on the split "
        "of the seed and val_frac given here"
    )
    first += "; the training settings were not recorded." if record["settings"] is None else "."
    record["notes"] = [first, *args.note]
    # The network alone: a state dict beside it is not published with it (.gitignore).
    print(f"wrote {manifest.write(args.evaluate, network_manifest([args.evaluate], record))}")


def train_network(
    args: argparse.Namespace,
    split: Split,
    vals: dict[str, Dataset],
    record: dict[str, Any],
    split_facts: dict[str, Any],
    dev: torch.device,
) -> None:
    """Trains a network on the training records and publishes it to `--out`."""
    data = split.training_data(dev)
    model = Net(args.hidden).to(dev)
    print(f"model {'-'.join(map(str, architecture(model)))}, {parameters(model):,} parameters")
    seconds = train(model, data, vals, args.epochs, args.batch, args.lr, args.wd)
    record["settings"] = {k: getattr(args, k) for k in ("hidden", "epochs", "batch", "lr", "wd", "val_frac", "seed")}
    record["contents"] = {
        "architecture": architecture(model),
        "parameters": parameters(model),
        "features": FEATURES,
        "split": split_facts,
        "errors": errors(model, vals),
        "training_seconds": round(sum(seconds), 1),
        "epoch_seconds": seconds,
    }
    record["notes"] = []
    args.out.parent.mkdir(parents=True, exist_ok=True)
    bin_path, pt_path = outputs(args.out)
    print(f"wrote {bin_path}, {pt_path} and {publish(args.out, model, record)}")


def architecture(model: Net) -> list[int]:
    """The layer widths, inputs first."""
    return [snn1.N_INPUTS, *(layer.out_features for layer in model.linears)]


def parameters(model: Net) -> int:
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    main()
