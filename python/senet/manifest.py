"""Integrity manifests (docs/FORMATS.md, the same format `senet manifest` writes): what a
database, a training-data file or a network consists of, how it was made, and the size and
SHA-256 of each of its files.

    python -m senet.manifest verify db/kendall5 runs/train.bin models/senet_net.bin

The manifest of a directory is MANIFEST.json in it; that of a file `name.ext` is
`name.manifest.json` beside it, which may also cover other files of that name (a network's
.bin and .pt). `verify` hashes every file; `check_sizes` only compares sizes, which is
quick and catches truncated or missing files.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

VERSION = 1
DIR_MANIFEST = "MANIFEST.json"
DATABASE = ("database", "senet-db/1")
TRAINING_DATA = ("training-data", "senet-records/1")
NETWORK = ("network", "snn1/1")


class ManifestError(ValueError):
    """A manifest that is missing or malformed, or files that do not match theirs."""


def manifest_path(path: str | os.PathLike[str]) -> Path:
    """The manifest covering `path`: `path` itself if it is a .json file, MANIFEST.json in it if
    it is a directory, else `name.manifest.json` beside it."""
    p = Path(path)
    if p.is_dir():
        return p / DIR_MANIFEST
    if p.suffix == ".json":
        return p
    return p.with_name(f"{p.stem}.manifest.json")


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(1 << 22):
            h.update(block)
    return h.hexdigest()


def file_entry(path: Path) -> dict[str, Any]:
    """A file as a manifest beside it lists it: its name, size and SHA-256."""
    return {"name": path.name, "bytes": path.stat().st_size, "sha256": file_sha256(path)}


def read(path: str | os.PathLike[str]) -> dict[str, Any]:
    """The manifest covering `path` (see `manifest_path`); ManifestError unless it is one of
    format version VERSION."""
    p = manifest_path(path)
    try:
        m = json.loads(p.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ManifestError(f"{p}: no manifest") from None
    except (OSError, ValueError) as e:
        raise ManifestError(f"{p}: {e}") from None
    if not isinstance(m, dict) or m.get("manifest") != VERSION or not isinstance(m.get("files"), list):
        raise ManifestError(f"{p}: not a manifest of format version {VERSION}")
    for f in m["files"]:
        if not (isinstance(f, dict) and isinstance(f.get("name"), str) and isinstance(f.get("bytes"), int)):
            raise ManifestError(f"{p}: malformed file entry {f!r}")
        if Path(f["name"]).name != f["name"] or f["name"].startswith("."):
            raise ManifestError(f"{p}: {f['name']}: not a plain file name")
    return m


def _problems(directory: Path, want: dict[str, Any], hashes: bool) -> str | None:
    path = directory / want["name"]
    try:
        size = path.stat().st_size
    except OSError as e:
        return f"{want['name']}: {e.strerror or e}"
    if size != want["bytes"]:
        return f"{want['name']}: {size} bytes, expected {want['bytes']}"
    if hashes and (sha := file_sha256(path)) != want.get("sha256"):
        return f"{want['name']}: SHA-256 {sha}, expected {want.get('sha256')}"
    return None


def verify(path: str | os.PathLike[str], hashes: bool = True) -> dict[str, Any]:
    """Checks every file the manifest covering `path` lists: present, of the recorded size and
    (unless `hashes` is False) SHA-256. A database's directory must also hold no layer file the
    manifest leaves out. Returns the manifest; ManifestError lists every problem."""
    p = manifest_path(path)
    m = read(p)
    directory = p.parent
    with ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 1)) as pool:
        problems = [x for x in pool.map(lambda f: _problems(directory, f, hashes), m["files"]) if x]
    if m.get("kind") == DATABASE[0]:
        listed = {f["name"] for f in m["files"]}
        problems += [f"{q.name}: not in the manifest" for q in directory.glob("L*.f32") if q.name not in listed]
    if problems:
        raise ManifestError(f"{p} does not match:\n  " + "\n  ".join(sorted(problems)))
    return m


def check_sizes(path: str | os.PathLike[str]) -> dict[str, Any]:
    """`verify` without hashing: quick, and enough to catch missing or truncated files."""
    return verify(path, hashes=False)


def identity(path: str | os.PathLike[str]) -> dict[str, Any]:
    """The identity of the artifact at `path` for another's manifest or report to record: its
    path, with `/` between its components, and the SHA-256 of its manifest (None if it has none)."""
    p = manifest_path(path)
    return {"path": Path(path).as_posix(), "manifest_sha256": file_sha256(p) if p.exists() else None}


def utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def new(kind: tuple[str, str], files: list[dict[str, Any]], created_by: dict[str, Any]) -> dict[str, Any]:
    """A new manifest of `kind` (DATABASE, TRAINING_DATA or NETWORK) listing `files`, with
    nothing else recorded yet."""
    return {
        "manifest": VERSION,
        "kind": kind[0],
        "format": kind[1],
        "rules": "kendall5",
        "created_utc": utc_now(),
        "created_by": created_by,
        "settings": None,
        "inputs": None,
        "contents": None,
        "notes": [],
        "files": files,
    }


def write_atomically(path: Path, data: bytes) -> None:
    """Writes `path` through `path.tmp`, flushed to the disk, then renamed into place: a reader
    finds the previous complete file or the new one, never a part."""
    tmp = path.with_name(path.name + ".tmp")
    try:
        with tmp.open("wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def write(path: str | os.PathLike[str], m: dict[str, Any]) -> Path:
    """Writes manifest `m` as the manifest covering `path`; returns where."""
    p = manifest_path(path)
    write_atomically(p, (json.dumps(m, indent=2) + "\n").encode("utf-8"))
    return p


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    v = sub.add_parser("verify", help="check files against their manifests (sizes and SHA-256)")
    v.add_argument("paths", nargs="+", type=Path, help="manifests, or the directories or files they cover")
    v.add_argument("--sizes-only", action="store_true", help="compare sizes only, without hashing")
    args = ap.parse_args(argv)
    failed = False
    for path in args.paths:
        t0 = time.perf_counter()
        try:
            m = verify(path, hashes=not args.sizes_only)
        except ManifestError as e:
            print(f"FAIL  {e}")
            failed = True
            continue
        total = sum(f["bytes"] for f in m["files"])
        what = "sizes" if args.sizes_only else "sizes and SHA-256"
        seconds = time.perf_counter() - t0
        print(f"ok    {manifest_path(path)}: {len(m['files'])} files, {total:,} bytes ({what}, {seconds:.1f}s)")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
