"""Integrity manifests (senet.manifest): reading, verifying, writing, and agreement with the
manifests of the `senet` program, each side checking what the other writes; and the
provenance that manifests and reports record (senet.provenance)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from conftest import DB, needs_db

import senet
from senet import _lib, manifest, provenance
from senet._lib import ROOT

SENET = ROOT / "target" / "release" / ("senet.exe" if sys.platform == "win32" else "senet")
NO_SENET = f"needs the senet program ({SENET.relative_to(ROOT)}: cargo build --release)"
# Layers of their true sizes (4 bytes per position), with made-up values.
LAYERS = {
    f"L{w}{b}.f32": bytes(i * k % 251 for i in range(n))
    for k, (w, b, n) in enumerate([(1, 1, 3248), (1, 2, 43848), (2, 1, 43848)], start=1)
}


def database(tmp_path: Path) -> Path:
    """A directory laid out as a database, with its manifest written from Python."""
    db = tmp_path / "db"
    db.mkdir()
    for name, data in LAYERS.items():
        (db / name).write_bytes(data)
    (db / "meta.json").write_text("{}\n", encoding="utf-8", newline="\n")
    files = [manifest.file_entry(p) for p in sorted(db.iterdir())]
    manifest.write(db, manifest.new(manifest.DATABASE, files, provenance.created_by()))
    return db


def senet_cli(*args: str | Path) -> subprocess.CompletedProcess[str]:
    if not SENET.exists():
        pytest.skip(NO_SENET)
    return subprocess.run([str(SENET), *map(str, args)], capture_output=True, text=True, check=False)


def test_manifest_paths(tmp_path: Path) -> None:
    assert manifest.manifest_path(tmp_path) == tmp_path / "MANIFEST.json"
    assert manifest.manifest_path("runs/train.bin") == Path("runs/train.manifest.json")
    assert manifest.manifest_path("models/net.manifest.json") == Path("models/net.manifest.json")


def test_an_intact_database_verifies(tmp_path: Path) -> None:
    db = database(tmp_path)
    m = manifest.verify(db)
    assert (m["kind"], m["format"], m["rules"]) == (*manifest.DATABASE, "kendall5")
    assert [f["name"] for f in m["files"]] == ["L11.f32", "L12.f32", "L21.f32", "meta.json"]
    assert manifest.check_sizes(db) == m


def change_a_byte(path: Path) -> None:
    data = bytearray(path.read_bytes())
    data[-1] ^= 1
    path.write_bytes(bytes(data))


SPOILS: list[tuple[str, Callable[[Path], object], str, bool]] = [
    # (what, how, the problem reported, whether comparing sizes finds it)
    ("missing", lambda db: (db / "L21.f32").unlink(), "L21.f32: ", True),
    ("truncated", lambda db: (db / "L21.f32").write_bytes(b"\x01" * 99), "L21.f32: 99 bytes, expected 43848", True),
    ("changed", lambda db: change_a_byte(db / "L11.f32"), "L11.f32: SHA-256 ", False),
    ("unlisted", lambda db: (db / "L31.f32").write_bytes(b"x"), "L31.f32: not in the manifest", True),
]


@pytest.mark.parametrize(("spoil", "problem", "by_size"), [s[1:] for s in SPOILS], ids=[s[0] for s in SPOILS])
def test_a_spoiled_database_does_not_verify(
    tmp_path: Path, spoil: Callable[[Path], object], problem: str, by_size: bool
) -> None:
    db = database(tmp_path)
    spoil(db)
    with pytest.raises(manifest.ManifestError, match="does not match") as e:
        manifest.verify(db)
    assert problem in str(e.value)
    if by_size:
        with pytest.raises(manifest.ManifestError):
            manifest.check_sizes(db)
    else:
        manifest.check_sizes(db)


def test_every_problem_is_reported(tmp_path: Path) -> None:
    db = database(tmp_path)
    for _, spoil, _, _ in SPOILS:
        spoil(db)
    with pytest.raises(manifest.ManifestError) as e:
        manifest.verify(db)
    for _, _, problem, _ in SPOILS:
        assert problem in str(e.value)


@pytest.mark.parametrize(
    ("text", "error"),
    [
        ('{"manifest": 2, "files": []}', "not a manifest of format version 1"),
        ('{"manifest": 1}', "not a manifest of format version 1"),
        ("[1, 2]", "not a manifest of format version 1"),
        ("{", "Expecting property name"),
        ('{"manifest": 1, "files": [{"name": "a"}]}', "malformed file entry"),
        ('{"manifest": 1, "files": [{"name": "../L11.f32", "bytes": 0}]}', "not a plain file name"),
        ('{"manifest": 1, "files": [{"name": "sub/L11.f32", "bytes": 0}]}', "not a plain file name"),
        ('{"manifest": 1, "files": [{"name": ".hidden", "bytes": 0}]}', "not a plain file name"),
    ],
)
def test_malformed_manifests_are_refused(tmp_path: Path, text: str, error: str) -> None:
    (tmp_path / "MANIFEST.json").write_text(text, encoding="utf-8")
    with pytest.raises(manifest.ManifestError, match=error):
        manifest.verify(tmp_path)


def test_a_missing_manifest_is_reported(tmp_path: Path) -> None:
    with pytest.raises(manifest.ManifestError, match="no manifest"):
        manifest.verify(tmp_path / "train.bin")


def test_a_failed_write_keeps_the_previous_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "net.bin"
    manifest.write_atomically(path, b"first")
    manifest.write_atomically(path, b"second")
    assert path.read_bytes() == b"second"

    def fail(src: Any, dst: Any) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(manifest.os, "replace", fail)
    with pytest.raises(OSError, match="disk full"):
        manifest.write_atomically(path, b"third")
    assert path.read_bytes() == b"second"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["net.bin"]


def test_identity(tmp_path: Path) -> None:
    db = database(tmp_path)
    sha = manifest.file_sha256(db / "MANIFEST.json")
    assert manifest.identity(db) == {"path": db.as_posix(), "manifest_sha256": sha}
    assert manifest.identity(tmp_path / "none.bin")["manifest_sha256"] is None


def test_the_command_line(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    db = database(tmp_path)
    assert manifest.main(["verify", str(db), str(db / "MANIFEST.json")]) == 0
    assert capsys.readouterr().out.count("ok    ") == 2
    change_a_byte(db / "L11.f32")
    assert manifest.main(["verify", "--sizes-only", str(db)]) == 0
    assert "(sizes," in capsys.readouterr().out
    assert manifest.main(["verify", str(db)]) == 1
    assert "FAIL" in capsys.readouterr().out


def test_senet_verifies_what_python_writes(tmp_path: Path) -> None:
    db = database(tmp_path)
    ok = senet_cli("verify", db)
    assert ok.returncode == 0, ok.stderr
    change_a_byte(db / "L11.f32")
    (db / "L31.f32").write_bytes(b"x")
    failed = senet_cli("verify", db)
    assert failed.returncode != 0
    assert "L11.f32: SHA-256" in failed.stderr and "L31.f32: not in the manifest" in failed.stderr


def test_python_verifies_what_senet_writes(tmp_path: Path) -> None:
    db = database(tmp_path)
    (db / "MANIFEST.json").unlink()
    written = senet_cli("manifest", "--db", db)
    assert written.returncode == 0, written.stderr
    rust = manifest.verify(db)
    assert rust["files"] == [manifest.file_entry(p) for p in sorted(db.iterdir()) if p.name != "MANIFEST.json"]
    assert rust["kind"] == manifest.DATABASE[0]
    assert rust["contents"]["complete"] is False  # a few layers, not the database

    data = tmp_path / "train.bin"
    data.write_bytes(bytes(12 * 5))
    written = senet_cli("manifest", "--data", data, "--note", "made by hand, it's a test")
    assert written.returncode == 0, written.stderr
    m = manifest.verify(data)
    assert (m["kind"], m["contents"]) == (manifest.TRAINING_DATA[0], {"records": 5})
    assert m["notes"][-1] == "made by hand, it's a test"
    # Both programs record their maker alike: the command (quoted as a shell reads it back),
    # the commit and the engine's build.
    by = m["created_by"]
    assert by.keys() == provenance.created_by().keys() == {"command", "commit", "engine"}
    assert by["command"].startswith("senet manifest --data ")
    assert by["command"].endswith(""" --note 'made by hand, it'"'"'s a test'""")
    assert by["engine"] == senet.build_info()


@needs_db
def test_generated_training_data_has_a_manifest(tmp_path: Path) -> None:
    data = tmp_path / "train.bin"
    made = senet_cli("gen-data", "--db", DB, "--out", data, "--games", "3", "--uniform", "50", "--seed", "7")
    assert made.returncode == 0, made.stderr
    m = manifest.verify(data)
    assert m["kind"] == manifest.TRAINING_DATA[0]
    assert m["contents"]["records"] == data.stat().st_size // 12 > 50
    assert m["settings"] == {"games": 3, "eps": 0.2, "uniform": 50, "seed": 7}
    assert m["contents"]["uniform"] == 50
    assert m["inputs"]["database"] == manifest.identity(DB)
    assert not data.with_name("train.bin.tmp").exists()


def test_created_by_and_the_command(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "__main__", SimpleNamespace(__spec__=SimpleNamespace(name="senet.bench")))
    monkeypatch.setattr(sys, "argv", ["bench.py", "--games", "10", "--note", "a b"])
    assert provenance.command() == "python -m senet.bench --games 10 --note 'a b'"
    by = provenance.created_by(environment={"device": "cpu"})
    assert by["command"] == provenance.command()
    assert by["engine"] == senet.build_info()
    assert by["environment"] == {"device": "cpu"}
    commit = by["commit"]
    assert commit is None or len(commit.removesuffix("-dirty")) == 40


def test_provenance_render(tmp_path: Path) -> None:
    net = tmp_path / "net.bin"
    net.write_bytes(b"network")
    db = database(tmp_path)
    p = provenance.provenance(db=db, net=net, games=100, mode="quick", seeds={"a": 1}, slow=True)
    json.dumps(p)  # reports record it as JSON
    # Outside the repository, paths are recorded whole; inside it, relative to it.
    assert p["network"] == {"path": net.resolve().as_posix(), "sha256": manifest.file_sha256(net)}
    assert p["database"] == manifest.identity(db.resolve())
    assert provenance.shown(ROOT / "db" / "kendall5") == "db/kendall5"
    text = provenance.render(p)
    assert text.startswith("## Provenance\n\n* Generated ")
    sha = p["database"]["manifest_sha256"]
    assert f"* Database: `{db.resolve().as_posix()}`, manifest SHA-256 `{sha}`." in text
    assert f"* Network: `{net.resolve().as_posix()}`, SHA-256 `{p['network']['sha256']}`." in text
    assert '* Settings: games = 100, mode = quick, seeds = {"a": 1}, slow = true.' in text
    assert text.endswith(".\n")

    bare = provenance.provenance()
    bare["machine"]["memory_bytes"] = None
    bare["engine"] = {**bare["engine"], "cpu_features": [], "optimized": False}
    bare["commit"] = None
    text = provenance.render(bare)
    assert "at commit `unknown`" in text and "debug build using no CPU features beyond the baseline" in text
    assert "Database" not in text and "Network" not in text and "Settings" not in text
    assert "GiB" not in text


def test_paths_through_a_link_in_the_checkout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # A database linked into the checkout from another disk is recorded where the checkout has it.
    checkout, elsewhere = tmp_path / "checkout", tmp_path / "elsewhere"
    (elsewhere / "kendall5").mkdir(parents=True)
    checkout.mkdir()
    (checkout / "Cargo.toml").touch()
    if sys.platform == "win32":
        import _winapi

        _winapi.CreateJunction(str(elsewhere), str(checkout / "db"))  # unlike a symbolic link, needs no privilege
    else:
        (checkout / "db").symlink_to(elsewhere, target_is_directory=True)
    monkeypatch.setattr(_lib, "ROOT", checkout)
    assert provenance.shown(checkout / "db" / "kendall5") == "db/kendall5"
    monkeypatch.chdir(checkout)
    assert provenance.shown("db/kendall5") == "db/kendall5"
    assert provenance.shown(elsewhere / "kendall5") == (elsewhere / "kendall5").resolve().as_posix()


def test_machine_facts() -> None:
    m = provenance.machine()
    assert m["cpu"] and m["logical_cpus"] == os.cpu_count()
    assert m["memory_bytes"] is None or m["memory_bytes"] > 2**28
