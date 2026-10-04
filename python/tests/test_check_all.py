"""tools/check_all.py's accounting of what it did not check: the skips it finds in pytest's
reports, and how the developer, --require and --strict modes treat them."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from senet._lib import ROOT


def load_check_all() -> ModuleType:
    spec = importlib.util.spec_from_file_location("check_all", ROOT / "tools" / "check_all.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["check_all"] = module  # for its dataclasses
    spec.loader.exec_module(module)
    return module


check_all = load_check_all()
Skip, Outcome = check_all.Skip, check_all.Outcome


def test_skip_reasons_name_what_is_missing() -> None:
    assert Skip("x", check_all.NO_NODE).need == "node"
    assert Skip("x", check_all.NO_TORCH).need == "torch"
    assert Skip("x", "needs the solved database (db/kendall5, complete)").need == "database"
    assert Skip("x", "needs the trained network (models/senet_net.bin)").need == "network"
    assert Skip("x", "not on this platform").need is None


def test_skipped_tests_from_a_pytest_report(tmp_path: Path) -> None:
    (tmp_path / "test_module.py").write_text(
        'import pytest\npytest.importorskip("no_such_module", reason="needs PyTorch")\n'
    )
    (tmp_path / "test_cases.py").write_text(
        "import pytest\n"
        '@pytest.mark.skipif(True, reason="needs Node.js")\n'
        "def test_marked(): pass\n"
        "def test_run(): pass\n"
        '@pytest.mark.parametrize("n", [1, 2])\n'
        'def test_inside(n): pytest.skip("needs the solved database")\n'
        '@pytest.mark.xfail(reason="expected")\n'
        "def test_xfail(): assert False\n"
    )
    (tmp_path / "pytest.ini").write_text("[pytest]\n")
    report = tmp_path / "report.xml"
    cmd = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", f"--junitxml={report}"]
    subprocess.run(cmd, cwd=tmp_path, capture_output=True, check=False)
    assert sorted((s.what, s.why, s.need) for s in check_all.skipped_tests(report)) == [
        ("test test_cases::test_inside[1]", "needs the solved database", "database"),
        ("test test_cases::test_inside[2]", "needs the solved database", "database"),
        ("test test_cases::test_marked", "needs Node.js", "node"),
        ("test test_module", "needs PyTorch", "torch"),
    ]


def test_missing_tools_skip_their_checks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ran: list[list[str]] = []

    def run(cmd: list[str | Path]) -> Any:
        ran.append([str(c) for c in cmd])
        return Outcome(True, "")

    monkeypatch.setattr(check_all, "run", run)

    def checks(node: str | None, torch: str | None) -> dict[str, Any]:
        return dict(check_all.all_checks(False, "cargo", node, torch, "python", tmp_path))

    def whys(node: str | None, torch: str | None) -> dict[str, str]:
        return {name: check for name, check in checks(node, torch).items() if isinstance(check, str)}

    assert whys("node", "python") == {}
    assert set(whys(None, "python").values()) == {check_all.NO_NODE} and len(whys(None, "python")) == 6
    assert whys("node", None) == {"network: Rust vs PyTorch vs browser": check_all.NO_TORCH}
    # With node, the network check must use it; without, it says that it leaves it out.
    assert checks("node", "python")["network: Rust vs PyTorch vs browser"]().skipped == []
    assert ran[-1][-1] == "--require-node"
    outcome = checks(None, "python")["network: Rust vs PyTorch vs browser"]()
    assert "--require-node" not in ran[-1]
    assert outcome.skipped == [Skip("the browser engine's network", check_all.NO_NODE)]


@pytest.fixture
def fake_checks(monkeypatch: pytest.MonkeyPatch) -> None:
    """Checks that pass, one skipped for want of node, and pytest skipping a PyTorch test."""
    checks = [
        ("passes", lambda: Outcome(True, "")),
        ("needs node", check_all.NO_NODE),
        ("pytest", lambda: Outcome(True, "", [Skip("test t", "needs PyTorch")])),
    ]
    monkeypatch.setattr(check_all, "all_checks", lambda *args: checks)
    monkeypatch.setattr(check_all, "tool", lambda name: name)
    monkeypatch.setattr(check_all, "python_with", lambda *modules: None)


def results(tmp_path: Path, *args: str) -> tuple[int, dict[str, Any]]:
    out = tmp_path / "results.json"
    code = check_all.main([*args, "--json", str(out)])
    return code, json.loads(out.read_text(encoding="utf-8"))


@pytest.mark.usefixtures("fake_checks")
def test_a_developer_run_says_what_it_did_not_check(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code, report = results(tmp_path)
    assert code == 0 and report["passed"]
    assert [c["status"] for c in report["checks"]] == ["ok", "skipped", "ok"]
    assert report["counts"] == {"ok": 2, "failed": 0, "skipped": 1, "skipped_within": 1}
    assert [(s["what"], s["allowed"]) for s in report["not_checked"]] == [
        ("needs node", True),
        ("test t", True),
        ("the database audits", True),
    ]
    out = capsys.readouterr().out
    assert "SKIP  needs node (needs Node.js)" in out
    assert "Not checked:\n  needs node: needs Node.js\n  test t: needs PyTorch\n  the database audits" in out
    assert "not everything was checked" in out


@pytest.mark.usefixtures("fake_checks")
def test_strict_and_required_runs_fail_on_skips(tmp_path: Path) -> None:
    code, report = results(tmp_path, "--strict")
    assert code == 1 and not report["passed"] and report["full"]
    assert [c["status"] for c in report["checks"]] == ["ok", "failed", "failed"]
    assert not any(s["allowed"] for s in report["not_checked"])

    code, report = results(tmp_path, "--require", "torch")
    assert code == 1
    assert [c["status"] for c in report["checks"]] == ["ok", "skipped", "failed"]

    code, report = results(tmp_path, "--require", "database", "network")
    assert code == 0
