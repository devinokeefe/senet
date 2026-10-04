"""Runs every check of the project and reports a summary.

Formatting, lints, type checks and tests of the Rust, Python and JavaScript code (the
engine's core tested a second time built for this machine's CPU, which on x86-64 with BMI2
takes other code paths: .cargo/native.toml), plus the cross-language parity checks: the
Python reference rules and the browser engine against a fresh Rust move dump, the browser
and server APIs against their shared contract, a smoke test of the web app's UI, the
portable page as built (twice, to the same bytes) and played offline with the real
network, the network against its manifest, its forward pass in Rust, PyTorch and
JavaScript, and its bots' play on held-out decisions (senet_train.strength).

    python tools/check_all.py            # a few minutes
    python tools/check_all.py --full     # also audit the solved database (db/kendall5)
    python tools/check_all.py --strict   # the release gate: everything, nothing skipped

--full also checks every file of the database against its manifest, audits its Bellman
residuals, compares every layer with w + b <= 5 against the Python reference solver (the
first such run computes the reference tables: about 10 minutes, cached in
python/senet_ref/_cache/), and the strength check's values with the database.

Every check runs even if an earlier one fails, and fails if one of its commands hangs (runs
past a generous timeout). What needs something missing here (Node.js, PyTorch, the solved
database or the trained network) is skipped: whole checks, tests, and parts of checks. The
summary lists everything that was not checked, and the exit code is 1 if any check failed.
--strict implies --full and fails on every skip; --require fails on the skips for want of
the things it names, e.g. `--require node torch network` where the database cannot be.
--skip leaves out the Rust or Python formatting, lints, type checks and tests, which CI
runs in other jobs. --json writes the results to a file as well.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from functools import partial
from pathlib import Path
from typing import IO
from xml.etree import ElementTree

ROOT = Path(__file__).resolve().parents[1]
EXE = ".exe" if sys.platform == "win32" else ""
SENET = ROOT / "target" / "release" / f"senet{EXE}"
DB = ROOT / "db" / "kendall5"
NET = ROOT / "models" / "senet_net.bin"
# Cargo options for a build for this machine's CPU, kept apart from the portable build.
NATIVE = ["--config", ".cargo/native.toml", "--target-dir", "target-native"]
SERVER_START_TIMEOUT = 60  # seconds for `senet serve` to load its database and network
RUN_TIMEOUT = 3600  # seconds for one command of a check, far more than any takes here
DB_TIMEOUT = 4 * 3600  # for one that reads the whole database
MOVE_DUMP = 1_000_000  # positions that the rules are compared on
SERVER_LINE = re.compile(r"Senet server on (http://\S+?)/?")  # the line it prints once it listens
REFERENCE_K = 5  # the Python reference solver covers every layer with w + b <= REFERENCE_K
JS_SOURCES = ("web/app.js", "web/portable/local-api.js", "web/portable/senet-engine.js")
OUTPUT_LINES = 40  # of a failed check's output, in the report
# What a check or test may need, keyed by the name --require takes. Skips for want of one
# give their reason as "needs <the thing>", here and in the tests.
NEEDS = {"node": "Node.js", "torch": "PyTorch", "database": "the solved database", "network": "the trained network"}
NO_NODE = f"needs {NEEDS['node']}"
NO_TORCH = f"needs {NEEDS['torch']}"
NO_DB = f"needs {NEEDS['database']} ({DB.relative_to(ROOT).as_posix()})"
NO_NET = f"needs {NEEDS['network']} ({NET.relative_to(ROOT).as_posix()})"
# The checks that --skip can leave out, by language.
LANGUAGE_CHECKS = {
    "rust": ("rustfmt", "clippy", "cargo test", "cargo test (senet-core built for this CPU)"),
    "python": ("ruff format", "ruff check", "mypy", "pytest"),
}


@dataclass
class Skip:
    """Something not checked, and why."""

    what: str
    why: str

    @property
    def need(self) -> str | None:
        """The key in NEEDS of the missing thing, or None if the skip has another reason."""
        return next((need for need, thing in NEEDS.items() if self.why.startswith(f"needs {thing}")), None)


@dataclass
class Outcome:
    """What a check found: whether it passed, its output, and what it left out."""

    ok: bool
    output: str
    skipped: list[Skip] = field(default_factory=list)


@dataclass
class Result:
    """A check's entry in the report. `status` is "ok", "failed" or "skipped"; a check that
    did not run lists itself as skipped."""

    name: str
    status: str
    ran: bool
    seconds: float
    skipped: list[Skip]
    output: str  # the end of a failed check's output


Check = Callable[[], Outcome]


def tool(name: str) -> str | None:
    """A tool on PATH, or in ~/.cargo/bin for the Rust ones; None if it is not installed."""
    return shutil.which(name) or shutil.which(name, path=str(Path.home() / ".cargo" / "bin"))


def python_with(*modules: str) -> str | None:
    """This interpreter if it can import `modules`, else the project's .venv if that can."""
    venv = ROOT / ".venv" / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    imports = f"import {', '.join(modules)}"
    for python in (sys.executable, str(venv)):
        if Path(python).exists() and subprocess.run([python, "-c", imports], capture_output=True).returncode == 0:
            return python
    return None


def run(cmd: Sequence[str | Path], timeout: float = RUN_TIMEOUT) -> Outcome:
    """Runs `cmd` in the project root, with python/ on PYTHONPATH, stopping it after
    `timeout` seconds."""
    paths = [str(ROOT / "python"), os.environ.get("PYTHONPATH", "")]
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(p for p in paths if p)}
    try:
        p = subprocess.run(
            [str(c) for c in cmd], cwd=ROOT, env=env, capture_output=True, text=True, errors="replace", timeout=timeout
        )
    except OSError as e:
        return Outcome(False, f"cannot run {cmd[0]}: {e}")
    except subprocess.TimeoutExpired as e:
        # The output captured so far, as bytes even with text=True.
        output = "".join(x.decode(errors="replace") if isinstance(x, bytes) else x or "" for x in (e.stdout, e.stderr))
        return Outcome(False, f"{output}\nstopped after {timeout:.0f} s")
    return Outcome(p.returncode == 0, p.stdout + p.stderr)


def run_each(cmds: Sequence[Sequence[str | Path]]) -> Outcome:
    """Runs every command, even after a failure; succeeds if all of them do."""
    outcomes = [run(cmd) for cmd in cmds]
    return Outcome(all(o.ok for o in outcomes), "".join(o.output for o in outcomes))


def partly(check: Check, *skips: Skip) -> Check:
    """`check`, which leaves out the parts that `skips` name."""

    def checked() -> Outcome:
        outcome = check()
        outcome.skipped += skips
        return outcome

    return checked


def skipped_tests(report: Path) -> list[Skip]:
    """The tests that a pytest JUnit XML report shows as skipped (a module skipped as a
    whole, as by importorskip, is one entry)."""
    skips = []
    for case in ElementTree.parse(report).iter("testcase"):
        for skipped in case.iter("skipped"):
            if skipped.get("type") == "pytest.xfail":
                continue  # an expected failure ran
            why = skipped.get("message", "")
            # A module skipped while collected reports (file, line, "Skipped: <why>").
            if found := re.search(r"Skipped: (.*)['\"]\)\s*$", skipped.text or ""):
                why = found[1]
            name = "::".join(filter(None, (case.get("classname"), case.get("name"))))
            skips.append(Skip(f"test {name}", why))
    return skips


def pytest_check(python: str, tmp: Path, timeout: float) -> Outcome:
    """The Python tests, with the ones they skipped."""
    report = tmp / "pytest.xml"
    outcome = run([python, "-m", "pytest", "-q", f"--basetemp={tmp / 'pytest'}", f"--junitxml={report}"], timeout)
    if report.exists():
        outcome.skipped = skipped_tests(report)
    return outcome


def portable_check(node: str, tmp: Path) -> Outcome:
    """Builds the portable page twice, which must give the same bytes, and plays it."""
    pages = [tmp / f"portable_{i}.html" for i in (1, 2)]
    built = run_each([[sys.executable, "tools/build_portable.py", "--out", page] for page in pages])
    if not built.ok:
        return built
    if pages[0].read_bytes() != pages[1].read_bytes():
        return Outcome(False, "two builds of the portable page differ")
    return run([node, "tools/check_portable.mjs", pages[0], NET])


def lines_of(stream: IO[str]) -> queue.Queue[str | None]:
    """The lines of `stream` as a thread reads them, then None at its end."""
    lines: queue.Queue[str | None] = queue.Queue()

    def read() -> None:
        for line in stream:
            lines.put(line)
        lines.put(None)

    threading.Thread(target=read, daemon=True).start()
    return lines


def with_server(cmd: Callable[[str], Sequence[str | Path]]) -> Check:
    """A check that runs `cmd(url)` while `senet serve` listens at `url`, on a port that the
    system assigns: a server left running from elsewhere cannot answer in its place."""

    def check() -> Outcome:
        try:
            server = subprocess.Popen(
                [str(SENET), "serve", "--port", "0"],
                cwd=ROOT,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                errors="replace",
            )
        except OSError as e:
            return Outcome(False, f"cannot start the server: {e}")
        try:
            assert server.stderr is not None
            lines = lines_of(server.stderr)
            log: list[str] = []
            deadline = time.monotonic() + SERVER_START_TIMEOUT
            while True:
                try:
                    line = lines.get(timeout=max(0.0, deadline - time.monotonic()))
                except queue.Empty:
                    return Outcome(False, "the server did not start in time:\n" + "".join(log))
                if line is None:
                    return Outcome(False, "the server exited:\n" + "".join(log))
                log.append(line)
                if match := SERVER_LINE.fullmatch(line.strip()):
                    return run(cmd(match[1]))
        finally:
            server.terminate()
            server.wait()

    return check


def all_checks(
    full: bool, cargo: str, node: str | None, torch_python: str | None, pytest_python: str, tmp: Path
) -> list[tuple[str, Check | str]]:
    """Every check, as (name, check) or, for one that cannot run here, (name, why it is skipped).
    `pytest_python` runs the tests: one with PyTorch, if there is one, so that its tests run."""
    py = sys.executable
    dump = tmp / "moves.jsonl"
    js_files = [*JS_SOURCES, *sorted(p.relative_to(ROOT).as_posix() for p in (ROOT / "tools").glob("*.mjs"))]
    have_net, have_db = NET.exists(), DB.is_dir()
    network: Check | str = NO_TORCH if have_net else NO_NET
    if torch_python and have_net:
        check_net: list[str | Path] = [torch_python, "-m", "senet_train.check_net", "--net", NET]
        if node:
            network = partial(run, [*check_net, "--require-node"])
        else:  # check_net leaves the browser engine out
            network = partly(partial(run, check_net), Skip("the browser engine's network", NO_NODE))
    strength: Check | str = NO_NET
    if have_net:
        measure: list[str | Path] = [py, "-m", "senet_train.strength", "--net", NET]
        if full and have_db:  # re-measure the corpus too
            strength = partial(run, [*measure, "--db", DB], DB_TIMEOUT)
        elif full:
            strength = partly(partial(run, measure), Skip("re-measuring the strength corpus", NO_DB))
        else:
            strength = partial(run, measure)
    portable: Check | str = NO_NET
    if not node:
        portable = NO_NODE
    elif have_net:
        portable = partial(portable_check, node, tmp)
    checks: list[tuple[str, Check | str]] = [
        ("rustfmt", partial(run, [cargo, "fmt", "--all", "--", "--check"])),
        (
            "clippy",
            partial(run, [cargo, "clippy", "--release", "--workspace", "--all-targets", "--", "-D", "warnings"]),
        ),
        ("cargo build", partial(run, [cargo, "build", "--release", "--workspace"])),
        ("cargo test", partial(run, [cargo, "test", "--release", "--workspace"])),
        ("cargo test (senet-core built for this CPU)", partial(run, [cargo, "test", "-p", "senet-core", *NATIVE])),
        ("ruff format", partial(run, [py, "-m", "ruff", "format", "--check", "python", "tools"])),
        ("ruff check", partial(run, [py, "-m", "ruff", "check", "python", "tools"])),
        ("mypy", partial(run, [py, "-m", "mypy"])),
        ("pytest", partial(pytest_check, pytest_python, tmp, DB_TIMEOUT if full else RUN_TIMEOUT)),
        ("JavaScript syntax", partial(run_each, [[node, "--check", f] for f in js_files]) if node else NO_NODE),
        (
            f"move dump ({MOVE_DUMP:,} positions)",
            partial(run, [SENET, "dump-moves", "--out", dump, "--n", str(MOVE_DUMP)]),
        ),
        ("rules: Python reference vs Rust", partial(run, [py, "-m", "senet_ref.check_movegen", dump])),
        ("rules: browser engine vs Rust", partial(run, [node, "tools/check_js_rules.mjs", dump]) if node else NO_NODE),
        ("API: browser vs contract", partial(run, [node, "tools/check_local_api.mjs"]) if node else NO_NODE),
        (
            "API: server vs contract",
            with_server(lambda url: [node, "tools/check_local_api.mjs", "--server", url]) if node else NO_NODE,
        ),
        ("web app: UI smoke test", partial(run, [node, "tools/check_web_app.mjs"]) if node else NO_NODE),
        ("portable page: reproducible, plays offline", portable),
        ("network: manifest", partial(run, [py, "-m", "senet.manifest", "verify", NET]) if have_net else NO_NET),
        ("network: Rust vs PyTorch vs browser", network),
        ("network: strength on held-out decisions" + (" (re-measured)" if full and have_db else ""), strength),
    ]
    if full:
        reference: list[str | Path] = [py, "-m", "senet_ref.check_db", DB, "--K", str(REFERENCE_K), "--solve"]
        on_db: list[tuple[str, Sequence[str | Path]]] = [
            ("database: manifest (every file's size and SHA-256)", [SENET, "verify", DB]),
            ("database: Bellman residual audit", [SENET, "check", "--db", DB]),
            (f"database vs Python reference solver (w + b <= {REFERENCE_K})", reference),
        ]
        checks += [(name, partial(run, cmd, DB_TIMEOUT) if have_db else NO_DB) for name, cmd in on_db]
    return checks


def run_check(name: str, check: Check | str, allowed: Callable[[Skip], bool]) -> Result:
    """Runs a check, or skips it if `check` is why, and prints a line about it. A skip that
    is not `allowed` fails the check."""
    if isinstance(check, str):
        skip = Skip(name, check)
        print(f"{'SKIP' if allowed(skip) else 'FAIL'}  {name} ({check})", flush=True)
        return Result(name, "skipped" if allowed(skip) else "failed", False, 0.0, [skip], "")
    t0 = time.monotonic()
    outcome = check()
    seconds = time.monotonic() - t0
    ok = outcome.ok and all(map(allowed, outcome.skipped))
    skips = f", {len(outcome.skipped)} skipped" if outcome.skipped else ""
    print(f"{'ok  ' if ok else 'FAIL'}  {name}  ({seconds:.0f}s{skips})", flush=True)
    tail = ""
    if not outcome.ok:
        tail = "\n".join(outcome.output.strip().splitlines()[-OUTPUT_LINES:])
        print("      " + tail.replace("\n", "\n      "), flush=True)
    return Result(name, "ok" if ok else "failed", True, seconds, outcome.skipped, tail)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--full", action="store_true", help="also audit the solved database")
    ap.add_argument("--strict", action="store_true", help="the release gate: --full, and fail on every skip")
    ap.add_argument(
        "--require",
        nargs="+",
        default=[],
        choices=NEEDS,
        metavar="NEED",
        help=f"fail on the skips for want of these ({', '.join(NEEDS)})",
    )
    ap.add_argument(
        "--skip",
        nargs="+",
        default=[],
        choices=LANGUAGE_CHECKS,
        metavar="LANGUAGE",
        help=f"leave out the formatting, lints, type checks and tests of these ({', '.join(LANGUAGE_CHECKS)})",
    )
    ap.add_argument("--json", type=Path, metavar="PATH", help="also write the results to PATH")
    args = ap.parse_args(argv)
    if args.strict and args.skip:
        ap.error("--strict runs every check; it cannot --skip any")
    full = args.full or args.strict
    required = set(NEEDS) if args.strict else set(args.require)
    left_out = {name: f"left out by --skip {lang}" for lang in args.skip for name in LANGUAGE_CHECKS[lang]}

    def allowed(skip: Skip) -> bool:
        return not args.strict and skip.need not in required

    cargo = tool("cargo")
    if cargo is None:
        raise SystemExit("cargo not found")
    node, torch_python = tool("node"), python_with("torch")
    pytest_python = python_with("torch", "pytest") or sys.executable
    tmp = Path(tempfile.mkdtemp(prefix="senet_check_"))
    try:
        checks = all_checks(full, cargo, node, torch_python, pytest_python, tmp)
        results = [run_check(name, left_out.get(name, check), allowed) for name, check in checks]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    unchecked = [skip for result in results for skip in result.skipped]
    if not full:
        unchecked.append(Skip("the database audits", "run with --full"))
    count = {status: sum(r.status == status for r in results) for status in ("ok", "failed", "skipped")}
    within = sum(len(r.skipped) for r in results if r.ran)
    if unchecked:
        print("\nNot checked:")
        for skip in unchecked:
            print(f"  {skip.what}: {skip.why}" + ("" if allowed(skip) else "  [FAIL]"))
    print(
        f"\n{len(results)} checks: {count['ok']} passed, {count['failed']} failed, {count['skipped']} skipped;"
        f" {within} tests or parts skipped in the checks that ran"
    )
    failed = [r.name for r in results if r.status == "failed"]
    if failed:
        print(f"FAILED: {', '.join(failed)}")
    elif unchecked:
        print("Passed, but not everything was checked (see above).")
    else:
        print("All checks passed; nothing was skipped.")
    if args.json:
        report = {
            "passed": not failed,
            "full": full,
            "strict": args.strict,
            "required": sorted(required),
            "skipped_languages": args.skip,
            "tools": {
                "cargo": cargo,
                "node": node,
                "python": sys.executable,
                "python with PyTorch": torch_python,
                "python for pytest": pytest_python,
            },
            "counts": {**count, "skipped_within": within},
            "not_checked": [{**asdict(skip), "allowed": allowed(skip)} for skip in unchecked],
            "checks": [asdict(r) for r in results],
        }
        args.json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
