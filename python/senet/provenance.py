"""Where a result comes from: the code, the machine, the inputs and the command.

Reports (senet.bench, senet.insights, senet.latency) record `provenance(...)` in their JSON,
render it at the end of their markdown and are written by `write_report`; manifests written
from Python record `created_by(...)`.
"""

from __future__ import annotations

import ctypes
import json
import os
import platform
import shlex
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from . import build_info
from ._lib import checkout
from .manifest import file_sha256, identity, utc_now


def home() -> Path:
    """Where the reports' default paths start: the source checkout, or the current directory
    for an installed copy of the package."""
    return checkout() or Path()


def git_commit() -> str | None:
    """The checkout's commit, with "-dirty" if the work tree has changes; None for an
    installed copy of the package or outside git."""
    root = checkout()
    if root is None:  # git would find whatever repository encloses the installation
        return None

    def git(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=True).stdout.strip()

    try:
        head, changes = git("rev-parse", "HEAD"), git("status", "--porcelain")
    except (OSError, subprocess.CalledProcessError):
        return None
    return head + ("-dirty" if changes else "")


def command() -> str:
    """The command line this process was started with, as `python -m module ...` for a module,
    its arguments quoted as a POSIX shell reads them back (as `senet` records its own)."""
    spec = getattr(sys.modules.get("__main__"), "__spec__", None)
    head = ["python", "-m", spec.name] if spec is not None else ["python", Path(sys.argv[0]).name]
    return " ".join(head + [shlex.quote(a) for a in sys.argv[1:]])


def cpu_name() -> str:
    """The processor's name, as the operating system reports it."""
    try:
        if sys.platform == "win32":
            import winreg

            key = r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key) as k:
                return str(winreg.QueryValueEx(k, "ProcessorNameString")[0]).strip()
        if sys.platform == "linux":
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
        if sys.platform == "darwin":
            out = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True)
            if out.returncode == 0 and out.stdout.strip():
                return out.stdout.strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def memory_bytes() -> int | None:
    """The machine's physical memory; None if it cannot be read."""
    if sys.platform == "win32":

        class Status(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_uint32),
                ("dwMemoryLoad", ctypes.c_uint32),
                ("ullTotalPhys", ctypes.c_uint64),
                ("ullAvailPhys", ctypes.c_uint64),
                ("ullTotalPageFile", ctypes.c_uint64),
                ("ullAvailPageFile", ctypes.c_uint64),
                ("ullTotalVirtual", ctypes.c_uint64),
                ("ullAvailVirtual", ctypes.c_uint64),
                ("ullAvailExtendedVirtual", ctypes.c_uint64),
            ]

        status = Status(dwLength=ctypes.sizeof(Status))
        ok = ctypes.WinDLL("kernel32").GlobalMemoryStatusEx(ctypes.byref(status))
        return int(status.ullTotalPhys) if ok else None
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        return None


def machine() -> dict[str, Any]:
    return {
        "os": platform.platform(),
        "cpu": cpu_name(),
        "logical_cpus": os.cpu_count(),
        "memory_bytes": memory_bytes(),
        "python": platform.python_version(),
    }


def shown(path: str | os.PathLike[str]) -> str:
    """`path` as a report records it, with `/`: relative to the checkout if it is in it (as
    given, or once symbolic links are resolved), so that the report does not depend on where
    the checkout is; else absolute."""
    resolved = Path(path).resolve()
    if (root := checkout()) is not None:
        for p in (Path(os.path.abspath(path)), resolved):
            if p.is_relative_to(root):
                return p.relative_to(root).as_posix()
    return resolved.as_posix()


def created_by(**extra: Any) -> dict[str, Any]:
    """What a manifest written by this process records as its maker, with `extra` facts: the
    same as `senet` records (senet_core::manifest::created_by_this_program)."""
    return {"command": command(), "commit": git_commit(), "engine": build_info(), **extra}


def provenance(
    db: str | os.PathLike[str] | None = None, net: str | os.PathLike[str] | None = None, **settings: Any
) -> dict[str, Any]:
    """The facts a report's numbers depend on: the time, command, commit, machine and engine
    build; the database (by its manifest) and network (by its SHA-256); and `settings` (seeds,
    game counts, ...)."""
    return {
        "generated_utc": utc_now(),
        "command": command(),
        "commit": git_commit(),
        "machine": machine(),
        "engine": build_info(),
        "database": {**identity(db), "path": shown(db)} if db is not None else None,
        "network": {"path": shown(net), "sha256": file_sha256(Path(net))} if net is not None else None,
        "settings": settings,
    }


def render(p: dict[str, Any]) -> str:
    """A markdown section on provenance `p`."""
    m, e = p["machine"], p["engine"]
    memory = f", {m['memory_bytes'] / 2**30:.0f} GiB" if m.get("memory_bytes") else ""
    lines = [
        "## Provenance",
        "",
        f"* Generated {p['generated_utc']} by `{p['command']}` at commit `{p['commit'] or 'unknown'}`.",
        f"* Machine: {m['cpu']}, {m['logical_cpus']} logical CPUs{memory}; {m['os']}; Python {m['python']}.",
        f"* Engine: senet {e['version']} for {e['target']}, {'optimized' if e['optimized'] else 'debug'} build"
        f" using {', '.join(e['cpu_features']) or 'no CPU features beyond the baseline'}.",
    ]
    if db := p["database"]:
        manifest = f"manifest SHA-256 `{db['manifest_sha256']}`" if db["manifest_sha256"] else "no manifest"
        lines.append(f"* Database: `{db['path']}`, {manifest}.")
    if net := p["network"]:
        lines.append(f"* Network: `{net['path']}`, SHA-256 `{net['sha256']}`.")
    if p["settings"]:
        settings = (f"{k} = {v if isinstance(v, str) else json.dumps(v)}" for k, v in p["settings"].items())
        lines.append("* Settings: " + ", ".join(settings) + ".")
    return "\n".join(lines) + "\n"


def write_report(results: dict[str, Any], out: Path, raw: Path, render: Callable[[dict[str, Any], str], str]) -> None:
    """Writes `results` as JSON to `raw`, and to `out` the markdown that `render` makes of
    that JSON, linking it."""
    for path in (raw, out):
        path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(results, indent=2) + "\n"
    raw.write_text(text, encoding="utf-8", newline="\n")
    try:
        link = Path(os.path.relpath(raw, out.parent)).as_posix()
    except ValueError:  # on another drive (Windows)
        link = raw.resolve().as_uri()
    out.write_text(render(json.loads(text), link), encoding="utf-8", newline="\n")
    print(f"wrote {out} and {raw}")
