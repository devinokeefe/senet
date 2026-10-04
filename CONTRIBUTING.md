# Contributing

Bug reports and pull requests are welcome. For a wrong value or an illegal move, give the
position (each side's squares and who is to throw) and the command or call that shows it.

## Setting up

You need Rust 1.85 or newer, Python 3.10 or newer and, for the browser checks, Node.js.
PyTorch is needed only to train or check the network. From the repository's root:

```bash
cargo build --release
python -m pip install -e ".[dev,train]"
```

## Checking a change

One command runs every check: formatting, lints, type checks, the Rust, Python and
JavaScript tests, and the cross-language parity checks. It lists whatever it could not
check (a missing Node.js, PyTorch or database) and fails if anything else fails:

```bash
python tools/check_all.py
```

Before a release, run the release gate on a machine with the solved database, Node.js,
PyTorch and a C compiler (which compiles the C header). It also audits the database,
compares its small layers with the reference solver, and fails on any check it would
otherwise skip:

```bash
python tools/check_all.py --strict --json runs/checks.json
```

CI (.github/workflows/ci.yml) runs the Rust and Python tests on Linux, macOS and Windows:
Rust 1.85, the oldest supported, and a pinned newer release that also formats and lints;
Python 3.10 and 3.13. The cross-language, web-app and network checks of `check_all` run on
Linux only. .github/constraints.txt pins the Python tools CI uses. No CI job can hold the
34 GB database, so the database checks run only in the release gate.

## Style

* Rust: `cargo fmt` (rustfmt.toml: lines of up to 120 columns) and `cargo clippy` with
  warnings as errors.
* Python: `ruff format` and `ruff check` (pyproject.toml: lines of up to 120 columns) and
  `mypy`, which checks the whole of `python/` and `tools/`.
* Text files end lines with LF (.gitattributes); programs that write text write LF on
  every platform.
* Keep the documentation in step with the code. Every number in it should come from a
  committed report or from a command the text names.

## What must stay in agreement

Some things exist in more than one language, and a check compares them:

| What | Where | Checked by |
|---|---|---|
| The rules | Rust `senet-core`, Python `senet_ref`, JavaScript `web/portable/senet-engine.js` | `senet dump-moves`, then `senet_ref.check_movegen` and `tools/check_js_rules.mjs` |
| The network's forward pass | Rust, PyTorch, JavaScript | `senet_train.check_net` |
| The C ABI | `crates/senet-ffi/src/lib.rs`, `crates/senet-ffi/include/senet.h`, `python/senet/_lib.py` | `python/tests/test_abi.py` |
| The HTTP API | `crates/senet-cli/src/server.rs`, `web/portable/local-api.js` | `tools/check_local_api.mjs` (docs/API.md) |
| Manifests | Rust `senet_core::manifest`, Python `senet.manifest` | `python/tests/test_manifest.py` |

A change to one of them changes the others, and the C ABI's version (`ABI_VERSION`)
changes with any change to its functions or types.
