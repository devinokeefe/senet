"""The C ABI's three descriptions agree: the header (crates/senet-ffi/include/senet.h), the
library's Rust source (crates/senet-ffi/src/lib.rs) and the Python binding (senet._lib):
the functions, their parameter and result types, the constants and the move layout. Drift
in any of them fails here, not in a caller."""

from __future__ import annotations

import ctypes as C
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from senet import KINDS, _lib
from senet._lib import ROOT

HEADER = ROOT / "crates" / "senet-ffi" / "include" / "senet.h"
RUST = ROOT / "crates" / "senet-ffi" / "src" / "lib.rs"
MOVEGEN = ROOT / "crates" / "senet-core" / "src" / "movegen.rs"

# C types in the header, as Rust names them.
C_TYPES = {
    "uint8_t": "u8",
    "uint32_t": "u32",
    "uint64_t": "u64",
    "int32_t": "i32",
    "int64_t": "i64",
    "size_t": "usize",
    "float": "f32",
    "double": "f64",
    "char": "c_char",
    "senet_move": "FfiMove",
    "senet_ctx": "Ctx",
}
Signature = tuple[str | None, list[tuple[str, str]]]  # result type, (name, type) of each parameter


def c_type(text: str) -> str:
    """A C type of the header, such as `const uint32_t *`, in Rust's terms: `*const u32`."""
    text = " ".join(text.split())
    base = C_TYPES[text.replace("const", "").replace("*", "").strip()]
    stars = text.count("*")
    assert stars <= 1, text
    return base if not stars else ("*const " if text.startswith("const ") else "*mut ") + base


def header() -> tuple[dict[str, Signature], dict[str, int], list[tuple[str, str]]]:
    """The header's functions, its integer constants, and the fields of senet_move."""
    text = HEADER.read_text(encoding="utf-8")
    functions: dict[str, Signature] = {}
    for line in text.splitlines():
        if found := re.fullmatch(r"(?P<result>.+?)\s*\b(?P<name>senet_\w+)\((?P<params>[^)]*)\);", line):
            params = []
            for p in found["params"].split(","):
                if p.strip() != "void":
                    name = re.search(r"(\w+)$", p.strip())
                    assert name, line
                    params.append((name[1], c_type(p.strip()[: -len(name[1])])))
            result = found["result"].strip()
            functions[found["name"]] = (None if result == "void" else c_type(result), params)
    constants = {name: int(v) for name, v in re.findall(r"#define SENET_(\w+) \(?(-?\d+)\)?", text)}
    struct = re.search(r"typedef struct senet_move \{(.*?)\} senet_move;", text, re.S)
    assert struct
    fields = [(name, C_TYPES[t]) for t, name in re.findall(r"^\s*(\w+) (\w+);", struct[1], re.M)]
    return functions, constants, fields


def rust() -> tuple[dict[str, Signature], dict[str, int], list[tuple[str, str]]]:
    """The library's exported functions, its constants, and the fields of FfiMove."""
    text = RUST.read_text(encoding="utf-8")
    functions: dict[str, Signature] = {}
    pattern = r'pub (?:unsafe )?extern "C" fn (senet_\w+)\(([^)]*)\)\s*(?:->\s*([^{]+?))?\s*\{'
    for name, params, result in re.findall(pattern, text, re.S):
        parsed = [tuple(" ".join(x.split()) for x in p.split(":")) for p in params.split(",") if p.strip()]
        functions[name] = (result.strip() or None, [(p[0], p[1]) for p in parsed])
    constants = {name: int(v) for name, v in re.findall(r"const (\w+): [iu]32 = (-?\d+);", text)}
    struct = re.search(r"pub struct FfiMove \{(.*?)\}", text, re.S)
    assert struct
    return functions, constants, re.findall(r"pub (\w+): (\w+),", struct[1])


def test_the_header_declares_what_the_library_exports() -> None:
    declared, _, _ = header()
    exported, _, _ = rust()
    assert len(declared) == 22
    assert declared == exported


def test_the_constants_agree() -> None:
    _, declared, _ = header()
    _, defined, _ = rust()
    assert declared["ABI_VERSION"] == defined["ABI_VERSION"] == _lib.ABI_VERSION == _lib.lib.senet_abi_version()
    for name in ("INVALID", "UNAVAILABLE", "NO_MOVE"):
        assert declared[name] == defined[name] == getattr(_lib, name)
    enum = re.search(r"pub enum Kind \{(.*?)\}", MOVEGEN.read_text(encoding="utf-8"), re.S)
    assert enum
    kinds = {name.upper(): int(v) for name, v in re.findall(r"(\w+) = (\d+),", enum[1])}
    assert kinds == {k.removeprefix("MOVE_"): v for k, v in declared.items() if k.startswith("MOVE_")}
    assert [KINDS[kinds[k]] for k in ("STEP", "SWAP", "OFF", "WATER")] == ["move", "swap", "off", "water"]


def test_the_move_layout_agrees() -> None:
    _, _, declared = header()
    _, _, defined = rust()
    fields = [("from", "u8"), ("to", "u8"), ("kind", "u8"), ("back", "u8"), ("me_after", "u32"), ("opp_after", "u32")]
    assert declared == defined == fields
    move = _lib.FfiMove
    names, types = [f[0] for f in move._fields_], [f[1] for f in move._fields_]
    assert names == ["frm", *[name for name, _ in fields[1:]]]  # `from` is a keyword in Python
    assert [python_type(t) for t in types] == [t for _, t in fields]
    assert (C.sizeof(move), C.alignment(move)) == (12, 4)
    assert [getattr(move, name).offset for name in names] == [0, 1, 2, 3, 4, 8]


def python_type(t: object) -> str:
    """A ctypes type in Rust's terms, without what ctypes cannot tell apart: whether a
    pointer is const, and size_t from the unsigned integer of its size."""
    if t is None:
        return "()"
    if t is C.c_char_p or t is C.c_char:
        return "*c_char" if t is C.c_char_p else "c_char"
    if t is C.c_void_p:
        return "*Ctx"  # handles are opaque pointers
    if t is _lib.FfiMove:
        return "FfiMove"
    if t is C.c_float or t is C.c_double:
        return "f32" if t is C.c_float else "f64"
    if isinstance(t, type) and issubclass(t, C._Pointer):
        return "*" + python_type(t._type_)
    assert isinstance(t, type) and issubclass(t, C._SimpleCData), t
    return f"{'i' if t(-1).value < 0 else 'u'}{8 * C.sizeof(t)}"


def as_python_sees_it(rust_type: str | None) -> str:
    if rust_type is None:
        return "()"
    t = rust_type.replace("*const ", "*").replace("*mut ", "*")
    return t.replace("usize", f"u{8 * C.sizeof(C.c_size_t)}")


def test_the_python_binding_declares_the_same_signatures() -> None:
    declared, _, _ = header()
    for name, (result, params) in declared.items():
        fn = getattr(_lib.lib, name)  # the library exports it
        argtypes = fn.argtypes or []  # senet_abi_version takes none, and is declared before the others
        assert python_type(fn.restype) == as_python_sees_it(result), name
        assert [python_type(t) for t in argtypes] == [as_python_sees_it(t) for _, t in params], name


def test_the_header_compiles(tmp_path: Path) -> None:
    compiler = shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")
    if compiler is None:
        pytest.skip("needs a C compiler (cc, gcc or clang)")
    source = tmp_path / "use_senet.c"
    source.write_text(
        '#include "senet.h"\n'
        '_Static_assert(sizeof(senet_move) == 12, "senet_move is 12 bytes");\n'
        '_Static_assert(_Alignof(senet_move) == 4, "senet_move is aligned to 4");\n'
        '_Static_assert(offsetof(senet_move, me_after) == 4, "me_after follows the four bytes");\n'
        "int main(void) { return senet_abi_version() == SENET_ABI_VERSION ? 0 : 1; }\n",
        encoding="utf-8",
    )
    flags = ["-std=c11", "-Wall", "-Wextra", "-Werror", "-pedantic", "-fsyntax-only"]
    result = subprocess.run(
        [compiler, *flags, f"-I{HEADER.parent}", str(source)], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
