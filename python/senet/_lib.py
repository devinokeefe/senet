"""Locates and loads the Rust engine library (senet_ffi) and declares its C ABI.

The ABI's conventions (failure codes, the text slot that `last_text()` reads, pointers and
handles) are described in crates/senet-ffi/include/senet.h.
"""

from __future__ import annotations

import ctypes as C
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ABI_VERSION = 6
INVALID, UNAVAILABLE, NO_MOVE = -1, -2, -3

_NAME = {"win32": "senet_ffi.dll", "darwin": "libsenet_ffi.dylib"}.get(sys.platform, "libsenet_ffi.so")


def checkout() -> Path | None:
    """The source checkout the package is in (ROOT); None for an installed copy (not an
    editable install), which is in no checkout."""
    return ROOT if (ROOT / "Cargo.toml").exists() else None


def _find() -> Path:
    """$SENET_FFI, else the release build in the source checkout the package is in
    (target/release). An installed copy of the package needs $SENET_FFI."""
    build = "cargo build --release -p senet-ffi"
    env = os.environ.get("SENET_FFI")
    if env:
        if not Path(env).is_file():
            raise OSError(f"SENET_FFI={env}: no such file; set it to the engine library ({_NAME})")
        # Absolute: a bare file name would be looked up on the system's library path.
        return Path(env).resolve()
    path = ROOT / "target" / "release" / _NAME
    if path.exists():
        return path
    if checkout() is not None:
        raise OSError(f"{path} not found; build it with: {build}")
    raise OSError(
        f"the senet package in {Path(__file__).parent} is not in its source checkout, so it cannot find the engine "
        f"library ({_NAME}): set SENET_FFI to its path. Build it in the checkout with: {build}"
    )


class FfiMove(C.Structure):
    _fields_ = [
        ("frm", C.c_uint8),
        ("to", C.c_uint8),
        ("kind", C.c_uint8),
        ("back", C.c_uint8),
        ("me_after", C.c_uint32),
        ("opp_after", C.c_uint32),
    ]


_PATH = _find()
lib = C.CDLL(str(_PATH))
lib.senet_abi_version.restype = C.c_uint32
if (_abi := lib.senet_abi_version()) != ABI_VERSION:
    raise OSError(f"{_PATH} has ABI version {_abi}, expected {ABI_VERSION}; rebuild it")

_P = C.POINTER
_u8, _u32, _u64, _i32, _i64 = C.c_uint8, C.c_uint32, C.c_uint64, C.c_int32, C.c_int64
_f32, _f64, _size, _ctx, _str = C.c_float, C.c_double, C.c_size_t, C.c_void_p, C.c_char_p

_SIGNATURES: dict[str, tuple[type | None, list[type]]] = {
    "senet_last_text": (_size, [_P(C.c_char), _size]),
    "senet_build_info": (_i64, []),
    "senet_max_moves": (_u32, []),
    "senet_n_features": (_u32, []),
    "senet_start": (_i32, [_P(_u32), _P(_u32)]),
    "senet_gen_moves": (_i32, [_u32, _u32, _u8, _P(FfiMove), _size]),
    "senet_index_of": (_i32, [_u32, _u32, _P(_u32), _P(_u32), _P(_u64)]),
    "senet_position_of": (_i32, [_u32, _u32, _u64, _P(_u32), _P(_u32)]),
    "senet_layer_size": (_u64, [_u32, _u32]),
    "senet_heuristic": (_f64, [_u32, _u32]),
    "senet_features": (_i32, [_u32, _u32, _P(_f32)]),
    "senet_ctx_new": (_ctx, [_str, _str]),
    "senet_ctx_close": (None, [_ctx]),
    "senet_ctx_free": (None, [_ctx]),
    "senet_db_value": (_f64, [_ctx, _u32, _u32]),
    "senet_db_values": (_i32, [_ctx, _P(_u32), _P(_u32), _size, _P(_f32)]),
    "senet_net_value": (_f64, [_ctx, _u32, _u32]),
    "senet_move_values": (_i32, [_ctx, _str, _u32, _u32, _u8, _P(_f64), _size]),
    "senet_bot_choose": (_i32, [_ctx, _str, _u32, _u32, _u8, _u64]),
    "senet_match": (_i64, [_ctx, _str, _str, _u64, _u64]),
    "senet_quality": (_i64, [_ctx, _str, _str, _u64, _u64]),
}
for _name, (_restype, _argtypes) in _SIGNATURES.items():
    _fn = getattr(lib, _name)
    _fn.restype = _restype
    _fn.argtypes = _argtypes

MAX_MOVES: int = lib.senet_max_moves()
N_FEATURES: int = lib.senet_n_features()


def last_text() -> str:
    """This thread's latest failure message, or JSON result of a match, quality run or build_info."""
    size = lib.senet_last_text(None, 0)
    buf = C.create_string_buffer(size)
    lib.senet_last_text(buf, size)
    return buf.value.decode("utf-8", errors="replace")
