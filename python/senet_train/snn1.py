"""The SNN1 network file (docs/FORMATS.md), read and written without PyTorch.

Little-endian: b"SNN1", u32 layer count, then per layer u32 n_in, u32 n_out,
f32 weights[n_out][n_in], f32 bias[n_out]. The layers form an N_INPUTS-input, 1-output
MLP of at most MAX_LAYERS layers with at most MAX_WIDTH outputs each, every parameter is
finite, no activation can overflow float32, and the file ends after the last layer: the
limits of the Rust engine (senet_core::net) and the browser engine
(web/portable/senet-engine.js).
"""

from __future__ import annotations

import struct
from collections.abc import Sequence

import numpy as np

MAGIC = b"SNN1"
N_INPUTS = 72
MAX_LAYERS = 16
MAX_WIDTH = 4096
MAX_FEATURE = 1.4  # a remaining distance, at most (30 + 29 + 28 + 27 + 26) / 100
F32_MAX = float(np.finfo(np.float32).max)

Layer = tuple[np.ndarray, np.ndarray]
"""A layer's float32 weights, of shape (n_out, n_in), and biases, of shape (n_out,)."""


def _check_count(n_layers: int) -> None:
    if not 1 <= n_layers <= MAX_LAYERS:
        raise ValueError(f"{n_layers} layers; a network has 1..{MAX_LAYERS}")


def _check_layer(k: int, n_in: int, n_out: int, n_in_expected: int) -> None:
    if n_in != n_in_expected or not 1 <= n_out <= MAX_WIDTH:
        raise ValueError(f"layer {k} is {n_in} -> {n_out}; expected {n_in_expected} inputs and 1..{MAX_WIDTH} outputs")


def _check_output(n_out: int) -> None:
    if n_out != 1:
        raise ValueError(f"the network has {n_out} outputs; it must have one")


def _check_finite(k: int, w: np.ndarray, b: np.ndarray) -> None:
    if not (np.isfinite(w).all() and np.isfinite(b).all()):
        raise ValueError(f"layer {k} has a parameter that is not finite")


def _check_range(layers: Sequence[Layer]) -> None:
    """Raises ValueError if an activation could overflow float32. `bound` holds bounds on
    the absolute values of a layer's inputs: the features, then the previous layer's
    outputs. Sums of terms within half the range of float32 cannot overflow it. The terms
    are added in order (cumsum), as the other engines add them."""
    bound = np.full(N_INPUTS, MAX_FEATURE)
    for k, (w, b) in enumerate(layers):
        terms = np.abs(w).astype(np.float64) * bound
        bound = np.abs(b).astype(np.float64) + np.cumsum(terms, axis=1)[:, -1]
        if bound.max() > F32_MAX / 2:
            raise ValueError(f"layer {k}'s outputs could overflow float32 (up to {bound.max():.1e})")


def check_shapes(shapes: Sequence[tuple[int, int]]) -> None:
    """Raises ValueError unless layers of these shapes, (n_in, n_out) each, form a network
    that the format allows."""
    _check_count(len(shapes))
    n_in_expected = N_INPUTS
    for k, (n_in, n_out) in enumerate(shapes):
        _check_layer(k, n_in, n_out, n_in_expected)
        n_in_expected = n_out
    _check_output(n_in_expected)


def read(data: bytes) -> list[Layer]:
    """The layers of an SNN1 file; ValueError if it is not a network that the format allows."""
    off = 0

    def take(n: int) -> bytes:
        nonlocal off
        if off + n > len(data):
            raise ValueError(f"truncated SNN1 file ({len(data)} bytes)")
        off += n
        return data[off - n : off]

    if take(4) != MAGIC:
        raise ValueError("not an SNN1 network file")
    (n_layers,) = struct.unpack("<I", take(4))
    _check_count(n_layers)
    layers = []
    n_in_expected = N_INPUTS
    for k in range(n_layers):
        n_in, n_out = struct.unpack("<II", take(8))
        _check_layer(k, n_in, n_out, n_in_expected)
        w = np.frombuffer(take(4 * n_in * n_out), dtype="<f4").reshape(n_out, n_in)
        b = np.frombuffer(take(4 * n_out), dtype="<f4")
        _check_finite(k, w, b)
        layers.append((w, b))
        n_in_expected = n_out
    _check_output(n_in_expected)
    if off != len(data):
        raise ValueError(f"{len(data) - off} unexpected bytes after the last layer")
    _check_range(layers)
    return layers


def write(layers: Sequence[Layer]) -> bytes:
    """The SNN1 file of `layers` (converted to float32); ValueError if they do not form a
    network that the format allows."""
    layers = [(np.asarray(w, dtype="<f4"), np.asarray(b, dtype="<f4")) for w, b in layers]
    for k, (w, b) in enumerate(layers):
        if w.ndim != 2 or b.shape != w.shape[:1]:
            raise ValueError(f"layer {k} has weights of shape {w.shape} and biases of shape {b.shape}")
    check_shapes([(w.shape[1], w.shape[0]) for w, _ in layers])
    parts = [MAGIC, struct.pack("<I", len(layers))]
    for k, (w, b) in enumerate(layers):
        _check_finite(k, w, b)
        parts += [struct.pack("<II", w.shape[1], w.shape[0]), w.tobytes(order="C"), b.tobytes()]
    _check_range(layers)
    return b"".join(parts)
