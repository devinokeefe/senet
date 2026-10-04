"""Independent pure-Python reference implementation of kendall5 Senet (RULES.md, docs/FORMATS.md).

Used to differentially test the Rust engine/solver; written from the specs only.
"""

from .indexing import compact, index_of, layer_size, position_of, random_position
from .rules import (
    EXTRA_THROW,
    THROW_PROBS,
    Move,
    flip,
    legal_moves,
    play_game,
    play_random_game,
    start_position,
)

__all__ = [
    "EXTRA_THROW",
    "THROW_PROBS",
    "Move",
    "compact",
    "flip",
    "index_of",
    "layer_size",
    "legal_moves",
    "play_game",
    "play_random_game",
    "position_of",
    "random_position",
    "start_position",
]
