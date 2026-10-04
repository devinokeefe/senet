"""Hand-constructed scenarios for every clause of RULES.md, plus indexing and solver checks.

Each test names the clause it exercises.  Positions are (me, opp) from the mover's view.
"""

from __future__ import annotations

import itertools
import random
from collections.abc import Iterable, Sequence
from math import comb

import numpy as np
import pytest

from senet_ref.indexing import USABLE_SQUARES, compact, index_of, layer_size, position_of, random_position
from senet_ref.rules import (
    EXTRA_THROW,
    THROW_PROBS,
    Move,
    blockade_squares,
    flip,
    legal_moves,
    play_game,
    play_random_game,
    start_position,
    throw_sticks,
    validate_position,
)
from senet_ref.solver import Layer, solve, value

Tables = dict[Layer, np.ndarray]


def M(frm: int, to: int, kind: str, direction: str, me: Iterable[int], opp: Iterable[int]) -> Move:
    return Move(frm, to, kind, direction, tuple(me), tuple(opp))


def fwd(frm: int, to: int, me: Iterable[int], opp: Iterable[int], kind: str = "move") -> Move:
    return M(frm, to, kind, "fwd", me, opp)


def back(frm: int, to: int, me: Iterable[int], opp: Iterable[int], kind: str = "move") -> Move:
    return M(frm, to, kind, "back", me, opp)


# ---------------------------------------------------------------------------
# Throws and setup
# ---------------------------------------------------------------------------


def test_throw_distribution() -> None:
    # "Throwing sticks": 4 fair sticks, zero light sides counts as 5.
    assert THROW_PROBS == {1: 4 / 16, 2: 6 / 16, 3: 4 / 16, 4: 1 / 16, 5: 1 / 16}
    assert sum(THROW_PROBS.values()) == 1.0
    counts = {t: 0 for t in range(1, 6)}
    for bits in range(16):
        light = bin(bits).count("1")
        counts[5 if light == 0 else light] += 1
    assert {t: c / 16 for t, c in counts.items()} == THROW_PROBS
    # "Turn structure" rule 3: 1, 4, 5 throw again.
    assert EXTRA_THROW == {1, 4, 5}


def test_start_position_and_flip() -> None:
    # "Pieces and setup": White on 1,3,5,7,9, Black on 2,4,6,8,10, White moves first.
    assert start_position() == ((1, 3, 5, 7, 9), (2, 4, 6, 8, 10))
    assert flip(*start_position()) == ((2, 4, 6, 8, 10), (1, 3, 5, 7, 9))
    assert flip((30, 1), (5,)) == ((5,), (1, 30))


def test_start_position_moves() -> None:
    me, opp = start_position()
    # t=2: every white piece except 9 would land on its own piece (rule 7).
    assert legal_moves(me, opp, 2) == [fwd(9, 11, (1, 3, 5, 7, 11), opp)]
    # t=1: every black piece is flanked by white pieces only -> unprotected -> 5 swaps (rule 8).
    moves = legal_moves(me, opp, 1)
    assert [(m.frm, m.to, m.kind) for m in moves] == [(a, a + 1, "swap") for a in me]
    assert moves[0] == fwd(1, 2, (2, 3, 5, 7, 9), (1, 4, 6, 8, 10), "swap")


# ---------------------------------------------------------------------------
# Forward moves, rules 2 + 9 / 7 / 8
# ---------------------------------------------------------------------------


def test_simple_move() -> None:
    # Forward rule 9: destination empty -> the piece moves there.
    assert legal_moves((1,), (10,), 3) == [fwd(1, 4, (4,), (10,))]
    assert legal_moves((1, 5), (20,), 2) == [fwd(1, 3, (3, 5), (20,)), fwd(5, 7, (1, 7), (20,))]


def test_cannot_land_on_own_piece() -> None:
    # Forward rule 7: destination holds a friendly piece -> illegal.
    assert legal_moves((1, 4), (20,), 3) == [fwd(4, 7, (1, 7), (20,))]


def test_swap_with_lone_enemy() -> None:
    # Forward rule 8: unprotected enemy -> swap (mover to d, enemy to a).
    assert legal_moves((1,), (4,), 3) == [fwd(1, 4, (4,), (1,), "swap")]


def test_friendly_neighbour_does_not_protect_enemy() -> None:
    # Definitions/protection: only *enemy* neighbours protect.  My piece on 5 next to the enemy on 4
    # does not protect it.
    assert legal_moves((1, 5), (4,), 3) == [
        fwd(1, 4, (4, 5), (1,), "swap"),
        fwd(5, 8, (1, 8), (4,)),
    ]


def test_protected_enemy_neighbour_above() -> None:
    # Protection: enemy on 4 has an enemy on s+1 = 5 -> cannot be swapped (rule 8).
    assert legal_moves((1, 10), (4, 5), 3) == [fwd(10, 13, (1, 13), (4, 5))]


def test_protected_enemy_neighbour_below() -> None:
    # Protection: enemy on 5 has an enemy on s-1 = 4 -> cannot be swapped (rule 8).
    # (Passing over the 2-run 4-5 is fine: it is not a blockade.)
    assert legal_moves((2, 10), (4, 5), 3) == [fwd(10, 13, (2, 13), (4, 5))]


def test_enemies_two_apart_are_not_protected() -> None:
    # Protection needs s-1 or s+1; an enemy on s+2 does not protect.
    assert legal_moves((1,), (4, 6), 3) == [fwd(1, 4, (4,), (1, 6), "swap")]


def test_26_and_28_are_not_adjacent() -> None:
    # Definitions: 26 and 28 are not neighbours (27 lies between), so an enemy on 28 does not
    # protect an enemy on 26.
    assert legal_moves((24,), (26, 28), 2) == [fwd(24, 26, (26,), (24, 28), "swap")]


def test_formats_md_example_position() -> None:
    # The example record of the move-generation dump in docs/FORMATS.md: 9->11 and the 26->28 swap
    # (the lone enemy on 28 is unprotected, since 27 and 29 are empty).  3->5 is illegal: the enemy
    # on 5 is protected by the enemy on 4.
    assert legal_moves((3, 9, 26), (4, 5, 28), 2) == [
        fwd(9, 11, (3, 11, 26), (4, 5, 28)),
        fwd(26, 28, (3, 9, 28), (4, 5, 26), "swap"),
    ]


# ---------------------------------------------------------------------------
# Blockades (forward rule 4, backward rule 2)
# ---------------------------------------------------------------------------


def test_blockade_squares_definition() -> None:
    # Definitions: a run of >= 3 consecutive enemy pieces; every square of the run counts.
    assert blockade_squares((3, 4, 5)) == {3, 4, 5}
    assert blockade_squares((3, 4)) == set()
    assert blockade_squares((3, 4, 5, 6, 10, 12)) == {3, 4, 5, 6}
    assert blockade_squares((26, 28, 29)) == set()  # 27 is never occupied
    assert blockade_squares((28, 29, 30)) == {28, 29, 30}


def test_blockade_blocks_passing() -> None:
    # Forward rule 4: 1 + 5 = 6 would pass the blockade 3-4-5.
    assert legal_moves((1, 20), (3, 4, 5), 5) == [fwd(20, 25, (1, 25), (3, 4, 5))]
    # A run of four blocks as well.
    assert legal_moves((2, 20), (3, 4, 5, 6), 5) == [fwd(20, 25, (2, 25), (3, 4, 5, 6))]


def test_blockade_only_counts_strictly_between() -> None:
    # Forward rule 4: squares strictly between a and d.  Stopping in front of a blockade is fine.
    assert legal_moves((1,), (3, 4, 5), 1) == [fwd(1, 2, (2,), (3, 4, 5))]
    assert legal_moves((2,), (5, 6, 7), 2) == [fwd(2, 4, (4,), (5, 6, 7))]


def test_two_run_does_not_block_passing() -> None:
    # Definitions: a blockade needs three pieces; a 2-run can be jumped.
    assert legal_moves((1,), (3, 4), 4) == [fwd(1, 5, (5,), (3, 4))]


def test_blocked_forward_falls_back_to_backward() -> None:
    # Forward rule 4 blocks 5 -> 9 over 6-7-8; with no forward move, backward 5 -> 1 is legal.
    assert legal_moves((5,), (6, 7, 8), 4) == [back(5, 1, (1,), (6, 7, 8))]


# ---------------------------------------------------------------------------
# House of Beauty (forward rule 3)
# ---------------------------------------------------------------------------


def test_cannot_pass_26() -> None:
    # Forward rule 3: 24 + 3 = 27 > 26 from below 26 -> illegal; only the other piece moves.
    assert legal_moves((10, 24), (1,), 3) == [fwd(10, 13, (13, 24), (1,))]
    # From 25 nothing but a 1 goes forward.
    for t in (2, 3, 4, 5):
        assert all(m.dir == "back" for m in legal_moves((25,), (1,), t))


def test_must_land_exactly_on_26() -> None:
    # Forward rule 3: landing exactly on 26 is fine.
    assert legal_moves((24,), (1,), 2) == [fwd(24, 26, (26,), (1,))]
    assert legal_moves((25,), (1,), 1) == [fwd(25, 26, (26,), (1,))]


# ---------------------------------------------------------------------------
# House of Water (forward rule 6)
# ---------------------------------------------------------------------------


def test_water_to_15() -> None:
    # Forward rule 6: 26 + 1 = 27 -> the piece is placed on 15.
    assert legal_moves((26,), (1,), 1) == [fwd(26, 15, (15,), (1,), "water")]


def test_water_15_occupied_by_enemy_goes_to_14_without_capture() -> None:
    # Forward rule 6: 15 not empty -> 14; never captures (enemy stays on 15).
    assert legal_moves((26,), (15,), 1) == [fwd(26, 14, (14,), (15,), "water")]


def test_water_15_occupied_by_friend_goes_to_14() -> None:
    # Forward rule 6 with a friendly piece on 15 (which may also move 15 -> 16).
    assert legal_moves((15, 26), (1,), 1) == [
        fwd(15, 16, (16, 26), (1,)),
        fwd(26, 14, (14, 15), (1,), "water"),
    ]


def test_water_15_and_14_occupied_goes_to_13() -> None:
    # Forward rule 6: 15 (enemy) and 14 (friend) occupied -> 13.  (14 -> 15 is a swap: lone enemy.)
    assert legal_moves((14, 26), (15,), 1) == [
        fwd(14, 15, (15, 26), (14,), "swap"),
        fwd(26, 13, (13, 14), (15,), "water"),
    ]


def test_water_ignores_blockade_below() -> None:
    # Forward rule 6: placement is not a move along the track; a blockade on 13-14-15 just means
    # the highest empty square below 15 is 12.
    assert legal_moves((26,), (13, 14, 15), 1) == [fwd(26, 12, (12,), (13, 14, 15), "water")]


# ---------------------------------------------------------------------------
# Leaving 26 (forward rules 2, 4, 5)
# ---------------------------------------------------------------------------


def test_26_plus_5_bears_off() -> None:
    # Forward rule 5: 26 + 5 = 31 -> borne off.
    assert legal_moves((26,), (1,), 5) == [fwd(26, 31, (), (1,), "off")]
    assert legal_moves((10, 26), (1,), 5) == [fwd(10, 15, (15, 26), (1,)), fwd(26, 31, (10,), (1,), "off")]


def test_26_plus_5_blocked_by_blockade_on_28_29_30() -> None:
    # Forward rule 4 (explicitly): bearing off from 26 over an enemy blockade 28-29-30 is illegal.
    assert legal_moves((10, 26), (28, 29, 30), 5) == [fwd(10, 15, (15, 26), (28, 29, 30))]
    # ...and with no other forward move the piece must go backward instead.
    assert legal_moves((26,), (28, 29, 30), 5) == [back(26, 21, (21,), (28, 29, 30))]


def test_26_plus_5_over_two_enemies_is_fine() -> None:
    # Forward rule 4: 28-29 is only a 2-run, not a blockade.
    assert legal_moves((26,), (28, 29), 5) == [fwd(26, 31, (), (28, 29), "off")]


@pytest.mark.parametrize("t,d", [(2, 28), (3, 29), (4, 30)])
def test_26_to_final_squares(t: int, d: int) -> None:
    # Forward rules 2 + 9: 26 + 2/3/4 lands on 28/29/30 (rule 3 only restricts a < 26).
    assert legal_moves((26,), (1,), t) == [fwd(26, d, (d,), (1,))]


@pytest.mark.parametrize("t,d", [(2, 28), (3, 29), (4, 30)])
def test_26_swaps_unprotected_enemy_on_final_square(t: int, d: int) -> None:
    # Forward rule 8 on 28/29/30: a lone enemy there is swapped back to 26.
    assert legal_moves((26,), (1, d), t) == [fwd(26, d, (d,), (1, 26), "swap")]


def test_protection_applies_on_final_squares() -> None:
    # Interpretation note: protection by adjacency applies on 28-30 too.
    assert legal_moves((10, 26), (29, 30), 3) == [fwd(10, 13, (13, 26), (29, 30))]
    assert legal_moves((10, 26), (29, 30), 4) == [fwd(10, 14, (14, 26), (29, 30))]
    # Alone, the 26 piece then has to go backward.
    assert legal_moves((26,), (29, 30), 3) == [back(26, 23, (23,), (29, 30))]


def test_26_to_30_passing_two_enemies() -> None:
    # Forward rule 4: 28-29 is a 2-run, so 26 -> 30 may pass it.
    assert legal_moves((26,), (28, 29), 4) == [fwd(26, 30, (30,), (28, 29))]


# ---------------------------------------------------------------------------
# Final squares (forward rule 1)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("a,need", [(28, 3), (29, 2), (30, 1)])
def test_final_squares_bear_off_only_with_exact_throw(a: int, need: int) -> None:
    # Forward rule 1: only bearing off with exactly 31 - a; nothing else (and no backward move).
    for t in range(1, 6):
        expected = [fwd(a, 31, (), (1,), "off")] if t == need else []
        assert legal_moves((a,), (1,), t) == expected, t


def test_final_square_cannot_step_to_another_final_square() -> None:
    # Forward rule 1 / interpretation note: 28 + 1 does not go to 29.
    assert legal_moves((10, 28), (1,), 1) == [fwd(10, 11, (11, 28), (1,))]
    assert legal_moves((28,), (1,), 2) == []


def test_bear_off_from_28_past_enemies() -> None:
    # Forward rule 1: 28 + 3 bears off even with enemies on 29 and 30.
    assert legal_moves((28,), (29, 30), 3) == [fwd(28, 31, (), (29, 30), "off")]


# ---------------------------------------------------------------------------
# Backward moves
# ---------------------------------------------------------------------------


def test_backward_only_when_no_forward_move_exists() -> None:
    # Backward moves: allowed only if no forward move exists for ANY friendly piece.
    assert legal_moves((10, 25), (1,), 2) == [fwd(10, 12, (12, 25), (1,))]
    assert legal_moves((25,), (1,), 2) == [back(25, 23, (23,), (1,))]


def test_backward_all_pieces_may_move_back() -> None:
    # Backward moves: once forward is impossible, every eligible piece may move back.
    assert legal_moves((24, 25), (1,), 3) == [
        back(24, 21, (21, 25), (1,)),
        back(25, 22, (22, 24), (1,)),
    ]


def test_backward_when_forward_blocked_by_protection() -> None:
    # Backward moves: 5 + 3 = 8 is a protected enemy -> no forward move -> 5 -> 2.
    assert legal_moves((5,), (8, 9), 3) == [back(5, 2, (2,), (8, 9))]


def test_backward_swap_sends_enemy_forward() -> None:
    # Backward rule 3: unprotected enemy on d swaps; the enemy goes forward to a.
    assert legal_moves((25,), (23,), 2) == [back(25, 23, (23,), (25,), "swap")]


def test_backward_protected_enemy_illegal() -> None:
    # Backward rule 3 (rule 8): protected enemy on d -> illegal -> forfeit here.
    assert legal_moves((25,), (22, 23), 2) == []


def test_backward_onto_own_piece_illegal() -> None:
    # Backward rule 3 (rule 7).
    assert legal_moves((23, 25), (1,), 2) == [back(23, 21, (21, 25), (1,))]


def test_backward_blocked_by_blockade() -> None:
    # Backward rule 2: 25 -> 21 would pass the blockade 22-23-24.
    assert legal_moves((25,), (22, 23, 24), 4) == []
    # A 2-run does not block backward moves.
    assert legal_moves((25,), (23, 24), 4) == [back(25, 21, (21,), (23, 24))]


def test_backward_cannot_go_below_1() -> None:
    # Backward rule 1: d = a - t < 1 is illegal.  (2 + 2 = 4 is a protected enemy.)
    assert legal_moves((2,), (4, 5), 2) == []
    # d = 1 is fine.
    assert legal_moves((2,), (3, 4), 1) == [back(2, 1, (1,), (3, 4))]


def test_final_square_pieces_never_move_backward() -> None:
    # Backward moves: only pieces with a <= 26; 30 -> 28 must not appear.
    assert legal_moves((25, 30), (1,), 2) == [back(25, 23, (23, 30), (1,))]
    assert legal_moves((25, 28), (1,), 5) == [back(25, 20, (20, 28), (1,))]


def test_backward_from_26() -> None:
    # Backward moves: 26 is <= 26 and may move back (here forward 26 + 2 hits a protected enemy).
    assert legal_moves((26,), (28, 29), 2) == [back(26, 24, (24,), (28, 29))]


# ---------------------------------------------------------------------------
# No move / forfeit
# ---------------------------------------------------------------------------


def test_forfeit() -> None:
    # "No move": no forward and no backward move -> empty list (turn forfeited).
    assert legal_moves((28,), (1,), 1) == []
    assert legal_moves((1,), (2, 3), 1) == []  # 2 is protected, 1 - 1 = 0 < 1


def test_moves_sorted_and_consistent() -> None:
    # Format: moves sorted by (from, to); every move keeps piece counts and the board valid.
    rng = random.Random(7)
    for _ in range(3000):
        me, opp = random_position(rng)
        for t in range(1, 6):
            moves = legal_moves(me, opp, t)
            assert moves == sorted(moves, key=lambda m: (m.frm, m.to))
            assert len({m.frm for m in moves}) == len(moves)
            for m in moves:
                validate_position(m.me_after, m.opp_after)
                assert len(m.opp_after) == len(opp)
                assert len(m.me_after) == len(me) - (m.kind == "off")
                assert (m.to == 31) == (m.kind == "off")
                assert m.kind != "water" or (m.frm == 26 and t == 1 and m.to <= 15)


class ScriptedSticks(random.Random):
    """A random.Random whose four stick tosses per throw give a scripted throw value."""

    def __new__(cls, throws: list[int]) -> ScriptedSticks:
        # Before Python 3.11, Random.__new__ takes its argument as a seed, which a list cannot be.
        return super().__new__(cls)

    def __init__(self, throws: list[int]) -> None:
        super().__init__(0)
        self.values: list[float] = []
        for t in throws:
            light = 0 if t == 5 else t  # zero light sides counts as 5
            self.values += [0.1] * light + [0.9] * (4 - light)

    def random(self) -> float:
        return self.values.pop(0)  # IndexError if the game asks for more throws than scripted


FIRST_MOVE = (lambda me, opp, t, moves: moves[0],) * 2


def test_throw_sticks_scripted() -> None:
    rng = ScriptedSticks([1, 2, 3, 4, 5])
    assert [throw_sticks(rng) for _ in range(5)] == [1, 2, 3, 4, 5]


def test_turn_extra_throw_after_1() -> None:
    # Turn rule 3: after moving with a 1 the same player throws again; turn rule 5: bearing off
    # the last piece wins immediately.  Player 0: 30 off with 1, then 29 off with 2.
    result = play_game(ScriptedSticks([1, 2]), (29, 30), (28,), FIRST_MOVE)
    assert result == (0, 2, 2, 0)


def test_turn_passes_after_2() -> None:
    # Turn rule 3: after moving with a 2 the turn passes.  Player 0: 26 -> 28; player 1: 30 off with 1.
    result = play_game(ScriptedSticks([2, 1]), (26,), (30,), FIRST_MOVE)
    assert result == (1, 2, 2, 0)


@pytest.mark.parametrize("t", [1, 4, 5])
def test_forfeit_ends_turn_even_after_extra_throw(t: int) -> None:
    # Turn rule 4 / interpretation note: no legal move -> the turn passes even after 1/4/5.
    # Player 0 on 28 cannot use t; player 1 on 30 then bears off with a 1.
    result = play_game(ScriptedSticks([t, 1]), (28,), (30,), FIRST_MOVE)
    assert result == (1, 2, 1, 1)


def test_random_games_finish() -> None:
    # Turn structure: games end when a side bears off its last piece.
    rng = random.Random(12345)
    for _ in range(50):
        result = play_random_game(rng)
        assert result.winner in (0, 1)
        assert result.moves > 0


def test_finished_games_have_no_moves() -> None:
    # Turn rule 5: the game ends when a side bears off its last piece.
    for me, opp in (((30,), ()), ((), (30,))):
        with pytest.raises(ValueError, match="already over"):
            legal_moves(me, opp, 1)
        with pytest.raises(ValueError, match="already over"):
            play_game(random.Random(0), me, opp, FIRST_MOVE)


# ---------------------------------------------------------------------------
# Indexing (docs/FORMATS.md)
# ---------------------------------------------------------------------------


def test_compact() -> None:
    assert [compact(s) for s in (1, 2, 26, 28, 29, 30)] == [0, 1, 25, 26, 27, 28]
    assert [compact(s) for s in USABLE_SQUARES] == list(range(29))
    with pytest.raises(ValueError):
        compact(27)


def test_layer_sizes() -> None:
    assert layer_size(1, 1) == 29 * 28 == 812
    assert layer_size(2, 1) == layer_size(1, 2) == 10_962
    assert layer_size(2, 2) == 406 * 351 == 142_506
    assert layer_size(5, 5) == comb(29, 5) * comb(24, 5) == 118_755 * 42_504 == 5_047_562_520
    for w in range(1, 6):
        for b in range(1, 6):
            assert layer_size(w, b) == comb(29, w) * comb(29 - w, b)


def test_index_hand_examples() -> None:
    # me (28, 30) -> compact (26, 28) -> me_rank = C(26,1) + C(28,2) = 404;
    # opp 29 -> compact 27 -> o' = 27 - 1 = 26 -> opp_rank = 26; index = 404 * C(27,1) + 26.
    assert index_of((28, 30), (29,)) == (2, 1, 404 * 27 + 26)
    assert index_of((1,), (2,)) == (1, 1, 0)
    assert index_of((29, 30), (28,)) == (2, 1, layer_size(2, 1) - 1)


@pytest.mark.parametrize("layer", [(1, 1), (1, 2), (2, 1), (2, 2)])
def test_index_is_bijection_on_small_layers(layer: Layer) -> None:
    w, b = layer
    seen = []
    for me in itertools.combinations(USABLE_SQUARES, w):
        rest = [s for s in USABLE_SQUARES if s not in me]
        for opp in itertools.combinations(rest, b):
            lw, lb, i = index_of(me, opp)
            assert (lw, lb) == layer
            seen.append(i)
            if i % 97 == 0:
                assert position_of(w, b, i) == (me, opp)
    assert sorted(seen) == list(range(layer_size(w, b)))


def test_index_round_trip_random() -> None:
    rng = random.Random(2024)
    for _ in range(20_000):
        me, opp = random_position(rng)
        w, b, i = index_of(me, opp)
        assert (w, b) == (len(me), len(opp)) and 0 <= i < layer_size(w, b)
        assert position_of(w, b, i) == (me, opp)
    for w in range(1, 6):
        for b in range(1, 6):
            for i in (0, layer_size(w, b) - 1, rng.randrange(layer_size(w, b))):
                assert index_of(*position_of(w, b, i)) == (w, b, i)


# ---------------------------------------------------------------------------
# Solver sanity
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tables() -> Tables:
    return solve(K=3, save=False, verbose=False)


def test_values_are_probabilities(tables: Tables) -> None:
    for arr in tables.values():
        assert np.all(np.isfinite(arr))
        assert arr.min() >= 0.0 and arr.max() <= 1.0


def _race(p: float, q: float) -> float:
    """V when I bear off with probability p per throw and the opponent with q, every other throw
    forfeiting the turn: V = p + (1 - p)(1 - q)V."""
    return p / (p + q - p * q)


@pytest.mark.parametrize(
    "me,opp,expected",
    [
        ((28,), (30,), _race(4 / 16, 4 / 16)),  # = 4/7
        ((29,), (30,), _race(6 / 16, 4 / 16)),  # = 12/17
        ((30,), (29,), _race(4 / 16, 6 / 16)),  # = 8/17
        ((29,), (28,), _race(6 / 16, 4 / 16)),
    ],
)
def test_final_square_races_closed_form(
    tables: Tables, me: tuple[int, ...], opp: tuple[int, ...], expected: float
) -> None:
    # Pieces on final squares can only bear off with the exact throw; anything else forfeits the
    # turn, even 1/4/5 (turn rule 4), giving V = p / (p + q - pq).
    assert value(tables, me, opp) == pytest.approx(expected, abs=1e-12)


def test_obvious_positions(tables: Tables) -> None:
    # About to bear off vs. an opponent at the very start: overwhelming favourite.
    assert value(tables, (30,), (1,)) > 0.9
    assert value(tables, (1,), (30,)) < 0.1
    # More pieces left is worse, all else equal.
    assert value(tables, (20, 21), (19,)) < value(tables, (21,), (19,))


def _bellman_direct(tables: Tables, me: Sequence[int], opp: Sequence[int]) -> float:
    """Bellman right-hand side evaluated straight from rules.legal_moves (no precomputed arrays)."""
    total = 0.0
    for t, p in THROW_PROBS.items():
        moves = legal_moves(me, opp, t)
        if not moves:
            q = 1.0 - value(tables, opp, me)
        else:
            q = max(
                1.0
                if not m.me_after
                else value(tables, m.me_after, m.opp_after)
                if t in EXTRA_THROW
                else 1.0 - value(tables, m.opp_after, m.me_after)
                for m in moves
            )
        total += p * q
    return total


def test_bellman_residual(tables: Tables) -> None:
    # The stored values are a fixed point of the Bellman equation (recomputed independently).
    worst = 0.0
    for (w, b), arr in tables.items():
        for i in range(len(arr)):
            me, opp = position_of(w, b, i)
            worst = max(worst, abs(_bellman_direct(tables, me, opp) - arr[i]))
    assert worst < 1e-11


def test_solution_independent_of_initial_values() -> None:
    a = solve(K=2, init=0.0, save=False, verbose=False)
    b = solve(K=2, init=1.0, save=False, verbose=False)
    assert np.max(np.abs(a[(1, 1)] - b[(1, 1)])) < 1e-11


def test_monte_carlo_matches_value(tables: Tables) -> None:
    # Play optimal-vs-optimal games with the simulator (turn structure from rules.play_game) and
    # compare the empirical win rate with V.
    def optimal(me: tuple[int, ...], opp: tuple[int, ...], t: int, moves: list[Move]) -> Move:
        def score(m: Move) -> float:
            if not m.me_after:
                return 1.0
            if t in EXTRA_THROW:
                return value(tables, m.me_after, m.opp_after)
            return 1.0 - value(tables, m.opp_after, m.me_after)

        return max(moves, key=score)

    me, opp = (10, 20), (15,)
    n = 6000
    rng = random.Random(99)
    wins = sum(play_game(rng, me, opp, (optimal, optimal)).winner == 0 for _ in range(n))
    v = value(tables, me, opp)
    sigma = (v * (1 - v) / n) ** 0.5
    assert abs(wins / n - v) < 4 * sigma
