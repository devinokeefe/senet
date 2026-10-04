"""The strength corpus (models/strength_positions.json) and its check (senet_train.strength)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from conftest import NET, needs_net

from senet import legal_moves, mask
from senet_train import strength


def corpus() -> dict[str, Any]:
    data: dict[str, Any] = json.loads(strength.CORPUS.read_text(encoding="utf-8"))
    return data


def test_blockades() -> None:
    assert strength.has_blockade(mask([3, 4, 5]))
    assert strength.has_blockade(mask([1, 20, 21, 22]))
    assert not strength.has_blockade(mask([3, 4, 6, 7]))
    assert not strength.has_blockade(0)


def test_categories_of_hand_made_decisions() -> None:
    water = legal_moves([20, 26], [1], 1)  # 26 -> 27 drowns
    assert strength.categories(mask([1]), water, [0.5, 0.6]) == ["water"]
    assert strength.categories(mask([1]), water, [0.5, 0.5 + strength.NEAR_TIE / 2]) == ["water", "near-tie"]
    assert strength.categories(mask([1]), water, [0.5, 0.5 + strength.NEAR_TIE * 1.5]) == ["water"]
    assert strength.categories(mask([1]), water, [0.5, 0.5]) == ["water"]  # a tie is no near-tie
    bear_off = legal_moves([10, 30], [1, 2, 3], 1)
    assert strength.categories(mask([1, 2, 3]), bear_off, [0.9, 0.2]) == ["blockade", "bear-off"]


def test_the_corpus_is_what_it_says() -> None:
    data = corpus()
    assert data["near_tie"] == strength.NEAR_TIE
    decisions = data["decisions"]
    assert len({(tuple(d["me"]), tuple(d["opp"]), d["t"]) for d in decisions}) == len(decisions)
    for d in decisions:
        moves = legal_moves(d["me"], d["opp"], d["t"])
        assert len(moves) == len(d["values"]) >= 2
        assert all(0 <= v <= 1 for v in d["values"])
        found = strength.categories(mask(d["opp"]), moves, d["values"])
        assert [c for c in d["categories"] if c != "typical"] == found, d
    for c in strength.CATEGORIES:
        assert sum(c in d["categories"] for d in decisions) >= strength.KEEP, c
    stats = data["reference"]["stats"]
    assert set(stats) == set(strength.BOTS) and all(set(stats[bot]) == set(strength.CATEGORIES) for bot in stats)


@needs_net
def test_the_check_fails_over_a_limit(tmp_path: Path) -> None:
    data = corpus()
    data["decisions"] = data["decisions"][:60]
    path = tmp_path / "corpus.json"

    def run(*args: str) -> int:
        return strength.main(["--net", str(NET), "--corpus", str(path), *args])

    def check_against(loss: float) -> int:
        stats = {bot: {c: {"n": 1, "mean": loss, "worst": loss} for c in strength.CATEGORIES} for bot in strength.BOTS}
        data["reference"]["stats"] = stats
        strength.write(path, data)
        return run()

    assert check_against(1.0) == 0
    assert check_against(-1.0) == 1  # limits below 0
    assert run("--rebase") == 0
    rebased = json.loads(path.read_text(encoding="utf-8"))
    assert rebased["decisions"] == data["decisions"] and rebased["reference"]["file"] == NET.name
    assert run() == 0
