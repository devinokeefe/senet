"""The benchmark and strategy reports, on hand-made inputs (no database needed)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import senet
from senet import _lib, bench, insights
from senet import provenance as prov
from senet._lib import ROOT
from senet.provenance import provenance
from senet.provenance import render as render_provenance


def _quality(
    games: int, decisions: int, errors: int, per_decision: float, per_game: float, worst: float
) -> dict[str, float]:
    return {
        "games": games,
        "decisions": decisions,
        "errors": errors,
        "avg_loss_per_decision": per_decision,
        "avg_loss_per_game": per_game,
        "max_loss": worst,
    }


RESULTS: dict[str, Any] = {
    "start": {"white_win_prob": 0.50250401},
    "quality": {
        "greedy": _quality(20_000, 1_060_203, 503_292, 0.0031517, 0.1670733, 0.3039033),
        "expectimax:3": _quality(5_000, 264_746, 102_531, 0.0015057, 0.0797281, 0.1849752),
        "net:2": _quality(400, 26_000, 1_300, 0.00001, 0.0006, 0.004),
        "perfect": _quality(20_000, 1_316_196, 0, 0.0, 0.0, 0.0),
    },
    "matches": {
        "perfect vs greedy": {"games": 100_000, "a_win_rate": 0.66852, "ci95": 0.00292},
        "perfect vs expectimax:3": {"games": 25_000, "a_win_rate": 0.58028, "ci95": 0.00612},
    },
}


def _tables(markdown: str) -> list[list[list[str]]]:
    """Every markdown table as a list of rows of stripped cells."""
    tables: list[list[list[str]]] = []
    previous_was_row = False
    for line in markdown.splitlines():
        is_row = line.startswith("|")
        if is_row:
            if not previous_was_row:
                tables.append([])
            tables[-1].append([cell.strip() for cell in line.strip("|").split("|")])
        previous_was_row = is_row
    return tables


def test_bench_render() -> None:
    md = bench.render(RESULTS)
    assert md.startswith("# Benchmarks\n") and md.endswith("\n")
    assert "**White (moving first) wins 50.2504%** of games." in md
    assert "Perfect self-play, which checks this value against actual play, is in docs/INSIGHTS.md." in md
    assert "An error is a decision that gives away more than 0.0002% (2e-6), twice\nthe database's" in md

    quality, matches = _tables(md)
    assert quality[0] == [
        "Bot",
        "Games",
        "Decisions",
        "Error rate",
        "Avg loss / decision",
        "Avg loss / game",
        "Worst single loss",
    ]
    assert quality[2:] == [
        ["Greedy heuristic (1 ply)", "20,000", "1,060,203", "47.47%", "0.315%", "16.71%", "30.4%"],
        ["Heuristic + expectimax, 3 throws", "5,000", "264,746", "38.73%", "0.151%", "7.97%", "18.5%"],
        ["net:2", "400", "26,000", "5.00%", "0.001%", "0.06%", "0.4%"],  # unknown spec: shown as is
        ["Perfect play (solved database)", "20,000", "1,316,196", "0.00%", "0.000%", "0.00%", "0.0%"],
    ]
    assert matches[0] == ["Match", "Games", "Pairs", "Perfect's win rate", "95% CI"]
    assert matches[2:] == [
        ["Perfect vs Greedy heuristic (1 ply)", "100,000", "50,000", "66.85%", "± 0.29%"],
        ["Perfect vs Heuristic + expectimax, 3 throws", "25,000", "12,500", "58.03%", "± 0.61%"],
    ]
    for table in (quality, matches):
        assert all(len(row) == len(table[0]) for row in table)
    assert "Provenance" not in md and "raw results" not in md
    assert bench.render(json.loads(json.dumps(RESULTS))) == md


def test_reports_record_their_provenance() -> None:
    origin = provenance(seeds=bench.SEEDS)
    md = bench.render({**RESULTS, "provenance": origin}, "BENCHMARKS.json")
    assert "; the raw results are in [BENCHMARKS.json](BENCHMARKS.json). Ruleset" in md
    assert md.endswith(render_provenance(origin))
    md = insights.render({**INSIGHTS, "provenance": origin}, "../docs/INSIGHTS.json")
    assert "The raw results are in [INSIGHTS.json](../docs/INSIGHTS.json)." in md
    assert md.endswith(render_provenance(origin))


def test_bench_output_paths() -> None:
    args = bench.parse_args([])
    assert (args.out, args.json) == (ROOT / "docs" / "BENCHMARKS.md", ROOT / "docs" / "BENCHMARKS.json")
    # --quick is for smoke tests: it must not overwrite the published results.
    args = bench.parse_args(["--quick"])
    assert (args.out, args.json) == (ROOT / "runs" / "bench_quick.md", ROOT / "runs" / "bench_quick.json")
    args = bench.parse_args(["--quick", "--out", "a.md", "--json", "b.json"])
    assert (args.out, args.json) == (Path("a.md"), Path("b.json"))


def test_an_installed_copy_works_in_the_current_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Installed (not editable), the package is in no checkout: the reports' default paths are
    # relative to the current directory, not to the installation, and no commit is recorded
    # (git would find whatever repository encloses the installation).
    monkeypatch.setattr(_lib, "ROOT", tmp_path / "site-packages")
    assert _lib.checkout() is None and prov.home() == Path()
    args = bench.parse_args([])
    assert (args.db, args.out) == (Path("db/kendall5"), Path("docs/BENCHMARKS.md"))
    assert prov.git_commit() is None and provenance()["commit"] is None
    assert prov.shown("db/kendall5") == Path("db/kendall5").resolve().as_posix()


@pytest.mark.parametrize(
    ("me", "opp", "t", "text"),
    [
        ([1, 3, 5, 7, 9], [2, 4, 6, 8, 10], 2, "9→11"),
        ([1], [4], 3, "1→4 swap"),
        ([25], [1], 2, "25→23 (backward)"),
        ([25], [23], 2, "25→23 swap (backward)"),
        ([26], [1], 1, "26→27 (drowns, back to 15)"),
        ([26], [1, 15], 1, "26→27 (drowns, back to 14)"),
        ([28], [1], 3, "28→off"),
    ],
)
def test_insights_describe(me: list[int], opp: list[int], t: int, text: str) -> None:
    (move,) = senet.legal_moves(me, opp, t)
    assert insights.describe(move) == text


INSIGHTS: dict[str, Any] = {
    "white_win_prob": 0.50250401,
    "openings": {
        1: [("1→2 swap", 0.51191), ("3→4 swap", 0.50747)],
        2: [("9→11", 0.48703)],
        4: [("7→11", 0.503494), ("9→13", 0.503494 - 1e-12), ("1→5", 0.49)],  # a tie within 1e-9: both optimal
    },
    "selfplay": {
        "games": 20_000,
        "white_wins": 10_114,
        "avg_throws": 197.3,
        "kinds": {"move": 0.714, "swap": 0.138, "move-back": 0.05, "swap-back": 0.0337, "water": 0.0197, "off": 0.044},
        "forfeits_per_game": 3.4,
        "choices_per_game": 131.8,
        "comeback_games": {0.25: 4088, 0.10: 492, 0.05: 27, 0.01: 0},
    },
}


def test_insights_render() -> None:
    md = insights.render(INSIGHTS)
    assert md.startswith("# What the solution says\n") and md.endswith("\n")
    assert "**White, who throws first, wins 50.2504%** of games, and Black wins 49.7496%." in md
    assert "Moving first outweighs Black's head start (front piece on 10 vs 9): White leads by 0.50 points." in md

    (openings,) = _tables(md)
    assert openings[0] == ["Throw (prob.)", "Move", "White wins"]
    assert openings[2:] == [
        ["1 (4/16)", "1→2 swap ★", "51.191%"],
        ["", "3→4 swap", "50.747%"],
        ["2 (6/16)", "9→11 ★", "48.703%"],
        ["4 (1/16)", "7→11 ★", "50.349%"],
        ["", "9→13 ★", "50.349%"],
        ["", "1→5", "49.000%"],
    ]
    assert "## Perfect self-play (20,000 games)\n\nRanges are 95% intervals.\n" in md
    assert "* White won 50.57% (49.88–51.26%); the database's value is 50.25%." in md
    assert "* A game lasts 197 throws on average, with 132 real decisions (two or more legal moves) and 3.4" in md
    assert "* Moves played: 71.4% ordinary, 13.8% swaps, 8.4% forced backward, 2.0% into the House of Water, " in md
    assert (
        "* Comebacks: winners had been below a 25% chance in 20.4% of games (19.9–21.0%), below 10% in 2.5% "
        "(2.3–2.7%), below 5% in 0.14% (0.09–0.20%), never below 1%." in md
    )
    # The report is made from the results as read back from their JSON, whose keys are strings.
    assert insights.render(json.loads(json.dumps(INSIGHTS))) == md
    black = insights.render({**INSIGHTS, "white_win_prob": 0.49})
    assert "Black's head start (front piece on 10 vs 9) outweighs moving first: Black leads by 2.00 points." in black


def test_intervals() -> None:
    assert insights.wilson(1, 10) == pytest.approx((0.017876, 0.404156), abs=1e-6)  # the textbook example
    lo, hi = insights.wilson(0, 20_000)
    assert lo == 0 and hi == pytest.approx(1.92e-4, rel=1e-2)
    assert insights.wilson(20_000, 20_000)[1] == 1
    assert [insights.places(p) for p in (0.714, 0.01, 0.0099, 0.00135, 0.0009, 0.0)] == [1, 1, 2, 2, 3, 1]
    assert insights.share(27, 20_000) == "0.14% (0.09–0.20%)"
    assert insights.share(10_114, 20_000, decimals=2) == "50.57% (49.88–51.26%)"


def test_insights_needs_a_game(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        insights.main(["--games", "0", "--out", str(tmp_path / "a.md"), "--json", str(tmp_path / "a.json")])
    assert "--games must be at least 1" in capsys.readouterr().err
    assert not list(tmp_path.iterdir())


def test_write_report(tmp_path: Path) -> None:
    out, raw = tmp_path / "docs" / "report.md", tmp_path / "data" / "report.json"
    results = {"by_count": {1: 0.5}}

    def render(r: dict[str, Any], link: str) -> str:
        assert r == {"by_count": {"1": 0.5}}  # rendered from the JSON as written
        return f"see {link}\n"

    prov.write_report(results, out, raw, render)
    assert json.loads(raw.read_text(encoding="utf-8")) == {"by_count": {"1": 0.5}}
    assert out.read_text(encoding="utf-8") == "see ../data/report.json\n"
    assert b"\r" not in out.read_bytes() + raw.read_bytes()
