"""Check a submitted build against the chapter's acceptance conditions."""

import os
import runpy
from pathlib import Path

import pytest

PATH = Path(os.environ.get("SUPERVISED_LEARNING_SOLUTION", Path(__file__).with_name("starter.py")))
SOLUTION = runpy.run_path(str(PATH))


@pytest.fixture(scope="module")
def result(tmp_path_factory):
    return SOLUTION["build"](tmp_path_factory.mktemp("exercise"), seed=7)


def test_august_matches_the_printed_calibration_row(result):
    assert result["threshold"] == pytest.approx(0.080)
    assert result["requests"] == 999
    assert round(result["calibration_error"], 4) == 0.0170
    assert result["referred"] == 125
    assert round(result["cost"], 2) == 0.82
    assert result["cost"] < result["placeholder_cost"]
    assert result["saving"] == pytest.approx(result["placeholder_cost"] - result["cost"])


def test_map_precedes_one_frozen_read_and_scores_never_deny(result):
    assert result["events"] == ["fit training", "fit validation map", "read August"]
    assert len(result["golden"].log) == 1
    assert set(result["actions"]) <= {"pay", "refer"}
    assert 0 < result["referred"] / result["requests"] < 1
    assert result["judgment"].strip() and result["measurement"].strip()


def test_seed_seven_reproduces_every_reported_digit(result, tmp_path):
    again = SOLUTION["build"](tmp_path, seed=7)
    for field in (
        "threshold",
        "requests",
        "calibration_error",
        "referred",
        "cost",
        "placeholder_cost",
        "saving",
    ):
        assert result[field] == again[field]
