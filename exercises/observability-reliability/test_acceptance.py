"""Reader acceptance: the untouched daily-rate starter fails these assertions."""

import math
import os
import runpy
from datetime import timedelta
from pathlib import Path

import pytest
from quotes_06 import START, market_open

SOLUTION = runpy.run_path(
    str(
        Path(
            os.environ.get(
                "OBSERVABILITY_RELIABILITY_SOLUTION", Path(__file__).with_name("detector_06.py")
            )
        )
    )
)
run_detector = SOLUTION["run_detector"]


@pytest.fixture(scope="module")
def pages():
    return run_detector()


def test_clean_days_and_market_closes_do_not_page(pages):
    assert all(market_open(p["at"]) for p in pages), "false page at market close"
    assert all(p["at"].date() == (START + timedelta(days=2)).date() for p in pages)


def test_planted_silence_pages_within_twenty_seconds(pages):
    start = START + timedelta(days=2, hours=11)
    assert any(p["venue"] == 1 and start <= p["at"] <= start + timedelta(seconds=20) for p in pages)


def test_threshold_uses_open_hours_and_one_expected_false_page():
    for rate, hours, false in ((3600, 325, 1), (1800, 325, 1), (3600, 65, 0.2)):
        assert SOLUTION["quiet_hours"](rate, hours, false) == pytest.approx(
            math.log(rate * hours / false) / rate
        )
