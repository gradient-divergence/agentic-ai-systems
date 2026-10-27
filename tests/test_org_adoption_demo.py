"""Chapter 38's exact regions, offline capture and exercise fixtures."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from demos import regions as _regions
from demos.org_adoption_demo import capture_org_adoption, write_data
from demos.org_adoption_fixtures import read_fixture
from refund_agent.rollout.org_adoption_desk import BalanceFixture, balance_case

ROOT = Path(__file__).resolve().parents[1]
PREFIX = "fnd.org-adoption."


@pytest.fixture(scope="module")
def captured(tmp_path_factory):
    return capture_org_adoption(
        tmp_path_factory.mktemp("org-adoption-capture")
    )


@pytest.mark.parametrize("name", ["plan-01", "balance-01", "keys-01"])
def test_every_output_reproduces_byte_for_byte(name, captured):
    text = captured[PREFIX + name]
    assert (
        text.encode()
        == (
            ROOT / "book_outputs" / f"_{PREFIX}{name}.txt"
        ).read_bytes()
    )
    assert "companion/" not in text and max(map(len, text.splitlines())) <= 79


def test_printed_reconciliation_runs_alone():
    source = ROOT / "refund_agent/rollout/org_adoption_reconcile.py"
    extractor = vars(_regions)
    region = next(
        s
        for s in extractor["extract_snippets_from_text"](
            source.read_text(), source
        )
        if s.snippet_id == PREFIX + "reconcile-01"
    )
    assert max(map(len, region.content.splitlines())) <= 79
    namespace = {}
    exec(compile(region.content, str(source), "exec"), namespace)
    value = json.loads(
        (ROOT / "demos/data/org_adoption_window_03.json").read_text()
    )
    count = namespace["reconcile"](**read_fixture(value))
    assert (
        count.received == 20
        and count.governed == 18
        and len(count.around) == 1
    )


def test_fixtures_regenerate_without_changing_the_exercise_givens(tmp_path):
    write_data(tmp_path)
    for name in (
        "org_adoption_window_03.json",
        "org_adoption_record_05.json",
        "org_adoption_balance_04.json",
    ):
        assert (tmp_path / name).read_bytes() == (
            ROOT / "demos/data" / name
        ).read_bytes()
    given = BalanceFixture.model_validate_json(
        (tmp_path / "org_adoption_balance_04.json").read_text()
    )
    assert (given.paid_cents, given.console_cents, given.request_cents) == (
        9000,
        6000,
        4500,
    )
    assert balance_case(given, early=False)["excess_cents"] == 1500


def test_import_is_silent():
    result = subprocess.run(
        [sys.executable, "-c", "import demos.org_adoption_demo"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0 and not result.stdout and not result.stderr


def test_no_predecessor_result_is_fabricated(captured):
    assert PREFIX + "window-01" not in captured
    assert "Admission validation: hold" in captured[PREFIX + "plan-01"]
    assert "10500 cents against 9000 paid" in captured[PREFIX + "keys-01"]
