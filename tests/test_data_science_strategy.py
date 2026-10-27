"""Tests for the listings of the chapter on decision strategy and evidence requirements."""

import contextlib
import importlib
import io
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from demos import data_science_strategy_records as records
from demos import regions as _regions

COMPANION = Path(__file__).resolve().parents[1]
DEMO = COMPANION / "demos/data_science_strategy_demo.py"
PREFIX = "fnd.data-science-strategy."


def demo():
    return importlib.import_module("demos.data_science_strategy_demo")


@pytest.fixture(scope="module")
def captured_outputs() -> dict[str, str]:
    extractor = vars(_regions)
    snippets = {
        snippet.snippet_id: snippet
        for snippet in extractor["extract_snippets_from_text"](DEMO.read_text(), DEMO)
        if snippet.snippet_id in {PREFIX + "bound-01", PREFIX + "bound-02", PREFIX + "record-02"}
    }
    return extractor["capture_outputs"](snippets, sys.executable)


def test_import_prints_nothing_and_writes_nothing(tmp_path: Path) -> None:
    code = (
        "import contextlib, io, os, sys\n"
        f"sys.path.insert(0, {str(COMPANION)!r})\n"
        "buffer = io.StringIO()\n"
        "with contextlib.redirect_stdout(buffer):\n"
        "    from demos.data_science_strategy_demo import EvidenceRequirement, check\n"
        "assert buffer.getvalue() == '', buffer.getvalue()\n"
        "assert os.listdir('.') == [], os.listdir('.')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=tmp_path, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


def test_bound_is_exact_and_close_to_three_over_n() -> None:
    bound, needed = demo().zero_failure_bound, demo().cases_needed
    for n in (10, 30, 40, 100, 300, 1000):
        rate = bound(n)
        assert (1 - rate) ** n == pytest.approx(0.05)
        assert rate < 3 / n
        if n >= 30:  # the two differ by less than half a percentage point
            assert 3 / n - rate < 0.005
    assert bound(40) == pytest.approx(0.072157524506)
    assert needed(0.01) == 299
    assert bound(299) < 0.01 < bound(298)
    assert needed(0.02) == 149
    assert needed(0.005) == 598
    assert bound(598) < 0.005 < bound(597)
    # Halving the 40-case bound takes 82 cases, about twice as many.
    assert needed(bound(40) / 2) == 82


@pytest.mark.parametrize("n", [0, 10, 40, 100, 300, 1000])
def test_flat_prior_bound_leaves_five_percent_posterior_tail(n: int) -> None:
    bound = demo().flat_prior_bound(n)
    assert (1 - bound) ** (n + 1) == pytest.approx(0.05)
    assert (1 - demo().flat_prior_bound(n, 0.1)) ** (n + 1) == pytest.approx(0.1)


def test_flat_prior_output_compares_bounds_and_the_minimum_run(captured_outputs) -> None:
    assert PREFIX + "bound-02" in captured_outputs
    lines = captured_outputs[PREFIX + "bound-02"].splitlines()
    rows = [line.split() for line in lines[2:7]]
    assert rows == [
        ["10", "0.2589", "0.2384"],
        ["40", "0.0722", "0.0705"],
        ["100", "0.0295", "0.0292"],
        ["300", "0.0099", "0.0099"],
        ["1000", "0.0030", "0.0030"],
    ]
    assert lines[-2] == "40 clean cases: frequentist=7.2%; flat prior=7.0%"
    assert lines[-1] == "cases for a bound of 0.01: frequentist=299; flat prior=298"
    assert demo().flat_prior_bound(298) < 0.01 < demo().flat_prior_bound(297)


def test_flat_prior_listing_runs_alone(tmp_path: Path, captured_outputs) -> None:
    extractor = vars(_regions)
    snippets = extractor["extract_snippets_from_text"](DEMO.read_text(), DEMO)
    regions = {snippet.snippet_id: snippet for snippet in snippets}
    assert PREFIX + "bound-02" in regions
    snippet = regions[PREFIX + "bound-02"]
    result = subprocess.run(
        [sys.executable, "-c", snippet.content],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == captured_outputs[PREFIX + "bound-02"]
    assert list(tmp_path.iterdir()) == []


def test_check_accepts_signed_in_advance_and_refuses_late_or_empty() -> None:
    module = demo()
    requirement, check = module.EvidenceRequirement, module.check
    verdicts = [check(requirement(**record), day) for record, day in records.CASES]
    assert verdicts == [[], ["signed after its run began"], ["no threshold"]]
    accepted = records.CASES[0][0]
    assert accepted["step"] == "tier 1, drafts to shadow"
    on_the_day = {**accepted, "signed_day": 10}
    assert check(requirement(**on_the_day), 10) == ["signed after its run began"]
    one_measure = {**accepted, "measures": ["Forbidden payments"]}
    assert check(requirement(**one_measure), 10) == ["no measure pair"]
    no_measures = {**accepted, "measures": []}
    assert check(requirement(**no_measures), 10) == ["no measures", "no measure pair"]
    assert check(requirement(**records.SHADOW_TO_CANARY), 10) == []


@pytest.mark.parametrize(
    "extra_measures", [[], ["Wrongful denials"], ["Wrongful denials", "Delay"]]
)
def test_check_accepts_the_pair_and_further_required_measures(extra_measures) -> None:
    module = demo()
    requirement = module.EvidenceRequirement(**records.DRAFTS_TO_SHADOW)
    requirement = replace(requirement, measures=[*requirement.measures, *extra_measures])
    assert module.check(requirement, 10) == []


@pytest.mark.parametrize(
    "measures",
    [
        ["", "Referral share"],
        ["Forbidden payments", ""],
        ["Forbidden payments", "Referral share", ""],
    ],
)
def test_check_refuses_an_empty_measure_even_after_the_pair(measures) -> None:
    module = demo()
    requirement = replace(module.EvidenceRequirement(**records.DRAFTS_TO_SHADOW), measures=measures)
    assert module.check(requirement, 10) == ["no measure pair"]


def test_canary_requires_wrongful_denials_against_the_baseline_only() -> None:
    requirement = records.SHADOW_TO_CANARY
    assert (
        "no more wrongful denials than the baseline on the same 299 requests"
        in requirement["claim"]
    )
    assert requirement["measures"] == [
        "Forbidden payments the agent would have made",
        "Share of tier 1 requests the agent sent to a person",
        "Wrongful denials: requests the agent denied that the person paid, "
        "with the payment not found false within 30 days",
    ]
    assert requirement["comparator"] == (
        "The baseline on the same requests; the person's decision settles whether a denial was wrongful"
    )
    assert (
        "wrongful denials no more than the baseline's on the same requests"
        in requirement["threshold"]
    )
    assert "no higher than the baseline's on the same requests" in requirement["threshold"]
    assert "next 299 live tier 1" in requirement["cases"]


def test_signed_requirement_output_has_all_seven_fields_and_both_days(captured_outputs) -> None:
    assert PREFIX + "record-02" in captured_outputs
    lines = captured_outputs[PREFIX + "record-02"].splitlines()
    assert lines[0] == "| Field | Tier 1, drafts to shadow |"
    assert all(len(line) <= 79 for line in lines)
    rows = {}
    label_before = None
    for line in lines[2:]:
        if not line.startswith("|"):
            break
        label, value = [cell.strip() for cell in line.strip("|").split("|")]
        if label:
            assert label not in rows
            rows[label] = value
        else:
            assert label_before is not None
            rows[label_before] += " " + value
        label_before = label or label_before
    assert set(rows) == {
        "Claim",
        "Comparator",
        "Cases",
        "Measures",
        "Threshold",
        "Signer",
        "Reversal",
    }
    for label, value in rows.items():
        expected = records.DRAFTS_TO_SHADOW[label.lower()]
        assert value == ("; ".join(expected) if isinstance(expected, list) else expected)
    assert "; share of tier 1 scenarios sent to a person" in rows["Measures"]
    assert "signed on day 8 for a run from day 10" in " ".join(" ".join(lines).split())
    assert lines[-1].endswith("{#tbl-data-science-strategy-drafts-to-shadow}")


def test_shared_failure_output_counts_one_case_instead_of_forty(captured_outputs) -> None:
    lines = captured_outputs[PREFIX + "bound-01"].splitlines()
    assert lines[-3:] == [
        "40 clean cases (constructed comparison):",
        "  independent: effective cases 40; bound 0.0722",
        "  one shared failure: effective cases 1; bound 0.95",
    ]
    assert demo().zero_failure_bound(1) == pytest.approx(0.95)


def test_exercise_5_passes_the_check_so_its_error_must_be_found_by_reading() -> None:
    module = demo()
    planted = module.EvidenceRequirement(**records.EXERCISE_5)
    assert module.check(planted, records.EXERCISE_5_RUN_START_DAY) == []
    differs = {
        key for key, value in records.EXERCISE_5.items() if value != records.SHADOW_TO_CANARY[key]
    }
    assert differs == {"cases", "signed_day"}
    no_undo = ("refund reversed", "refunds reversed", "undo", "claw back", "paid back")
    for record in (records.DRAFTS_TO_SHADOW, records.SHADOW_TO_CANARY, records.TIER_2):
        assert not any(phrase in record["reversal"].lower() for phrase in no_undo)


def test_canary_cases_prespecify_one_version_and_a_policy_for_searching_versions() -> None:
    cases = records.SHADOW_TO_CANARY["cases"]
    assert "frozen version" in cases
    assert "prespecified" in cases
    assert "unchanged" not in cases
    assert "repeated-testing policy" in cases


def test_scenario_tiers_and_baseline_line(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    with contextlib.redirect_stdout(io.StringIO()):
        lines = demo().baseline_lines().splitlines()
    assert lines[0].startswith("baseline: 40/40 agreement with generated labels")
    assert lines[1].endswith("executed=0")
    assert lines[2] == "public scenarios by tier: tier 1=15 tier 2=10 tier 3=15"
    assert "15 tier 1 requests" in records.DRAFTS_TO_SHADOW["cases"]
