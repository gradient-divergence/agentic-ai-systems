"""Behavior and output checks for the causal chapter's plan, never a live test."""

import importlib
import io
import json
import math
import os
import re
import runpy
import subprocess
import sys
from contextlib import redirect_stdout
from dataclasses import replace
from datetime import timedelta
from pathlib import Path

import numpy as np
import pytest

from demos import regions as _regions

COMPANION = Path(__file__).resolve().parents[1]
_EXTRACT = vars(_regions)
capture_outputs = _EXTRACT["capture_outputs"]
extract_snippets_from_text = _EXTRACT["extract_snippets_from_text"]
DEMO = COMPANION / "demos" / "causal_inference_demo.py"
OUTPUTS = COMPANION / "book_outputs"
PREFIX = "fnd.causal-inference."
PRINTED = ("adjust-01", "assign-01")
REGIONS = (*PRINTED[:1], "support-01", "review-01", "did-01", "plan-01", "plan-full-01", "plan-02",
           "assign-01", "size-01", "numbers-01", "keys-01", "figures-01")


def demo():
    assert DEMO.is_file(), "the chapter's demo is missing"
    return importlib.import_module("demos.causal_inference_demo")


def snippets():
    assert DEMO.is_file(), "the chapter's demo is missing"
    return {s.snippet_id: s for s in extract_snippets_from_text(DEMO.read_text(), DEMO)}


@pytest.fixture(scope="module")
def captured():
    return capture_outputs(snippets(), sys.executable)


@pytest.fixture(scope="module")
def planning():
    return demo().planning_history()


def test_printed_listings_run_in_chapter_order_on_their_own(tmp_path):
    marked = snippets()
    script = tmp_path / "printed_listings.py"
    printed = "\n\n".join(marked[PREFIX + name].content for name in PRINTED)
    # Exercise the function defined in the printed listing, as well as its helper's
    # rehearsal. An undeclared global in the printed function must fail here.
    printed += "\nassert [assign(rehearsal_plan, r, float(p)) for r, p in "
    printed += "zip(rehearsal['validation'], rehearsal['score'])] == records\n"
    script.write_text(printed)
    # A reader's setting, as in chapter 5's test: the companion importable, nothing else.
    env = {**os.environ, "PYTHONPATH": str(COMPANION)}
    result = subprocess.run([sys.executable, str(script)], cwd=tmp_path, env=env,
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    expected = b"".join((OUTPUTS / f"_{PREFIX}{name}.txt").read_bytes() for name in PRINTED)
    assert result.stdout.encode() == expected


def test_hospital_and_remediation_counts():
    d = demo()
    hospital = [("high", 400, 120, 100, 40), ("low", 100, 10, 400, 80)]
    assert d.naive_effect(hospital) == pytest.approx(0.02)
    assert d.standardized_effect(hospital) == pytest.approx(-0.10)
    incidents = [("critical", 200, 60, 50, 10), ("other", 50, 40, 200, 150)]
    assert d.naive_effect(incidents) == pytest.approx(-0.24)
    assert d.standardized_effect(incidents) == pytest.approx(0.075)
    with pytest.raises(ValueError, match="both actions"):
        d.standardized_effect([("unsupported", 0, 0, 10, 2)])
    with pytest.raises(ValueError):
        d.naive_effect([])


def test_seeded_hospital_samples_and_intervals():
    d = demo()
    assert d.hospital_samples(1000, seed=7) == d.hospital_samples(1000, seed=7)
    assert d.hospital_samples(1000, seed=8) != d.hospital_samples(1000, seed=7)
    counts = d.hospital_samples(100000, seed=7)
    naive, adjusted = d.effect_intervals(counts)
    assert naive[1] > -0.10 and abs(naive[0] - 0.02) < 0.01
    assert abs(adjusted[0] + 0.10) < 0.01
    assert naive[1] < naive[0] < naive[2]
    assert adjusted[1] < adjusted[0] < adjusted[2]


def test_review_saving_prices_both_kinds_of_error():
    d = demo()
    y, amount = np.array([1, 0, 1, 0]), np.array([40., 40., 10., 10.])
    referred = np.array([True, True, False, False])
    assert d.review_saving(y, amount, referred, q=0, f=0) == 8.0
    assert d.review_saving(y, amount, referred, q=0.2, f=0.1) == 5.5
    assert d.review_saving(y, amount, referred, q=1, f=1) == -7.0
    with pytest.raises(ValueError):
        d.review_saving(y, amount, referred, q=-0.1, f=0)


def test_did_and_sample_size_hand_worked_values():
    d = demo()
    assert d.difference_in_differences(.12, .18, .11, .13) == pytest.approx(.04)
    assert d.difference_in_differences(.05, .038, .048, .046) == pytest.approx(-.010)
    assert d.requests_per_arm(12, 2) == 566
    assert d.requests_per_arm(10, 2.5) == 252
    assert d.requests_per_arm(12, 1) == 4 * 565 + 1
    for sd, delta in ((-1, 2), (10, 0), (float("nan"), 2)):
        with pytest.raises(ValueError):
            d.requests_per_arm(sd, delta)


def test_did_worked_case_hand_worked_values(captured):
    output = captured[PREFIX + "did-01"]
    assert output.splitlines() == [
        "Share resolved within the hour: switched 0.12 to 0.18; others 0.11 to 0.13",
        "Before and after at switched site +0.06; after only, switched less others +0.05",
        "Difference-in-differences: (0.18 - 0.12) - (0.13 - 0.11) = +0.04",
        "Switched site untreated after: 0.12 + (0.13 - 0.11) = 0.14; gap to 0.18 = +0.04",
        "Switched site's effect if: one action version; others untreated, unaffected;",
        "same kind of unit counted the same way; parallel trends; no anticipation.",
    ]
    assert all(len(line) <= 79 for line in output.splitlines())


@pytest.mark.parametrize("field,value", [
    ("refer_probability", 0), ("refer_probability", 1), ("refer_probability", float("nan")),
    ("dollar_ceiling", None), ("dollar_ceiling", 0), ("dollar_ceiling", float("inf")),
    ("segments", ()), ("segments", []), ("segments", ("",)), ("owner", ""), ("outcome", ""),
    ("pay_action", ""), ("refer_action", ""), ("eligibility", ""), ("estimand", ""),
    ("assignment_unit", ""), ("policy_version", ""), ("denial_check", ""),
    ("stop_rule", ""), ("cap", 4), ("label_window_days", 0),
])
def test_plan_refuses_missing_or_unusable_fields(field, value):
    d = demo()
    with pytest.raises(ValueError):
        d.check_plan(replace(d.ThresholdExperimentPlan(), **{field: value}))


def test_plan_signature_and_label_maturity():
    d = demo()
    plan = d.ThresholdExperimentPlan()
    assert d.check_plan(plan) == "accepted"
    for changed in (
        replace(plan, analysis_at=plan.last_assignment_at + timedelta(days=29)),
        replace(plan, signed_at=plan.first_assignment_at),
        replace(plan, signed_at=None),
        replace(plan, last_assignment_at=plan.first_assignment_at - timedelta(days=1)),
    ):
        with pytest.raises(ValueError):
            d.check_plan(changed)


def test_assignment_boundaries_customer_identity_and_no_outcome_read(planning):
    d = demo()
    row = planning["validation"][0]
    plan = d.ThresholdExperimentPlan()
    row = row.model_copy(update={"amount_cents": 4000})
    assert d.assign(plan, row, .079)["action"] == "pay"
    assert d.assign(plan, row, .099)["action"] == "refer"  # the gate's extra referral
    for score, inside in ((.1, True), (.249, True), (.25, False)):
        record = d.assign(plan, row, score)
        assert record["in_set"] is inside
        assert record["probability"] == (0.5 if inside else 1.0)
        if not inside:
            assert record["action"] == "refer"
    gate_extra = row.model_copy(update={"amount_cents": 1000})
    assert d.assign(plan, gate_extra, .1)["action"] == "refer"
    # Different requests of the same customer see the same coin.
    another = row.model_copy(update={"request_id": "another-request", "found_false_30d": None})
    assert d.assign(plan, another, .15)["action"] == d.assign(plan, row, .15)["action"]
    class WithoutOutcome:
        def __getattr__(self, name):
            if name in ("found_false_30d", "label_final_at"):
                raise AssertionError("assignment read an outcome")
            return getattr(row, name)
    assert d.assign(plan, WithoutOutcome(), .15) == d.assign(plan, row, .15)
    for field, value in (("amount_cents", 5000), ("policy_decision", "deny"),
                         ("order_evidence_age_seconds", 1801)):
        record = d.assign(plan, row.model_copy(update={field: value}), .15)
        assert not record["in_set"] and record["probability"] == 1
    with pytest.raises(ValueError):
        d.assign(plan, row, float("nan"))


def test_assignment_refuses_a_degenerate_coin_before_returning_a_record(planning):
    d = demo()
    row = planning["validation"][0].model_copy(update={"amount_cents": 4000})
    for probability in (0, 1, float("nan")):
        with pytest.raises(ValueError):
            d.assign(replace(d.ThresholdExperimentPlan(), refer_probability=probability), row, .15)


def test_assignment_record_keeps_decision_time_inputs_and_no_outcome(planning):
    d = demo()
    row = planning["validation"][0]
    record = d.assign(d.ThresholdExperimentPlan(), row, .15)
    assert record["decision_at"] == row.at.isoformat()
    assert record["source_policy_version"] == row.policy_version
    assert record["inputs"]["amount_cents"] == row.amount_cents
    assert record["inputs"]["reason"] == row.reason
    assert record["score"] == .15
    assert "found_false_30d" not in record and "found_false_30d" not in record["inputs"]


def test_plan_refuses_blank_design_text_and_noninteger_window():
    d = demo()
    for field, value in (("owner", "   "), ("label_window_days", .5), ("seed", 7.5)):
        with pytest.raises(ValueError):
            d.check_plan(replace(d.ThresholdExperimentPlan(), **{field: value}))


def test_plan_records_the_difference_named_before_its_schedule(planning):
    d = demo()
    plan = d.build_plan(planning, difference=4)
    assert plan.minimum_difference == 4
    size = d.sizing(plan, planning["validation"], planning["score"])
    assert size["difference"] == 4 and size["per_arm"] == 246 and size["months"] == 9
    for value in (0, float("nan"), None):
        with pytest.raises(ValueError):
            d.check_plan(replace(plan, minimum_difference=value))


def test_rehearsal_support_and_size_use_only_validation(planning):
    d = demo()
    rows, score = planning["validation"], planning["score"]
    assert len(rows) == 988
    assert planning["census"] == (36000, 17704, 18296)
    assert max(r.label_final_at for r in rows) <= d.GO_LIVE
    bands = d.support_by_band(rows, score)
    assert sum(b[2] for b in bands) == 988 and all(b[3] == 0 for b in bands)
    for cap in (10, 15):
        plan = replace(d.ThresholdExperimentPlan(), cap=cap)
        records = d.rehearse(plan, rows, score)
        assert records == d.rehearse(plan, rows, score)
        selected = [r for r in records if r["in_set"]]
        n, pay = len(selected), sum(r["action"] == "pay" for r in selected)
        low, high = d.coin_range(n)
        assert low <= pay <= high
        assert all(r["probability"] == (.5 if r["in_set"] else 1.) for r in records)
        size = d.sizing(plan, rows, score, difference=2)
        assert size["set_requests"] == n
        assert size["months"] == math.ceil(2 * size["per_arm"] / n)
        assert size["worst_case_loss"] == pytest.approx(size["per_arm"] * 49.99)
    size = d.sizing(d.ThresholdExperimentPlan(), rows, score)
    assert size["set_requests"] == 59 and size["per_arm"] == 982 and size["months"] == 34
    assert size["mean"] == pytest.approx(454.29/59)
    assert size["worst_case_loss"] == 49090.18
    other_seed = replace(d.ThresholdExperimentPlan(), seed=8)
    assert d.rehearse(other_seed, rows, score) != d.rehearse(d.ThresholdExperimentPlan(), rows, score)


def test_sizing_counts_one_eligible_request_per_customer(planning):
    d = demo()
    plan = d.ThresholdExperimentPlan()
    rows, score = list(planning["validation"]), planning["score"]
    inside = [i for i, r in enumerate(d.rehearse(plan, rows, score)) if r["in_set"]]
    assert len({rows[i].requester_id for i in inside}) == len(inside) == 59
    # A second eligible request from one customer shares its arm, so the two are
    # not independent draws and the sample-size formula no longer counts them.
    first, second = inside[:2]
    rows[second] = rows[second].model_copy(update={"requester_id": rows[first].requester_id})
    records = d.rehearse(plan, rows, score)
    assert records[first]["action"] == records[second]["action"]
    with pytest.raises(ValueError, match="one eligible request per customer"):
        d.sizing(plan, rows, score)


def test_no_action_outside_the_set_depends_on_the_seed(planning):
    d = demo()
    rows, score = planning["validation"], planning["score"]
    for cap in (10, 15):
        plan = replace(d.ThresholdExperimentPlan(), cap=cap)
        first = d.rehearse(plan, rows, score)
        for seed in (8, 9, 2026):
            other = d.rehearse(replace(plan, seed=seed), rows, score)
            assert [r["in_set"] for r in other] == [r["in_set"] for r in first]
            for a, b in zip(first, other):
                if not a["in_set"]:
                    assert (a["action"], a["probability"]) == (b["action"], b["probability"])
                    assert a["probability"] == 1.0
    # Outside the set the action is the tier's gate alone: pay under 0.080, else refer.
    for record, row in zip(first, rows):
        if not record["in_set"]:
            gate = "pay" if record["score"] < d.TIER_THRESHOLD else "refer"
            assert record["action"] == gate


def test_frozen_windows_are_never_consulted_or_read(monkeypatch, planning):
    d = demo()
    from demos import supervised_learning_demo as ch5
    from refund_agent.data import history
    from refund_agent.split import GoldenSet
    full = d.generate_history(seed=7)
    window = ch5.windows([r for r in full if r.paid])
    august, september = (ch5.frozen(window[name], name) for name in ("test", "shifted"))
    before = [list(g.log) for g in (august, september)]
    class Sealed:
        def __init__(self, row):
            self.row = row
        def __getattr__(self, name):
            if name in ("found_false_30d", "label_final_at"):
                raise AssertionError("a label beyond the validation window was read")
            return getattr(self.row, name)
    sealed = [Sealed(r) if r.at > d.VALIDATION_LAST_AT else r for r in full]
    monkeypatch.setattr(d, "generate_history", lambda seed: sealed)
    def refuse(*args, **kwargs):
        raise AssertionError("a frozen set was consulted")
    monkeypatch.setattr(GoldenSet, "consult", refuse)
    again = d.planning_history()
    np.testing.assert_array_equal(again["score"], planning["score"])
    # Seal the generator used by a fresh script namespace too: every region, not
    # only the fitting helper, must respect the same access boundary.
    monkeypatch.setattr(history, "generate_history", lambda seed: sealed)
    printed = io.StringIO()
    with redirect_stdout(printed):
        runpy.run_path(str(DEMO), run_name="__main__")
    assert printed.getvalue() == "".join(
        (OUTPUTS / f"_{PREFIX}{name}.txt").read_text() for name in REGIONS
    )
    assert [g.log for g in (august, september)] == before


def test_figure_data_prices_the_same_assumptions_as_review_output(captured, planning):
    d = demo()
    data = json.loads(captured[PREFIX + "figures-01"])["review_errors"]
    _, y, amount = d.arrays(planning["validation"])
    selected = planning["score"] * amount >= 4
    for curve in data["curves"]:
        for f, saving in zip(data["wrongly_deny_sound"], curve["saving"]):
            assert saving == pytest.approx(d.review_saving(y, amount, selected,
                                                        curve["missed_false"], f), abs=5e-7)


def test_plan_table_keeps_currency_literal_when_included_as_markdown(captured):
    table = captured[PREFIX + "plan-01"]
    assert not re.search(r"(?<!\\)\$", table)
    assert r"amount: below \$25 or at least \$25" in " ".join(table.replace("|", " ").split())


def test_compact_plan_preserves_the_complete_checked_record(captured, planning):
    from dataclasses import asdict

    d = demo()
    plan = d.build_plan(planning)
    table = captured[PREFIX + "plan-01"]
    assert all(len(line) <= 79 for line in table.splitlines())
    rows = [line for line in table.splitlines()[2:] if line.startswith("|")]
    assert sum(bool(line.split("|")[1].strip()) for line in rows) == 17
    table = " ".join(table.replace("|", " ").split())
    complete = json.loads(captured[PREFIX + "plan-full-01"])
    expected = json.loads(json.dumps(asdict(plan), default=lambda at: at.isoformat()))
    assert complete == expected
    assert d.check_plan(plan) == "accepted"
    for field in (
        "eligibility", "assignment_unit", "estimand", "outcome", "stop_rule",
        "interference", "missing_outcomes", "denial_check", "owner", "launch",
    ):
        assert complete[field].replace("$", r"\$") in table
    for field in ("first_assignment_at", "last_assignment_at", "analysis_at", "signed_at"):
        assert complete[field][:10] in table
    assert "P(refer)=0.5; seed 7" in table
    assert r"review \$4.00; wrong denial \$20.00" in table
    assert r"cap \$10.00; payout ceiling \$100000.00" in table


def test_import_and_demo_call_no_agent_or_payment_tool(tmp_path):
    code = f"""
import os, runpy, sys
sys.path.insert(0, {str(COMPANION)!r})
from refund_agent.tools import Tools
from refund_agent.agent import Agent
def forbidden(*args, **kwargs):
    raise AssertionError('the plan called the agent or a tool')
Tools.call = forbidden
Agent.run = forbidden
import demos.causal_inference_demo as d
assert os.listdir('.') == []
runpy.run_path({str(DEMO)!r}, run_name='__main__')
assert os.listdir('.') == []
"""
    result = subprocess.run([sys.executable, "-c", code], cwd=tmp_path,
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr


def test_fixture_has_one_selected_denominator_error():
    d = demo()
    stored = json.loads((COMPANION / "demos/data/causal_inference_effect_05.json").read_text())
    assert stored == d.effect_report()
    assert stored["headline"]["pay_rate"] == .12
    assert stored["headline"]["refer_paid_rate"] == .05
    costs = d.arm_costs(400, 40, 48, 80, 16, 50)
    assert costs["pay"] == 4.8 and costs["refer"] == 8.1
    assert costs["difference"] == pytest.approx(3.3)
    assert costs["found_false_difference"] == 32


def test_constructed_arm_variance_includes_review_and_denial_cost():
    result = demo().arm_costs(1000, 40, 100, 150, 30, 80)
    assert result["pay"] == 4 and result["refer"] == 6.8
    assert result["difference"] == pytest.approx(2.8)
    assert result["se"] == pytest.approx(.46516, abs=.00001)
    assert result["interval"] == pytest.approx((1.88828, 3.71172), abs=.00001)


def test_every_stored_output_matches_capture_and_repeated_seed(captured):
    assert tuple(name.removeprefix(PREFIX) for name in captured) == REGIONS
    for name, text in captured.items():
        assert (OUTPUTS / f"_{name}.txt").read_bytes() == text.encode()
    assert capture_outputs(snippets(), sys.executable) == captured
    figure = json.loads(captured[PREFIX + "figures-01"])
    assert len(figure["review_errors"]["curves"]) == 3
    assert figure["pay_probability"]["levels"] == [None, .5, 0]
