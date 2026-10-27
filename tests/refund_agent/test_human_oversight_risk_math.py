from dataclasses import replace
from datetime import datetime, timezone
from fractions import Fraction
from math import exp

import pytest

from refund_agent.oversight.human_oversight_risk_controller import Controller
from refund_agent.oversight.human_oversight_risk_demand import Stage6Demand
from refund_agent.oversight.human_oversight_risk_design import (
    accept_months,
    check_design,
    default_design,
)
from refund_agent.oversight.human_oversight_risk_monitor import (
    ReviewObservation,
    alarm,
    alarm_power,
    binom_cdf,
    trailing_alarm,
    wilson_interval,
    zero_failure_bound,
)
from refund_agent.oversight.human_oversight_risk_queueing import mm1, recovery

AT = datetime(2026, 10, 1, tzinfo=timezone.utc)


@pytest.mark.parametrize("lam,mu", [(1, 2), (Fraction(3, 2), 2), (Fraction(9, 5), 2), (12, 15)])
def test_mm1_equals_the_direct_geometric_sum(lam, mu):
    row = mm1(lam, mu)
    summed = sum(n * float(row.occupancy(n)) for n in range(1000))
    assert summed == pytest.approx(float(row.L), abs=1e-10)
    assert row.L == lam * row.W
    assert row.tail(4) == pytest.approx(exp(-float(mu - lam) * 4))


def test_exact_worked_bank_and_reviewer_numbers():
    bank = mm1(12, 15)
    assert (bank.u, bank.p0, bank.L, bank.W * 60, bank.W_wait * 60) == (Fraction(4, 5), Fraction(1, 5), 4, 20, 16)
    assert [mm1(lam, 2).W for lam in (1, Fraction(3, 2), Fraction(9, 5))] == [1, 2, 5]
    assert recovery(28, 16, 5.6) == pytest.approx(2.69230769)
    assert recovery(28, 8, 5.6) == pytest.approx(11.66666667)
    assert recovery(28, 5, 5.6) is None


@pytest.mark.parametrize("lam,mu", [(2, 2), (18, 15)])
def test_unstable_has_no_numbers(lam, mu):
    row = mm1(lam, mu)
    assert row.status == "unstable"
    assert (row.u, row.p0, row.L, row.W, row.W_wait, row.spare_rate) == (None,) * 6
    with pytest.raises(ValueError):
        row.tail(4)


@pytest.mark.parametrize("lam,mu", [(-1, 2), (1, 0), (float("nan"), 2), (1, float("inf")), (True, 2)])
def test_invalid_mm1_rates(lam, mu):
    with pytest.raises(ValueError):
        mm1(lam, mu)


def test_alarm_edge_and_power_are_exact():
    assert alarm(6, 10) and not alarm(7, 10)
    assert float(binom_cdf(6, 10)) == pytest.approx(.0127951984)
    assert float(binom_cdf(7, 10)) == pytest.approx(.0701908264)
    assert alarm_power(10) == Fraction(53, 64)


@pytest.mark.parametrize("queue_length", [16, 20])
def test_trailing_alarm_excludes_overloaded_only_cases(queue_length):
    rows = [ReviewObservation(str(i), AT, "alex", queue_length, True,
                              "refused" if i < 6 else "approved", AT) for i in range(10)]
    state = trailing_alarm(rows, AT)
    assert state["decided"] == 0 and state["caught"] == 0
    assert not state["alarm"] and state["p"] is None


@pytest.mark.parametrize("queue_length", [6, 15])
def test_trailing_alarm_counts_decided_busy_band_case(queue_length):
    row = ReviewObservation("busy", AT, "alex", queue_length, True, "refused", AT)
    state = trailing_alarm([row], AT)
    assert state["decided"] == 1 and state["caught"] == 1


def test_stats_match_the_earlier_chapter_functions():
    from demos.data_science_strategy_demo import zero_failure_bound as earlier_bound
    from demos.ml_data_evaluation_demo import wilson_interval as earlier_wilson

    for n in (1, 10, 73, 183):
        assert zero_failure_bound(n) == earlier_bound(n)
        for k in (0, n // 2, n):
            assert wilson_interval(k, n) == pytest.approx(earlier_wilson(k, n))


def test_counting_conditions_name_the_group_that_is_short():
    demand = Stage6Demand()
    d1, d2, d3 = [check_design(default_design(name, AT), demand) for name in ("D1", "D2", "D3")]
    assert "reviewers_unstable" in d1["failures"]
    assert d2["monday_margin"] == pytest.approx(-.4, abs=.02)
    assert "pooled_monday_infeasible" in d2["failures"]
    assert d3["monday_margin"] == pytest.approx(7.6, abs=.02)
    assert d3["failures"] == ()
    short = check_design(default_design("D3", AT), replace(demand, tier3_daily=8))
    assert "tier3_monday_infeasible" in short["failures"]
    assert short["tier3_monday_capacity"] == 8


def test_rework_is_counted_and_cannot_pass_without_owner_capacity():
    demand = Stage6Demand(modified_share=.1, refused_share=.05, blind_disagreement_share=.1)
    report = check_design(default_design("D3", AT), demand)
    assert report["weekly_hours"]["modification"] == pytest.approx(112 * .1 * 20 / 60)
    assert report["weekly_hours"]["refusal_followup"] == pytest.approx(112 * .05 * 15 / 60)
    assert report["weekly_hours"]["adjudication"] == pytest.approx(112 * .1 * .1 * 30 / 60)
    assert "decision_owner_followups_unstable" in report["failures"]


def test_revised_design_counts_morgans_overlapping_roles_once():
    demand = Stage6Demand(modified_share=.1, refused_share=.05, blind_disagreement_share=.1)
    report = check_design(default_design("D4", AT), demand)
    assert report["capacity_hours"] == 96 and report["monday_capacity"] == 24
    assert report["failures"] == ()
    # Ordinary reviewers cannot carry the owner's follow-up work either.
    assert next(row for row in report["classes"] if row["class"] == "adjudication")["allowed"] == ("morgan",)


def test_one_ten_percent_month_among_99_clean_months_is_rejected():
    result = accept_months([{"late": 0, "blocking": 100}] * 99 + [{"late": 10, "blocking": 100}])
    assert result["pooled_share"] == .001
    assert result["breach_months"] == 1 and result["breach_frequency"] == .01
    assert not result["accepted"]
    assert accept_months([{"late": 1, "blocking": 100}])["accepted"]


def test_controller_records_changes_and_exposes_no_tier_or_blocking_actuator():
    ctl = Controller()
    assert ctl.weekly(.8, AT) == ()
    changed = ctl.weekly(.81, AT)
    assert [row["kind"] for row in changed] == ["recheck_share", "blind_share", "staffing_request"]
    assert (ctl.recheck_share, ctl.blind_share) == (.1, .05)
    assert not hasattr(ctl, "tier") and not hasattr(ctl, "blocking")
    with pytest.raises(ValueError, match="floors"):
        Controller(recheck_share=.01)


@pytest.mark.parametrize("corruption,expected", [
    ("shipment_contradiction", "already_delivered"),
    ("balance_plus_cent", "amount_over_policy"),
    ("false_window_denial", "eligible"),
])
def test_corrupted_packets_contain_the_three_required_policy_mistakes(corruption, expected):
    from test_human_oversight_risk_lifecycle import fixture

    from refund_agent import contracts as c
    from refund_agent.oversight.human_oversight_risk_monitor import planted_packet
    from refund_agent.rules import draft

    _, _, desk, ticket = fixture()
    prototype = desk.packets[ticket.review_id][0]
    order = prototype.evidence.order.model_copy(update={
        "paid_cents": 30000,
        "prior_refunds": [c.PriorRefund(payment_id="earlier-refund", amount_cents=5000, at=prototype.created_at)],
    })
    prototype = prototype.model_copy(update={"evidence": prototype.evidence.model_copy(update={"order": order})})
    before = prototype.model_dump()
    packet = planted_packet(prototype, "human-oversight-risk-plant", corruption)
    evidence, proposal = packet.proposal.evidence, packet.proposal
    reading = draft(c.State(request=proposal.request, trace_id=evidence.trace_id,
                            order=evidence.order, policy=evidence.policy,
                            shipment_events=evidence.shipment_events, execution_at=proposal.created_at,
                            refunded_cents=5000))
    assert reading.reason == expected and reading.outcome != packet.proposed.outcome
    assert packet.review_id == "human-oversight-risk-plant" and proposal.amount_cents <= 50000
    assert prototype.model_dump() == before
    if corruption == "balance_plus_cent":
        assert proposal.amount_cents == 25001  # $300 paid less $50 already refunded, plus one cent.
    elif corruption == "false_window_denial":
        assert packet.proposed.reason == "outside_return_window"
        assert (proposal.created_at - evidence.shipment_events[0].at).days < evidence.policy.return_days
    else:
        assert proposal.reason == "not_delivered" and evidence.shipment_events[0].status == "delivered"


def test_alarm_timing_includes_undetected_runs_and_respects_weekly_looks():
    from datetime import timedelta

    from refund_agent.oversight.human_oversight_risk_monitor import (
        alarm_timing_summary,
        first_alarm,
    )

    rows = [ReviewObservation(str(i), AT, "alex", 6, True,
                              "refused" if i < 6 else "approved", AT + timedelta(days=8)) for i in range(10)]
    assert first_alarm(rows, AT) == (2, 10)
    assert first_alarm(rows, AT, days=7) is None
    assert alarm_timing_summary([1, 1, 2, 3, None]) == {
        "replications": 5, "median": 2, "p10": 1, "p90": None,
        "alarmed_by_week2": .6, "undetected": 1,
    }
    with pytest.raises(ValueError):
        alarm_timing_summary([])
