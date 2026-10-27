from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from refund_agent.oversight.human_oversight_risk_demand import (
    Stage6Demand,
    WorkClass,
    generate_stage6,
    generator_mix_counterfactual,
)
from refund_agent.oversight.human_oversight_risk_monitor import ReviewObservation, load_report
from refund_agent.oversight.human_oversight_risk_queueing import mm1
from refund_agent.oversight.human_oversight_risk_simulate import (
    Person,
    QueueArrival,
    simulate,
    simulate_one,
    staffing,
)

START = datetime(2026, 10, 1, tzinfo=timezone.utc)


def test_seeded_demand_never_calls_label_builders_or_reads_a_golden_file(monkeypatch):
    import builtins
    import io

    from refund_agent.data import generate as gen

    def forbidden(*args, **kwargs):
        raise AssertionError("demand must not consult labels")

    monkeypatch.setattr(gen, "_label", forbidden)
    file_reads = []

    def no_reads(*args, **kwargs):
        file_reads.append(args[0])
        raise AssertionError("demand must not read any file, including a golden case")

    with monkeypatch.context() as reads:
        reads.setattr(builtins, "open", no_reads)
        reads.setattr(io, "open", no_reads)
        a = generate_stage6(7, START, 3, Stage6Demand())
        b = generate_stage6(7, START, 3, Stage6Demand())
    assert file_reads == []
    assert a == b and a.correct == [] and a.instructor_scenarios == []
    assert {r.amount_cents for r in a.requests} <= {2499, 4999, 5000, 50000, 50001, 65000, 120000}
    assert generator_mix_counterfactual() == 100


def test_validation_matches_mm1_on_a_long_seeded_run():
    for u in (.5, .9):
        result = simulate_one(7, u, n=100000)
        exact = mm1(2 * u, 2)
        assert result["W"] == pytest.approx(float(exact.W), rel=.08)
        assert result["L"] == pytest.approx(float(exact.L), rel=.08)
        assert result["relative_gap"] < 1e-12


def test_weekday_marks_use_staffed_hours_and_are_above_the_curve():
    for u in (.5, .7, .8, .9):
        result = simulate_one(7, u, n=10000, desk=True)
        assert result["calendar_lambda"] == pytest.approx(2 * u * 40 / 168)
        assert result["W"] > float(mm1(2 * u, 2).W)


@pytest.mark.parametrize("name", ["D1", "D2", "D3"])
def test_finite_window_counts_starting_backlog_and_censored_unfinished_stays(name):
    backlog = [QueueArrival("human-oversight-risk-inherited", "tier2", START - timedelta(days=1))]
    result = simulate(7, START, 7, people=staffing(name), initial_backlog=backlog)
    assert result.starting_backlog == 1
    assert result.conservation["tier2"]["starting_backlog"] == 1
    assert all(row["relative_gap"] < 1e-10 for row in result.conservation.values())
    assert all(row["occupancy_hours"] == pytest.approx(row["clipped_stay_hours"]) for row in result.conservation.values())
    assert "finite-window conservation" in " ".join(result.notes)


def test_ordinary_reviewers_cannot_fix_a_tier3_monday_shortfall():
    friday = datetime(2026, 10, 2, 18, tzinfo=timezone.utc)
    arrivals = [QueueArrival(f"human-oversight-risk-heavy-{i}", "tier3", friday + timedelta(seconds=i))
                for i in range(17)]
    result = simulate(7, START, 10, arrivals=arrivals, classes=(WorkClass("tier3", 4, 30, 3),),
                      service_model="fixed", people=staffing("D3") + (Person("extra", "reviewer"),))
    assert result.late_blocking == 1 and result.monday_breaches == 1
    assert all(j.reviewer == "morgan" for j in result.items)
    assert any(j.ticket and j.ticket.assignee == "decision-owner" for j in result.items)


def test_provided_arrivals_replace_draws_and_blind_review_excludes_first_person():
    result = simulate(7, START, 7, arrivals=[QueueArrival("human-oversight-risk-blind", "blind", START,
                                                       first_reviewer="alex")],
                      classes=(WorkClass("blind", 1000, 15, 2, False, 5),), service_model="fixed")
    assert result.classes["blind"]["arrivals"] == 1
    assert result.items[0].reviewer == "riley"


def test_nonblocking_deadline_breach_is_visible_without_entering_the_blocking_ceiling():
    result = simulate(7, START, 7, arrivals=[QueueArrival("human-oversight-risk-pending-blind", "blind", START,
                                                       first_reviewer="alex")], people=staffing("D1"),
                      classes=(WorkClass("blind", 1, 15, 2, False, 5),), service_model="fixed")
    assert result.classes["blind"]["deadline_breaches"] == 1
    assert result.classes["blind"]["completed_after_drain"] == 0
    assert result.late_blocking == 0


def test_unresolved_escalations_do_not_count_as_planted_catches():
    rows = [ReviewObservation(str(i), START, "alex", 20, True, "escalated", None) for i in range(6)]
    rows += [ReviewObservation("miss1", START, "alex", 20, True, "approved", START),
             ReviewObservation("miss2", START, "alex", 20, True, "approved", START)]
    result = load_report(rows)["overloaded"]
    assert result["caught"] == 0 and result["decided"] == 2 and result["unresolved"] == 6


def test_unopened_and_open_pending_cases_are_reported_without_future_outcomes():
    from refund_agent.oversight.human_oversight_risk_monitor import observed_by, trailing_alarm

    at = START + timedelta(days=7)
    rows = [ReviewObservation("opened", START, "alex", 6, True, "refused", at + timedelta(days=1)),
            ReviewObservation("unopened", at + timedelta(days=1), "riley", 6, True, "approved",
                              at + timedelta(days=1, hours=1), START),
            ReviewObservation("not_arrived", at + timedelta(days=1), None, None, True, "open", None)]
    visible = observed_by(rows, at)
    assert len(visible) == 2 and all(r.closed_at is None and r.action == "open" for r in visible)
    report = load_report(visible)
    assert report["busy"]["unresolved"] == 1 and report["unopened"]["unresolved"] == 1
    assert sum(r["decided"] for r in report.values()) == 0
    assert trailing_alarm(rows, at)["decided"] == 0
    assert trailing_alarm(visible, at)["decided"] == 0


def test_attention_identifies_actual_planted_records_instead_of_work_class():
    from types import SimpleNamespace

    from test_human_oversight_risk_lifecycle import fixture

    from demos.human_oversight_risk_demo import attention_observations
    from refund_agent.oversight.human_oversight_risk_monitor import planted_packet

    def job(key, work_class, *, opened=START):
        return SimpleNamespace(arrival=QueueArrival(key, work_class, START),
                               work=SimpleNamespace(name=work_class), opened_at=opened,
                               closed_at=START + timedelta(minutes=15) if opened else None,
                               reviewer="riley" if opened else None, queue_length=20)

    result = SimpleNamespace(end=START + timedelta(days=30), items=[
        job("actual-record", "tier2"), job("class-only", "planted"), job("waiting-record", "planted", opened=None)])
    _, _, desk, ticket = fixture()
    prototype = desk.packets[ticket.review_id][0]
    records = {key: planted_packet(prototype, key, "shipment_contradiction")
               for key in ("actual-record", "waiting-record")}
    rows = attention_observations(result, records, 7)
    by_id = {r.review_id: r for r in rows}
    assert by_id["actual-record"].planted and not by_id["class-only"].planted
    assert by_id["waiting-record"].planted and by_id["waiting-record"].closed_at is None
    assert by_id["waiting-record"].queue_length is None
    with pytest.raises(ValueError, match="does not belong"):
        attention_observations(result, {"actual-record": records["waiting-record"]}, 7)


def test_absent_approver_leaves_tier3_waiting_and_escalated():
    away = frozenset((START + timedelta(days=i)).date() for i in range(7))
    people = tuple(replace(p, absent=away) if p.name == "morgan" else p for p in staffing("D3"))
    arrivals = [QueueArrival("human-oversight-risk-away", "tier3", START)]
    result = simulate(7, START, 7, arrivals=arrivals, people=people, service_model="fixed")
    item = result.items[0]
    assert item.closed_at >= START + timedelta(days=7)
    assert item.reviewer == "morgan" and item.ticket.assignee == "decision-owner"


def test_saturday_capacity_reduces_actual_monday_residual_not_the_arrival_count():
    friday = datetime(2026, 10, 2, 18, tzinfo=timezone.utc)
    arrivals = [QueueArrival(f"human-oversight-risk-saturday-{i}", "tier3", friday + timedelta(seconds=i))
                for i in range(17)]
    result = simulate(7, START, 10, arrivals=arrivals, classes=(WorkClass("tier3", 4, 30, 3),),
                      service_model="fixed", people=staffing("D4"))
    monday = result.weekends[0]
    assert monday["counts"]["tier3"] == 17
    assert monday["incoming_weekend_hours"] == 8.5
    assert monday["tier3_hours"] == .5 and result.late_blocking == 0
