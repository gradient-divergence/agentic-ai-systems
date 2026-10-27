"""The review desk schedules work without reading labels or granting authority."""

import json
from dataclasses import replace
from datetime import date
from types import SimpleNamespace

import pytest

from refund_agent.data.generate import generate


@pytest.fixture(scope="module")
def windows():
    from refund_agent.schedule.window import run_window

    return {policy: run_window(policy) for policy in ("plain", "count", "count_then_wait")}


def test_queue_reads_only_training_times_and_amounts_and_never_labels(monkeypatch):
    from refund_agent import split
    from refund_agent.schedule.queue import waiting

    data = generate()
    training = [r for r in data.requests if r.at < split.CUTS[0]]

    class HeldRequest:
        at = split.CUTS[0]

        def __getattr__(self, field):
            raise AssertionError(f"read held-out field {field}")

    class GuardedData:
        requests = training + [HeldRequest()]
        reviewers = data.reviewers
        policy_at = data.policy_at

        @property
        def correct(self):
            raise AssertionError("read labels")

    monkeypatch.setattr(split.GoldenSet, "consult", lambda *a, **k: pytest.fail("golden access"))
    monkeypatch.setattr(split, "freeze", lambda *a, **k: pytest.fail("golden freeze"))
    queue = waiting(GuardedData(), date(2026, 3, 23), [])
    assert len(queue) == 13
    assert all(d.at < split.CUTS[0] for d in queue)
    # Scheduling needs none of the request's reason, text, order, failure or outcome.
    minimal = [
        SimpleNamespace(at=r.at, request_id=r.request_id, amount_cents=r.amount_cents)
        for r in training
    ]
    assert (
        waiting(
            SimpleNamespace(requests=minimal, reviewers=data.reviewers, policy_at=data.policy_at),
            date(2026, 3, 23),
            [],
        )
        == queue
    )
    from refund_agent.schedule.window import run_window

    assert run_window("count_then_wait", data=GuardedData())["answered"] == 397


def test_working_mornings_and_monday_counts(windows):
    for result in windows.values():
        assert len(result["days"]) == 114
        assert result["requests"] == 398
        assert result["answered"] == 397
        assert len(result["remaining"]) == 1
    plain = next(d for d in windows["plain"]["days"] if d["day"] == "2026-03-23")
    best = next(d for d in windows["count_then_wait"]["days"] if d["day"] == "2026-03-23")
    assert (plain["answered"], best["answered"], best["bound"]) == (9, 10, 10)
    assert [d["day"] for d in windows["plain"]["days"] if d["answered"] < d["rounded_bound"]] == [
        "2026-03-23",
        "2026-04-13",
        "2026-04-27",
        "2026-05-11",
    ]


def test_every_morning_is_permitted_and_bounded(windows):
    for result in windows.values():
        for day in result["days"]:
            record = day["record"]
            assert not record.plan_check
            assert record.certificate["certificate"]
            assert record.certificate["permitted"]
            assert day["answered"] <= day["rounded_bound"]
            assert day["best_count"] == day["rounded_bound"]
    assert [d["answered"] for d in windows["count"]["days"]] == [
        d["answered"] for d in windows["count_then_wait"]["days"]
    ]
    assert (
        windows["count_then_wait"]["tier3"]["longest_days"]
        <= windows["plain"]["tier3"]["longest_days"]
    )


@pytest.mark.parametrize(
    "mutation,error",
    [
        ("tier", "not allowed"),
        ("overlap", "overlap"),
        ("duplicate", "twice"),
        ("minutes", "minutes"),
        ("duration", "duration"),
    ],
)
def test_plan_checker_rejects_bad_assignments(mutation, error):
    from refund_agent.schedule.check import check_plan
    from refund_agent.schedule.inputs import MINUTES
    from refund_agent.schedule.model import solve_morning
    from refund_agent.schedule.queue import waiting

    queue = waiting(generate(), date(2026, 3, 23), [])
    plan = solve_morning(queue, MINUTES, 10000).plan
    bad, minutes = list(plan), dict(MINUTES)
    if mutation == "tier":
        index = next(i for i, a in enumerate(bad) if a.person == "morgan")
        bad[index] = replace(bad[index], person="alex")
    elif mutation == "overlap":
        bad[1] = replace(bad[1], start=bad[0].start, end=bad[0].start + (bad[1].end - bad[1].start))
    elif mutation == "duplicate":
        bad.append(bad[0])
    elif mutation == "minutes":
        minutes["alex"] = 5
    else:
        bad[0] = replace(bad[0], end=bad[0].start)
    assert any(error in failure for failure in check_plan(bad, queue, minutes))


@pytest.mark.parametrize(
    "value,bound,absolute,relative", [(0, 0, 0, 0), (0, 2, 2, "undefined"), (6, 7.5, 1.5, 0.25)]
)
def test_zero_and_nonzero_gaps_round_trip(value, bound, absolute, relative, tmp_path):
    from refund_agent.schedule.certificate import check_certificate
    from refund_agent.schedule.record import SolveRecord

    certificate = check_certificate([[1]], [bound], [1], [value], [1])
    assert certificate["absolute_gap"] == absolute
    assert certificate["relative_gap"] == relative
    record = SolveRecord(
        queue_fingerprint="test",
        budget=10000,
        status="feasible",
        plan=[],
        value=value,
        bound=bound,
        multipliers=[1],
        certificate=certificate,
        plan_check=[],
    )
    assert record.absolute_gap == absolute
    assert record.relative_gap == relative
    path = tmp_path / "record.json"
    record.write_json(path)
    restored = SolveRecord.from_dict(json.loads(path.read_text()))
    assert restored == record
    assert bool(record.relative_gap_reason) == (relative == "undefined")


def test_empty_morning_records_zero(windows):
    from refund_agent.schedule.inputs import MINUTES
    from refund_agent.schedule.model import solve_morning

    record = solve_morning([], MINUTES, 10000)
    assert record.status == "optimal"
    assert (record.value, record.bound, record.absolute_gap, record.relative_gap) == (0, 0, 0, 0)


def test_seeded_window_is_deterministic(windows):
    from refund_agent.schedule.window import run_window

    rerun = run_window("count_then_wait", seed=7)
    assert rerun == windows["count_then_wait"]


def test_schedule_does_not_call_agent_tools_or_change_the_corpus(monkeypatch):
    from refund_agent.schedule.window import run_window

    data = generate()
    before = data.model_dump_json()
    # Payment and approval tools are not reachable through the planner.
    import refund_agent.tools as tools

    monkeypatch.setattr(tools.Tools, "__init__", lambda *a, **k: pytest.fail("tool construction"))
    from refund_agent import contracts

    monkeypatch.setattr(contracts, "Approval", lambda *a, **k: pytest.fail("approval creation"))
    monkeypatch.setattr(
        contracts, "Receipt", lambda *a, **k: pytest.fail("payment receipt creation")
    )
    run_window("count_then_wait", data=data)
    assert data.model_dump_json() == before


def test_respond_by_counting_key_has_recomputable_minimum_extra_time():
    from demos.constrained_optimization_demo import deadline_report

    result = deadline_report()
    assert result["infeasible_mornings"] == result["counting_flags"] == 49
    for proof in result["proofs"]:
        assert proof["tier3_minutes"] == 30 * len(proof["required"])
        assert proof["tier3_minutes"] > proof["morgan_minutes"]
        assert proof["extra_morgan_minutes"] == proof["tier3_minutes"] - proof["morgan_minutes"]
        assert not proof["plan_check"]


def test_planted_plan_has_only_a_tier_error():
    from demos.constrained_optimization_demo import exercise_five
    from refund_agent.schedule.check import check_plan
    from refund_agent.schedule.inputs import MINUTES
    from refund_agent.schedule.queue import waiting
    from refund_agent.schedule.record import Assignment

    fixture = exercise_five()
    plan = [Assignment.from_dict(a) for a in fixture["plan"]]
    failures = check_plan(plan, waiting(generate(), date(2026, 3, 23), []), MINUTES)
    assert len(failures) == 1
    assert "not allowed" in failures[0]


def test_morning_records_include_inputs_and_recompute_after_json_round_trip(windows):
    from refund_agent.schedule.record import SolveRecord, queue_fingerprint

    for policy in windows.values():
        for day in policy["days"]:
            record = day["record"]
            assert record.inputs["queue"] or not day["queue"]
            assert record.queue_fingerprint == queue_fingerprint(
                day["queue"], record.inputs["minutes"]
            )
            assert SolveRecord.from_dict(json.loads(json.dumps(record.to_dict()))) == record


@pytest.mark.parametrize("found,status", [(False, "unknown"), (True, "feasible")])
def test_budget_stop_is_unknown_without_an_incumbent(monkeypatch, found, status):
    import numpy as np

    from refund_agent.schedule import model
    from refund_agent.schedule.inputs import BUDGET, MINUTES
    from refund_agent.schedule.queue import waiting

    queue = waiting(generate(), date(2026, 3, 23), [])
    columns = len(model.count_rows(queue, MINUTES)[0])
    monkeypatch.setattr(
        model,
        "milp",
        lambda *a, **k: SimpleNamespace(status=1, x=np.zeros(columns) if found else None),
    )
    record = model.solve_morning(queue, MINUTES, BUDGET)
    assert record.status == status
    assert record.relative_gap == "undefined"
    assert record.absolute_gap == record.bound == 10


@pytest.mark.parametrize("found", [False, True])
def test_other_solver_error_is_unknown_and_proposes_no_plan(monkeypatch, found):
    from refund_agent.schedule import model
    from refund_agent.schedule.inputs import BUDGET, MINUTES
    from refund_agent.schedule.queue import waiting

    original = model.milp

    def errored(*args, **kwargs):
        result = original(*args, **kwargs)
        result.status = 4  # SciPy's generic error does not diagnose an invalid model.
        if not found:
            result.x = None
        return result

    monkeypatch.setattr(model, "milp", errored)
    record = model.solve_morning(waiting(generate(), date(2026, 3, 23), []), MINUTES, BUDGET)
    assert record.status == "unknown"
    assert record.plan == []
    assert record.value == 0


@pytest.mark.parametrize(
    "status,description",
    [(1, "budget exhausted"), (2, "infeasible"), (3, "unbounded"), (4, "unknown")],
)
def test_waiting_time_solve_retains_the_count_and_reports_failure(monkeypatch, status, description):
    from refund_agent.schedule import model
    from refund_agent.schedule.inputs import BUDGET, MINUTES
    from refund_agent.schedule.queue import waiting

    original, calls = model.milp, []

    def stopped(*args, **kwargs):
        calls.append(kwargs)
        return (
            original(*args, **kwargs) if len(calls) == 1 else SimpleNamespace(status=status, x=None)
        )

    monkeypatch.setattr(model, "milp", stopped)
    record = model.solve_morning(waiting(generate(), date(2026, 3, 23), []), MINUTES, BUDGET)
    assert record.status == "optimal"
    assert record.secondary_status == description
    assert record.value == 10
    assert not record.plan_check


def test_secondary_solver_error_cannot_replace_the_proven_count_plan(monkeypatch):
    import numpy as np

    from refund_agent.schedule import model
    from refund_agent.schedule.inputs import BUDGET, MINUTES
    from refund_agent.schedule.queue import waiting

    original, calls = model.milp, []

    def errored(*args, **kwargs):
        calls.append(kwargs)
        result = original(*args, **kwargs)
        if len(calls) == 2:
            result.status, result.x = 4, np.zeros_like(result.x)
        return result

    monkeypatch.setattr(model, "milp", errored)
    record = model.solve_morning(waiting(generate(), date(2026, 3, 23), []), MINUTES, BUDGET)
    assert record.status == "optimal"
    assert record.secondary_status == "unknown"
    assert record.value == 10
    assert not record.plan_check
