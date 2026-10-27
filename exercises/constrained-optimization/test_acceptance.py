"""Exercise 8 checks the selected submission across the training window."""

import importlib.util
import json
import os
from datetime import timedelta
from pathlib import Path

import pytest

from refund_agent.data.generate import generate
from refund_agent.labels import LABEL_POLICY
from refund_agent.schedule.check import check_plan
from refund_agent.schedule.inputs import BUDGET, END, MINUTES, START, WORKING_DAYS
from refund_agent.schedule.model import solve_morning
from refund_agent.schedule.queue import waiting
from refund_agent.schedule.record import SolveRecord
from refund_agent.split import GoldenSet, freeze

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def golden_evidence(tmp_path):
    data = generate()
    golden = freeze(data, data.requests, LABEL_POLICY, name="chapter4-deadline-guard")
    directory = tmp_path / "golden-sets"
    golden.persist(
        directory,
        prior_log=[
            dict(entry=1, set=golden.manifest["name"], reader="chapter4", purpose="prior evidence")
        ],
    )
    return golden, directory, {p: p.read_bytes() for p in directory.glob("*")}


@pytest.fixture
def solution(monkeypatch, golden_evidence):
    def forbidden(*args, **kwargs):
        raise AssertionError("deadline exercise consulted or changed the golden set")

    monkeypatch.setattr(GoldenSet, "consult", forbidden)
    monkeypatch.setattr(GoldenSet, "persist", forbidden)
    source = Path(
        os.environ.get("CONSTRAINED_OPTIMIZATION_SOLUTION", Path(__file__).with_name("starter.py"))
    ).resolve()
    monkeypatch.chdir(golden_evidence[1].parent)
    spec = importlib.util.spec_from_file_location("deadline_submission", source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def due_day(draft):
    day = draft.at.date() + timedelta(days=1)
    while day.weekday() not in WORKING_DAYS:
        day += timedelta(days=1)
    return day


def test_submission_deadlines_fallbacks_minimum_time_tiers_and_golden(solution, golden_evidence):
    golden, directory, before = golden_evidence
    data, day, carried, infeasible = generate(), START, [], 0
    corpus = data.model_dump_json()
    while day <= END:
        if day.weekday() in WORKING_DAYS:
            queue = waiting(data, day, carried)
            inputs = tuple(queue)
            required = {d.request_id for d in queue if due_day(d) <= day}
            tier3 = {d.request_id for d in queue if d.request_id in required and d.tier == 3}
            tier3_minutes = sum(d.minutes for d in queue if d.request_id in tier3)
            result = solution.plan_morning(queue, dict(MINUTES), BUDGET)
            assert tuple(queue) == inputs, "submission changed tiers or queue inputs"
            assert set(result["required"]) == required
            record = result["record"]
            assert isinstance(record, SolveRecord)
            assert not check_plan(record.plan, queue, MINUTES)
            assert SolveRecord.from_dict(json.loads(json.dumps(record.to_dict()))) == record
            taken = {a.request_id for a in record.plan}
            missing = required - taken
            assert set(result["missing_rule"]) == missing
            if tier3_minutes > MINUTES["morgan"]:
                infeasible += 1
                assert result["status"] == "infeasible"
                usual = solve_morning(queue, MINUTES, BUDGET)
                assert record.plan == usual.plan, "fallback must be the planner's usual plan"
                assert missing, "infeasible fallback must mark the missed rule"
                proof = result["proof"]
                assert set(proof["required"]) == tier3
                assert proof["tier3_minutes"] == tier3_minutes
                assert proof["morgan_minutes"] == MINUTES["morgan"]
                extra = proof["extra_morgan_minutes"]
                assert extra == tier3_minutes - MINUTES["morgan"]
                assert tier3_minutes > MINUTES["morgan"] + extra - 1
                repair = result["repair"]
                assert not check_plan(repair.plan, queue, {**MINUTES, "morgan": 60 + extra})
                assert required <= {a.request_id for a in repair.plan}
            else:
                assert result["status"] in ("optimal", "feasible")
                assert not missing and result["proof"] is None and result["repair"] is None
            carried = [d for d in queue if d.request_id not in taken]
        day += timedelta(days=1)
    assert infeasible == 49
    assert data.model_dump_json() == corpus
    assert before == {p: p.read_bytes() for p in directory.glob("*")}
    assert len(golden.log) == 1
