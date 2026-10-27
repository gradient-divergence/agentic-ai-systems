"""Pin the stage-0 reads, drafts and audit effects printed in Chapter 1."""

import ast
import contextlib
import inspect
import io
import os
import runpy
import subprocess
import sys
from pathlib import Path

import pytest

from refund_agent.baseline import baseline
from refund_agent.contracts import Outcome, Tier
from refund_agent.data.generate import generate

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "demos/what_is_agentic_ai_demo.py"


def test_baseline_reprint_matches_the_package() -> None:
    from demos.what_is_agentic_ai_baseline import baseline as reprint

    assert ast.dump(ast.parse(inspect.getsource(reprint)), include_attributes=False) == ast.dump(
        ast.parse(inspect.getsource(baseline)), include_attributes=False
    )
    data = generate(seed=7)
    assert reprint(data) == baseline(data)


def test_demo_runs_as_a_script() -> None:
    result = subprocess.run(
        [sys.executable, str(DEMO)],
        cwd=ROOT,
        env=os.environ,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert len(result.stdout.splitlines()) <= 12
    assert max(map(len, result.stdout.splitlines())) <= 79


@pytest.fixture(scope="module")
def run() -> tuple[dict, str]:
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        namespace = runpy.run_path(str(DEMO))
    return namespace, stdout.getvalue()


def test_demo_runs_the_package_baseline(run: tuple[dict, str]) -> None:
    assert run[0]["baseline"] is baseline


def test_request_reason_matches_printed_wording(run: tuple[dict, str]) -> None:
    assert run[0]["request"].reason == "changed_mind"


def test_as_sent_draft_and_numeric_reading(run: tuple[dict, str]) -> None:
    approved = run[0]["approved"]
    chosen, state = approved.decision, approved.state
    assert chosen.request_id == "r-7-0007"
    assert chosen.amount_cents == 3800
    assert chosen.outcome == Outcome.APPROVE
    assert chosen.tier == Tier.AGENT
    assert chosen.reason == "eligible"
    assert chosen.policy_citation == "refund_cap_v4@2026-03-01#eligibility"
    assert state.order.order_id == "o-0007"
    assert state.order.paid_cents == 4200
    assert state.order.prior_refunds == []
    assert state.refunded_cents == 0
    assert state.policy.return_days == 21
    delivered = max(e.at for e in state.shipment_events if e.status == "delivered")
    assert (state.request.at - delivered).total_seconds() == 12 * 86400


def test_reads_return_the_order_shipment_and_policy(run: tuple[dict, str]) -> None:
    reads = run[0]["approved"].reads
    assert [name for name, result in reads] == [
        "order.lookup",
        "shipment.status",
        "policy.retrieve",
    ]
    assert all(result.ok for name, result in reads)
    assert [result.age_seconds for name, result in reads] == [60.0, 120.0, 3600.0]
    assert reads[0][1].order.paid_cents == 4200
    assert reads[1][1].shipment.events[-1].status == "delivered"
    assert reads[2][1].policy.version == "2026-03-01"


def test_ambiguous_lookup_stops_reading_and_refers(run: tuple[dict, str]) -> None:
    ambiguous = run[0]["ambiguous"]
    chosen = ambiguous.decision
    assert chosen.request_id == "r-7-0000"
    assert chosen.outcome == Outcome.REFER
    assert chosen.tier == Tier.AGENT
    assert chosen.reason == "ambiguous_order"
    assert chosen.policy_citation is None
    assert len(ambiguous.reads) == 1
    name, result = ambiguous.reads[0]
    assert (name, result.ok, result.reason) == ("order.lookup", False, "ambiguous_order")
    assert ambiguous.state.order is None
    assert ambiguous.state.shipment_events == []
    assert ambiguous.state.policy is None
    assert run[0]["matches"] == 2


def test_audit_records_are_counted_from_the_actual_runs(run: tuple[dict, str]) -> None:
    approved, ambiguous = run[0]["approved"], run[0]["ambiguous"]
    assert approved.tools.audits == {}
    records = list(ambiguous.tools.audits.values())
    assert len(records) == 1
    record = records[0]
    assert record.event == "decision"
    assert record.decision == ambiguous.decision
    assert record.trace_id == "baseline-r-7-0000"
    assert record.decision.reason == "ambiguous_order"
    assert not record.money_moved
    assert record.paid_amount_cents == 0


def test_baseline_calls_no_write_tool_and_executes_nothing(run: tuple[dict, str]) -> None:
    for result in (run[0]["approved"], run[0]["ambiguous"]):
        assert not result.decision.executed
        assert result.tools.payment_for(result.state.request.request_id) is None
        assert result.tools.proposals == {}
        assert result.tools.reviews == {}
        assert all(
            name in {"order.lookup", "shipment.status", "policy.retrieve"}
            for name, reply in result.reads
        )


def test_printed_facts_and_draft_choices(run: tuple[dict, str]) -> None:
    assert run[1].splitlines() == [
        "r-7-0007 asks 3800 cents for a change of mind",
        "order.lookup: o-0007 paid 4200 cents; refunded 0 cents",
        "shipment.status: delivered 12 days before the request",
        "policy.retrieve: refund_cap_v4 version 2026-03-01; window 21 days",
        "draft: approve, tier 1, eligible",
        "cites refund_cap_v4@2026-03-01#eligibility",
        "audit records: 0",
        "r-7-0000: order.lookup failed: ambiguous_order (2 matching orders)",
        "draft: refer, tier 1, ambiguous_order; no citation",
        "audit records: 1; decision refer, ambiguous_order",
        "draft kinds: 3 (approve, deny, refer); write tool calls: 0; executions: 0",
    ]
    assert set(Outcome) == {Outcome.APPROVE, Outcome.DENY, Outcome.REFER}
