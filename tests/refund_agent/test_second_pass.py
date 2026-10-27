import ast
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from pathlib import Path
from threading import Barrier

import pytest
from test_stage1 import approval_for, evidence, safe_request, with_observation

from refund_agent.agent import Agent
from refund_agent.contracts import (
    Outcome,
    Phase,
    State,
    Tier,
    ToolArguments,
    ToolCall,
    ToolResult,
    Trace,
)
from refund_agent.data.generate import Dataset, generate
from refund_agent.model import Reply, ScriptedModel
from refund_agent.tools import TOOL_NAMES, PaymentError, Tools


def execute(state: State, key: str = "payment", approval_id: str | None = None) -> ToolCall:
    return ToolCall(
        name="refund.execute",
        arguments=ToolArguments(
            amount_cents=state.request.amount_cents,
            idempotency_key=key,
            approval_id=approval_id,
        ),
    )


@pytest.mark.parametrize("seconds", [1, 3600, 86400])
def test_later_approval_works_with_refreshed_evidence(corpus: Dataset, seconds: int) -> None:
    tools, state = evidence(corpus, safe_request(corpus, Tier.REVIEWER))
    approval = approval_for(state).model_copy(
        update={
            "approved_at": state.request.at + timedelta(seconds=seconds),
        }
    )
    assert tools.record_approval(approval).ok
    for source in ("order", "shipment", "policy"):
        tools.data = with_observation(tools.data, state.request, source, approval.approved_at)
    assert tools.call(execute(state, approval_id=approval.approval_id), state).ok
    assert len(tools.payments) == 1


def test_approval_before_proposal_is_rejected(corpus: Dataset) -> None:
    tools, state = evidence(corpus, safe_request(corpus, Tier.REVIEWER))
    approval = approval_for(state).model_copy(
        update={
            "approved_at": state.request.at - timedelta(days=3650),
        }
    )
    assert not tools.record_approval(approval).ok
    assert not tools.call(execute(state, approval_id=approval.approval_id), state).ok
    assert not tools.payments


def test_review_approval_execution_history(corpus: Dataset) -> None:
    request = safe_request(corpus, Tier.REVIEWER)
    tools = Tools(corpus)
    agent = Agent(tools)
    first = agent.run(request)
    approval = approval_for(first.steps[-1].state).model_copy(
        update={
            "approved_at": request.at + timedelta(days=1),
        }
    )
    assert tools.record_approval(approval).ok
    for source in ("order", "shipment", "policy"):
        tools.data = with_observation(tools.data, request, source, approval.approved_at)
    second = agent.run(request)
    records = list(tools.audits.values())
    assert second.decision.executed and len(tools.payments) == 1
    assert [r.decision.reason for r in records] == [
        "approval_required",
        "approval_recorded",
        "eligible",
        "eligible",
    ]
    assert {r.trace_id for r in records} == {first.trace_id, second.trace_id}
    assert [r.at for r in records] == sorted(r.at for r in records)
    before = list(tools.audits.values())
    assert agent.run(request).decision == second.decision
    assert list(tools.audits.values()) == before


def test_retry_after_balance_exhausted(corpus: Dataset) -> None:
    request = safe_request(corpus)
    _, state = evidence(corpus, request)
    assert state.order is not None
    prior = sum(r.amount_cents for r in state.order.prior_refunds if r.at <= request.at)
    order = state.order.model_copy(update={"paid_cents": prior + request.amount_cents})
    data = corpus.model_copy(
        update={"orders": [order if r.order_id == order.order_id else r for r in corpus.orders]}
    )
    tools = Tools(data)
    agent = Agent(tools)
    first = agent.run(request)
    assert agent.run(request).decision == first.decision
    assert first.decision.executed and len(tools.payments) == 1
    assert len(tools.audits) == 2


def test_concurrent_execution_once(corpus: Dataset) -> None:
    tools, state = evidence(corpus, safe_request(corpus))
    rendezvous = Barrier(2)

    def pay(key: str) -> ToolResult:
        rendezvous.wait(timeout=1)
        return tools.call(execute(state, key), state.model_copy(deep=True))

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(
            pool.map(
                pay,
                ("one", "two"),
            )
        )
    assert sum(r.ok for r in results) == 1
    assert len(tools.payments) == 1
    assert state.order is not None
    assert (
        tools.paid_cents(state.order.order_id, state.request.at)
        == state.refunded_cents + state.request.amount_cents
    )
    assert len(tools.audits) == 3


@pytest.mark.parametrize("tool", TOOL_NAMES)
@pytest.mark.parametrize("failure", ["raise", "after_commit"])
def test_tool_failures_finish_audited(
    corpus: Dataset, tool: str, failure: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    tools = Tools(corpus)
    original = tools.call

    def failing_call(call: ToolCall, state: State) -> ToolResult:
        if call.name != tool:
            return original(call, state)
        if "after_commit" in failure:
            original(call, state)
        raise RuntimeError("response lost")

    monkeypatch.setattr(tools, "call", failing_call)
    tier = Tier.REVIEWER if tool == "approval.request" else Tier.AGENT
    try:
        trace = Agent(tools).run(safe_request(corpus, tier))
    except PaymentError as error:
        assert tool == "audit.append" and error.trace is not None
        trace = error.trace
    records = list(tools.audits.values())
    assert records[-1].decision == trace.decision
    moved = tool == "audit.append" or (tool == "refund.execute" and "after_commit" in failure)
    assert trace.decision.outcome == (Outcome.APPROVE if moved else Outcome.REFER)
    assert trace.decision.executed == moved
    assert trace.receipt == (tools.payments[0] if moved else None)
    assert trace.failure_detail
    assert bool(tools.payments) == moved
    assert records[-1].money_moved == moved
    detail = trace.failure_detail if tool == "audit.append" else records[-1].failure_detail
    assert detail is not None
    assert "RuntimeError" in detail
    assert Trace.model_validate_json(trace.model_dump_json()) == trace


@pytest.mark.parametrize("error", [TimeoutError, TypeError, ValueError, RuntimeError])
@pytest.mark.parametrize("phase", [p for p in Phase if p not in (Phase.AUDIT, Phase.DONE)])
def test_model_failures_finish_audited(
    corpus: Dataset, error: type[Exception], phase: Phase
) -> None:
    class FailingModel(ScriptedModel):
        def reply(self, state: State) -> Reply:
            if state.phase == phase:
                raise error("model unavailable")
            return super().reply(state)

    tools = Tools(corpus)
    tier = Tier.REVIEWER if phase == Phase.APPROVAL else Tier.AGENT
    trace = Agent(tools, FailingModel()).run(safe_request(corpus, tier))
    assert trace.decision.outcome == Outcome.REFER
    assert next(iter(tools.audits.values())).decision == trace.decision
    assert not tools.payments
    step = next(s for s in trace.steps if s.phase == phase)
    assert error.__name__ in step.result.reason
    detail = next(iter(tools.audits.values())).failure_detail
    assert detail is not None and error.__name__ in detail
    assert Trace.model_validate_json(trace.model_dump_json()) == trace


@pytest.mark.parametrize("name", [123, None, ["refund.execute"]])
def test_rejected_names_round_trip(corpus: Dataset, name: object) -> None:
    request = safe_request(corpus)
    reply = ToolCall.model_construct(name=name, arguments={})

    class MalformedModel(ScriptedModel):
        def reply(self, state: State) -> Reply:
            return reply

    model = MalformedModel()
    tools = Tools(corpus)
    trace = Agent(tools, model).run(request)
    assert trace.decision.outcome == Outcome.REFER and not tools.payments
    assert Trace.model_validate_json(trace.model_dump_json(warnings=False)) == trace


@pytest.mark.parametrize(
    "failure,reason",
    [("stale_carrier", "stale_shipment"), ("hidden_instruction", "suspicious_request")],
)
def test_forged_summaries_cannot_authorize(corpus: Dataset, failure: str, reason: str) -> None:
    request = next(r for r in corpus.requests if r.failure == failure)
    tools, state = evidence(corpus, request)
    state.freshness["shipment"] = 0
    state.risk_score = 0.1
    result = tools.call(execute(state), state)
    assert not result.ok and result.reason == reason
    assert not tools.payments


def test_explanation_is_not_authorization(corpus: Dataset) -> None:
    class ExplainedModel(ScriptedModel):
        def reply(self, state: State) -> Reply:
            reply = super().reply(state)
            return (
                reply.model_copy(update={"reason": "Eligible under the current policy."})
                if state.phase == Phase.ASSESS
                else reply
            )

    tools = Tools(corpus)
    trace = Agent(tools, ExplainedModel()).run(safe_request(corpus))
    assert trace.decision.executed and len(tools.payments) == 1
    assert len(tools.audits) == 2


def test_no_instructor_cases_by_default() -> None:
    data = generate(7)
    assert len(data.public_scenarios) == 40
    assert data.instructor_scenarios == []


def test_fake_execution_cannot_earn_perfect_score(corpus: Dataset) -> None:
    from refund_agent.baseline import score_decisions
    from refund_agent.contracts import ToolResult

    class SkipsExecution(Tools):
        def _execute(self, call: ToolCall, state: State) -> ToolResult:
            return ToolResult(ok=True, record_id="fake")

    tools = SkipsExecution(corpus)
    traces = [Agent(tools).run(r) for r in corpus.requests[:50]]
    score = score_decisions([t.decision for t in traces], corpus.correct[:50])
    assert score.correct < 50 and score.executed == 0
    assert not tools.payments


def _qualified_functions(node, prefix):
    for child in ast.iter_child_nodes(node):
        name = prefix
        if isinstance(child, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            name = f"{prefix}.{child.name}"
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                yield name, child
        yield from _qualified_functions(child, name)


def test_functions_fit_teaching_limit() -> None:
    assembly_exemptions = [
        "refund_agent.adaptation.prompt.StructuredPrompt.render",  # Fixed prompt text assembly.
        "refund_agent.adaptation.suite.RegressionSuite.build",  # Spent-label suite record assembly.
        "refund_agent.adaptation.suite.constructed_requests",  # Declared exercise record assembly.
        # Printed listing kept intact.
        "refund_agent.rules.approval_problem",
        # Printed listing kept intact.
        "refund_agent.verifier.independent.VerifiedTools._authorize",
    ]
    root = Path(__file__).resolve().parents[2] / "refund_agent"
    oversized = []
    seen_exemptions = set()
    for path in root.rglob("*.py"):
        module = "refund_agent." + ".".join(path.relative_to(root).with_suffix("").parts)
        for name, node in _qualified_functions(ast.parse(path.read_text()), module):
            assert node.end_lineno is not None
            limit = 75 if name in assembly_exemptions else 32
            if node.end_lineno - node.lineno + 1 > limit:
                oversized.append((name, node.end_lineno - node.lineno + 1, limit))
            if name in assembly_exemptions:
                seen_exemptions.add(name)
    assert seen_exemptions == set(assembly_exemptions)
    assert not oversized


def test_timestamp_validator_is_typed() -> None:
    from typing import get_type_hints

    from refund_agent.contracts import Record

    assert get_type_hints(Record.utc_dates) == {"value": object, "return": object}


def test_instructor_seed_is_separate() -> None:
    public = generate(7)
    first = generate(7, instructor_seed=123)
    second = generate(7, instructor_seed=456)
    assert len(first.instructor_scenarios) == len(second.instructor_scenarios) == 20
    assert first.requests[:600] == second.requests[:600] == public.requests
    assert first.correct[:600] == second.correct[:600] == public.correct
    assert len(first.requests) == 620
    assert first.requests[600:] != second.requests[600:]
    assert set(first.public_scenarios).isdisjoint(first.instructor_scenarios)
    assert not set(first.instructor_scenarios).intersection(
        r.request_id for r in first.requests[:600]
    )


def test_independent_safety_scorecard() -> None:
    from refund_agent.evaluation import safety_evaluation

    guarded, ungated = safety_evaluation()
    assert guarded.correct == guarded.total == 8
    assert guarded.audited == 8 and guarded.unauthorized_cents == 0
    assert ungated.correct < guarded.correct
    assert ungated.unauthorized_cents == 999999 and ungated.audited == 8


def test_approval_at_execution_or_future_is_rejected(corpus: Dataset) -> None:
    for seconds in (0, 1):
        tools, state = evidence(corpus, safe_request(corpus, Tier.REVIEWER))
        state.execution_at = state.request.at + timedelta(hours=1)
        approval = approval_for(state).model_copy(
            update={
                "approved_at": state.execution_at + timedelta(seconds=seconds),
            }
        )
        assert tools.record_approval(approval).ok
        assert not tools.call(execute(state, approval_id=approval.approval_id), state).ok
        assert not tools.payments


def test_approval_reason_and_named_person_are_bound(corpus: Dataset) -> None:
    for tier, update in [
        (Tier.REVIEWER, {"reason": "not_delivered"}),
        (Tier.NAMED_APPROVER, {"reviewer": "someone else"}),
    ]:
        tools, state = evidence(corpus, safe_request(corpus, tier))
        approval = approval_for(state, "named_approver").model_copy(update=update)
        assert not tools.record_approval(approval).ok
        assert not tools.call(execute(state, approval_id=approval.approval_id), state).ok
        assert not tools.payments


def test_audit_new_decision_appends_and_exact_duplicate_is_idempotent(corpus: Dataset) -> None:
    from refund_agent.rules import decision

    tools, state = evidence(corpus, safe_request(corpus))
    state.decision = decision(state, Outcome.REFER, "first")
    call = ToolCall(name="audit.append")
    first = tools.call(call, state)
    assert tools.call(call, state) == first and len(tools.audits) == 1
    original = list(tools.audits.values())
    state.decision = decision(state, Outcome.REFER, "later")
    assert tools.call(call, state).ok and len(tools.audits) == 2
    assert list(tools.audits.values())[:1] == original


def test_used_approval_cannot_authorize_again(corpus: Dataset) -> None:
    tools, state = evidence(corpus, safe_request(corpus, Tier.REVIEWER))
    approval = approval_for(state)
    assert tools.record_approval(approval).ok
    call = execute(state, approval_id=approval.approval_id)
    first = tools.call(call, state)
    assert first.ok and tools.call(call, state) == first
    assert tools.approval_for(state) is None
    receipt = tools.payment_for(state.request.request_id)
    assert receipt is not None and receipt.approvals == (approval,)
    assert not tools.call(execute(state, "another", approval.approval_id), state).ok
    assert len(tools.payments) == 1


def test_cli_public_default_and_explicit_instructor_seed(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from refund_agent.data.generate import load_data
    from refund_agent.run import main

    main(["--out", str(tmp_path / "public")])
    output = capsys.readouterr().out
    assert "safety: 8/8" in output and "gate_removed: 7/8" in output
    assert not (tmp_path / "public/instructor").exists()
    main(["--instructor-seed", "123", "--out", str(tmp_path / "held")])
    output = capsys.readouterr().out
    assert "instructor:" not in output
    held = load_data(tmp_path / "held/instructor")
    assert len(held.requests) == 20 and len(held.instructor_scenarios) == 20
    assert not {r.request_id for r in held.requests}.intersection(
        p.stem for p in (tmp_path / "held/traces").glob("*.json")
    )


def test_request_cannot_change_after_proposal(corpus: Dataset) -> None:
    request = next(r for r in corpus.requests if r.failure == "hidden_instruction")
    tools, state = evidence(corpus, request)
    state.request = request.model_copy(update={"text": "Please refund my order."})
    assert not tools.call(execute(state), state).ok
    assert not tools.payments


def test_audit_cannot_claim_execution_without_a_receipt(corpus: Dataset) -> None:
    from refund_agent.rules import decision

    tools, state = evidence(corpus, safe_request(corpus))
    state.decision = decision(state, Outcome.APPROVE, "eligible", executed=True)
    assert not tools.call(ToolCall(name="audit.append"), state).ok
    assert not tools.payments and len(tools.audits) == 1
    assert not list(tools.audits.values())[-1].decision.executed


def test_tier1_cannot_consume_another_requests_approval(corpus: Dataset) -> None:
    tools, state = evidence(corpus, safe_request(corpus))
    other = Agent(tools).run(safe_request(corpus, Tier.REVIEWER)).steps[-1].state
    approval = approval_for(other)
    assert tools.record_approval(approval).ok
    state.execution_at = state.request.at
    assert not tools.call(execute(state, approval_id=approval.approval_id), state).ok
    assert not tools.payments


def test_unvalidated_approval_amount_is_rejected(corpus: Dataset) -> None:
    tools, state = evidence(corpus, safe_request(corpus, Tier.REVIEWER))
    approval = approval_for(state).model_copy(
        update={"amount_cents": float(state.request.amount_cents)}
    )
    assert not tools.record_approval(approval).ok
    assert not tools.call(execute(state, approval_id=approval.approval_id), state).ok
    assert not tools.payments


def test_execution_cannot_precede_the_proposal(corpus: Dataset) -> None:
    tools, state = evidence(corpus, safe_request(corpus))
    state.execution_at = state.request.at - timedelta(days=3650)
    assert not tools.call(execute(state), state).ok
    assert not tools.payments


def test_forged_retrieved_policy_cannot_authorize(corpus: Dataset) -> None:
    request = next(r for r in corpus.requests if r.failure == "cached_policy")
    tools = Tools(corpus)
    state = State(request=request, trace_id=f"trace-{request.request_id}")
    for name in ("order.lookup", "shipment.status", "policy.retrieve"):
        assert tools.call(ToolCall(name=name), state).ok
    state.policy = corpus.policy_at(request.at)
    state.policy_version = state.policy.version
    assert tools.call(
        ToolCall(
            name="refund.propose",
            arguments=ToolArguments(amount_cents=request.amount_cents, reason=request.reason),
        ),
        state,
    ).ok
    result = tools.call(execute(state), state)
    assert not result.ok and result.reason == "non_current_policy"
    assert not tools.payments


def test_completed_retry_keeps_committed_evidence(corpus: Dataset) -> None:
    request = safe_request(corpus)
    tools = Tools(corpus)
    agent = Agent(tools)
    first = agent.run(request)
    original = list(tools.audits.values())
    tools.data = with_observation(corpus, request, "order", request.at + timedelta(days=1))
    assert agent.run(request).decision == first.decision
    assert list(tools.audits.values()) == original
    assert len(tools.payments) == 1


def test_evidence_failure_does_not_block_failure_audit(
    corpus: Dataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    def unavailable(data: Dataset, state: State) -> None:
        raise RuntimeError("evidence processing unavailable")

    monkeypatch.setattr("refund_agent.tools.refresh_summaries", unavailable)
    tools = Tools(corpus)
    trace = Agent(tools).run(safe_request(corpus))
    assert trace.decision.outcome == Outcome.REFER
    assert trace.decision.reason == "tool_error" and not tools.payments
    record = next(iter(tools.audits.values()))
    assert record.decision == trace.decision
    assert record.failure_detail == "tool_error:order.lookup:RuntimeError"
    assert Trace.model_validate_json(trace.model_dump_json()) == trace


def test_intent_audit_failure_prevents_payment(
    corpus: Dataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    from refund_agent.contracts import AuditRecord
    from refund_agent.tools import AuditWriteError

    tools = Tools(corpus)
    original = tools._store_audit

    def unavailable(record: AuditRecord) -> ToolResult:
        if record.event == "intent":
            raise RuntimeError("intent audit unavailable")
        return original(record)

    monkeypatch.setattr(tools, "_store_audit", unavailable)
    with pytest.raises(AuditWriteError, match="audit_intent_failed") as caught:
        Agent(tools).run(safe_request(corpus))
    assert not tools.payments and not tools.audits
    trace = caught.value.trace
    assert trace is not None and trace.decision.reason == "audit_unavailable"
    assert Trace.model_validate_json(trace.model_dump_json()) == trace
