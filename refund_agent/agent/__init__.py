"""A bounded state machine; model text never authorizes a payment."""

from datetime import datetime
from types import MappingProxyType as _ReadOnly

from pydantic import ValidationError

from refund_agent import contracts as c
from refund_agent.contracts import Trace, TraceStep, _detach, _utc_time, _validated
from refund_agent.model import ModelClient, Reply, ScriptedModel
from refund_agent.rules import decision, tier_for
from refund_agent.tools import AuditWriteError, PaymentError, ReconciliationError, Tools

READ_NEXT = _ReadOnly(
    {
        c.Phase.ORDER: c.Phase.SHIPMENT,
        c.Phase.SHIPMENT: c.Phase.POLICY,
        c.Phase.POLICY: c.Phase.ASSESS,
    }
)


def _review_call(tools, state, call, steps):
    try:
        result = tools.call(call, state)
    except BaseException as error:
        steps.append(
            _review_step(state, call, c.ToolResult(ok=False, reason=type(error).__name__), steps)
        )
        raise
    steps.append(_review_step(state, call, result, steps))
    if (
        not result.ok
        and call.name != "shipment.status"
        and not (call.name == "order.lookup" and state.order is None)
    ):
        raise RuntimeError(f"named_review_unavailable:{result.reason}")
    return result


def _review_step(state, call, result, steps):
    phase = next(p for p, name in c.PHASE_TO_TOOL.items() if name == call.name)
    return c.TraceStep(
        index=len(steps), phase=phase, call=call, result=result, state=_detach(state)
    )


def _review_calls(state):
    for name, needed in (
        ("order.lookup", state.order is None),
        ("shipment.status", state.order is None or not state.shipment_events),
        ("policy.retrieve", state.policy is None),
    ):
        if needed:
            yield c.ToolCall(name=name)
    if state.proposal_id is None:
        yield c.ToolCall(
            name="refund.propose",
            arguments=c.ToolArguments(
                amount_cents=state.request.amount_cents, reason=state.request.reason
            ),
        )
    yield c.ToolCall(name="approval.request")


def _open_named_review(tools, state, steps):
    for call in _review_calls(state):
        result = _review_call(tools, state, call, steps)
        if call.name == "order.lookup" and not result.ok:
            result = tools._request_review(state, result.reason)
            steps.append(_review_step(state, c.ToolCall(name="approval.request"), result, steps))
            break
    review = tools._named_reviews().get(result.record_id or "")
    if review is None or review.required_role != "named_approver":
        raise RuntimeError(f"named_review_unavailable:{result.reason}")
    if not review.open:
        tools._write("reviews", review.review_id, review.model_copy(update={"open": True}))


def ensure_named_review(tools: Tools, state: c.State, steps: list[TraceStep]) -> None:
    """Complete required review even over budget; trace every finalization call."""
    if state.request.amount_cents <= 50000 or state.receipt:
        return
    assert state.decision is not None
    reason = state.decision.reason
    state.decision = decision(state, c.Outcome.REFER, reason)
    reviews = tools._named_reviews()
    if any(
        r.open
        and r.required_role == "named_approver"
        and (r.proposal.request if r.proposal is not None else r.request) == state.request
        for r in reviews.values()
    ):
        return
    review_state = _detach(state)
    review_state.execution_at = max(state.execution_at or state.request.at, state.request.at)
    _open_named_review(tools, review_state, steps)
    for field in (
        "order",
        "shipment_events",
        "policy",
        "policy_version",
        "freshness",
        "risk_score",
        "proposal_id",
    ):
        setattr(state, field, getattr(review_state, field))
    state.decision = decision(state, c.Outcome.REFER, reason)


class Agent:
    def __init__(self, tools: Tools, model: ModelClient | None = None, max_steps: int = 12) -> None:
        if type(max_steps) is not int or max_steps < 1:
            raise ValueError("step budget must be a positive integer")
        self._tools = tools
        self.model = model if model is not None else ScriptedModel()
        self.max_steps = max_steps

    @property
    def tools(self) -> Tools:
        return self._tools

    # region book:fnd.agentic-loop.run-01
    def run(self, request: c.Request, *, at: datetime | None = None) -> Trace:
        request = _validated(c.Request, request)
        state = c.State(request=request, trace_id=f"trace-{request.request_id}")
        steps: list[TraceStep] = []
        try:
            with self.tools._boundary(state, "run"):
                self._start(state, at)
                # Reserve the final step for audit, including budget exhaustion.
                while state.phase != c.Phase.AUDIT and len(steps) < self.max_steps - 1:
                    steps.append(self._step(state, len(steps)))
                if state.phase != c.Phase.AUDIT:
                    self._refer(state, "step_budget_exhausted")
                ensure_named_review(self.tools, state, steps)
                return self._trace(state, steps, self.tools.record_decision(state))
        except ReconciliationError:
            raise
        except (AuditWriteError, PaymentError) as error:
            if not isinstance(error, PaymentError) or error.confirmed:
                self._error_trace(state, steps, error)
            raise

    # endregion book:fnd.agentic-loop.run-01

    def _start(self, state: c.State, at: datetime | None) -> None:
        try:
            state.execution_at = (
                _utc_time(at) if at is not None else self.tools.run_time(state.request)
            )
        except ValueError:
            state.failure_detail = "invalid_execution_time"
            self._refer(state, state.failure_detail)
            return
        if state.execution_at < state.request.at:
            self._refer(state, "execution_before_request")
            return
        completed = self.tools.completed_state(state.request)
        if completed:
            state.__dict__.update(vars(completed))
        if state.decision is not None:
            state.phase = c.Phase.AUDIT

    def _error_trace(
        self, state: c.State, steps: list[TraceStep], error: AuditWriteError | PaymentError
    ) -> None:
        try:
            self._refer(state, "audit_unavailable")
            state.failure_detail = str(error)
            ensure_named_review(self.tools, state, steps)
            error.trace = self._trace(state, steps, c.ToolResult(ok=False, reason=str(error)))
        except BaseException:
            pass  # Ledger-derived error facts remain available if the trace fails.

    def _refer(self, state: c.State, reason: str) -> None:
        receipt = state.receipt
        state.decision = receipt.decision if receipt else decision(state, c.Outcome.REFER, reason)
        state.phase = c.Phase.AUDIT
        if receipt:
            state.failure_detail = reason

    def _validated_reply(self, state: c.State) -> Reply:
        try:
            reply = self.model.reply(_detach(state))
        except BaseException as error:
            if isinstance(error, (AuditWriteError, PaymentError)) and error._owner is self.tools:
                raise
            raise RuntimeError("model_failed") from error
        if type(reply) in (c.ToolCall, c.Decision):
            return _validated(type(reply), reply)
        raise TypeError("model reply must be a c.ToolCall or c.Decision")

    def _step(self, state: c.State, index: int) -> TraceStep:
        phase, source = state.phase, "model"
        call, chosen, rejected = None, None, None
        try:
            reply = self._validated_reply(state)
            call = reply if isinstance(reply, c.ToolCall) else None
            chosen = reply if isinstance(reply, c.Decision) else None
            source = "tool" if call and call.name == c.PHASE_TO_TOOL.get(phase) else "model"
            result = self._process_reply(state, reply)
        except (AuditWriteError, PaymentError):
            raise
        except Exception as error:
            result, rejected = self._step_failure(state, phase, source, error)
        return TraceStep(
            index=index,
            phase=phase,
            call=call,
            reply=chosen,
            result=result,
            rejected_reply=rejected,
            state=_detach(state),
        )

    def _step_failure(
        self, state: c.State, phase: c.Phase, source: str, error: Exception
    ) -> tuple[c.ToolResult, str | None]:
        state.receipt = self.tools.payment_for(state.request.request_id)
        invalid = isinstance(error, ValidationError)
        reason = "invalid_tool_arguments" if invalid else f"{source}_error"
        rejected = "Rejected reply: contract validation failed." if invalid else None
        self._refer(state, reason)
        operation = (
            c.PHASE_TO_TOOL.get(phase, phase.value) if source == "tool" or rejected else phase.value
        )
        detail = f"{reason}:{operation}:{type(error.__cause__ or error).__name__}"
        state.failure_detail = detail
        return c.ToolResult(ok=False, reason=reason if rejected else detail), rejected

    # region book:fnd.agentic-loop.step-01
    def _process_reply(self, state: c.State, reply: Reply) -> c.ToolResult:
        if isinstance(reply, c.ToolCall) and reply.name == "refund.execute":
            if state.request.amount_cents > 50000:
                self._refer(state, "approval_required")
                return c.ToolResult(ok=False, reason="approval_required")
        if state.phase == c.Phase.ASSESS and isinstance(reply, c.Decision):
            return self._assess(state, reply)
        if not isinstance(reply, c.ToolCall) or reply.name != c.PHASE_TO_TOOL.get(state.phase):
            self._refer(state, "unexpected_tool")
            return c.ToolResult(ok=False, reason="unexpected_tool")
        result = self.tools.call(reply, state)
        if result.ok:
            self._advance(state, state.phase)
        else:
            reason = result.reason or "tool_rejected"
            self._refer(state, reason)
            state.failure_detail = state.failure_detail or f"{reply.name}:{reason}"
        return result

    def _assess(self, state: c.State, reply: c.Decision) -> c.ToolResult:
        assessed = self.tools.assess(state)
        changed = self.tools.changed_approval(state)
        if changed:
            state.decision, state.phase = decision(state, c.Outcome.REFER, changed), c.Phase.PROPOSE
        elif assessed.tier == c.Tier.NAMED_APPROVER and assessed.outcome != c.Outcome.APPROVE:
            state.decision = decision(state, c.Outcome.REFER, assessed.reason)
            state.phase = c.Phase.PROPOSE
        elif assessed.outcome != c.Outcome.APPROVE:
            state.decision, state.phase = assessed, c.Phase.AUDIT
        elif reply.model_dump(exclude={"reason"}) != assessed.model_dump(exclude={"reason"}):
            self._refer(state, "invalid_decision")
        else:
            state.phase = c.Phase.PROPOSE
        return c.ToolResult(
            ok=state.decision is None or state.decision.reason != "invalid_decision",
            reason=state.decision.reason if state.decision else "eligible",
        )

    def _advance(self, state: c.State, phase: c.Phase) -> None:
        if phase in READ_NEXT:
            state.phase = READ_NEXT[phase]
        elif phase == c.Phase.PROPOSE:
            tier = tier_for(state.request.amount_cents)
            approval = self.tools.approval_for(state)
            state.approvals = [approval] if approval else []
            may_execute = tier == c.Tier.AGENT or (tier == c.Tier.REVIEWER and bool(approval))
            ready = state.decision is None and may_execute
            state.phase = c.Phase.EXECUTE if ready else c.Phase.APPROVAL
        elif phase == c.Phase.APPROVAL:
            self._refer(state, state.decision.reason if state.decision else "approval_required")
        elif phase == c.Phase.EXECUTE:
            state.phase = c.Phase.AUDIT
            if state.receipt is None:
                self._refer(state, "execution_not_recorded")

    # endregion book:fnd.agentic-loop.step-01

    def _trace(self, state: c.State, steps: list[TraceStep], result: c.ToolResult) -> Trace:
        if state.receipt:
            state.decision = state.receipt.decision
        assert state.decision is not None
        state.phase = c.Phase.DONE
        final = TraceStep(
            index=len(steps),
            phase=c.Phase.AUDIT,
            call=c.ToolCall(name="audit.append"),
            result=result,
            state=_detach(state),
        )
        steps.append(final)
        return Trace(
            trace_id=state.trace_id,
            steps=steps,
            decision=state.decision,
            receipt=state.receipt,
            failure_detail=state.failure_detail,
        )
