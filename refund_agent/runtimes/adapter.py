"""Reimplement the own loop's step semantics at a generic runtime's hooks.

The original Agent and Tools are unchanged. The comparison tests this separate
implementation, including tools composed by the caller after the payment gate.
"""

from dataclasses import dataclass
from datetime import datetime

from pydantic import ValidationError

from refund_agent import contracts as c
from refund_agent.agent import ensure_named_review
from refund_agent.contracts import _detach, _utc_time, _validated
from refund_agent.model import ScriptedModel
from refund_agent.rules import decision, tier_for
from refund_agent.runtimes.runner import Call, Reply, Runner, RuntimeSettings
from refund_agent.tools import AuditWriteError, PaymentError, ReconciliationError

READ_NEXT = {
    c.Phase.ORDER: c.Phase.SHIPMENT,
    c.Phase.SHIPMENT: c.Phase.POLICY,
    c.Phase.POLICY: c.Phase.ASSESS,
}
ADAPTER_VERSION = "1"


@dataclass(frozen=True)
class AdapterSettings:
    end_at_audit: bool = True
    one_tool_per_phase: bool = True
    refer_on_failure: bool = True
    no_retry: bool = True
    agent_budget: bool = True
    max_steps: int = 12
    retries: int = 1


class RefundAdapter:
    def __init__(self, tools, model=None, settings=None, runner_type=Runner):
        self.tools = tools
        self.model = model if model is not None else ScriptedModel()
        self.settings = settings or AdapterSettings()
        if type(self.settings.max_steps) is not int or self.settings.max_steps < 1:
            raise ValueError("step budget must be a positive integer")
        self.runner = runner_type(
            RuntimeSettings(
                stop_on_no_calls=not self.settings.end_at_audit,
                retries=0 if self.settings.no_retry else self.settings.retries,
                errors_to_model=not self.settings.refer_on_failure,
                max_turns=self.settings.max_steps if self.settings.agent_budget else 8,
                reserve_turns=int(self.settings.end_at_audit),
            )
        )
        self.registry = {name: self._dispatch for name in c.PHASE_TO_TOOL.values()}
        self.session = None
        self._steps = []
        self._source = "model"

    def _refer(self, state, reason):
        state.decision = (
            state.receipt.decision if state.receipt else decision(state, c.Outcome.REFER, reason)
        )
        state.phase = c.Phase.AUDIT
        if state.receipt:
            state.failure_detail = reason

    def _start(self, state, at):
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

    def ask(self, state, session):
        self._source = "model"
        self._phase = state.phase
        self._rejected_before = False
        try:
            reply = self.model.reply(_detach(state))
        except BaseException as error:
            owner = getattr(self.tools, "inner", self.tools)
            if isinstance(error, (AuditWriteError, PaymentError)) and error._owner is owner:
                raise
            raise RuntimeError("model_failed") from error
        if type(reply) in (c.ToolCall, c.Decision):
            reply = _validated(type(reply), reply)
        elif isinstance(reply, list) and not self.settings.one_tool_per_phase:
            calls = [_validated(c.ToolCall, call) for call in reply]
            return Reply([Call(call.name, call) for call in calls], reply)
        else:
            raise TypeError("model reply must be a c.ToolCall or c.Decision")
        calls = [Call(reply.name, reply)] if isinstance(reply, c.ToolCall) else []
        # Decision 1 is authority, not one of the runtime settings being ablated.
        final = not (state.phase == c.Phase.ASSESS and state.request.amount_cents > 50000)
        return Reply(calls, reply, final=final)

    # region book:fnd.agent-frameworks.callbacks-01
    def before_call(self, state, reply):
        if not reply.calls:
            return None
        call = reply.calls[0].arguments
        self._source = (
            "tool" if call.name == c.PHASE_TO_TOOL.get(state.phase) else "model"
        )
        if call.name == "refund.execute" and state.request.amount_cents > 50000:
            self._refer(state, "approval_required")
            self._rejected_before = True
            return [c.ToolResult(ok=False, reason="approval_required")]
        if self.settings.one_tool_per_phase and (
            len(reply.calls) != 1 or call.name != c.PHASE_TO_TOOL.get(state.phase)
        ):
            self._refer(state, "unexpected_tool")
            self._rejected_before = True
            return [c.ToolResult(ok=False, reason="unexpected_tool")]
        return None

    # endregion book:fnd.agent-frameworks.callbacks-01

    def _dispatch(self, state, call):
        return self.tools.call(call.arguments, state)

    def _assess(self, state, reply):
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

    def _advance(self, state, phase):
        if phase in READ_NEXT:
            state.phase = READ_NEXT[phase]
        elif phase == c.Phase.PROPOSE:
            tier = tier_for(state.request.amount_cents)
            approval = self.tools.approval_for(state)
            state.approvals = [approval] if approval else []
            may_execute = tier == c.Tier.AGENT or (tier == c.Tier.REVIEWER and bool(approval))
            state.phase = (
                c.Phase.EXECUTE if state.decision is None and may_execute else c.Phase.APPROVAL
            )
        elif phase == c.Phase.APPROVAL:
            self._refer(state, state.decision.reason if state.decision else "approval_required")
        elif phase == c.Phase.EXECUTE:
            state.phase = c.Phase.AUDIT
            if state.receipt is None:
                self._refer(state, "execution_not_recorded")

    # region book:fnd.agent-frameworks.callbacks-02
    def after_call(self, state, reply, results):
        phase = self._phase
        raw = reply.value
        call = raw if isinstance(raw, c.ToolCall) else None
        chosen = raw if isinstance(raw, c.Decision) else None
        if not reply.calls:
            if phase == c.Phase.ASSESS and chosen is not None:
                result = self._assess(state, chosen)
            else:
                self._refer(state, "unexpected_tool")
                result = c.ToolResult(ok=False, reason="unexpected_tool")
        else:
            result = results[-1]
            if self._rejected_before:
                pass
            elif result.ok:
                self._advance(state, phase)
            elif self.settings.refer_on_failure:
                reason = result.reason or "tool_rejected"
                self._refer(state, reason)
                state.failure_detail = (
                    state.failure_detail or f"{reply.calls[-1].name}:{reason}"
                )
        # endregion book:fnd.agent-frameworks.callbacks-02
        self._record_step(state, phase, call, chosen, result)

    def _record_step(self, state, phase, call, chosen, result):
        self._steps.append(
            c.TraceStep(
                index=len(self._steps),
                phase=phase,
                call=call,
                reply=chosen,
                result=result,
                state=_detach(state),
            )
        )

    # region book:fnd.agent-frameworks.callbacks-03
    def on_error(self, state, reply, error):
        phase = state.phase
        invalid = isinstance(error, ValidationError)
        reason = "invalid_tool_arguments" if invalid else f"{self._source}_error"
        rejected = (
            "Rejected reply: contract validation failed." if invalid else None
        )
        operation = (
            c.PHASE_TO_TOOL.get(phase, phase.value)
            if self._source == "tool" or rejected
            else phase.value
        )
        detail = f"{reason}:{operation}:{type(error.__cause__ or error).__name__}"
        if self.settings.refer_on_failure:
            state.receipt = self.tools.payment_for(state.request.request_id)
            self._refer(state, reason)
            state.failure_detail = detail
        # endregion book:fnd.agent-frameworks.callbacks-03
        raw = reply.value if reply else None
        self._steps.append(
            c.TraceStep(
                index=len(self._steps),
                phase=phase,
                call=raw if isinstance(raw, c.ToolCall) else None,
                reply=raw if isinstance(raw, c.Decision) else None,
                result=c.ToolResult(ok=False, reason=reason if rejected else detail),
                rejected_reply=rejected,
                state=_detach(state),
            )
        )
        return self.settings.refer_on_failure

    def _stop(self, state):
        return state.phase == c.Phase.AUDIT

    def run(self, request, *, at: datetime | None = None) -> c.Trace:
        request = _validated(c.Request, request)
        state = c.State(request=request, trace_id=f"trace-{request.request_id}")
        self._steps = []
        try:
            with self.tools._boundary(state, "run"):
                self._start(state, at)
                self._run_session(state)
                self._finalize(state)
                return self._trace(state)
        except ReconciliationError:
            raise
        except (AuditWriteError, PaymentError) as error:
            if not isinstance(error, PaymentError) or error.confirmed:
                self._error_trace(state, error)
            raise

    def _run_session(self, state):
        self.session = self.runner.run(
            state,
            self.ask,
            self.registry,
            trace_id=state.trace_id,
            before=self.before_call,
            after=self.after_call,
            on_error=self.on_error,
            stop=self._stop,
            fatal_errors=(AuditWriteError, PaymentError),
        )

    def _audit(self, state):
        assert self.session is not None
        if state.phase != c.Phase.AUDIT:
            self._refer(state, "step_budget_exhausted")
        ensure_named_review(self.tools, state, self._steps)
        result = self.runner.invoke(
            state,
            Call("audit.append"),
            {"audit.append": lambda state, call: self.tools.record_decision(state)},
            self.session,
            fatal_errors=(AuditWriteError, PaymentError),
        )
        self.session.messages.append({"role": "tool", "name": "audit.append", "result": result})
        self.session.turns += 1
        state.phase = c.Phase.DONE
        self._steps.append(
            c.TraceStep(
                index=len(self._steps),
                phase=c.Phase.AUDIT,
                call=c.ToolCall(name="audit.append"),
                result=result,
                state=_detach(state),
            )
        )

    def _finalize(self, state):
        if self.settings.end_at_audit:
            self._audit(state)
        else:
            if state.decision is None:
                last = self._steps[-1].reply if self._steps else None
                state.decision = last or decision(state, c.Outcome.REFER, "runtime_turn_limit")
            ensure_named_review(self.tools, state, self._steps)

    def _trace(self, state):
        if state.receipt:
            state.decision = state.receipt.decision
        assert state.decision is not None
        return c.Trace(
            trace_id=state.trace_id,
            steps=self._steps,
            decision=state.decision,
            receipt=state.receipt,
            failure_detail=state.failure_detail,
        )

    def _error_trace(self, state, error):
        self._refer(state, "audit_unavailable")
        state.failure_detail = str(error)
        ensure_named_review(self.tools, state, self._steps)
        state.phase = c.Phase.DONE
        final = c.TraceStep(
            index=len(self._steps),
            phase=c.Phase.AUDIT,
            call=c.ToolCall(name="audit.append"),
            result=c.ToolResult(ok=False, reason=str(error)),
            state=_detach(state),
        )
        error.trace = c.Trace(
            trace_id=state.trace_id,
            steps=[*self._steps, final],
            decision=state.decision,
            receipt=state.receipt,
            failure_detail=state.failure_detail,
        )
