"""Chapter 36's stage-6 regions, before the core decision becomes effective.

Chapters 38 and 37 wrap this path in authority, then tier-routing order. Their
components are required at integration, not replaced by empty handlers here.
"""

from collections.abc import Callable
from dataclasses import dataclass

from refund_agent import contracts as c
from refund_agent.audit.audits_explanations_explain import (
    DENIAL_REASONS,
    DenialExplanation,
    explain,
    replay,
)
from refund_agent.audit.audits_explanations_holds import AuditHolds
from refund_agent.audit.audits_explanations_rates import AuditRule
from refund_agent.audit.audits_explanations_records import (
    EvaluationRecord,
    ExplanationContext,
    ReviewRoute,
    SegmentAssignments,
)
from refund_agent.memory.register import Snapshot
from refund_agent.rules import sources
from refund_agent.tools import Tools


@dataclass(frozen=True)
class AuditInputs:
    rule: AuditRule
    segments: SegmentAssignments
    register: Snapshot
    history: dict[str, tuple[EvaluationRecord, ...]]
    review_routes: dict[str, ReviewRoute]
    may_act: Callable[[str, str], bool]
    holds: AuditHolds


class ExplanationAttachment(c.Record):
    request_id: str
    proposed_decision: c.Decision
    explanation: DenialExplanation | None
    checks: dict[str, bool]
    failure: str | None
    evaluation: EvaluationRecord | None = None


class AuditsExplanationsEvaluatedTools(Tools):
    """Placed beside the core, inside every partner's temporary source view."""

    _evaluations: dict[str, EvaluationRecord]

    def assess(self, state):
        chosen = super().assess(state)
        if chosen.reason in DENIAL_REASONS:
            order, shipment, snapshot = sources(self._data, state)
            if order is None or shipment is None or snapshot is None or state.policy is None:
                raise ValueError("complete evaluated source records are required")
            record = EvaluationRecord(
                request=state.request,
                order=order,
                shipment=shipment,
                policy=state.policy,
                policy_observed_at=snapshot.observed_at,
                at=state.execution_at or state.request.at,
                refunded_cents=state.refunded_cents,
            )
            self._evaluations[state.trace_id] = c._detach(record)
            state.freshness = record.state().freshness
        return chosen


class AuditsExplanationsTools(Tools):
    version = "audits-explanations-stage6-1"

    def __init__(self, data, *, audit_inputs: AuditInputs, **kwargs):
        self.audit_inputs = audit_inputs
        self._explanations = {}
        self._held_requests = set()
        self._evaluations = {}
        super().__init__(data, **kwargs)

    @property
    def explanations(self) -> dict[str, ExplanationAttachment]:
        return c._detach(self._explanations)

    @property
    def held_requests(self) -> frozenset[str]:
        return frozenset(self._held_requests)

    # region chapter36:audits-explanations-audit-hold
    def audit_hold(self, state, chosen):
        if chosen.outcome == c.Outcome.DENY and state.order is not None:
            segment = self.audit_inputs.segments.region_of(
                state.order.order_id
            )
            if self.audit_inputs.holds.is_held(
                segment, chosen.reason, state.execution_at or state.request.at
            ):
                self._held_requests.add(state.request.request_id)
                return chosen.model_copy(update={"outcome": c.Outcome.REFER})
        return chosen

    def assess(self, state):
        return self.audit_hold(state, super().assess(state))

    # endregion chapter36:audits-explanations-audit-hold

    def _explanation_context(self, state):
        current = self._evaluations.get(state.trace_id)
        if current is None:
            raise ValueError("the rule's retained evaluation record is required")
        inputs = self.audit_inputs
        if state.request.request_id not in inputs.review_routes:
            raise ValueError(
                "a person-review ticket and deadline are required"
            )
        return ExplanationContext(
            current,
            inputs.register,
            inputs.history.get(state.request.request_id, ()),
            inputs.may_act,
            inputs.review_routes[state.request.request_id],
        )

    # region chapter36:audits-explanations-explanation
    def _decision(self, state, reason=None):
        chosen = state.decision
        if reason is None and state.receipt is None and chosen is not None:
            state.decision = chosen = self.audit_hold(state, chosen)
            if (
                chosen.outcome in (c.Outcome.DENY, c.Outcome.REFER)
                and chosen.reason in DENIAL_REASONS
            ):
                generated, checks, failure = None, {}, None
                try:
                    context = self._explanation_context(state)
                    generated = explain(
                        state,
                        chosen,
                        context.current.refunded_cents,
                        context=context,
                    )
                    checks = replay(generated, state, context=context)
                    if not all(checks.values()):
                        failure = "replay_failed"
                except ValueError:
                    failure = "source_record_refused"
                if failure:
                    state.decision = chosen.model_copy(
                        update={
                            "outcome": c.Outcome.REFER,
                            "reason": "explanation_failed",
                        }
                    )
                    state.failure_detail = "audits-explanations:" + failure
                self._explanations[state.request.request_id] = ExplanationAttachment(
                    request_id=state.request.request_id,
                    proposed_decision=chosen,
                    explanation=generated,
                    checks=checks,
                    failure=failure,
                    evaluation=self._evaluations.get(state.trace_id),
                )
        return super()._decision(state, reason)

    # endregion chapter36:audits-explanations-explanation


def after_run(
    *, value, tools, request, candidate, settings, at, runtime_settings
):
    chosen = value.decision
    if (
        chosen.outcome == c.Outcome.DENY
        or chosen.outcome == c.Outcome.REFER
        and chosen.reason in DENIAL_REASONS
    ):
        attachment = tools.explanations.get(request.request_id)
        if (
            attachment is None
            or attachment.failure
            or not all(attachment.checks.values())
        ):
            raise ValueError(
                "effective denial has no passing audits-explanations replay"
            )
    return value
