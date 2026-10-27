"""Own registration and routing; the shared stage6 region joins this at merge."""

from refund_agent import contracts as c
from refund_agent.oversight.human_oversight_risk_desk import ProposedExplanation, ReviewDesk
from refund_agent.release.extensions import Component, register
from refund_agent.rules import draft
from refund_agent.tools import Tools

SELECTOR = {"name": "human-oversight-risk.review", "module": __name__}


class OversightTools(Tools):
    version = "human-oversight-risk.review@1"

    def __init__(self, data, *, reviewer_authenticate, denial_explainer, decision_owner,
                 assigned_reviewer, oversight_mode, **kwargs):
        if not callable(denial_explainer):
            raise ValueError("chapter 36's denial generator/replay interface is required")
        self.denial_explainer = denial_explainer
        if assigned_reviewer not in {r.reviewer_id for r in data.reviewers} or assigned_reviewer == decision_owner:
            raise ValueError("a listed assigned reviewer separate from the decision owner is required")
        self.assigned_reviewer = assigned_reviewer
        self.oversight_mode = oversight_mode
        super().__init__(data, **kwargs)
        self.review_desk = ReviewDesk(self, reviewer_authenticate, decision_owner=decision_owner,
                                      denial_explainer=denial_explainer, mode=oversight_mode)
        self.proposed_decisions = {}

    # region chapter:human-oversight-risk.tier-routing
    def assess(self, state):
        proposed = super().assess(state)
        if proposed.tier == c.Tier.REVIEWER and proposed.outcome == c.Outcome.DENY:
            if self.review_desk.confirmed_denials.get(proposed.request_id) != proposed:
                self.proposed_decisions[proposed.request_id] = proposed
                return proposed.model_copy(update={"outcome": c.Outcome.REFER})
        return proposed

    def _review(self, call, state):
        result = super()._review(call, state)
        if result.ok:
            assert result.record_id is not None
            review = self.reviews[result.record_id]
            self._open_packet(review, state)
        return result
    # endregion chapter:human-oversight-risk.tier-routing

    def _open_packet(self, review, state):
        if review.proposal is None:
            self.review_desk.open(review, state.decision, reviewer=self.assigned_reviewer,
                                  opened_at=state.execution_at or state.request.at)
            return
        proposed = self.proposed_decisions.get(state.request.request_id, draft(state))
        explanation = None
        if proposed.outcome == c.Outcome.DENY:
            explanation = c._validated(ProposedExplanation, self.denial_explainer(state, proposed))
        self.review_desk.open(review.proposal, proposed, explanation, reviewer=self.assigned_reviewer)


def after_run(*, value, tools, request, at, runtime_settings, candidate, settings):
    from refund_agent.runtimes.faults import FaultTools

    tools = tools.inner if isinstance(tools, FaultTools) else tools
    if value.decision.tier == c.Tier.NAMED_APPROVER and value.receipt is None:
        state = value.steps[-1].state.model_copy(deep=True)
        for review in tools._named_reviews().values():
            if not review.open:
                continue
            packet_request = review.proposal.request if review.proposal is not None else review.request
            if packet_request == request and (review.proposal is None or tools.oversight_mode == "shadow"):
                tools._open_packet(review, state)
        return value
    if value.decision.tier != c.Tier.REVIEWER or value.receipt is not None:
        return value
    if request.request_id not in tools.proposed_decisions and tools.oversight_mode != "shadow":
        return value
    state = value.steps[-1].state.model_copy(deep=True)
    if state.order is None or state.policy is None:
        raise ValueError("tier2_review_packet_unprepared")
    steps = list(value.steps)
    calls = ("refund.propose", "approval.request") if request.request_id in tools.proposed_decisions else ()
    for name in calls:
        call = c.ToolCall(name=name, arguments=c.ToolArguments(
            amount_cents=request.amount_cents, reason=request.reason))
        result = tools.call(call, state)
        if not result.ok:
            raise ValueError("tier2_review_intake:" + result.reason)
        steps.append(c.TraceStep(index=len(steps), phase=c.Phase.PROPOSE if name == "refund.propose" else c.Phase.APPROVAL,
                                 call=call, result=result, state=state.model_copy(deep=True)))
    if tools.oversight_mode == "shadow":
        for review in tools.shadow_reviews.values():
            packet_request = review.proposal.request if review.proposal is not None else review.request
            if packet_request == request:
                tools._open_packet(review, state)
    return value.model_copy(update={"steps": steps})


def compose(*, base, kwargs, candidate, settings, mode, context):
    required = ("reviewer_authenticate", "denial_explainer", "decision_owner", "assigned_reviewer")
    if any(key not in context for key in required):
        raise ValueError("human-oversight-risk requires authenticated review and explanations")
    return type("HumanOversightRiskTools", (OversightTools, base), {}), {
        **kwargs, **{key: context[key] for key in required},
        "oversight_mode": mode,
    }


register(Component(
    name=SELECTOR["name"], version=OversightTools.version, module=__name__, order=37,
    modules=("refund_agent.oversight.human_oversight_risk_desk",
             "refund_agent.oversight.human_oversight_risk_lifecycle",
             "refund_agent.oversight.human_oversight_risk_deadline"),
    files=("release/owners.json",),
    hooks={"compose": compose, "after_run": after_run},
    slots={"37": "credential-bound tier2 review and deadline escalation"},
))
