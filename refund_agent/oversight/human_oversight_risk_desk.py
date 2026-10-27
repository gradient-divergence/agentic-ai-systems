"""Authenticated review intake; payment authorization still belongs to the gate.

The required authenticate callback is chapter 35's full credential check. It must
return a ReviewerContext only after verifying seal, issuer, scope, clock and
revocation. Chapter 37 provides no issuer, bypass or substitute check.
"""

from datetime import datetime, timedelta
from typing import Callable, Literal

from refund_agent import contracts as c
from refund_agent.oversight.human_oversight_risk_lifecycle import (
    Notice,
    ReviewEvent,
    add_desk_hours,
    expiry_risk,
    open_ticket,
)
from refund_agent.rules import draft, evidence_problem, tier_for


class ReviewerContext(c.Record):
    credential_id: c.Identity
    reviewer: c.Identity
    role: Literal["reviewer", "named_approver"]


class ProposedExplanation(c.Record):
    """Required view of chapter 36's generated AND replayed DenialExplanation."""
    request_id: str
    policy_version: str
    proposed: bool
    rendered: str


class ReviewResult(c.Record):
    action: Literal["approved", "refused", "modified", "escalated"]
    request_state: Literal["denial_confirmed", "paid", "renewed_review", "waiting"]
    reason: str
    review_id: str
    receipt_id: str | None = None
    new_review_id: str | None = None


Authenticate = Callable[[object, str, datetime], ReviewerContext]


class ReviewDesk:
    def __init__(self, tools, authenticate: Authenticate, *, decision_owner: str, denial_explainer, mode: str):
        if not callable(authenticate) or not callable(denial_explainer) or not decision_owner:
            raise ValueError("authenticated reviewer intake and decision owner required")
        self.tools, self.authenticate, self.decision_owner = tools, authenticate, decision_owner
        self.denial_explainer = denial_explainer
        if mode not in ("execute", "shadow"):
            raise ValueError("review desk needs its execution mode")
        self.mode = mode
        self.tickets, self.packets, self.confirmed_denials = {}, {}, {}
        self.work = []

    def open(self, proposal: c.Proposal | c.ReviewItem, proposed: c.Decision,
             explanation: ProposedExplanation | None = None, *, reviewer: str,
             opened_at: datetime | None = None):
        proposed = c._validated(c.Decision, proposed)
        if isinstance(proposal, c.ReviewItem):
            proposal = c._validated(c.ReviewItem, proposal)
            if (
                proposal.proposal is not None or proposal.request is None or proposal.named_approver is None
                or not proposal.open or proposed.outcome != c.Outcome.REFER or explanation is not None
                or self.tools._named_reviews().get(proposal.review_id) != proposal
            ):
                raise ValueError("request_only_review_packet")
            named_approver = proposal.named_approver
        else:
            proposal = c._validated(c.Proposal, proposal)
            named_approver = proposal.evidence.policy.named_approver
        request = proposal.request
        if request is None:
            raise ValueError("review_packet_mismatch")
        if (proposed.request_id != request.request_id or proposed.amount_cents != request.amount_cents
                or proposed.tier != tier_for(request.amount_cents)):
            raise ValueError("review_packet_mismatch")
        if proposed.outcome == c.Outcome.DENY:
            if not isinstance(proposal, c.Proposal) or explanation is None:
                raise ValueError("denial requires its replayed proposed explanation")
            explanation = c._validated(ProposedExplanation, explanation)
            if (explanation.request_id != request.request_id
                    or explanation.policy_version != proposal.evidence.policy.version
                    or not explanation.proposed or not explanation.rendered.strip()):
                raise ValueError("denial requires its replayed proposed explanation")
        ticket = open_ticket(proposal, named_approver=named_approver,
                             decision_owner=self.decision_owner, reviewer=reviewer,
                             queue_length=sum(t.closed_at is None for t in self.tickets.values()),
                             opened_at=opened_at)
        if ticket.review_id not in self.tickets:
            self.tickets[ticket.review_id] = ticket
            self.packets[ticket.review_id] = (proposal, proposed, explanation)
        return self.tickets[ticket.review_id]

    def _identity(self, ticket, reviewer, credential, at):
        if self.mode != "execute":
            raise ValueError("review_outcome_requires_executed_lane")
        if credential is None:
            raise ValueError("credential_missing")
        proof = c._validated(ReviewerContext, self.authenticate(credential, reviewer, at))
        person = next((p for p in self.tools.data.reviewers if p.reviewer_id == reviewer), None)
        proposal = self.packets[ticket.review_id][0]
        policy = self.tools.data.policy_at(at)
        if (person is None or proof.reviewer != reviewer or proof.role != person.role
                or reviewer in (proposal.request.preparer_id, proposal.request.requester_id)
                or reviewer == self.decision_owner
                or (reviewer not in ticket.path[:-1] and reviewer != policy.named_approver)):
            raise ValueError("credential_reviewer")
        if ticket.tier == c.Tier.NAMED_APPROVER and (
            reviewer != policy.named_approver or person.role != "named_approver"
        ):
            raise ValueError("named_approver_required")
        return proof

    def _append(self, ticket, kind, owner, at, reason, proof, *, outcome=None):
        event = ReviewEvent(kind=kind, owner=owner, at=at, reason=reason,
                            credential_id=proof.credential_id)
        ticket = ticket.model_copy(update={
            "events": ticket.events + (event,),
            **({"outcome": outcome, "closed_at": at} if outcome else {}),
        })
        self.tickets[ticket.review_id] = ticket
        return ticket

    def _close_core_review(self, proposal):
        key = proposal.review_id if isinstance(proposal, c.ReviewItem) else "review-" + proposal.proposal_id
        review = self.tools.reviews[key]
        self.tools._write("reviews", key, review.model_copy(update={"open": False}))
        return key

    def _fresh(self, request, at):
        state = c.State(request=request, trace_id="trace-" + request.request_id,
                        execution_at=at)
        for name in ("order.lookup", "shipment.status", "policy.retrieve"):
            result = self.tools.call(c.ToolCall(name=name), state)
            if not result.ok:
                return state, result.reason
        return state, evidence_problem(self.tools.data, state)

    def _escalate(self, ticket, at, reason):
        # Execution refusals and human refusals escalate even before a deadline.
        step = min(ticket.step + 1, len(ticket.path) - 1)
        due, owner = add_desk_hours(at, 4), ticket.path[step]
        event = ReviewEvent(kind="escalation", owner=owner, at=at, reason=reason, due_at=due)
        notice = Notice(requester_id=ticket.requester_id, owner=owner, at=at, reason=reason, due_at=due)
        updated = ticket.model_copy(update={
            "step": step, "assignee": owner, "due_at": due, "outcome": "escalated",
            "closed_at": None, "events": ticket.events + (event,),
            "notices": ticket.notices + (notice,),
        })
        self.tickets[updated.review_id] = updated
        self.work.append({"kind": "escalation", "minutes": 15, "at": at,
                          "review_id": ticket.review_id, "owner": owner})
        return updated

    def record_refusal(self, review_id, reviewer, credential, reason, at):
        at = c._utc_time(at)
        ticket = self.tickets[review_id]
        if ticket.closed_at is not None or at < ticket.opened_at or not reason.strip():
            raise ValueError("open review and a refusal reason required")
        proof = self._identity(ticket, reviewer, credential, at)
        proposal = self.packets[review_id][0]
        closed = self._append(ticket, "refusal", reviewer, at, reason, proof, outcome="refused")
        key = self._close_core_review(proposal)
        followup = closed.model_copy(update={
            "review_id": review_id + "-refusal", "outcome": "open", "closed_at": None,
        })
        self.packets[followup.review_id] = self.packets[review_id]
        followup = self._escalate(followup, at, "review_refused:" + reason)
        self.tools._write("reviews", key, self.tools.reviews[key].model_copy(update={"open": True}))
        return ReviewResult(action="refused", request_state="waiting", reason=reason,
                            review_id=review_id, new_review_id=followup.review_id)

    def approve(self, review_id, reviewer, credential, at):
        at = c._utc_time(at)
        ticket = self.tickets[review_id]
        if ticket.closed_at is not None or at < ticket.opened_at:
            raise ValueError("open review required")
        proof = self._identity(ticket, reviewer, credential, at)
        proposal, proposed, _ = self.packets[review_id]
        if isinstance(proposal, c.ReviewItem):
            return self._renew_policy(review_id, reviewer, proof, at)
        if self.tools.data.policy_at(at).version != proposal.evidence.policy.version:
            return self._renew_policy(review_id, reviewer, proof, at)
        if proposed.outcome == c.Outcome.DENY:
            state, problem = self._fresh(proposal.request, at)
            if problem:
                self._escalate(ticket, at, problem)
                return ReviewResult(action="escalated", request_state="waiting", reason=problem, review_id=review_id)
            assert state.policy is not None
            current = draft(state)
            if (state.policy.version != proposal.evidence.policy.version
                    or current.outcome != c.Outcome.DENY or current.reason != proposed.reason):
                self._escalate(ticket, at, "denial_proposal_changed")
                return ReviewResult(action="escalated", request_state="waiting",
                                    reason="denial_proposal_changed", review_id=review_id)
            state.decision = current
            self.tools.record_decision(state)
            self.confirmed_denials[proposal.request_id] = current
            self._close_core_review(proposal)
            self._append(ticket, "denial_confirmation", reviewer, at, current.reason, proof, outcome="approved")
            return ReviewResult(action="approved", request_state="denial_confirmed",
                                reason=current.reason, review_id=review_id)
        approval = c.Approval(
            approval_id="human-oversight-risk-approval-" + proposal.proposal_id,
            request_id=proposal.request_id, order_id=proposal.evidence.order.order_id,
            proposal_id=proposal.proposal_id, amount_cents=proposal.amount_cents,
            reason=proposal.reason, policy_version=proposal.evidence.policy.version,
            reviewer=reviewer, role=proof.role, approved_at=at,
        )
        # This is intentionally the required chapter-35 signature, with no retry
        # through unauthenticated Tools.record_approval when it is unavailable.
        result = self.tools.record_approval(approval, credential=credential, at=at)
        if not result.ok:
            return ReviewResult(action="escalated", request_state="waiting",
                                reason=result.reason, review_id=review_id)
        self._append(ticket, "approval", reviewer, at, "authorized_review", proof)
        state = c.State(request=proposal.request, trace_id="trace-" + proposal.request_id,
                        execution_at=at + timedelta(microseconds=1), proposal_id=proposal.proposal_id,
                        order=proposal.evidence.order)
        paid = self.tools.call(c.ToolCall(name="refund.execute", arguments=c.ToolArguments(
            amount_cents=proposal.amount_cents, approval_id=approval.approval_id,
            idempotency_key="refund-" + proposal.request_id)), state)
        ticket = self.tickets[review_id]
        if not paid.ok:
            self._append(ticket, "execution_refused", reviewer, state.execution_at, paid.reason, proof)
            self._escalate(self.tickets[review_id], state.execution_at, "execution_refused:" + paid.reason)
            return ReviewResult(action="escalated", request_state="waiting",
                                reason=paid.reason, review_id=review_id)
        self._append(ticket, "payment", reviewer, state.execution_at, "gated_payment", proof, outcome="approved")
        return ReviewResult(action="approved", request_state="paid", reason="eligible",
                            review_id=review_id, receipt_id=paid.record_id)

    def _renew_policy(self, review_id, reviewer, proof, at):
        """Recovered evidence or an owner-changed policy needs a new bound review."""
        ticket = self.tickets[review_id]
        old = self.packets[review_id][0]
        reason = "request_evidence_available" if isinstance(old, c.ReviewItem) else "policy_version_changed"
        state, problem = self._fresh(old.request, at)
        if problem:
            self._escalate(ticket, at, "policy_renewal:" + problem)
            return ReviewResult(action="escalated", request_state="waiting", reason=problem, review_id=review_id)
        if isinstance(old, c.Proposal) and state.policy.version == old.evidence.policy.version:
            raise ValueError("renewal requires a new policy version")
        result = self.tools.call(c.ToolCall(name="refund.propose", arguments=c.ToolArguments(
            amount_cents=old.request.amount_cents, reason=old.request.reason)), state)
        if not result.ok:
            raise ValueError("policy_renewal:" + result.reason)
        result = self.tools.call(c.ToolCall(name="approval.request"), state)
        if not result.ok:
            raise ValueError("policy_renewal:" + result.reason)
        proposal, proposed = self.tools.proposals[state.proposal_id], draft(state)
        explanation = self.denial_explainer(state, proposed) if proposed.outcome == c.Outcome.DENY else None
        renewed = self.open(proposal, proposed, explanation, reviewer=reviewer)
        self._close_core_review(old)
        self._append(ticket, "modification", reviewer, at, reason, proof, outcome="modified")
        self.work.extend({"kind": kind, "minutes": minutes, "at": at, "review_id": renewed.review_id}
                         for kind, minutes in (("new_proposal", 5), ("new_review", 30 if tier_for(old.request.amount_cents) == 3 else 15)))
        return ReviewResult(action="modified", request_state="renewed_review", reason=reason,
                            review_id=review_id, new_review_id=renewed.review_id)

    def escalate(self, review_id, reviewer, credential, reason, at):
        at = c._utc_time(at)
        ticket = self.tickets[review_id]
        if ticket.closed_at is not None or at < ticket.opened_at or not reason.strip():
            raise ValueError("open review and an escalation reason required")
        proof = self._identity(ticket, reviewer, credential, at)
        updated = self._escalate(ticket, at, reason)
        last = updated.events[-1].model_copy(update={"credential_id": proof.credential_id})
        self.tickets[review_id] = updated.model_copy(update={"events": updated.events[:-1] + (last,)})
        return ReviewResult(action="escalated", request_state="waiting", reason=reason, review_id=review_id)

    def modify(self, review_id, reviewer, credential, new_request, at):
        at = c._utc_time(at)
        ticket = self.tickets[review_id]
        if ticket.closed_at is not None or at < ticket.opened_at:
            raise ValueError("open review required")
        proof = self._identity(ticket, reviewer, credential, at)
        old = self.packets[review_id][0]
        new_request = c._validated(c.Request, new_request)
        if (new_request.request_id == old.request.request_id or new_request.requester_id != old.request.requester_id
                or new_request.order_reference != old.request.order_reference or new_request.at != at):
            raise ValueError("modification needs a new bound request at the review clock")
        data = self.tools.data
        data = data.model_copy(update={"requests": data.requests + [new_request], "retrievals": data.retrievals + [
            c.PolicySnapshot(request_id=new_request.request_id, version=data.policy_at(at).version, observed_at=at)]})
        self.tools.data = data
        state, problem = self._fresh(new_request, at)
        if problem:
            raise ValueError("modification_unprepared:" + problem)
        prepared = self.tools.call(c.ToolCall(name="refund.propose", arguments=c.ToolArguments(
            amount_cents=new_request.amount_cents, reason=new_request.reason)), state)
        if not prepared.ok:
            raise ValueError(prepared.reason)
        result = self.tools.call(c.ToolCall(name="approval.request"), state)
        if not result.ok:
            raise ValueError(result.reason)
        proposal = self.tools.proposals[state.proposal_id]
        proposed = draft(state)
        if proposed.outcome == c.Outcome.DENY:
            raise ValueError("a modified denial needs chapter 36's new explanation")
        renewed = self.open(proposal, proposed, reviewer=reviewer)
        self._close_core_review(old)
        self._append(ticket, "modification", reviewer, at, "new_proposal_required", proof, outcome="modified")
        self.work.extend({"kind": kind, "minutes": minutes, "at": at, "review_id": renewed.review_id}
                         for kind, minutes in (("new_proposal", 5), ("new_review", 15)))
        return ReviewResult(action="modified", request_state="renewed_review", reason="new_proposal_required",
                            review_id=review_id, new_review_id=renewed.review_id)

    def waiting_expiry(self, at, horizon_hours=48):
        proposals = [self.packets[key][0] for key, t in self.tickets.items()
                     if t.closed_at is None and isinstance(self.packets[key][0], c.Proposal)]
        unique = {p.proposal_id: p for p in proposals}
        return expiry_risk(list(unique.values()), at, horizon_hours=horizon_hours)
