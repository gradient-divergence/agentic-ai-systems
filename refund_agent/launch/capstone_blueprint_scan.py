"""Detect missing reviews on effective decisions as well as money movements."""

from datetime import datetime
from typing import Literal

from pydantic import model_validator

from refund_agent import contracts as c
from refund_agent.launch.capstone_blueprint_requirement import (
    Day,
    LaunchRecord,
    Text,
)
from refund_agent.rules import tier_for
from refund_agent.selection import fingerprint


class ReviewOutcome(LaunchRecord):
    request_id: Text
    proposal_id: Text
    proposal_fingerprint: Text
    actor: Text
    actor_kind: Literal["person", "deadline_handler"]
    outcome: Literal["approve", "deny"]
    at: datetime


class EffectiveDecision(LaunchRecord):
    record_id: Text
    day: Day
    requested_day: Day
    request: c.Request
    decision: c.Decision
    at: datetime
    proposal: c.Proposal | None
    actor: Text
    approval: c.Approval | None = None
    review: ReviewOutcome | None = None
    receipt: c.Receipt | None = None

    @model_validator(mode="after")
    def coherent_clock(self):
        c._utc_time(self.at)
        if self.review is not None:
            c._utc_time(self.review.at)
        if self.approval is not None:
            c._utc_time(self.approval.approved_at)
        if self.receipt is not None:
            c._utc_time(self.receipt.at)
        if self.requested_day > self.day:
            raise ValueError("effective decision precedes request day")
        return self


class ReversalEvent(LaunchRecord):
    record_id: Text
    condition_id: Literal[
        "forbidden_payment",
        "budget_spent",
        "tier2_unbound_approval",
        "tier3_without_named_approver",
        "launch_gate_held",
    ]
    day: Day
    at: datetime
    reason: Text
    planned_time_box: bool = False

    @model_validator(mode="after")
    def time_box_is_only_planned_end(self):
        if self.planned_time_box and (
            self.condition_id != "forbidden_payment"
            or self.reason != "canary ended: time_box"
        ):
            raise ValueError(
                "only a planned time-box end is exempt from reversal"
            )
        return self


class LaunchLogs(LaunchRecord):
    decisions: tuple[EffectiveDecision, ...] = ()
    events: tuple[ReversalEvent, ...] = ()
    policy: c.Policy | None = None


def _review_matches(action: EffectiveDecision) -> bool:
    review, proposal = action.review, action.proposal
    return bool(
        review is not None
        and proposal is not None
        and review.actor_kind == "person"
        and review.request_id == action.request.request_id
        and review.proposal_id == proposal.proposal_id
        and review.proposal_fingerprint
        == fingerprint(proposal.model_dump(mode="json"))
        and proposal.request == action.request
        and proposal.request_id == action.request.request_id
        and review.outcome == action.decision.outcome.value
        and proposal.created_at <= review.at <= action.at
    )


def _approval_matches(action: EffectiveDecision, policy: c.Policy) -> bool:
    approval, proposal, receipt = (
        action.approval,
        action.proposal,
        action.receipt,
    )
    if (
        approval is None
        or proposal is None
        or receipt is None
        or action.review is None
        or not _review_matches(action)
    ):
        return False
    return (
        approval in receipt.approvals
        and receipt.order_id == approval.order_id
        and approval.request_id == action.request.request_id
        and approval.proposal_id == proposal.proposal_id
        and approval.order_id == proposal.evidence.order.order_id
        and approval.amount_cents
        == proposal.amount_cents
        == action.request.amount_cents
        and approval.reason == proposal.reason == action.request.reason
        and approval.policy_version
        == proposal.evidence.policy.version
        == policy.version
        and approval.reviewer == action.review.actor
        and approval.role in ("reviewer", "named_approver")
        and approval.reviewer
        not in (
            "refund-agent",
            action.request.requester_id,
            action.request.preparer_id,
        )
        and approval.approved_at == action.review.at <= receipt.at
    )


def scan(
    logs: LaunchLogs, registry, policy: c.Policy, since: float
) -> tuple[ReversalEvent, ...]:
    """Normalized log input includes every terminal audit decision and receipt.

    Intake-to-terminal/receipt reconciliation is a producer responsibility. This
    scan refuses conflicting identities and never interprets a pending referral
    as a decision. Chapter 37 supplies proposal-bound denial confirmations.
    """
    logs = LaunchLogs.model_validate_json(logs.model_dump_json())
    found = {}
    for action in logs.decisions:
        if action.day <= since:
            continue
        request, decision = action.request, action.decision
        if (
            decision.request_id != request.request_id
            or decision.tier != tier_for(request.amount_cents)
            or decision.amount_cents != request.amount_cents
        ):
            raise ValueError("effective decision conflicts with its request")
        if decision.outcome == c.Outcome.REFER:
            continue
        tier = tier_for(request.amount_cents)
        reason = None
        if tier == c.Tier.REVIEWER:
            if action.actor == "deadline_handler" or (
                action.review is not None
                and action.review.actor_kind == "deadline_handler"
            ):
                reason = "deadline handler recorded approval or denial"
            elif not _review_matches(action):
                reason = (
                    "tier 2 effective decision has no bound person's outcome"
                )
            elif (decision.executed or action.receipt is not None) and (
                action.receipt is None or not _approval_matches(action, policy)
            ):
                reason = "tier 2 payment has no approval bound to proposal"
            condition = "tier2_unbound_approval"
        elif tier == c.Tier.NAMED_APPROVER:
            if (
                action.review is None
                or not _review_matches(action)
                or action.review.actor != policy.named_approver
                or (decision.executed or action.receipt is not None)
                and (
                    action.receipt is None
                    or not _approval_matches(action, policy)
                    or action.approval is None
                    or action.approval.role != "named_approver"
                )
            ):
                reason = "tier 3 payment or denial lacks the named approver's outcome"
            condition = "tier3_without_named_approver"
        else:
            continue
        if action.receipt is not None and (
            action.receipt.request != request
            or action.receipt.proposal_id
            != (action.proposal.proposal_id if action.proposal else None)
            or action.receipt.amount_cents != request.amount_cents
        ):
            reason = "payment does not bind to the effective decision's request/proposal"
        if reason:
            found[action.record_id] = ReversalEvent(
                record_id="capstone-blueprint-scan-" + action.record_id,
                condition_id=condition,
                day=action.day,
                at=action.at,
                reason=reason,
            )
    return tuple(sorted(found.values(), key=lambda event: (event.at, event.record_id)))


def gate_events(
    logs: LaunchLogs, registry, since: float
) -> tuple[ReversalEvent, ...]:
    logs = LaunchLogs.model_validate_json(logs.model_dump_json())
    if not logs.decisions:
        return logs.events
    if logs.policy is None:
        raise ValueError("launch scan needs the in-force policy")
    detected = scan(logs, registry, logs.policy, since)
    return logs.events + detected


def stop_event(
    *, record_id: str, trigger: str, day: float, at: datetime
) -> ReversalEvent:
    if trigger not in {"time_box", "defect", "budget_spent"}:
        raise ValueError("only an ended stop is a reversal candidate")
    return ReversalEvent(
        record_id=record_id,
        condition_id="budget_spent"
        if trigger == "budget_spent"
        else "forbidden_payment",
        day=day,
        at=at,
        reason="canary ended: " + trigger,
        planned_time_box=trigger == "time_box",
    )
