"""Review records are separate from the refund's resulting decision and ledger."""

from datetime import datetime, timedelta
from typing import Literal

from pydantic import Field, model_validator

from refund_agent import contracts as c


def respond_by(at: datetime, working_days: int = 1) -> datetime:
    at = c._utc_time(at)
    if type(working_days) is not int or working_days < 1:
        raise ValueError("deadline needs a positive number of working days")
    day = at.replace(hour=17, minute=0, second=0, microsecond=0)
    while working_days:
        day += timedelta(days=1)
        if day.weekday() < 5:
            working_days -= 1
    return day


def add_desk_hours(at: datetime, hours: float) -> datetime:
    at = c._utc_time(at)
    if not 0 < hours < float("inf"):
        raise ValueError("desk hours must be positive and finite")
    remaining = timedelta(hours=hours)
    while remaining:
        opening = at.replace(hour=9, minute=0, second=0, microsecond=0)
        closing = opening + timedelta(hours=8)
        if at.weekday() >= 5 or at >= closing:
            at = opening + timedelta(days=1)
            continue
        at = max(at, opening)
        used = min(remaining, closing - at)
        at += used
        remaining -= used
    return at


class ReviewEvent(c.Record):
    kind: Literal[
        "approval", "refusal", "modification", "escalation",
        "denial_confirmation", "execution_refused", "payment",
    ]
    owner: str
    at: datetime
    reason: str
    due_at: datetime | None = None
    credential_id: str | None = None


class Notice(c.Record):
    requester_id: str
    owner: str
    at: datetime
    due_at: datetime
    reason: str


class ReviewTicket(c.Record):
    review_id: str
    request_id: str
    requester_id: str
    work_class: str
    tier: c.Tier
    opened_at: datetime
    due_at: datetime
    assignee: str
    path: tuple[str, ...]
    queue_length: int = Field(ge=0, strict=True)
    blocking: bool = True
    step: int = Field(default=0, ge=0, strict=True)
    outcome: Literal["approved", "refused", "modified", "escalated", "open"] = "open"
    closed_at: datetime | None = None
    events: tuple[ReviewEvent, ...] = ()
    notices: tuple[Notice, ...] = ()

    @model_validator(mode="after")
    def coherent(self):
        if (not self.path or len(set(self.path)) != len(self.path)
                or self.step >= len(self.path) or self.assignee != self.path[self.step]):
            raise ValueError("ticket needs a distinct path and its current owner")
        if self.due_at < self.opened_at:
            raise ValueError("deadline precedes arrival")
        if self.closed_at is not None and self.closed_at < self.opened_at:
            raise ValueError("close precedes arrival")
        if (self.closed_at is None) != (self.outcome in ("open", "escalated")):
            raise ValueError("terminal review actions close; escalations stay open")
        if self.tier == c.Tier.NAMED_APPROVER and len(self.path) != 2:
            raise ValueError("tier 3 starts with its named approver, then decision owner")
        return self


def open_ticket(proposal: c.Proposal | c.ReviewItem, *, named_approver: str,
                decision_owner: str, reviewer: str, queue_length: int = 0,
                opened_at: datetime | None = None) -> ReviewTicket:
    from refund_agent.rules import tier_for

    request = proposal.request
    if request is None:
        raise ValueError("review ticket needs its request")
    if isinstance(proposal, c.ReviewItem):
        packet_id, packet_at = proposal.review_id, request.at
    else:
        packet_id, packet_at = proposal.proposal_id, proposal.created_at
    at = c._utc_time(opened_at or packet_at)
    tier = tier_for(request.amount_cents)
    path = (named_approver, decision_owner) if tier == c.Tier.NAMED_APPROVER or reviewer == named_approver else (
        reviewer, named_approver, decision_owner,
    )
    return ReviewTicket(
        review_id="human-oversight-risk-" + packet_id,
        request_id=request.request_id, requester_id=request.requester_id,
        work_class=f"tier{int(tier)}", tier=tier, opened_at=at,
        due_at=respond_by(at), assignee=path[0], path=path,
        queue_length=queue_length,
    )


def expiry_risk(proposals: list[c.Proposal], at: datetime, *, horizon_hours: float = 48):
    """Return-window equality is eligible; a later execution is refused by the gate."""
    at = c._utc_time(at)
    if not 0 <= horizon_hours < float("inf"):
        raise ValueError("risk horizon must be nonnegative and finite")
    rows = []
    for proposal in proposals:
        if proposal.reason == "not_delivered":
            continue
        delivered = [e.at for e in proposal.evidence.shipment_events if e.status == "delivered"]
        if not delivered:
            continue
        expires = max(delivered) + timedelta(days=proposal.evidence.policy.return_days)
        if expires <= at + timedelta(hours=horizon_hours):
            rows.append({"request_id": proposal.request_id, "eligible_through": expires,
                         "hours_left": (expires - at).total_seconds() / 3600,
                         "expired": at > expires})
    return rows
