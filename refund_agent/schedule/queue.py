"""Training-window drafts, carrying unanswered work without reading outcomes."""

from dataclasses import dataclass, replace
from datetime import datetime, timedelta

from refund_agent.rules import tier_for
from refund_agent.split import CUTS, time_split

from .inputs import REVIEW_MINUTES, START, WORKING_DAYS, morning


@dataclass(frozen=True)
class Draft:
    request_id: str
    at: datetime
    tier: int
    allowed: tuple[str, ...]
    minutes: int
    plan_at: datetime


def training_requests(data):
    # Only timestamps are inspected to exclude held-out requests, before any other
    # field is read. The canonical split then supplies the training window.
    before_cut = [r for r in data.requests if r.at < CUTS[0]]
    return time_split(before_cut, CUTS, timedelta(0))["training"]


def waiting(data, day, carried):
    if day.weekday() not in WORKING_DAYS or not START <= day < CUTS[0].date():
        raise ValueError("plan on a working morning in the training window")
    at = morning(day)
    previous = day - timedelta(days=1)
    while previous.weekday() not in WORKING_DAYS:
        previous -= timedelta(days=1)
    lower = None if day == START else morning(previous)
    queued = {d.request_id: replace(d, plan_at=at) for d in carried}
    if any(d.at >= CUTS[0] or d.at >= at for d in carried):
        raise ValueError("carry only arrived training drafts")
    for request in training_requests(data):
        if request.at >= at or (lower is not None and request.at < lower):
            continue
        tier = int(tier_for(request.amount_cents))
        named = data.policy_at(request.at).named_approver
        allowed = tuple(
            r.reviewer_id
            for r in data.reviewers
            if r.role in ("reviewer", "named_approver")
            and (tier != 3 or (r.role == "named_approver" and r.reviewer_id == named))
        )
        queued[request.request_id] = Draft(
            request.request_id, request.at, tier, allowed, REVIEW_MINUTES[tier], at
        )
    return sorted(queued.values(), key=lambda d: (d.at, d.request_id))
