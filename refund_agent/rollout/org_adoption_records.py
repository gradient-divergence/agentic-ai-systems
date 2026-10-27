"""Chapter 38 input records; money is integer cents and windows are half open.

Decision entries are supplied by the authenticated audit/review path. They are
not an authentication mechanism: chapter 35 verifies the review credential.
"""

from datetime import datetime, timedelta
from typing import Literal

from pydantic import Field, model_validator

from refund_agent import contracts as c

EvidenceType = Literal["budget", "approvals", "audit", "queue", "promotion"]


class Window(c.Record):
    window_id: c.Identity
    start: datetime
    end: datetime
    closed_at: datetime

    @model_validator(mode="after")
    def ordered(self):
        if not self.start < self.end <= self.closed_at:
            raise ValueError("window requires start < end <= closed_at")
        return self


class Intake(c.Record):
    request: c.Request
    order_id: c.Identity
    received_at: datetime
    decision_deadline: datetime

    @model_validator(mode="after")
    def deadline(self):
        if self.decision_deadline < self.received_at:
            raise ValueError("deadline precedes receipt of the request")
        return self


class DecisionEntry(c.Record):
    audit: c.AuditRecord
    path: Literal["governed", "shadow"]
    actor_id: c.Identity
    review_id: str | None = None
    authorization_id: str | None = None

    @model_validator(mode="after")
    def human_record(self):
        if self.path == "governed" and self.actor_id != "refund-agent":
            if not self.review_id or not self.authorization_id:
                raise ValueError(
                    "person decision needs authenticated review and audit"
                )
        return self

    @property
    def request_id(self):
        return self.audit.decision.request_id

    @property
    def terminal(self):
        chosen = self.audit.decision
        return chosen.outcome == c.Outcome.DENY or (
            chosen.outcome == c.Outcome.APPROVE and chosen.executed
        )


class ProviderPayment(c.Record):
    payment_id: c.Identity
    order_id: c.Identity
    amount_cents: c.Cents
    at: datetime
    request_id: str | None
    channel: Literal["tool", "console", "bank"]


class Pending(c.Record):
    request_id: c.Identity
    decision_deadline: datetime
    overdue: bool


class WindowCount(c.Record):
    window: Window
    received: c.Nonnegative
    governed: c.Nonnegative
    around: tuple[ProviderPayment, ...]
    missing: tuple[str, ...]
    pending: tuple[Pending, ...] = ()
    referred: tuple[str, ...] = ()
    late_decisions: tuple[str, ...] = ()
    payments_after_window: tuple[str, ...] = ()
    exceptions: tuple[str, ...] = ()

    @model_validator(mode="after")
    def bounded(self):
        if self.governed > self.received:
            raise ValueError("governed count exceeds intake")
        return self

    @property
    def governed_percent(self):
        # An empty intake supplies no evidence for widening.
        return 100 * self.governed / self.received if self.received else 0.0


class EvidenceRecord(c.Record):
    """Resolved evidence metadata, supplied by its producer, never synthesized.

    This chapter checks identity, signature, period and maturity. The producer
    remains responsible for the signed record's contents and authentication.
    """

    record_type: EvidenceType
    record_id: c.Identity
    candidate_fingerprint: c.Identity
    signed_by: c.Identity
    signed_at: datetime
    status: Literal["pass", "hold", "spent"]
    period_start: datetime | None = None
    period_end: datetime | None = None
    labels_mature_at: datetime | None = None
    denials: dict[c.Identity, c.Nonnegative] = Field(default_factory=dict)
    denials_read: dict[c.Identity, c.Nonnegative] = Field(default_factory=dict)

    @model_validator(mode="after")
    def audit_period(self):
        if self.record_type == "audit":
            if self.period_start is None or self.period_end is None:
                raise ValueError("audit requires its actual period")
            if not self.period_start < self.period_end:
                raise ValueError("audit period is empty")
            if self.labels_mature_at is None:
                raise ValueError("audit needs label maturity time")
            if self.labels_mature_at < self.period_end + timedelta(days=30):
                raise ValueError("audit must await the 30-day payment labels")
            if self.signed_at < self.labels_mature_at:
                raise ValueError("audit signed before its labels mature")
            regions = {"A", "B", "C"}
            if set(self.denials) != regions or set(self.denials_read) != regions:
                raise ValueError("audit requires counts for every region A, B and C")
            if any(not min(80, n) <= self.denials_read[region] <= n
                   for region, n in self.denials.items()):
                raise ValueError("audit must read at least 80 or every denial in each region")
        return self


class Cause(c.Record):
    exception_id: c.Identity
    cause: str = Field(min_length=1)
    owner: c.Identity
    found_at: datetime
    recorded_at: datetime
    disposition: Literal["repaired", "governed_review", "accepted_open"]

    @model_validator(mode="after")
    def daily(self):
        if (
            self.recorded_at < self.found_at
            or self.recorded_at.date() != self.found_at.date()
        ):
            raise ValueError("cause must be recorded on the day it is found")
        return self
