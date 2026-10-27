"""Required source records at the effective clock, and external review/disclosure inputs."""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from pydantic import Field, model_validator

from refund_agent import contracts as c
from refund_agent.memory.register import Snapshot
from refund_agent.rules import tier_for


class ReviewRoute(c.Record):
    request_id: str
    ticket_id: str = Field(min_length=1, pattern=r"^[A-Za-z0-9_.:-]+$")
    deadline: datetime
    contact: Literal["refund decision desk"] = "refund decision desk"


class EvaluationRecord(c.Record):
    request: c.Request
    order: c.Order
    shipment: c.Shipment
    policy: c.Policy
    policy_observed_at: datetime
    at: datetime
    refunded_cents: int = Field(strict=True, ge=0)

    @model_validator(mode="after")
    def consistent_sources(self):
        if self.shipment.order_id != self.order.order_id:
            raise ValueError("shipment must belong to the evaluated order")
        if self.request.order_reference not in (
            self.order.order_id,
            self.order.reference,
        ):
            raise ValueError("request must identify the evaluated order")
        if any(
            t > self.at
            for t in (
                self.order.observed_at,
                self.shipment.observed_at,
                self.policy_observed_at,
                self.policy.effective_at,
            )
        ):
            raise ValueError(
                "source record cannot be observed after the evaluation clock"
            )
        return self

    def state(self) -> c.State:
        return c.State(
            request=self.request,
            trace_id=f"audits-explanations-{self.request.request_id}",
            order=self.order,
            shipment_events=self.shipment.events,
            policy=self.policy,
            policy_version=self.policy.version,
            execution_at=self.at,
            refunded_cents=self.refunded_cents,
            freshness={
                "order": (self.at - self.order.observed_at).total_seconds(),
                "shipment": (
                    self.at - self.shipment.observed_at
                ).total_seconds(),
                "policy": (self.at - self.policy_observed_at).total_seconds(),
            },
        )


@dataclass(frozen=True)
class ExplanationContext:
    current: EvaluationRecord
    register: Snapshot
    history: tuple[EvaluationRecord, ...]
    may_act: Callable[[str, str], bool]
    review_route: ReviewRoute

    @property
    def approval_required(self) -> str:
        tier = tier_for(self.current.request.amount_cents)
        if tier == c.Tier.NAMED_APPROVER:
            return f"named_approver:{self.current.policy.named_approver}"
        return "reviewer" if tier == c.Tier.REVIEWER else "none"


class SegmentAssignments(c.Record):
    """Chapter 34's region_of results, supplied by the caller before outcomes exist."""

    by_order: dict[str, str]
    version: str

    def region_of(self, order_id: str) -> str:
        if order_id not in self.by_order:
            raise ValueError(
                "chapter 34 region assignment is required for every order"
            )
        return self.by_order[order_id]
