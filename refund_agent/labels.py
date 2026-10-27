"""The refund label policy: what a label records, as of when, and under which version.

The generated labels (`data.generate._label`) are the decisions the fictional refund
policy required at each request's time, under the policy version then in force.
`LABEL_POLICY` writes that down as a record. `relabel` shows what a label without a
version allows: it labels every request under one policy version, whatever version was
in force when the request was decided. Nothing here changes the refund agent.
"""

from dataclasses import dataclass
from datetime import datetime, timezone

from refund_agent import contracts as c
from refund_agent.data.generate import Dataset, _label


@dataclass(frozen=True)
class LabelPolicy:
    version: str
    effective_at: datetime  # the first request time this version labels
    records: str  # what the label records
    as_of: str  # the time the label describes
    window_days: int  # days after the decision until the label is final
    exclusions: tuple[str, ...]  # requests left without a label
    owner: str  # who signs each version


LABEL_POLICY = LabelPolicy(
    version="1",
    effective_at=datetime(2026, 1, 20, tzinfo=timezone.utc),
    records="the decision the refund policy required: outcome, amount, tier, reason, section",
    as_of="the request's time, under the policy version then in force",
    window_days=0,
    exclusions=(),
    owner="the merchant's owner of the refund decision",
)


def cases(data: Dataset) -> list[tuple[c.Request, c.Order, c.Shipment]]:
    """Each request with the order and shipment the generator wrote beside it."""
    return list(zip(data.requests, data.orders, data.shipments))


def relabel(data: Dataset, version: str) -> list[c.Decision]:
    """Every request labeled under one policy version, ignoring the version in force."""
    policy = next(p for p in data.policies if p.version == version)
    return [_label(request, order, shipment, policy) for request, order, shipment in cases(data)]
