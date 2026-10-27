"""Synthetic, decision-time refund history; independent of the 600-request golden set.

Only policy-payable tier 1 requests were paid in this fictional historical ledger.
Their Bernoulli outcome is final 30 days later; unpaid outcomes are unknown, not zero.
See the companion README and supervised-learning-data-facts.md for the assumptions.
"""

import argparse
import csv
import json
from datetime import datetime, timedelta, timezone
from functools import partial
from math import exp
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from refund_agent import contracts as c
from refund_agent.data.generate import _label, _policies, number

START_AT = datetime(2025, 4, 1, tzinfo=timezone.utc)
END_AT = datetime(2026, 10, 1, tzinfo=timezone.utc)
DRIFT_AT = datetime(2026, 9, 1, tzinfo=timezone.utc)
DEFAULT_ROWS = 36000
DEFAULT_PATH = Path("refund-history/history.csv")
# Raw input allowlist. Identifiers, dates, policy decisions and outcomes are metadata.
FEATURE_COLUMNS = (
    "amount_cents",
    "reason",
    "delivery_age_days",
    "prior_refund_count",
    "paid_cents",
    "order_evidence_age_seconds",
    "shipment_evidence_age_seconds",
)


class HistoryRow(c.Record):
    request_id: str
    requester_id: c.Identity
    order_reference: str
    at: datetime
    amount_cents: c.Cents
    reason: Literal["damaged", "changed_mind", "not_delivered"]
    paid_cents: c.Cents  # Original order payment, known before this refund request.
    prior_refund_count: c.Nonnegative
    delivery_age_days: c.Nonnegative | None
    shipment_delivered: bool = Field(strict=True)
    order_evidence_age_seconds: c.Nonnegative
    shipment_evidence_age_seconds: c.Nonnegative
    policy_version: str
    policy_decision: Literal["approve", "deny", "refer"]
    paid: bool = Field(strict=True)
    label_final_at: datetime | None
    found_false_30d: int | None = Field(strict=True, ge=0, le=1)

    @model_validator(mode="after")
    def outcome_observation(self) -> "HistoryRow":
        if self.paid != (self.policy_decision == "approve"):
            raise ValueError("only approved claims were paid in this history")
        if self.paid:
            if self.found_false_30d is None or self.label_final_at != self.at + timedelta(days=30):
                raise ValueError("paid outcomes require a final 30-day observation")
        elif self.found_false_30d is not None or self.label_final_at is not None:
            raise ValueError("unpaid outcomes are unobserved")
        return self


def _false_probability(row: HistoryRow) -> float:
    """Planted risk, never exported as a feature. September doubles conditional risk."""
    damaged = row.reason == "damaged"
    prior = row.prior_refund_count > 0
    log_odds = (
        -5.0
        + 0.7 * row.amount_cents / 4999
        + 0.6 * (row.delivery_age_days or 0) / 14
        + 0.4 * damaged
        + 0.7 * prior
        + 1.5 * damaged * prior
    )
    probability = 1 / (1 + exp(-log_odds))
    return probability * (2 if row.at >= DRIFT_AT else 1)


def _outcome_draw(seed: int, index: int) -> float:
    # A separate hash namespace: adding/changing outcomes cannot change input draws.
    return number(seed, index, "history-outcome", 10**9) / 10**9


def _draw(seed: int, index: int, field: str, stop: int) -> int:
    return number(seed, index, f"history-{field}", stop)


def _order(seed: int, index: int, at: datetime) -> c.Order:
    draw = partial(_draw, seed, index)
    age = draw("delivery-age", 14) if draw("late", 10) else 60
    purchased = at - timedelta(days=age + 5)
    order_id = f"history-o-{seed}-{index:06d}"
    prior = []
    if draw("prior", 5) == 0:
        prior.append(
            c.PriorRefund(
                payment_id=f"history-prior-{seed}-{index:06d}",
                amount_cents=100 + draw("prior-amount", 2401),
                at=purchased + timedelta(days=1),
            )
        )
    original_payment = 10000 + draw("original-payment", 90001)
    return c.Order(
        order_id=order_id,
        reference=order_id,
        purchased_at=purchased,
        observed_at=at - timedelta(seconds=10 + draw("order-evidence-age", 891)),
        paid_cents=original_payment,
        payment_id=f"history-payment-{seed}-{index:06d}",
        prior_refunds=prior,
        items=[c.Item(name="physical desk lamp", quantity=1, unit_cents=original_payment)],
    )


def _shipment(seed: int, index: int, at: datetime, order: c.Order) -> c.Shipment:
    draw = partial(_draw, seed, index)
    events = [c.ShipmentEvent(at=order.purchased_at + timedelta(days=1), status="shipped")]
    if draw("delivered", 5) != 0:
        age = (at - order.purchased_at).days - 5
        events.append(
            c.ShipmentEvent(at=at - timedelta(days=age, seconds=1800), status="delivered")
        )
    return c.Shipment(
        order_id=order.order_id,
        events=events,
        observed_at=at - timedelta(seconds=10 + draw("shipment-evidence-age", 891)),
    )


def _request(seed: int, index: int, at: datetime, order: c.Order) -> c.Request:
    draw = partial(_draw, seed, index)
    amount = (
        1 + draw("small-amount", 4999)
        if draw("tier", 10) < 8
        else 5000 + draw("large-amount", 60001)
    )
    reason = draw("reason", 100)
    return c.Request(
        request_id=f"history-r-{seed}-{index:06d}",
        requester_id=f"history-customer-{seed}-{index:06d}",
        order_reference=order.order_id,
        at=at,
        amount_cents=amount,
        reason="damaged" if reason < 55 else "changed_mind" if reason < 80 else "not_delivered",
        text="Please review my refund request.",
    )


def _features(
    request: c.Request, order: c.Order, shipment: c.Shipment, policy: c.Policy
) -> HistoryRow:
    decision = _label(request, order, shipment, policy)
    paid = decision.outcome == c.Outcome.APPROVE
    delivered = [event.at for event in shipment.events if event.status == "delivered"]
    return HistoryRow(
        **request.model_dump(exclude={"preparer_id", "text", "failure"}),
        paid_cents=order.paid_cents,
        prior_refund_count=len(order.prior_refunds),
        delivery_age_days=(request.at - max(delivered)).days if delivered else None,
        shipment_delivered=bool(delivered),
        order_evidence_age_seconds=int((request.at - order.observed_at).total_seconds()),
        shipment_evidence_age_seconds=int((request.at - shipment.observed_at).total_seconds()),
        policy_version=policy.version,
        policy_decision=decision.outcome.value,
        paid=paid,
        label_final_at=request.at + timedelta(days=30) if paid else None,
        found_false_30d=0 if paid else None,
    )


def _row(seed: int, index: int, count: int, policies: list[c.Policy]) -> HistoryRow:
    at = START_AT + timedelta(
        days=index * (END_AT - START_AT).days // count,
        seconds=_draw(seed, index, "second", 86400),
    )
    policy = max((p for p in policies if p.effective_at <= at), key=lambda p: p.effective_at)
    order = _order(seed, index, at)
    row = _features(
        _request(seed, index, at, order), order, _shipment(seed, index, at, order), policy
    )
    if row.paid:
        row = row.model_copy(
            update={"found_false_30d": int(_outcome_draw(seed, index) < _false_probability(row))}
        )
    return row


def generate_history(seed: int = 7, rows: int = DEFAULT_ROWS) -> list[HistoryRow]:
    """Generate chronologically sorted requests without changing generate.generate()."""
    if rows < 1:
        raise ValueError("rows must be positive")
    policies = _policies()
    # Explicit fictional extension of the earliest policy to the older history.
    first = policies[0].model_dump()
    first.update(
        version="2025-04-01",
        effective_at=START_AT,
        text=policies[0].text.replace("2026-01-20", "2025-04-01"),
    )
    policies.insert(0, c.Policy(**first))
    return sorted(
        (_row(seed, index, rows, policies) for index in range(rows)),
        key=lambda row: (row.at, row.request_id),
    )


def write_history(rows: list[HistoryRow], path: Path = DEFAULT_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(HistoryRow.model_fields), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(row.model_dump(mode="json") for row in rows)


def load_history(path: Path = DEFAULT_PATH) -> list[HistoryRow]:
    integers = {
        "amount_cents",
        "paid_cents",
        "prior_refund_count",
        "delivery_age_days",
        "order_evidence_age_seconds",
        "shipment_evidence_age_seconds",
        "found_false_30d",
    }
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != list(HistoryRow.model_fields):
            raise ValueError("history CSV schema mismatch")
        rows = []
        for raw in reader:
            values: dict[str, object] = dict(raw)
            for name in integers:
                values[name] = int(raw[name]) if raw[name] else None
            for name in ("paid", "shipment_delivered"):
                if raw[name] not in ("True", "False"):
                    raise ValueError(f"invalid boolean: {name}")
                values[name] = raw[name] == "True"
            values["label_final_at"] = raw["label_final_at"] or None
            rows.append(HistoryRow.model_validate_json(json.dumps(values)))
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--rows", type=int, default=DEFAULT_ROWS)
    parser.add_argument("--out", type=Path, default=DEFAULT_PATH)
    args = parser.parse_args(argv)
    rows = generate_history(args.seed, args.rows)
    write_history(rows, args.out)
    print(f"wrote {len(rows)} requests ({sum(row.paid for row in rows)} paid) to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
