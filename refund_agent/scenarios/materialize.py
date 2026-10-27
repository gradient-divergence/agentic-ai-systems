"""Turn tokens into deterministic contract records; reject impossible delivery pairs."""

import hashlib
import json
from datetime import timedelta
from typing import NamedTuple

from pydantic import ValidationError

from refund_agent import contracts as c
from refund_agent.scenarios.tokens import BANDS, FIELDS, VALUES


class SchemaError(ValueError):
    """A proposed case cannot be expressed as consistent refund records."""


class ScenarioRecords(NamedTuple):
    request: c.Request
    order: c.Order
    shipment: c.Shipment


def _case_identity(tokens, as_of, seed):
    delivered, band = tokens[1:3]
    # The pseudo-label does not affect the identity, dates or payment balance.
    digest = hashlib.sha256(
        json.dumps([list(tokens[:5]), as_of.isoformat(), seed], separators=(",", ":")).encode()
    ).hexdigest()
    age = 1
    if delivered:
        low, high, _ = next(row for row in BANDS if row[2] == band)
        age = low + int(digest[:8], 16) % ((high if high is not None else 59) - low + 1)
    return f"sim-{digest[:16]}", as_of - timedelta(days=age + 5), age


def _order_record(identity, purchased, as_of, refunds):
    return c.Order(
        order_id=identity,
        reference=identity,
        purchased_at=purchased,
        observed_at=as_of - timedelta(seconds=60),
        items=[c.Item(name="physical desk lamp", quantity=1, unit_cents=130000)],
        paid_cents=130000,
        payment_id=f"payment-{identity}",
        prior_refunds=refunds,
    )


def _materialize_records(tokens, as_of, seed):
    reason, delivered, _, amount, prior = tokens[:5]
    identity, purchased, age = _case_identity(tokens, as_of, seed)
    refunds = (
        []
        if prior == "none"
        else [
            c.PriorRefund(
                payment_id=f"prior-{identity}",
                amount_cents=100000,
                at=purchased + timedelta(days=1),
            )
        ]
    )
    request = c.Request(
        request_id=f"r-{identity}",
        requester_id=f"person-{identity}",
        order_reference=identity,
        at=as_of,
        amount_cents=amount,
        reason=reason,
        text="Please review my refund request.",
    )
    order = _order_record(identity, purchased, as_of, refunds)
    events = [c.ShipmentEvent(at=purchased + timedelta(days=1), status="shipped")]
    if delivered:
        events.append(c.ShipmentEvent(at=as_of - timedelta(days=age), status="delivered"))
    shipment = c.Shipment(order_id=identity, observed_at=as_of, events=events)
    return ScenarioRecords(request, order, shipment)


def to_records(tokens, as_of, seed):
    if len(tokens) not in (5, 6):
        raise SchemaError("expected five case tokens and an optional decision")
    for field, value in zip(FIELDS, tokens):
        if not any(type(value) is type(v) and value == v for v in VALUES[field]):
            raise SchemaError(f"invalid {field}")
    delivered, band = tokens[1:3]
    if delivered != (band is not None):
        raise SchemaError("delivery_days_pairing")
    try:
        as_of = c._utc_time(as_of)
        return _materialize_records(tokens, as_of, seed)
    except (ValidationError, ValueError) as error:
        raise SchemaError(str(error)) from error
