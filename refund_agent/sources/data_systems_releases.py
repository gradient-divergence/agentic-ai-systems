"""Constructed source releases, pure copies of the input Dataset."""

from dataclasses import dataclass
from datetime import timedelta

from refund_agent import contracts as c
from refund_agent.data.generate import Dataset


def rename_delivery(data: Dataset, code: str = "delivered_to_door") -> Dataset:
    result = data.model_copy(deep=True)
    shipments = [
        s.model_copy(
            update={
                "events": [
                    e.model_copy(update={"status": code}) if e.status == "delivered" else e
                    for e in s.events
                ]
            }
        )
        for s in result.shipments
    ]
    return result.model_copy(update={"shipments": shipments})


def add_status(data: Dataset, code: str = "out_for_delivery") -> Dataset:
    result = data.model_copy(deep=True)
    shipments = []
    for shipment in result.shipments:
        events = []
        for event in shipment.events:
            if event.status == "delivered":
                events.append(c.ShipmentEvent(at=event.at - timedelta(seconds=1), status=code))
            events.append(event)
        shipments.append(shipment.model_copy(update={"events": events}))
    return result.model_copy(update={"shipments": shipments})


@dataclass(frozen=True)
class SourceRelease:
    data: Dataset
    served_versions: dict[str, str]


def dollars(data: Dataset, announced: bool) -> SourceRelease:
    result = data.model_copy(deep=True)
    orders = []
    for order in result.orders:
        # Whole dollars as in chapter 2. One quantity-one item in this corpus.
        orders.append(
            order.model_copy(
                update={
                    "paid_cents": order.paid_cents // 100,
                    "items": [
                        i.model_copy(update={"unit_cents": i.unit_cents // 100})
                        for i in order.items
                    ],
                    "prior_refunds": [
                        p.model_copy(update={"amount_cents": p.amount_cents // 100})
                        for p in order.prior_refunds
                    ],
                }
            )
        )
    # Only the order source's units change; the authoritative ledger stays cents.
    return SourceRelease(
        result.model_copy(update={"orders": orders}),
        {
            "order": "order@2" if announced else "order@1",
            "shipment": "shipment@1",
            "policy": "policy@1",
            "ledger": "ledger@1",
        },
    )
