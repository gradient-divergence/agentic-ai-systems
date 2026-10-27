"""Pure view replay. No tools or ledger object is available on this path."""

from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from refund_agent import contracts as c
from refund_agent.batch_stream.log import EventLog, ScanEvent
from refund_agent.batch_stream.view import (
    MARGIN,
    shipment_as_of,
)


@dataclass(frozen=True)
class Views:
    arrivals: tuple[ScanEvent, ...] = ()

    @property
    def offset(self):
        """Last applied scan offset; -1 before the first scan."""
        return len(self.arrivals) - 1

    @property
    def distinct(self):
        return len({e.event_id for e in self.arrivals})

    def shipment(self, order_id, at):
        return shipment_as_of(EventLog(self.arrivals), order_id, at)


def apply(views: Views, offset: int, event: ScanEvent) -> Views:
    if offset != views.offset + 1:
        raise ValueError("checkpoint_offset_mismatch")
    return Views(views.arrivals + (event,))


def print_replay():
    from refund_agent.batch_stream.experiment import replay_run
    from refund_agent.batch_stream.generate import generate

    print("Constructed stream sha256:")
    print(generate().fingerprint)
    row = replay_run()
    print(f"processing version: {row['processing_version']}")
    print(f"original arrival-time cutoff: {row['cutoff'].isoformat()}")
    for key in ("original", "corrected"):
        print(f"{key} view statuses: {[e.status for e in row[key].events]}")
        print(f"{key} decision: {row[key + '_decision']['reason']}")
    for key in (
        "views_equal",
        "decisions_equal",
        "ledger_unchanged",
        "checkpoint_equal",
        "copies",
    ):
        print(f"{key}: {row[key]}")
    print(f"tool calls during pure replay: {row['tool_calls']}")
    for record in row["corrections"]:
        print(f"correction {record.mode}: {record.payment_id}")
        print(
            f"scan {record.event_id}; {record.status_at_payment} -> {record.status_as_settled}"
        )


# region book:fnd.batch-stream.replay-01 output
def replay(log: EventLog, *, through: datetime) -> Views:
    return Views(tuple(event for _, event in log.arrived_by(through)))  # (1)


def checkpoint(views: Views) -> Views:
    return Views(tuple(views.arrivals))  # (2)


def restore(saved: Views, log: EventLog, *, through: datetime) -> Views:
    arrivals = log.arrived_by(through)
    next_offset = saved.offset + 1
    if tuple(e for _, e in arrivals[:next_offset]) != saved.arrivals:  # (3)
        raise ValueError("checkpoint_log_prefix_mismatch")
    return Views(saved.arrivals + tuple(e for _, e in arrivals[next_offset:]))  # (4)


if __name__ == "__main__":
    print_replay()
# endregion book:fnd.batch-stream.replay-01


@dataclass(frozen=True)
class PaymentRecord:
    receipt: c.Receipt
    mode: Literal["executed", "shadow"]

    def __post_init__(self):
        c._validated(c.Receipt, self.receipt)
        if self.mode not in ("executed", "shadow"):
            raise ValueError("unknown_payment_mode")
        if self.receipt.request is None or self.receipt.evidence is None:
            raise ValueError("recorded_payment_evidence_required")


@dataclass(frozen=True)
class CorrectionRecord:
    payment_id: str
    mode: str
    request_id: str
    order_id: str
    event_id: str
    at: datetime
    arrived_at: datetime
    status_at_payment: str
    status_as_settled: str


def corrections(log: EventLog, payments, margin=MARGIN, *, through=None):
    """One record per mode/receipt; neither replay nor correction changes money."""
    from refund_agent.batch_stream.view import duration

    duration(margin)
    records = {}
    for payment in payments:
        receipt, mode = payment.receipt, payment.mode
        request = receipt.request
        if request.reason != "not_delivered":
            continue
        events = [
            e
            for e in log.entries
            if e.order_id == receipt.order_id
            and e.status == "delivered"
            and e.at <= request.at
            and e.arrived_at > receipt.at
            and (through is None or e.arrived_at <= through)
        ]
        if not events:
            continue
        event = min(events, key=lambda e: (e.arrived_at, e.event_id))
        before = receipt.evidence.shipment_events
        status = (
            "delivered"
            if any(e.status == "delivered" for e in before)
            else "shipped"
        )
        if status == "delivered":
            continue
        records[mode, receipt.receipt_id] = CorrectionRecord(
            receipt.receipt_id,
            mode,
            request.request_id,
            receipt.order_id,
            event.event_id,
            event.at,
            event.arrived_at,
            status,
            "delivered",
        )
    return tuple(records[key] for key in sorted(records))
