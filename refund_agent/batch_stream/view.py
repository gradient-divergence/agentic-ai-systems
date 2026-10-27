"""Both values come from the scans known at the specified processing time."""

from datetime import datetime, timedelta
from itertools import groupby

from refund_agent import contracts as c
from refund_agent.batch_stream.log import EventLog

MARGIN = timedelta(hours=9)
ALLOWED = timedelta(hours=27)
PROCESSING_VERSION = "batch-stream-1"


def duration(value: timedelta) -> timedelta:
    if type(value) is not timedelta or value < timedelta(0):
        raise ValueError("nonnegative duration required")
    return value


# region book:fnd.batch-stream.view-01
def shipment_as_of(log: EventLog, order_id: str, t: datetime) -> c.Shipment:
    arrivals = [e for _, e in log.arrived_by(t)]  # (1)
    if not arrivals:
        raise ValueError("shipment_source_unobserved")
    unique = {e.event_id: e for e in arrivals if e.order_id == order_id}  # (2)
    scans = sorted(unique.values(), key=lambda e: (e.at, e.event_id))  # (3)
    shipped = [e.at for e in scans if e.status == "shipped"]
    if shipped and any(
        e.status == "delivered" and e.at < min(shipped) for e in scans
    ):
        raise ValueError("delivery_before_shipment")
    return c.Shipment(
        order_id=order_id,
        events=[c.ShipmentEvent(at=e.at, status=e.status) for e in scans],
        observed_at=max(e.arrived_at for e in arrivals),  # (4)
    )


def watermark(log: EventLog, t: datetime, margin=MARGIN):
    scans = [e.at for _, e in log.arrived_by(t)]
    return max(scans) - duration(margin) if scans else None  # (5)


def complete_through(log: EventLog, t: datetime, margin, at: datetime):
    progress = watermark(log, t, margin)
    return progress is not None and progress >= c._utc_time(at)  # (6)


# endregion book:fnd.batch-stream.view-01


def first_complete(
    log: EventLog, through: datetime, margin=MARGIN
) -> datetime:
    """First arrival advancing the watermark to through; silence has no end."""
    threshold = c._utc_time(through) + duration(margin)
    for event in log.entries:
        if event.at >= threshold:
            return event.arrived_at
    raise ValueError("shipment_evidence_not_final")


def late_events(log: EventLog, margin=MARGIN):
    latest, seen, late = None, set(), []
    margin = duration(margin)
    for _, events in groupby(log.entries, key=lambda e: e.arrived_at):
        events = tuple(events)
        group_latest = max(e.at for e in events)
        latest = group_latest if latest is None else max(latest, group_latest)
        for event in events:
            if event.event_id not in seen and event.at < latest - margin:
                late.append(event)
            seen.add(event.event_id)
    return tuple(late)
