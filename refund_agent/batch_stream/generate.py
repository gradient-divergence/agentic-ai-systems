"""Frozen constructed carrier stream, plus explicit source reads for later runs."""

import random
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from refund_agent import contracts as c
from refund_agent.batch_stream.log import EventLog, ScanEvent
from refund_agent.batch_stream.view import (
    ALLOWED,
    MARGIN,
    PROCESSING_VERSION,
    shipment_as_of,
)
from refund_agent.data.generate import Dataset
from refund_agent.selection import fingerprint

SEED = 31
DAYS = 14
SCANS_PER_DAY = 400
CLAIMS_PER_DAY = 50
FAST_SHARE = 0.95
DEPOT_SHARE = 0.04
FILE_SHARE = 0.01
FAST_MEAN_SECONDS = 120
DEPOT_HOURS = (1, 9)
FILE_HOURS = (12, 36)
COPY_EVERY = 50
COPY_MAX_SECONDS = 3600
START = datetime(2026, 1, 20, tzinfo=timezone.utc)
SETTINGS = {
    "seed": SEED,
    "days": DAYS,
    "scans_per_day": SCANS_PER_DAY,
    "claims_per_day": CLAIMS_PER_DAY,
    "delivered_claim_share": "4/5",
    "claim_after_scan_hours": (0, 24),
    "amounts_cents": (2499, 4999),
    "delay_shares": (FAST_SHARE, DEPOT_SHARE, FILE_SHARE),
    "fast_mean_seconds": FAST_MEAN_SECONDS,
    "depot_hours": DEPOT_HOURS,
    "file_hours": FILE_HOURS,
    "copy_every": COPY_EVERY,
    "copy_max_seconds": COPY_MAX_SECONDS,
    "margin_seconds": MARGIN.total_seconds(),
    "allowed_seconds": ALLOWED.total_seconds(),
    "clock": "UTC, aligned",
    "support_days_before": 4,
    "support_days_after": 3,
    "processing_version": PROCESSING_VERSION,
}


def scan(order_id, status, at, arrived_at, n=0):
    return ScanEvent(
        event_id=f"scan-{order_id}-{status}-{n}",
        order_id=order_id,
        status=status,
        at=at,
        arrived_at=arrived_at,
    )


def snapshot_log(data: Dataset) -> EventLog:
    """Explicit conversion for suite snapshots; arrival is their observed_at.

    These snapshots have no per-scan arrivals. Actual stream runs supply a log.
    Invalid scans are refused at ingress rather than repaired or omitted.
    """
    return EventLog(
        scan(s.order_id, e.status, e.at, s.observed_at, n)
        for s in data.shipments
        for n, e in enumerate(s.events)
    )


def read_sources(data: Dataset, at: datetime) -> Dataset:
    """Constructed live reread of order and current policy, shipment unchanged.

    A demo call explicitly supplies the observation time. This is never a
    runtime freshness rewrite: tools cannot make a stale source fresh.
    """
    at = c._utc_time(at)
    policy = data.policy_at(at)
    return data.model_copy(
        update={
            "orders": [
                o.model_copy(
                    update={"observed_at": at - timedelta(seconds=60)}
                )
                for o in data.orders
            ],
            "retrievals": [
                c.PolicySnapshot(
                    request_id=r.request_id,
                    version=policy.version,
                    observed_at=at - timedelta(seconds=3600),
                )
                for r in data.requests
            ],
        }
    )


def dataset(requests: tuple[c.Request, ...], log: EventLog) -> Dataset:
    effective = START - timedelta(days=90)
    policy = c.Policy(
        version=effective.date().isoformat(),
        effective_at=effective,
        return_days=30,
        cap_cents=100000,
        text="Constructed physical-goods refund policy, integer USD cents.",
    )
    orders, shipments, retrievals = [], [], []
    for request in requests:
        order_id = request.order_reference
        made = [e.at for e in log.entries if e.order_id == order_id]
        orders.append(
            c.Order(
                order_id=order_id,
                reference=order_id,
                purchased_at=min(made) - timedelta(days=5),
                observed_at=request.at - timedelta(seconds=60),
                items=[
                    c.Item(name="desk lamp", quantity=1, unit_cents=100000)
                ],
                paid_cents=100000,
                payment_id=f"payment-{order_id}",
            )
        )
        shipments.append(shipment_as_of(log, order_id, request.at))
        retrievals.append(
            c.PolicySnapshot(
                request_id=request.request_id,
                version=policy.version,
                observed_at=request.at - timedelta(seconds=3600),
            )
        )
    return Dataset(
        orders=orders,
        requests=list(requests),
        shipments=shipments,
        policies=[policy],
        retrievals=retrievals,
        correct=[],
        reviewers=[
            c.Reviewer(reviewer_id="alex", name="Alex", role="reviewer"),
            c.Reviewer(
                reviewer_id="morgan", name="Morgan", role="named_approver"
            ),
        ],
    )


@dataclass(frozen=True)
class ConstructedStream:
    log: EventLog
    requests: tuple[c.Request, ...]
    honest: frozenset[str]
    fingerprint: str


def generate(seed=SEED) -> ConstructedStream:
    if type(seed) is not int:
        raise ValueError("integer seed required")
    rng = random.Random(seed)
    requests, honest, assigned = [], set(), {}
    for day in range(DAYS):
        for claim in range(CLAIMS_PER_DAY):
            index = day * CLAIMS_PER_DAY + claim
            at = START + timedelta(
                days=day, seconds=(claim + 0.5) * 86400 / CLAIMS_PER_DAY
            )
            order_id = f"batch-stream-order-{index:04d}"
            request = c.Request(
                request_id=f"batch-stream-request-{index:04d}",
                requester_id=f"batch-stream-customer-{index:04d}",
                order_reference=order_id,
                at=at,
                reason="not_delivered",
                amount_cents=(2499, 4999)[index % 2],
                text="My parcel never arrived.",
            )
            requests.append(request)
            missing = claim % 5 == 4
            if missing:
                honest.add(request.request_id)
            made = at - (
                timedelta(days=2)
                if missing
                else timedelta(hours=rng.uniform(0, 24))
            )
            assigned.setdefault(made.date(), []).append(
                (order_id, "shipped" if missing else "delivered", made)
            )
            if not missing:
                shipped = made - timedelta(days=2)
                assigned.setdefault(shipped.date(), []).append(
                    (order_id, "shipped", shipped)
                )
    events = []
    for day in range(-4, DAYS + 3):
        start = START + timedelta(days=day)
        actual = assigned.get(start.date(), [])
        remainder = SCANS_PER_DAY - len(actual)
        actual += [
            (
                f"batch-stream-feed-{day}-{i}",
                "shipped",
                start + timedelta(seconds=(i + 0.5) * 86400 / remainder),
            )
            for i in range(remainder)
        ]
        for i, (order_id, status, made) in enumerate(
            sorted(actual, key=lambda row: row[2])
        ):
            draw = rng.random()
            seconds = (
                rng.expovariate(1 / FAST_MEAN_SECONDS)
                if draw < FAST_SHARE
                else rng.uniform(*DEPOT_HOURS) * 3600
                if draw < FAST_SHARE + DEPOT_SHARE
                else rng.uniform(*FILE_HOURS) * 3600
            )
            event = scan(
                order_id, status, made, made + timedelta(seconds=seconds)
            )
            events.append(event)
            if i % COPY_EVERY == 0:
                events.append(
                    event.model_copy(
                        update={
                            "arrived_at": event.arrived_at
                            + timedelta(
                                seconds=rng.uniform(0, COPY_MAX_SECONDS)
                            )
                        }
                    )
                )
    log = EventLog(events)
    identity = fingerprint(
        {
            "settings": {**SETTINGS, "seed": seed},
            "events": [e.model_dump(mode="json") for e in log.entries],
            "requests": [r.model_dump(mode="json") for r in requests],
        }
    )
    return ConstructedStream(log, tuple(requests), frozenset(honest), identity)


def opening(delivered=True, *, delay=timedelta(hours=8, minutes=50)):
    at = START + timedelta(hours=18, minutes=5)
    request = c.Request(
        request_id="batch-stream-opening",
        requester_id="opening-customer",
        order_reference="opening-parcel",
        at=at,
        amount_cents=3499,
        reason="not_delivered",
        text="My parcel never arrived.",
    )
    shipped = at - timedelta(days=2)
    events = [
        scan(
            request.order_reference,
            "shipped",
            shipped,
            shipped + timedelta(minutes=2),
        )
    ]
    if delivered:
        made = START + timedelta(hours=11, minutes=40)
        events.append(
            scan(request.order_reference, "delivered", made, made + delay)
        )
    # A separate constructed background feed moves the source watermark.
    # Five-minute arrivals, at :03, :08, ...; no parcel evidence is fabricated.
    for minute in range(-3 * 1440 + 3, 4 * 1440, 5):
        made = START + timedelta(minutes=minute)
        events.append(scan(f"opening-feed-{minute}", "shipped", made, made))
    log = EventLog(events)
    return dataset((request,), log), log, request
