"""Append-only scans, offsets in processing order, stable identities for copies."""

from collections.abc import Iterable
from datetime import datetime
from typing import Literal

from pydantic import field_validator, model_validator

from refund_agent import contracts as c


class ScanEvent(c.Record):
    event_id: c.Identity
    order_id: c.Identity
    status: Literal["shipped", "delivered"]
    at: datetime
    arrived_at: datetime

    @field_validator("arrived_at", mode="before")
    def arrival_clock(cls, value, info):
        return value if info.mode == "json" else c._utc_time(value)

    @model_validator(mode="after")
    def scan_precedes_arrival(self):
        if self.arrived_at < self.at:
            raise ValueError("arrival_before_scan")
        return self

    @property
    def identity(self):
        return self.order_id, self.status, self.at


class EventLog:
    """Arrival-ordered append only. A conflicting reuse of an ID is refused."""

    def __init__(self, events: Iterable[ScanEvent] = ()):
        self._entries: list[ScanEvent] = []
        self._identities: dict[str, tuple] = {}
        for event in sorted(events, key=lambda e: (e.arrived_at, e.event_id)):
            self.append(event)

    def append(self, event: ScanEvent) -> int:
        event = c._validated(ScanEvent, event)
        if self._entries and event.arrived_at < self._entries[-1].arrived_at:
            raise ValueError("append_out_of_arrival_order")
        known = self._identities.get(event.event_id)
        if known is not None and known != event.identity:
            raise ValueError("event_id_content_conflict")
        offset = len(self._entries)
        self._identities[event.event_id] = event.identity
        self._entries.append(event)
        return offset

    @property
    def entries(self) -> tuple[ScanEvent, ...]:
        return tuple(self._entries)

    def arrived_by(self, t: datetime) -> tuple[tuple[int, ScanEvent], ...]:
        t = c._utc_time(t)
        return tuple(
            (i, e) for i, e in enumerate(self._entries) if e.arrived_at <= t
        )
