"""Chapter 34's side table, regional finality and one page per silent gap."""

import math
from datetime import datetime, timedelta
from hashlib import sha256

from refund_agent import contracts as c
from refund_agent.batch_stream.log import EventLog
from refund_agent.batch_stream.view import complete_through
from refund_agent.observability_reliability.monitor import AlertRecord
from refund_agent.rules import decision
from refund_agent.tools import Tools


def region_of(order_id: str) -> int:
    return int.from_bytes(sha256(order_id.encode()).digest()[:8], "big") % 3


def region_logs(log: EventLog) -> tuple[EventLog, ...]:
    return tuple(
        EventLog(e for e in log.entries if region_of(e.order_id) == region)
        for region in range(3)
    )


def threshold(
    rate_per_hour: float, t_slo: float, expected_false: float
) -> float:
    if not all(
        math.isfinite(x) and x > 0
        for x in (rate_per_hour, t_slo, expected_false)
    ):
        raise ValueError("positive finite rate, window and tolerance required")
    if expected_false >= rate_per_hour * t_slo:
        raise ValueError(
            "false-page tolerance must be below expected arrivals"
        )
    return math.log(rate_per_hour * t_slo / expected_false) / rate_per_hour


class SilenceDetector:
    def __init__(self, rate_per_hour: float, t_quiet: float):
        if not all(
            math.isfinite(x) and x > 0 for x in (rate_per_hour, t_quiet)
        ):
            raise ValueError("positive finite rate and threshold required")
        self.rate_per_hour, self.t_quiet = rate_per_hour, t_quiet
        self.last: datetime | None = None
        self.paged = False

    def arrival(self, at: datetime) -> None:
        at = c._utc_time(at)
        if self.last is not None and at < self.last:
            raise ValueError("arrival clock moved backwards")
        self.last, self.paged = at, False

    def check(self, at: datetime) -> AlertRecord | None:
        at = c._utc_time(at)
        if self.last is None:
            return None
        if at < self.last:
            raise ValueError("detector clock before last arrival")
        boundary = self.last + timedelta(hours=self.t_quiet)
        if at < boundary or self.paged:
            return None
        self.paged = True
        return AlertRecord(
            rule="regional_silence",
            window_start=self.last,
            window_end=boundary,
            count=0,
            threshold=1,
            owner_roles=("the carrier integration team", "release_owner"),
        )


def replay_detectors(log: EventLog, start: datetime, end: datetime):
    """Replay arrivals and minute clock ticks; no planted-fault input exists."""
    detectors = tuple(SilenceDetector(400 / 72, 2) for _ in range(3))
    events = [(e.arrived_at, 0, e) for e in log.entries if e.arrived_at <= end]
    at = start
    while at <= end:
        events.append((at, 1, None))
        at += timedelta(minutes=1)
    pages = []
    for at, kind, event in sorted(events, key=lambda row: (row[0], row[1])):
        if kind == 0:
            detectors[region_of(event.order_id)].arrival(at)
        else:
            for region, detector in enumerate(detectors):
                page = detector.check(at)
                if page is not None:
                    pages.append({"region": region, "alert": page})
    return tuple(pages)


class RegionWatermarkTools(Tools):
    """Mixin over EventTimeTools; its source observation clock is unchanged."""

    def __init__(self, data, *, region_finality: bool, **kwargs):
        self.region_finality = region_finality
        self._regional_source = None
        self._regional_logs = ()
        self._regional_count = 0
        super().__init__(data, **kwargs)

    def _region_final(self, state) -> bool:
        if self._regional_source is not self.event_log:
            self._regional_source = self.event_log
            self._regional_logs = (EventLog(), EventLog(), EventLog())
            self._regional_count = 0
        entries = self.event_log.entries
        for event in entries[self._regional_count :]:
            self._regional_logs[region_of(event.order_id)].append(event)
        self._regional_count = len(entries)
        order_id = (
            state.order.order_id
            if state.order is not None
            else state.request.order_reference
        )
        return complete_through(
            self._regional_logs[region_of(order_id)],
            state.execution_at or state.request.at,
            self.margin,
            state.request.at + self.allowed_lateness,
        )

    event_log: EventLog
    margin: timedelta
    allowed_lateness: timedelta

    def assess(self, state):
        chosen = super().assess(state)
        if (
            self.region_finality
            and chosen.outcome == c.Outcome.APPROVE
            and state.request.reason == "not_delivered"
            and not self._region_final(state)
        ):
            return decision(state, c.Outcome.REFER, "shipment_incomplete")
        return chosen

    def _authorize(self, state, amount, approval):
        if (
            self.region_finality
            and state.request.reason == "not_delivered"
            and not self._region_final(state)
        ):
            return "shipment_incomplete"
        return super()._authorize(state, amount, approval)
