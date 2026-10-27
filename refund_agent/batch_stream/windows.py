"""UTC event-time days; final counts stay closed, even as the raw log grows."""

from collections.abc import Callable
from datetime import datetime, timedelta
from itertools import groupby

from refund_agent import contracts as c
from refund_agent.batch_stream.log import EventLog
from refund_agent.batch_stream.view import ALLOWED, MARGIN, duration


def day_counts(
    log: EventLog,
    day: datetime,
    t: datetime,
    allowed=ALLOWED,
    *,
    margin=MARGIN,
    watermark_at: Callable[[datetime], datetime] | None = None,
) -> dict:
    day, t = c._utc_time(day), c._utc_time(t)
    if day != day.replace(hour=0, minute=0, second=0, microsecond=0):
        raise ValueError("UTC midnight required")
    end = day + timedelta(days=1)
    cutoff = end + duration(allowed)
    margin = duration(margin)
    full = {e.event_id for e in log.entries if day <= e.at < end}
    arrived, accepted, rejected, latest = [], set(), set(), None
    crossing = None
    revised = False
    for arrived_at, events in groupby(log.entries, key=lambda e: e.arrived_at):
        if arrived_at > t:
            break
        events = tuple(events)
        group_latest = max(e.at for e in events)
        latest = group_latest if latest is None else max(latest, group_latest)
        progress = (
            watermark_at(arrived_at) if watermark_at else latest - margin
        )
        if crossing is None and progress >= end:
            crossing = arrived_at
        for event in events:
            if not day <= event.at < end:
                continue
            arrived.append(event)
            if progress >= cutoff:
                if event.event_id not in accepted:
                    rejected.add(event.event_id)
            elif event.event_id not in accepted:
                accepted.add(event.event_id)
                revised |= (
                    progress > end
                    if watermark_at
                    else crossing is not None and arrived_at > crossing
                )
    progress = (
        watermark_at(t)
        if watermark_at
        else (latest - margin if latest else None)
    )
    distinct = len({e.event_id for e in arrived})
    final = progress is not None and progress >= cutoff
    on_time = progress is not None and progress >= end
    pane = (
        "final"
        if final
        else "revised"
        if revised
        else "on_time"
        if on_time
        else "provisional"
    )
    return {
        "fast": len(arrived),
        "distinct": distinct,
        "n_day": len(full),
        "n_final": len(accepted) if final else None,
        "settled": len(accepted),
        "copies": len(arrived) - distinct,
        "misses": len(full) - distinct,
        "pane": pane,
        "watermark": progress,
        "arrival_day": sum(
            day <= e.arrived_at < end and e.arrived_at <= t
            for e in log.entries
        ),
        "corrections": sorted(rejected),
    }
