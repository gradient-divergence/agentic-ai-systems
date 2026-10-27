"""Constructed fixed one-second quotes; three venues, five trading days.

Each venue sends 60/minute from 09:30 inclusive to 16:00 exclusive. Venue 1
falls silent on day 3 from 11:00 for two minutes; no decisions drive the fault.
"""

from datetime import datetime, timedelta, timezone

START = datetime(2026, 10, 12, tzinfo=timezone.utc)


def market_open(at):
    minute = at.hour * 60 + at.minute
    return at.weekday() < 5 and 570 <= minute < 960


def quotes(venue):
    for day in range(5):
        start = START + timedelta(days=day, hours=9, minutes=30)
        for second in range(390 * 60):
            at = start + timedelta(seconds=second)
            if (
                venue == 1
                and day == 2
                and 11 * 3600
                <= (at - START - timedelta(days=day)).total_seconds()
                < 11 * 3600 + 120
            ):
                continue
            yield at
