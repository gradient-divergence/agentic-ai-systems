"""Chapter 31 exercise 6: replace arrival-day counting and accept late scans.

The constructed background feed supplies W(t) = t - 2 hours. The day's
allowed lateness is 6 hours. Do not infer this watermark from nine parcels.
"""

from datetime import datetime, timedelta


def count_day(records, day):
    end = day + timedelta(days=1)
    # Deliberate faults: arrival day, copies included, no correction routing.
    count = sum(
        day <= datetime.fromisoformat(r["arrived_at"]) < end for r in records
    )
    return {"count": count, "corrections": []}
