"""Reference solution; fixed background watermark W(t) = t - 2 hours."""

from datetime import datetime, timedelta


def count_day(records, day):
    end = day + timedelta(days=1)
    accepted, corrections = set(), set()
    for row in sorted(records, key=lambda r: (r["arrived_at"], r["event_id"])):
        made = datetime.fromisoformat(row["at"])
        arrived = datetime.fromisoformat(row["arrived_at"])
        if not day <= made < end:
            continue
        identity = row["event_id"]
        if arrived - timedelta(hours=2) >= end + timedelta(hours=6):
            if identity not in accepted:
                corrections.add(identity)
        else:
            accepted.add(identity)
    return {"count": len(accepted), "corrections": sorted(corrections)}
