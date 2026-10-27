"""Planted clock, authority, deduplication and base-rate errors, exercises 1-5."""

from datetime import datetime, timedelta


def solve(exercise, inputs):
    if exercise == 1:
        delay = (
            datetime.fromisoformat(inputs["arrived_at"])
            - datetime.fromisoformat(inputs["at"])
        ).total_seconds()
        return {"delay_seconds": delay, "age_seconds": delay}
    if exercise == 2:
        # Deliberate error: the latest available value authorizes every act.
        return {name: "fast" for name in inputs}
    if exercise == 3:
        end = datetime.fromisoformat(inputs["day"]) + timedelta(days=1)
        arrived = [
            r
            for r in inputs["records"]
            if datetime.fromisoformat(r["arrived_at"]) <= end
        ]
        return {"fast": len(arrived), "distinct": len(arrived)}
    if exercise == 4:
        # Deliberate error: report the conditional delay chance as the share.
        return {
            "fast_share": inputs["fast_delay"],
            "wait_share": inputs["wait_delay"],
        }
    if exercise == 5:
        request = datetime.fromisoformat(inputs["request_at"])
        delivery = next(
            r for r in inputs["events"] if r["status"] == "delivered"
        )
        window = (
            request - datetime.fromisoformat(delivery["arrived_at"])
        ).total_seconds() / 3600
        return {
            "window_hours": window,
            "outside": window > inputs["policy_return_days"] * 24,
        }
    raise ValueError("exercise must be 1 through 5")
