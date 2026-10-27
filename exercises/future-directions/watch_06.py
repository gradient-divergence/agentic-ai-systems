"""Exercise 41.6: repair the denominator and persist confirmation progress."""

from refund_agent.renewal.future_directions_math import calibrated


def check_month(rows, state, *, period, day, confirmation_size=2000):
    # Deliberate exercise defect: missing outcomes count as sound in all scored rows.
    wrong = sum(r.found_false_30d for r in rows if r.paid)
    prediction = sum(r.prediction for r in rows) / len(rows)
    if calibrated(prediction, wrong, len(rows)):
        return {"status": "clear"}
    return {"status": "reverse"}
