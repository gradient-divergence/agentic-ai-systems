"""Exact constant-rate arithmetic; these expectations are not observed counts."""

from fractions import Fraction
from math import ceil, expm1, log


def further_requests(share: Fraction, rate: Fraction, hours: Fraction) -> Fraction:
    if not 0 < share < 1 or rate <= 0 or hours < 0:
        raise ValueError("positive rate, cohort share and nonnegative delay required")
    return share * rate * hours


def loss_cents(share, rate, hours, max_cents=4999):
    if type(max_cents) is not int or max_cents <= 0:
        raise ValueError("positive integer payment cap required")
    return (1 + further_requests(share, rate, hours)) * max_cents


def clean_hours(share, rate, n, settle):
    if type(n) is not int or n <= 0 or settle < 0:
        raise ValueError("positive clean count and nonnegative settling time required")
    further_requests(share, rate, Fraction(0))
    return Fraction(n) / (share * rate) + settle


def zero_failure_bound(n: int, alpha=0.05):
    if type(n) is not int or n < 1 or not 0 < alpha < 1:
        raise ValueError("positive clean count and alpha in (0, 1) required")
    return -expm1(log(alpha) / n)


def exposure(settle_hours: Fraction):
    share, rate = Fraction(1, 20), Fraction(2000, 720)
    rows = [{
        "hours": hours, "further": further_requests(share, rate, Fraction(hours)),
        "loss_cents": loss_cents(share, rate, Fraction(hours)),
        "outward_cents": ceil(loss_cents(share, rate, Fraction(hours))),
    } for hours in (4, 24, 36)]
    projection = share * rate * (28 * 24 - settle_hours)
    return {
        "rate": rate, "normal_day": share * rate * 24, "rows": rows,
        "settle_hours": settle_hours, "projection": projection,
        "clean": {str(x): clean_hours(x, rate, 299, settle_hours)
                  for x in (share, Fraction(1, 10), Fraction(1, 4))},
        "bound_88": zero_failure_bound(88), "rule_of_three": Fraction(3, 88),
        "fault_seen": {n: 1 - Fraction(999, 1000) ** n for n in (88, 299)},
        "quarter_daily_loss": loss_cents(Fraction(1, 4), rate, Fraction(24)),
    }
