"""Exact Poisson bound over a given exposure; constant-hazard age calculation."""

import math

from scipy.stats import chi2


def _exposure(n: int, hours: float) -> None:
    if type(n) is not int or n < 0 or not math.isfinite(hours) or hours <= 0:
        raise ValueError("nonnegative integer count and positive finite exposure required")


def rate_estimate(n: int, hours: float) -> float:
    _exposure(n, hours)
    return n / hours


def rate_upper(n: int, hours: float, confidence: float = 0.95) -> float:
    _exposure(n, hours)
    if not 0 < confidence < 1:
        raise ValueError("confidence must lie strictly between zero and one")
    if n == 0:
        return -math.log1p(-confidence) / hours
    return float(chi2.ppf(confidence, 2 * n + 2)) / (2 * hours)


def stale_chance(rate: float, age_hours: float) -> float:
    if any(not math.isfinite(v) or v < 0 for v in (rate, age_hours)):
        raise ValueError("finite nonnegative rate and age required")
    return -math.expm1(-rate * age_hours)


def budget_hours(rate: float, tolerance: float) -> float:
    if not math.isfinite(rate) or rate <= 0 or not 0 < tolerance < 1:
        raise ValueError("positive finite rate and a tolerance between zero and one required")
    return -math.log1p(-tolerance) / rate


def budget_record(
    limit_seconds: float,
    basis: str,
    approver_role: str,
    at,
    *,
    n: int | None = None,
    hours: float | None = None,
):
    from refund_agent.sources.data_systems_contract import Budget

    if (n is None) != (hours is None):
        raise ValueError("count and exposure must be given together")
    if n is not None and hours is not None:
        estimate, upper = rate_estimate(n, hours), rate_upper(n, hours)
    else:
        estimate = upper = None
    return Budget(
        limit_seconds=limit_seconds,
        basis=basis,
        approver_role=approver_role,
        at=at,
        count=n,
        exposure_hours=hours,
        estimate=estimate,
        upper=upper,
        chance_estimate=stale_chance(estimate, limit_seconds / 3600)
        if estimate is not None
        else None,
        chance_upper=stale_chance(upper, limit_seconds / 3600) if upper is not None else None,
    )
