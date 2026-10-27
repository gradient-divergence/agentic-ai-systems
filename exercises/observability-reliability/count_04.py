"""Starter: this mistakenly judges the cohort rather than all payments."""

from refund_agent.observability_reliability.budget import binom_tail, min_count


def daily_rule():
    n = 2
    k = min_count(n, 0.01, 0.01)
    return {
        "eligible": n,
        "threshold": k,
        "power": float(binom_tail(k, n, 0.1)),
    }
