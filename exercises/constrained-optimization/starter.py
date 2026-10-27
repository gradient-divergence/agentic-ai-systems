"""Exercise 8: return a checked deadline plan, or a marked usual-plan fallback.

Set CONSTRAINED_OPTIMIZATION_SOLUTION to this file when running test_acceptance.py.
Return status, record (a SolveRecord), required, missing_rule, proof and repair.
proof is None on a feasible morning; otherwise it gives the tier-3 required IDs,
tier3_minutes, morgan_minutes and extra_morgan_minutes. repair is the SolveRecord
with that minimum extra time, or None when no repair is needed.
"""


def plan_morning(queue, minutes, budget):
    raise NotImplementedError("enforce the respond-by rule and prove the minimum extra time")
