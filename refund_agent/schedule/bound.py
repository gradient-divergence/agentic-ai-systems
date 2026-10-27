"""The count relaxation, omitting redundant per-variable upper bounds."""

import numpy as np
from scipy.optimize import linprog

from .certificate import check_certificate


def count_rows(queue, minutes):
    if any(type(v) is not int or v < 0 for v in minutes.values()):
        raise ValueError("capacities must be nonnegative integer minutes")
    pairs = [(j, p) for j, d in enumerate(queue) for p in d.allowed if p in minutes]
    people = list(minutes)
    rows = np.zeros((len(queue) + len(people), len(pairs)))
    for k, (j, person) in enumerate(pairs):
        rows[j, k] = 1
        rows[len(queue) + people.index(person), k] = queue[j].minutes
    rhs = np.array([1] * len(queue) + list(minutes.values()), dtype=float)
    return pairs, rows, rhs, np.ones(len(pairs))


def count_bound(queue, minutes):
    pairs, rows, rhs, objective = count_rows(queue, minutes)
    if not pairs:
        plan, multipliers = np.zeros(0), np.zeros(len(rhs))
    else:
        result = linprog(-objective, A_ub=rows, b_ub=rhs, bounds=(0, None), method="highs")
        if not result.success:
            raise RuntimeError(f"count relaxation status {result.status}")
        plan, multipliers = result.x, -result.ineqlin.marginals
        multipliers = np.where(multipliers == 0, 0.0, multipliers)
    check = check_certificate(rows, rhs, objective, plan, multipliers)
    if not check["optimal"]:
        raise RuntimeError("relaxation certificate failed independent arithmetic")
    return dict(
        value=check["bound"], multipliers=multipliers.tolist(), plan=plan.tolist(), check=check
    )
