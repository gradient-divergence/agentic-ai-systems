"""Count first, then maximize hours already waited with that count fixed."""

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from .bound import count_bound, count_rows
from .certificate import check_certificate
from .check import check_plan
from .record import SolveRecord, queue_fingerprint, queue_inputs
from .rule import ordered_plan

_STATUSES = {
    0: "optimal",
    1: "budget exhausted",
    2: "infeasible",
    3: "unbounded",
    4: "unknown",  # SciPy reports "other", not a diagnosis of an invalid model.
}


def _constraints(queue, rows, rhs, required):
    lower = np.full(len(rhs), -np.inf)
    for j, draft in enumerate(queue):
        if draft.request_id in required:
            lower[j] = 1
    return [LinearConstraint(rows, lower, rhs)]


def _waiting_solution(queue, pairs, objective, count, constraints, options, budget):
    constraints.append(LinearConstraint(objective, round(-count.fun), round(-count.fun)))
    waited = np.array([(queue[j].plan_at - queue[j].at).total_seconds() / 3600 for j, _ in pairs])
    options["node_limit"] = max(0, budget - int(count.mip_node_count or 0))
    later = milp(
        -waited,
        integrality=np.ones(len(pairs)),
        bounds=Bounds(0, 1),
        constraints=constraints,
        options=options,
    )
    usable = later.status in (0, 1) and later.x is not None
    return later.x if usable else count.x, _STATUSES[later.status]


def _choose(queue, problem, budget, prefer_wait, required):
    pairs, rows, rhs, objective = problem
    constraints = _constraints(queue, rows, rhs, required)
    options = dict(node_limit=budget, mip_rel_gap=0.0)
    count = milp(
        -objective,
        integrality=np.ones(len(pairs)),
        bounds=Bounds(0, 1),
        constraints=constraints,
        options=options,
    )
    status = _STATUSES[count.status]
    if count.status == 1:
        status = "feasible" if count.x is not None else "unknown"
    solution, secondary = count.x if count.status in (0, 1) else None, "not requested"
    if count.status == 0 and prefer_wait:
        solution, secondary = _waiting_solution(
            queue, pairs, objective, count, constraints, options, budget
        )
    chosen = {}
    if solution is not None:
        chosen = {queue[j].request_id: p for k, (j, p) in enumerate(pairs) if solution[k] > 0.5}
    return chosen, status, secondary


def _checked_record(queue, minutes, budget, status, secondary, chosen, bound, problem):
    pairs, rows, rhs, objective = problem
    plan = ordered_plan(queue, chosen)
    vector = [int(chosen.get(queue[j].request_id) == p) for j, p in pairs]
    certificate = check_certificate(rows, rhs, objective, vector, bound["multipliers"])
    failures = check_plan(plan, queue, minutes)
    if failures or not certificate["permitted"] or not certificate["certificate"]:
        raise RuntimeError("returned incumbent failed the independent check")
    return SolveRecord(
        queue_fingerprint=queue_fingerprint(queue, minutes),
        budget=budget,
        status=status,
        plan=plan,
        value=len(plan),
        bound=bound["value"],
        multipliers=bound["multipliers"],
        certificate=certificate,
        plan_check=failures,
        secondary_status=secondary,
        inputs=queue_inputs(queue, minutes),
    )


def solve_morning(queue, minutes, budget, *, prefer_wait=True, required=()):
    if type(budget) is not int or budget < 0:
        raise ValueError("budget is a nonnegative branch-node count")
    if not set(required) <= {d.request_id for d in queue}:
        raise ValueError("required drafts must be in the waiting queue")
    problem, bound = count_rows(queue, minutes), count_bound(queue, minutes)
    status, secondary, chosen = "optimal", "not requested", {}
    if problem[0]:
        chosen, status, secondary = _choose(queue, problem, budget, prefer_wait, required)
    elif required:
        status = "infeasible"
    return _checked_record(queue, minutes, budget, status, secondary, chosen, bound, problem)
