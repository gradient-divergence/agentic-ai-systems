"""Every working morning of the training window, with carry-over and start waits."""

from datetime import timedelta
from math import floor
from statistics import mean

from refund_agent.data.generate import generate

from .bound import count_bound, count_rows
from .certificate import check_certificate
from .check import check_plan
from .inputs import BUDGET, END, MINUTES, START, TOL, WORKING_DAYS
from .model import solve_morning
from .queue import training_requests, waiting
from .record import SolveRecord, queue_fingerprint, queue_inputs
from .rule import plain_rule


def _plain_record(queue):
    plan, bound = plain_rule(queue, MINUTES), count_bound(queue, MINUTES)
    pairs, rows, rhs, objective = count_rows(queue, MINUTES)
    chosen = {a.request_id: a.person for a in plan}
    vector = [int(chosen.get(queue[j].request_id) == p) for j, p in pairs]
    check = check_certificate(rows, rhs, objective, vector, bound["multipliers"])
    return SolveRecord(
        queue_fingerprint(queue, MINUTES),
        BUDGET,
        "feasible",
        plan,
        len(plan),
        bound["value"],
        bound["multipliers"],
        check,
        check_plan(plan, queue, MINUTES),
        inputs=queue_inputs(queue, MINUTES),
    )


def _waits(queue, plan):
    lookup = {d.request_id: d for d in queue}
    return [
        dict(
            request_id=a.request_id,
            tier=lookup[a.request_id].tier,
            start=a.start.isoformat(),
            days=(a.start - lookup[a.request_id].at).total_seconds() / 86400,
        )
        for a in plan
    ]


def _day_result(day, queue, record, policy):
    rounded = floor(record.bound + TOL)
    best_count = int(record.value) if record.status == "optimal" else None
    if record.value == rounded:
        best_count = rounded  # this permitted whole plan meets the rounded bound
    elif policy == "plain":
        best_count = int(solve_morning(queue, MINUTES, BUDGET, prefer_wait=False).value)
    return dict(
        day=day.isoformat(),
        queue=queue,
        answered=len(record.plan),
        best_count=best_count,
        bound=record.bound,
        rounded_bound=rounded,
        record=record,
    )


def _window_summary(policy, seed, data, days, waits, carried):
    tier3 = [w["days"] for w in waits if w["tier"] == 3]
    return dict(
        policy=policy,
        seed=seed,
        requests=len(training_requests(data)),
        days=days,
        answered=len(waits),
        remaining=carried,
        waits=waits,
        tier3=dict(mean_days=mean(tier3), longest_days=max(tier3)),
        at_bound=sum(d["answered"] == d["rounded_bound"] for d in days),
    )


def run_window(policy, seed=7, data=None):
    if policy not in ("plain", "count", "count_then_wait"):
        raise ValueError("unknown review policy")
    data = generate(seed) if data is None else data
    day, carried, days, waits = START, [], [], []
    while day <= END:
        if day.weekday() in WORKING_DAYS:
            queue = waiting(data, day, carried)
            record = (
                _plain_record(queue)
                if policy == "plain"
                else solve_morning(queue, MINUTES, BUDGET, prefer_wait=policy == "count_then_wait")
            )
            waits.extend(_waits(queue, record.plan))
            answered = {a.request_id for a in record.plan}
            carried = [d for d in queue if d.request_id not in answered]
            days.append(_day_result(day, queue, record, policy))
        day += timedelta(days=1)
    return _window_summary(policy, seed, data, days, waits, carried)
