"""Reference solution: deadlines remain hard; infeasibility carries a permitted fallback."""

from demos.constrained_optimization_demo import respond_by_day
from refund_agent.schedule.model import solve_morning


def plan_morning(queue, minutes, budget):
    required = [d.request_id for d in queue if respond_by_day(d.at) <= d.plan_at.date()]
    record = solve_morning(queue, minutes, budget, required=required)
    status, proof, repair = record.status, None, None
    if status == "infeasible":
        tier3 = [d for d in queue if d.request_id in required and d.tier == 3]
        needed = sum(d.minutes for d in tier3)
        extra = needed - minutes["morgan"]
        assert extra > 0, "this fixture's conflict must be the tier-3 minutes"
        proof = dict(
            required=[d.request_id for d in tier3],
            tier3_minutes=needed,
            morgan_minutes=minutes["morgan"],
            extra_morgan_minutes=extra,
        )
        record = solve_morning(queue, minutes, budget)
        repair = solve_morning(queue, {**minutes, "morgan": needed}, budget, required=required)
    taken = {a.request_id for a in record.plan}
    return dict(
        status=status,
        record=record,
        required=required,
        missing_rule=[key for key in required if key not in taken],
        proof=proof,
        repair=repair,
    )
