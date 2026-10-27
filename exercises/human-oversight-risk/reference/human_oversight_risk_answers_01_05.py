"""Reference solutions; formulas reuse the chapter's checked implementations."""

from math import ceil

from refund_agent.oversight.human_oversight_risk_lifecycle import respond_by
from refund_agent.oversight.human_oversight_risk_monitor import load_report
from refund_agent.oversight.human_oversight_risk_queueing import mm1


def exercise_1():
    return {"accumulates": "review_work", "sensor": "backlog_and_item_timestamps"}


def exercise_2(arrival):
    return {"due": respond_by(arrival), "timeout": "escalate_and_notify"}


def exercise_3(lam, mu, deadline):
    row = mm1(lam, mu)
    return {"u": row.u, "L": row.L, "W": row.W,
            "W_wait": row.W_wait, "tail": row.tail(deadline)}


def exercise_4():
    ordinary, senior = 30 * 7 * 20 / 60, 6 * 7 * 40 / 60
    return {"ordinary_hours": ordinary, "senior_hours": senior,
            "adjusters": ceil(ordinary / (40 * .8)),
            "senior_monday_feasible": 6 * 55 / 24 * 40 / 60 <= 8}


def exercise_5(rows):
    planted = [r for r in rows if r.planted]
    decided = [r for r in planted if r.closed_at and r.action in ("approved", "refused", "modified")]
    return {"caught": sum(r.action in ("refused", "modified") for r in decided),
            "decided": len(decided), "bands": load_report(rows)}
