"""Intentional mistakes for exercises 1 to 5; replace each answer and rerun."""

from math import ceil


def exercise_1():
    return {"accumulates": "payments", "sensor": "receipt_count"}


def exercise_2(arrival):
    from datetime import timedelta

    return {"due": arrival + timedelta(hours=24), "timeout": "approve"}


def exercise_3(lam, mu, deadline):
    return {"u": lam / mu, "L": lam / mu, "W": 1 / mu,
            "W_wait": 0, "tail": 0}


def exercise_4():
    return {"ordinary_hours": 70, "senior_hours": 28,
            "adjusters": ceil(70 / (7 * 8 * .8)), "senior_monday_feasible": True}


def exercise_5(rows):
    planted = [r for r in rows if r.planted]
    return {"caught": sum(r.action in ("refused", "escalated") for r in planted),
            "decided": len(planted), "bands": {}}
