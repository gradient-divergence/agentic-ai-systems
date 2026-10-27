"""Deliberately incorrect starters, one designed error per exercise."""

from fractions import Fraction


def exercise_01():
    # Fix: distinguish the four checks, instead of checking presence alone.
    return ("complete", "complete", "complete", "complete")


def exercise_02():
    # Fix: a launch hold grants nothing; a reversal undoes no ledger entries.
    return {
        "stage": 7,
        "launch_signed": True,
        "withdraws": "tier1",
        "mechanism": "chapter 34 monitor -> chapter 38 stage authority",
        "decider_after": "person",
        "paid_refunds_undone": True,
    }


def exercise_03(packet):
    # Fix: a passing record can still name a bundle whose evidence does not count.
    return {"verdict": "launch", "not_counted": ()}


def exercise_04():
    # Fix: read the reopened-ticket countermeasure as well as the hours.
    return {
        "manual_hours": Fraction(3700, 60),
        "agent_hours": Fraction(2884, 60),
        "manual_reopened": 2,
        "agent_reopened": 3,
        "verdict": "launch",
    }


def exercise_05(decision):
    return {
        "verdict": decision["verdict"],
        "target": decision["promotion"]["rollback_target"],
    }


def check_maturity(record, gate_day, mode="companion"):
    # Fix: deployed mode must read the last payment day and wait for its labels.
    return ()
