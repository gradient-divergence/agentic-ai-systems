"""Reference solutions and and worked arithmetic for exercises 1–5."""

from datetime import datetime

from refund_agent.sources.data_systems_budget import (
    budget_hours,
    rate_upper,
    stale_chance,
)


def exercise_1():
    return {"response": "refer", "reason": "contract_vocabulary:shipment.status"}


def exercise_2():
    return {"caught_by": "none", "responsible": "the carrier integration team"}


def exercise_3():
    upper = rate_upper(15, 1000)
    return {
        "estimate": 15 / 1000,
        "upper": upper,
        "chance": stale_chance(upper, 10 / 60),
        "minutes": 60 * budget_hours(upper, 0.005),
    }


def exercise_4():
    upper = rate_upper(0, 6000)
    return {
        "quiet_upper": upper,
        "quiet_chance": stale_chance(upper, 0.5),
        "daytime_rate": 48 / 1200,
        "daytime_upper": rate_upper(48, 1200),
    }


def exercise_5(fixture):
    """The corrupted export simulates an overwrite outside the immutable store.

    Its shipment fingerprint matches the later record, observed 40 minutes after
    payment. The original input is absent. Ask the carrier integration team to
    recover the original input by its fingerprint; the payment's evidence stays
    unreconstructed until it is found.
    """
    at = datetime.fromisoformat(fixture["lineage"]["decision_at"])
    shipment = next(r for r in fixture["lineage"]["inputs"] if r["source"] == "shipment")
    stored = fixture["records"][shipment["fingerprint"]]
    observed = datetime.fromisoformat(stored["observed_at"])
    if observed <= at:
        raise ValueError("fixture has no planted later shipment")
    return {
        "condition": "lineage",
        "evidence": "shipment observed after decision",
        "rebuild": False,
        "reply": "Carrier integration team: recover the original fingerprint's record; "
        "the payment's evidence stays unreconstructed until it is found.",
    }
