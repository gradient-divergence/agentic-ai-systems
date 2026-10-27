"""Exercises 1–5: one deliberate conceptual fault in each response."""


def exercise_1():
    return {"response": "deny", "reason": "contract_vocabulary:shipment.status"}


def exercise_2():
    return {"caught_by": "vocabulary", "responsible": "the carrier integration team"}


def exercise_3():
    return {"estimate": 0.015, "upper": 0.015, "chance": 0.0025, "minutes": 20.05}


def exercise_4():
    return {"quiet_upper": 0, "quiet_chance": 0, "daytime_rate": 0.04, "daytime_upper": 0.04}


def exercise_5(fixture):
    return {
        "condition": "freshness",
        "evidence": "policy too old",
        "rebuild": True,
        "reply": "Please refresh the policy.",
    }
