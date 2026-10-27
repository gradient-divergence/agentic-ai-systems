"""Submit these five exercises by correcting the copied refund answers."""


def responses():
    return {
        "01": {"packet": ["b", "d", "f", "g"], "refund": ["a", "c", "e", "h"]},
        "02": {
            "triage": ("clock", "number"),
            "hospital": ("clock", "number"),
            "bank": ("clock", "number"),
        },
        "03": {"budget_seconds": 1800, "chance_30_minutes": 0.02},
        "04": {
            "clear_edge": "2/25",
            "hold_edge": "4/5",
            "decisions": ["clear", "refer"],
            "day_hold_edge": "1/7",
        },
        "05": {"window_days": 30, "decision": "go", "hidden_fraud_share": 0.0},
    }
