"""Starter: a correct policy referral is currently misclassified as failure."""


def classify_outcomes(rows):
    return {
        row["id"]: {
            "timeliness": "bad" if row["outcome"] == "refer" else "good",
            "corrections": "not_counted",
            "breaches": [],
        }
        for row in rows
    }
