"""Intentionally wrong exercise 5: appeals used as wrongful-denial readings."""


def analyze_report(report):
    rates = {
        g: report["appeals"][g] / report["denials"][g]
        for g in report["denials"]
        if report["denials"][g]
    }
    region = max(rates, key=rates.__getitem__)
    return {
        "holds": {region: ["already_delivered"]},
        "reply": "The region with the most appeals per denial should be held.",
    }
