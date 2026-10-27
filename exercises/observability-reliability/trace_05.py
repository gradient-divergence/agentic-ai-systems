"""Starter: this trusts the frozen audit age and ignores the execution gap."""


def diagnose(record):
    return {
        "age_at_payment_seconds": record["audit_age_seconds"],
        "breach": False,
        "action": "continue",
    }
