"""Exercise 6 starter: open vocabulary and no version pin are planted faults."""

from refund_agent.sources.data_systems_budget import budget_hours, rate_upper

OWNER = "the card feed team"
VERSION = "card@2"
UNITS = "USD cents"
EFFECTIVE = "2026-10-01"
ESTIMATE = 30 / 600
UPPER = rate_upper(30, 600)
BUDGET_SECONDS = 3600 * budget_hours(UPPER, 0.01)
NOTICE_DAYS = 7
BASIS = "constructed: 30 refunds in 600 hours of disputes awaiting a decision"


def problem(record):
    # Intended fault 1: no closed transaction-type vocabulary.
    # Intended fault 2: no pin on the served contract versions.
    amount = record["amount_cents"]
    if type(amount) is not int or amount == 0:
        return "integer_nonzero_cents"
    if record["type"] == "purchase" and amount < 0:
        return "purchase_sign"
    if record["type"] == "refund":
        if record["version"] == "card@1" and amount < 0:
            return "refund_sign"
        if record["version"] == "card@2" and amount > 0:
            return "refund_sign"
    if record["version"] == "card@2" and record["at"] < EFFECTIVE:
        return "version_not_effective"
    if not 0 <= record["age_seconds"] <= BUDGET_SECONDS:
        return "budget"
    return None
