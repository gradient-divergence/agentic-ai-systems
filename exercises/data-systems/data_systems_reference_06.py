"""Reference solution contract for the announced card-feed release."""

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
    if record["version"] not in ("card@1", "card@2"):
        return "version_unknown"
    if record["type"] not in ("purchase", "refund"):
        return "transaction_vocabulary"
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
