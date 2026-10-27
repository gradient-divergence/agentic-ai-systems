"""The three notices, in the order used for deterministic ties and probability vectors."""

from dataclasses import dataclass
from datetime import datetime

from refund_agent.rules import tier_for

OFFERS = ("standard", "dated", "referenced")
BASELINE = "standard"
DESCRIPTIONS = {
    "standard": "The amount, policy clause and original payment method.",
    "dated": "The standard notice and the date the money should arrive.",
    "referenced": "The dated notice and a payment reference to quote to the bank.",
}


@dataclass(frozen=True)
class OfferRequest:
    """An allowlist of decision-time inputs; no false-claim label can reach a policy."""

    request_id: str
    requester_id: str
    at: datetime
    reason: str
    amount_cents: int
    policy_version: str
    paid: bool = True


def eligible(row):
    return OFFERS if row.paid and int(tier_for(row.amount_cents)) == 1 else ()
