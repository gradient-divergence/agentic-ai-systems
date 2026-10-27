"""Constructed potential answers, drawn independently of the policy's choices."""

from dataclasses import dataclass
from datetime import datetime, timedelta

from refund_agent.data.generate import number
from refund_agent.offers.notices import OFFERS

WINDOW_DAYS = 14
NO_CONTACT_RATE = {
    "damaged": (0.80, 0.86, 0.84),
    "changed_mind": (0.80, 0.83, 0.81),
    "not_delivered": (0.80, 0.82, 0.88),
}


@dataclass(frozen=True)
class Answer:
    offer: str
    contact_at: datetime | None
    matures_at: datetime
    reward: int


def true_rate(reason, offer):
    return NO_CONTACT_RATE[reason][OFFERS.index(offer)]


def answers(row):
    """The history's seed and row index identify the same world under every policy."""
    seed, index = map(int, row.request_id.rsplit("-", 2)[-2:])
    result = {}
    for offer in OFFERS:
        no_contact = number(seed, index, f"history-offer-{offer}", 10**9) / 10**9 < true_rate(
            row.reason, offer
        )
        day = 1 + number(seed, index, f"history-contact-day-{offer}", WINDOW_DAYS)
        result[offer] = Answer(
            offer,
            None if no_contact else row.at + timedelta(days=day),
            row.at + timedelta(days=WINDOW_DAYS),
            int(no_contact),
        )
    return result
