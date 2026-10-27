"""Constructed stage-6 rates, independent of generated labels and golden cases."""

from dataclasses import dataclass
from datetime import datetime, timedelta
from fractions import Fraction
from math import isfinite
from random import Random

from refund_agent import contracts as c
from refund_agent.data.generate import Dataset, _order, _policies, _request, _shipment
from refund_agent.rules import tier_for
from refund_agent.schedule.inputs import REVIEW_MINUTES


@dataclass(frozen=True)
class WorkClass:
    name: str
    rate_per_day: float
    minutes: float
    tier: int
    blocking: bool = True
    working_days: int = 1
    role: str = "reviewer"

    def __post_init__(self):
        if (not self.name or not isfinite(self.rate_per_day) or self.rate_per_day < 0
                or not isfinite(self.minutes) or self.minutes <= 0
                or self.tier not in (1, 2, 3) or self.working_days < 1
                or self.role not in ("reviewer", "named_approver", "decision_owner")):
            raise ValueError("invalid constructed work class")


@dataclass(frozen=True)
class Stage6Demand:
    tier1_monthly: float = 2000
    month_days: int = 30
    tier2_daily: float = 16
    tier3_daily: float = 4
    planted_weekly: float = 10
    referral_share: float = 25 / 175
    payment_share: float = 64 / 175
    recheck_share: float = .25
    blind_share: float = .10
    held_denials_monthly: float = 0
    modified_share: float = 0
    refused_share: float = 0
    blind_disagreement_share: float = 0
    reversal: bool = False

    def __post_init__(self):
        shares = (self.referral_share, self.payment_share, self.recheck_share,
                  self.blind_share, self.modified_share, self.refused_share,
                  self.blind_disagreement_share)
        rates = (self.tier1_monthly, self.tier2_daily, self.tier3_daily,
                 self.planted_weekly, self.held_denials_monthly)
        if (type(self.month_days) is not int or self.month_days < 1
                or any(not isfinite(x) or not 0 <= x <= 1 for x in shares)
                or any(not isfinite(x) or x < 0 for x in rates)
                or self.referral_share + self.payment_share > 1):
            raise ValueError("invalid stage-6 rates or shares")

    def classes(self) -> tuple[WorkClass, ...]:
        tier1 = self.tier1_monthly / self.month_days
        rows = (
            WorkClass("tier2", self.tier2_daily, REVIEW_MINUTES[2], 2),
            WorkClass("tier3", self.tier3_daily, REVIEW_MINUTES[3], 3, role="named_approver"),
            WorkClass("referral", tier1 if self.reversal else tier1 * self.referral_share, 5, 1),
            WorkClass("recheck", 0 if self.reversal else tier1 * self.payment_share * self.recheck_share,
                      5, 1, False, 2),
            WorkClass("blind", self.tier2_daily * self.blind_share, 15, 2, False, 5),
            WorkClass("planted", self.planted_weekly / 7, 15, 2),
            WorkClass("held_denial", self.held_denials_monthly / self.month_days, 5, 1),
            # A modification consumes new-proposal preparation AND a new review.
            WorkClass("modification", self.tier2_daily * self.modified_share, 20, 2),
            WorkClass("refusal_followup", self.tier2_daily * self.refused_share,
                      15, 2, role="decision_owner"),
            WorkClass("adjudication", self.tier2_daily * self.blind_share * self.blind_disagreement_share,
                      30, 2, role="decision_owner"),
        )
        return rows


def generate_stage6(seed: int, start: datetime, days: int, demand: Stage6Demand) -> Dataset:
    """Poisson requests; amount choices within a tier use the original builders.

    This function never generates or reads labels. Planted review packets and
    post-decision samples are separate work, not customer requests in this corpus.
    """
    start = c._utc_time(start)
    if type(seed) is not int or type(days) is not int or days < 1:
        raise ValueError("seed must be an integer and days positive")
    policies = [p for p in _policies() if p.effective_at <= start]
    if not policies:
        raise ValueError("no policy in force at the month's start")
    policy = max(policies, key=lambda p: p.effective_at)
    data = Dataset(orders=[], requests=[], shipments=[], policies=[policy], retrievals=[], correct=[],
                   reviewers=[c.Reviewer(reviewer_id="alex", name="Alex", role="reviewer"),
                              c.Reviewer(reviewer_id="riley", name="Riley", role="reviewer"),
                              c.Reviewer(reviewer_id="morgan", name="Morgan", role="named_approver")])
    rng = Random(seed)
    events = []
    amounts = {1: (2499, 4999), 2: (5000, 50000), 3: (50001, 65000, 120000)}
    for tier, rate in ((1, demand.tier1_monthly / demand.month_days),
                       (2, demand.tier2_daily), (3, demand.tier3_daily)):
        t = 0.0
        if rate == 0:
            continue
        while (t := t + rng.expovariate(rate / 86400)) < days * 86400:
            events.append((t, tier, rng.choice(amounts[tier])))
    for index, (seconds, tier, amount) in enumerate(sorted(events), start=7):
        at = start + timedelta(seconds=seconds)
        prefix = "human-oversight-risk-"
        order = _order(seed, index, at, None, prefix)
        shipment = _shipment(seed, index, at, order, None)
        request = _request(seed, index, at, order, None, prefix).model_copy(
            update={"amount_cents": amount})
        assert int(tier_for(request.amount_cents)) == tier
        data.orders.append(order)
        data.shipments.append(shipment)
        data.requests.append(request)
        data.retrievals.append(c.PolicySnapshot(request_id=request.request_id, version=policy.version,
                                                observed_at=at - timedelta(hours=1)))
    return data


def generator_mix_counterfactual(tier1_monthly=2000, month_days=30) -> Fraction:
    return Fraction(tier1_monthly * 3, 2 * month_days)
