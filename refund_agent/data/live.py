"""A frozen generated shadow stream, distinct from historical and instructor sets.

September's fictional policy is fixed. Evidence ages are drawn independently from
constructed masses at a normal age, the age limit and one second past it. Labels
repeat the policy and stand in for blind person decisions, not observed people.
"""

from dataclasses import dataclass, field
from datetime import timedelta

from refund_agent import contracts as c
from refund_agent.data.generate import (
    Dataset,
    _label,
    _order,
    _policies,
    _request,
    _shipment,
    number,
)
from refund_agent.selection import fingerprint

AGES = {
    "order": ((60, 80), (1800, 10), (1801, 10)),
    "shipment": ((120, 80), (1800, 10), (1801, 10)),
    "policy": ((3600, 90), (2592000, 5), (2592001, 5)),
}


def _age(seed, index, source):
    draw = number(seed, index, f"live-{source}-age", 100)
    total = 0
    for age, mass in AGES[source]:
        total += mass
        if draw < total:
            return age
    raise AssertionError("age masses must sum to one hundred")


@dataclass(frozen=True)
class LiveStream:
    data: Dataset
    manifest: dict
    sha256: str = field(init=False)

    def __post_init__(self):
        object.__setattr__(self, "sha256", self._fingerprint())

    def _fingerprint(self):
        return fingerprint({"manifest": self.manifest, "data": self.data.model_dump(mode="json")})

    def intact(self):
        return self.sha256 == self._fingerprint()

    @property
    def fingerprint(self):
        return self.sha256


def generate_live(seed, start, days, per_day):
    start = c._utc_time(start)
    if start.date().isoformat() <= "2026-09-27" or any(
        type(n) is not int or n < 1 for n in (days, per_day)
    ):
        raise ValueError("a new post-corpus start and positive days and requests per day required")
    policy = _policies()[-1]
    reviewers = [
        c.Reviewer(reviewer_id="alex", name="Alex", role="reviewer"),
        c.Reviewer(reviewer_id="morgan", name="Morgan", role="named_approver"),
    ]
    data = Dataset(
        orders=[],
        requests=[],
        shipments=[],
        policies=[policy],
        retrievals=[],
        correct=[],
        reviewers=reviewers,
    )
    for i in range(days * per_day):
        at = start + timedelta(days=i // per_day, seconds=number(seed, i, "live-second", 86400))
        records = _live_records(seed, i, at, policy)
        for name, record in zip(
            ("requests", "orders", "shipments", "retrievals", "correct"), records
        ):
            getattr(data, name).append(record)
    return LiveStream(data, _live_manifest(seed, start, days, per_day, policy))


def _live_records(seed, index, at, policy):
    j = index + 7  # Unique references; the original seven-amount cycle.
    order = _order(seed, j, at, None, "live-")
    order = order.model_copy(
        update={"observed_at": at - timedelta(seconds=_age(seed, index, "order"))}
    )
    shipment = _shipment(seed, j, at, order, None)
    shipment = shipment.model_copy(
        update={"observed_at": at - timedelta(seconds=_age(seed, index, "shipment"))}
    )
    request = _request(seed, j, at, order, None, "live-")
    retrieval = c.PolicySnapshot(
        request_id=request.request_id,
        version=policy.version,
        observed_at=at - timedelta(seconds=_age(seed, index, "policy")),
    )
    return request, order, shipment, retrieval, _label(request, order, shipment, policy)


def _live_manifest(seed, start, days, per_day, policy):
    return {
        "seed": seed,
        "frozen_at": start.isoformat(),
        "start": start.isoformat(),
        "days": days,
        "per_day": per_day,
        "policy_version": policy.version,
        "age_distribution": AGES,
        "person_reference": "generated labels; blind to agent shadow decisions",
        "outcome_rule": "constructed paid-only 30-day outcome; never observed before final time",
    }
