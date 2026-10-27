"""Append-only decision and maturity records; earlier events are never rewritten."""

import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path

from refund_agent.offers.notices import OFFERS
from refund_agent.offers.policies import fingerprint
from refund_agent.offers.response import WINDOW_DAYS


@dataclass(frozen=True)
class OfferRow:
    decision_id: str
    request_id: str
    customer_id: str
    decided_at: datetime
    reason: str
    amount_cents: int
    history_policy_version: str
    eligible_offers: tuple[str, ...]
    policy_version: str
    policy_fingerprint: str
    seed: int
    cap: float
    coin: bool
    probabilities: tuple[float, ...]
    offer: str
    propensity: float
    override: bool
    contact_at: datetime | None
    matures_at: datetime
    reward: int | None
    observed_at: datetime | None = None
    table_key: str = ""
    table_fingerprint: str = ""
    table_answers: tuple[int, ...] = (0, 0, 0)
    table_no_contacts: tuple[int, ...] = (0, 0, 0)

    def __post_init__(self):
        p = self.probabilities
        if (
            len(p) != 3
            or any(not math.isfinite(x) or x < 0 for x in p)
            or not math.isclose(sum(p), 1, abs_tol=1e-12)
        ):
            raise ValueError("three probabilities summing to 1 required")
        if (
            not self.eligible_offers
            or len(set(self.eligible_offers)) != len(self.eligible_offers)
            or any(a not in OFFERS for a in self.eligible_offers)
        ):
            raise ValueError("an eligible-offer mask is required")
        if any(p[i] != 0 for i, offer in enumerate(OFFERS) if offer not in self.eligible_offers):
            raise ValueError("ineligible offers must have zero probability")
        if self.offer not in self.eligible_offers or self.propensity != p[OFFERS.index(self.offer)]:
            raise ValueError("propensity must be the applied probability of the sent offer")
        if self.propensity == 0 and not self.override:
            raise ValueError("a zero-probability sent offer requires a recorded human override")
        if self.decided_at.utcoffset() is None or self.matures_at != self.decided_at + timedelta(
            days=WINDOW_DAYS
        ):
            raise ValueError("an aware decision time and a 14-day maturity time are required")
        if not 0 <= self.cap <= 1 or type(self.coin) is not bool or type(self.override) is not bool:
            raise ValueError("valid cap and explicit coin/override flags required")
        self._validate_answer()

    def _validate_answer(self):
        if self.reward is None:
            if self.contact_at is not None or self.observed_at is not None:
                raise ValueError("pending rows expose no outcome")
        elif (
            self.reward not in (0, 1)
            or self.observed_at is None
            or self.observed_at < self.matures_at
        ):
            raise ValueError("a reward is observed only once the window matures")
        elif (self.contact_at is None) != bool(self.reward):
            raise ValueError("no contact has reward 1")
        if self.contact_at is not None and not self.decided_at < self.contact_at <= self.matures_at:
            raise ValueError("contact must belong to this decision's window")

    @classmethod
    def draft(cls, row, policy, seed, offers, probabilities, offer, *, cap, coin, **snapshot):
        return cls(
            decision_id=f"offer-{row.request_id}",
            request_id=row.request_id,
            customer_id=row.requester_id,
            decided_at=row.at,
            reason=row.reason,
            amount_cents=row.amount_cents,
            history_policy_version=row.policy_version,
            eligible_offers=offers,
            policy_version=policy.version,
            policy_fingerprint=policy.fingerprint,
            seed=seed,
            cap=cap,
            coin=coin,
            probabilities=tuple(map(float, probabilities)),
            offer=offer,
            propensity=float(probabilities[OFFERS.index(offer)]),
            override=False,
            contact_at=None,
            matures_at=row.at + timedelta(days=WINDOW_DAYS),
            reward=None,
            **snapshot,
        )

    def record(self):
        value = asdict(self)
        for key in ("decided_at", "contact_at", "matures_at", "observed_at"):
            value[key] = value[key].isoformat() if value[key] is not None else None
        return value

    @classmethod
    def from_record(cls, record):
        value = dict(record)
        for key in ("decided_at", "contact_at", "matures_at", "observed_at"):
            value[key] = datetime.fromisoformat(value[key]) if value[key] is not None else None
        for key in ("eligible_offers", "probabilities", "table_answers", "table_no_contacts"):
            value[key] = tuple(value[key])
        return cls(**value)


class OfferLog:
    def __init__(self):
        self._events = []
        self._rows = {}

    @property
    def rows(self):
        return tuple(self._rows.values())

    @property
    def events(self):
        return tuple(self._events)

    @property
    def fingerprint(self):
        return fingerprint([row.record() for row in self._events])

    def append(self, row):
        previous = self._rows.get(row.decision_id)
        if previous is not None:
            fixed = set(row.record()) - {"reward", "contact_at", "observed_at"}
            if (
                previous.reward is not None
                or row.reward is None
                or any(getattr(previous, key) != getattr(row, key) for key in fixed)
            ):
                raise ValueError("only one maturity event may follow a decision")
        self._events.append(row)
        self._rows[row.decision_id] = row

    def write(self, path):
        import fcntl

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = [json.dumps(row.record(), sort_keys=True) + "\n" for row in self._events]
        with path.open("a+", encoding="utf-8") as stream:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            stream.seek(0)
            stored = stream.readlines()
            if stored != payload[: len(stored)]:
                raise ValueError("existing offer events differ; refusing to rewrite")
            stream.writelines(payload[len(stored) :])

    @classmethod
    def read(cls, path):
        result = cls()
        with Path(path).open(encoding="utf-8") as stream:
            for line in stream:
                result.append(OfferRow.from_record(json.loads(line)))
        return result
