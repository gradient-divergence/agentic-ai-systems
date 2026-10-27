"""Read chapter 21's verdict record only; no offer outcomes or sealed package."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from refund_agent.offpolicy.policies import Policy
from refund_agent.offpolicy.record import EvaluationRecord
from refund_agent.selection import fingerprint

RECORD = Path(__file__).resolve().parents[2] / "demos/data/practical_rl_evaluation_record.json"


@dataclass(frozen=True)
class OfferSelection:
    name: str
    policy: Policy
    candidates: dict
    limits: tuple
    record_fingerprint: str
    access: dict

    def record(self):
        return dict(
            name=self.name,
            fingerprint=self.policy.fingerprint,
            version=self.policy.version,
            policy=self.policy.record(),
            evaluation_fingerprint=self.record_fingerprint,
            candidates=self.candidates,
            limits=list(self.limits),
        )

    def notice(self, reason, *, seed=7, index=0):
        distribution = self.policy.probabilities(reason)
        rng = np.random.default_rng(np.random.SeedSequence([seed, index]))
        return str(rng.choice(list(distribution), p=list(distribution.values())))


def select_offers(record, *, access=None):
    policies = record.data["policies"]
    candidates = {
        name: {k: card[k] for k in ("fingerprint", "decision", "reasons")}
        for name, card in policies.items()
        if name != "fixed"
    }
    eligible = [
        name
        for name, card in candidates.items()
        if card["decision"] == "eligible for the shadow run"
    ]
    if len(eligible) > 1:
        raise ValueError("multiple eligible offers require the release owner to select one")
    name = eligible[0] if eligible else "fixed"
    fields = policies[name]["policy"]
    policy = Policy(
        fields["version"],
        tuple((cell, tuple((offer, p) for offer, p in table)) for cell, table in fields["tables"]),
        fields["source_fingerprint"],
        fields["frozen_at"],
    )
    if policy.fingerprint != policies[name]["fingerprint"]:
        raise ValueError("offer policy fingerprint differs from evaluation record")
    return OfferSelection(
        name,
        policy,
        candidates,
        tuple(record.data["limits"]),
        fingerprint(record.data),
        access or {"purpose": "read constructed evaluation record; no outcomes read"},
    )


def load_offers(path=RECORD):
    access = {
        "path": "demos/data/" + Path(path).name,
        "purpose": "read chapter 21 evaluation record; no outcomes read",
    }
    return select_offers(EvaluationRecord.read(path), access=access)
