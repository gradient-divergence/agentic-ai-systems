"""Frozen context-to-action distributions, including adapters for Chapter 17."""

from dataclasses import dataclass

import numpy as np

from refund_agent.offers.notices import BASELINE, OFFERS
from refund_agent.offers.policies import fingerprint


@dataclass(frozen=True)
class Policy:
    version: str
    tables: tuple
    source_fingerprint: str = ""
    frozen_at: str = ""

    def __post_init__(self):
        cells = [cell for cell, _ in self.tables]
        if not cells or len(cells) != len(set(cells)):
            raise ValueError("unique, nonempty policy cells required")
        for _, distribution in self.tables:
            p = list(dict(distribution).values())
            if (
                len(p) != len(distribution)
                or not p
                or any(not np.isfinite(x) or x < 0 for x in p)
                or not np.isclose(sum(p), 1)
            ):
                raise ValueError("a probability distribution is required in every cell")

    def probabilities(self, context):
        return dict(dict(self.tables)[context])

    @property
    def fingerprint(self):
        return fingerprint(self.record())

    def record(self):
        return dict(
            version=self.version,
            tables=self.tables,
            source_fingerprint=self.source_fingerprint,
            frozen_at=self.frozen_at,
        )

    @classmethod
    def deterministic(cls, version, choices, actions, **metadata):
        return cls(
            version,
            tuple(
                (cell, tuple((a, float(a == chosen)) for a in actions))
                for cell, chosen in sorted(choices.items())
            ),
            **metadata,
        )

    @classmethod
    def from_offer_policy(cls, policy):
        return cls.deterministic(
            policy.version,
            {cell: offer for cell, offer, *_ in policy.tables},
            OFFERS,
            source_fingerprint=policy.fingerprint,
            frozen_at=policy.frozen_at.isoformat(),
        )


def offer_candidates(package):
    a = Policy.from_offer_policy(package.candidate)
    b = Policy.deterministic(
        "not-delivered-only-1",
        {
            cell: offer if cell == "not_delivered" else BASELINE
            for cell, offer, *_ in package.candidate.tables
        },
        OFFERS,
        source_fingerprint=package.candidate.fingerprint,
        frozen_at=package.candidate.frozen_at.isoformat(),
    )
    return {"A": a, "B": b}, Policy.from_offer_policy(package.baseline)
