"""Split anchored policy text without rewriting it, and retain its provenance."""

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime

from refund_agent.contracts import Policy, _utc_time


def fingerprint(value) -> str:
    text = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(text.encode()).hexdigest()


@dataclass(frozen=True)
class Clause:
    citation: str
    section: str
    version: str
    effective_from: datetime
    effective_to: datetime | None
    text: str
    text_hash: str
    observed_at: datetime
    encoder: str = ""
    vector: tuple[float, ...] = ()

    def in_force(self, at: datetime) -> bool:
        at = _utc_time(at)
        return self.effective_from <= at and (self.effective_to is None or at < self.effective_to)

    def record(self) -> dict:
        return {
            "citation": self.citation,
            "section": self.section,
            "version": self.version,
            "effective_from": self.effective_from.isoformat(),
            "effective_to": self.effective_to.isoformat() if self.effective_to else None,
            "text": self.text,
            "text_hash": self.text_hash,
            "observed_at": self.observed_at.isoformat(),
            "encoder": self.encoder,
            "vector": list(self.vector),
        }


def policy_clauses(policies: list[Policy], observed_at: datetime) -> list[Clause]:
    observed_at = _utc_time(observed_at)
    ordered = sorted(policies, key=lambda p: p.effective_at)
    if len({p.effective_at for p in ordered}) != len(ordered):
        raise ValueError("duplicate policy effective dates")
    if len({p.policy_id for p in ordered}) > 1:
        raise ValueError("one policy history per index")
    clauses = []
    for i, policy in enumerate(ordered):
        anchors = list(re.finditer(r"#([a-z_]+):\s*", policy.text))
        if not anchors or len({a.group(1) for a in anchors}) != len(anchors):
            raise ValueError("policy needs distinct section anchors")
        end = ordered[i + 1].effective_at if i + 1 < len(ordered) else None
        for j, anchor in enumerate(anchors):
            stop = anchors[j + 1].start() if j + 1 < len(anchors) else len(policy.text)
            text = policy.text[anchor.end() : stop].strip()
            if not text:
                raise ValueError("empty policy clause")
            section = anchor.group(1)
            clauses.append(
                Clause(
                    citation=f"{policy.policy_id}@{policy.version}#{section}",
                    section=section,
                    version=policy.version,
                    effective_from=_utc_time(policy.effective_at),
                    effective_to=end,
                    text=text,
                    text_hash=hashlib.sha256(text.encode()).hexdigest(),
                    observed_at=observed_at,
                )
            )
    return clauses
