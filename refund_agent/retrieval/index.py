"""Filter by effective interval and read age before exact cosine ranking."""

from copy import deepcopy
from dataclasses import dataclass, replace
from datetime import datetime

from refund_agent.contracts import _utc_time
from refund_agent.retrieval.clauses import Clause, fingerprint, policy_clauses
from refund_agent.retrieval.encoders import cosine
from refund_agent.rules import _FRESHNESS_LIMITS


@dataclass(frozen=True)
class Hit:
    clause: Clause
    score: float


@dataclass(frozen=True)
class Retrieval:
    hits: tuple[Hit, ...] = ()
    refusal: str | None = None


def rank(clauses, vector, k, newest=False) -> tuple[Hit, ...]:
    hits = [Hit(clause, cosine(vector, clause.vector)) for clause in clauses]
    if not hits or all(abs(hit.score) < 1e-12 for hit in hits):
        return ()
    # Round only the sort key so numerical noise does not order mathematical ties.
    hits.sort(
        key=lambda h: (
            -round(h.score, 12),
            -h.clause.effective_from.timestamp() if newest else 0,
            h.clause.citation,
        )
    )
    return tuple(hits[:k])


@dataclass
class PolicyIndex:
    clauses: tuple[Clause, ...]
    encoder: object
    manifest: dict

    @classmethod
    def build(cls, policies, observed_at, encoder):
        encoder = deepcopy(encoder)  # a later fit must not mix coordinate systems
        clauses = policy_clauses(policies, observed_at)
        state = encoder.manifest()
        encoder_id = fingerprint(state)
        vectors = encoder.transform([c.text for c in clauses])
        records = tuple(
            replace(c, encoder=encoder_id, vector=tuple(v)) for c, v in zip(clauses, vectors)
        )
        content = {
            "encoder": state,
            "records": [c.record() for c in records],
            "freshness_seconds": _FRESHNESS_LIMITS["policy"],
        }
        manifest = {
            "clauses": len(records),
            "policy_versions": sorted({c.version for c in records}),
            "encoder": encoder_id,
            "observed_at": _utc_time(observed_at).isoformat(),
            "sha256": fingerprint(content),
        }
        return cls(records, encoder, manifest)

    def retrieve(self, question: str, as_of: datetime, k: int) -> Retrieval:
        as_of = _utc_time(as_of)
        if type(k) is not int or k < 1:
            raise ValueError("k must be a positive integer")
        active = [c for c in self.clauses if c.in_force(as_of)]
        if not active:
            return Retrieval(refusal="no_clause_in_force")
        ages = [(as_of - c.observed_at).total_seconds() for c in active]
        if any(age < 0 for age in ages):
            return Retrieval(refusal="future_policy")
        if any(age > _FRESHNESS_LIMITS["policy"] for age in ages):
            return Retrieval(refusal="stale_policy")
        if not self.encoder.has_terms(question):
            return Retrieval(refusal="no_matching_terms")
        vector = self.encoder.transform([question])[0]
        hits = rank(active, vector, k)
        return Retrieval(hits, None if hits else "no_matching_terms")
