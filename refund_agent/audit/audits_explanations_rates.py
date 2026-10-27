"""Exact finite-population audits of a signed ceiling on owed customers denied.

Blind readings and mature payment labels are distinct inputs. A constructed truth
can stand in for a reading in a demonstration, never in a deployed audit.
"""

from datetime import datetime
from fractions import Fraction
from hashlib import sha256
from math import comb, sqrt
from typing import Literal

from pydantic import Field, model_validator

from refund_agent.contracts import Record
from refund_agent.selection import fingerprint


def _counts(N: int, K: int, m: int) -> None:
    if any(type(v) is not int for v in (N, K, m)) or not (
        0 <= K <= N and 0 <= m <= N
    ):
        raise ValueError("invalid hypergeometric counts")


def hypergeom_tail(k: int, N: int, K: int, m: int) -> Fraction:
    _counts(N, K, m)
    if type(k) is not int or k < 0:
        raise ValueError("invalid sample count")
    return sum(
        (
            Fraction(comb(K, j) * comb(N - K, m - j), comb(N, m))
            for j in range(max(k, m - (N - K), 0), min(m, K) + 1)
        ),
        Fraction(0),
    )


def _ceiling(value: Fraction | float) -> Fraction:
    value = value if isinstance(value, Fraction) else Fraction(str(value))
    if not 0 < value < 1:
        raise ValueError("ceiling must be strictly between zero and one")
    return value


# region book:fnd.audits-explanations.audit-core-01
def wrongful_denial_rate(denied_share, wrongful_share, found_false_share):
    if not 0 <= denied_share <= 1:
        raise ValueError("denied share must be between zero and one")
    if not (0 <= wrongful_share <= 1 and 0 <= found_false_share <= 1):
        raise ValueError("label shares must be between zero and one")
    wrong = denied_share * wrongful_share  # (1)
    owed = wrong + (1 - denied_share) * (1 - found_false_share)  # (2)
    return wrong / owed if owed else None  # (3)


def ceiling_test(k, N, owed, m, ceiling):
    _counts(N, 0, m)
    if type(owed) is not int or owed < 0 or not 0 <= k <= m:
        raise ValueError("invalid owed or sample count")
    ceiling = _ceiling(ceiling)
    K = min(N, int(ceiling * owed / (1 - ceiling)))  # (4)
    return K, hypergeom_tail(k, N, K, m)  # (5)


def breaches(chance, alpha_family, segments):
    if not 0 < alpha_family < 1 or segments < 1:
        raise ValueError("invalid family level or segment count")
    return chance < Fraction(str(alpha_family)) / segments  # (6)


# endregion book:fnd.audits-explanations.audit-core-01


def wilson_interval(k: int, n: int) -> tuple[float, float]:
    if type(k) is not int or type(n) is not int or not 0 <= k <= n or n == 0:
        raise ValueError("Wilson interval needs a nonempty valid sample")
    z = 1.96  # Same rounded normal cutoff as the chapter-4 interval.
    p, zz = k / n, z * z
    center = (p + zz / (2 * n)) / (1 + zz / n)
    half = z * sqrt(p * (1 - p) / n + zz / (4 * n * n)) / (1 + zz / n)
    return (0.0 if k == 0 else center - half, 1.0 if k == n else center + half)


def fpr_from_identity(base_rate: float, ppv: float, tpr: float) -> float:
    if not (0 < base_rate < 1 and 0 < ppv < 1 and 0 < tpr <= 1):
        raise ValueError(
            "identity requires interior base rate/PPV and positive TPR"
        )
    fpr = base_rate / (1 - base_rate) * (1 - ppv) / ppv * tpr
    if fpr > 1:
        raise ValueError("these rates imply an impossible false-positive rate")
    return fpr


def exact_upper_count(
    k: int, N: int, m: int, alpha: Fraction = Fraction(1, 20)
) -> int:
    """Invert Pr_K(X <= k) >= alpha to give the exact one-sided upper count."""
    _counts(N, 0, m)
    if type(k) is not int or not 0 <= k <= m or not 0 < alpha < 1:
        raise ValueError("invalid confidence-bound inputs")
    low, high = k, N - m + k
    while low < high:
        mid = (low + high + 1) // 2
        if 1 - hypergeom_tail(k + 1, N, mid, m) >= alpha:
            low = mid
        else:
            high = mid - 1
    return low


def minimum_release_sample(
    N: int, owed: int, ceiling: Fraction | float
) -> int | None:
    """Minimum sample with zero wrongful readings to prove the harm below ceiling."""
    _counts(N, 0, 0)
    if type(owed) is not int or owed < 0:
        raise ValueError("invalid owed payments")
    c = _ceiling(ceiling)
    if owed == 0:
        return None
    boundary = c * owed / (1 - c)
    unsafe = (
        boundary.numerator + boundary.denominator - 1
    ) // boundary.denominator
    if unsafe > N:
        return 0
    low, high = 0, N
    while low < high:
        m = (low + high) // 2
        zero_chance = (
            Fraction(comb(N - unsafe, m), comb(N, m))
            if m <= N - unsafe
            else Fraction(0)
        )
        if zero_chance < Fraction(1, 20):
            high = m
        else:
            low = m + 1
    return low


class Period(Record):
    start: datetime
    end: datetime

    @model_validator(mode="after")
    def ordered(self):
        if self.end <= self.start:
            raise ValueError("audit period must be nonempty")
        return self


class AuditRule(Record):
    version: str = "audits-explanations-rule-1"
    segments: tuple[str, ...] = ("A", "B", "C")
    read_per_segment: int = Field(default=80, strict=True, gt=0)
    ceiling: float = Field(default=0.01, gt=0, lt=1, allow_inf_nan=False)
    alpha_family: float = Field(default=0.05, gt=0, lt=1, allow_inf_nan=False)
    test: Literal["hypergeometric_ceiling"] = "hypergeometric_ceiling"
    correction: Literal["bonferroni"] = "bonferroni"
    owner: str = "merchant's owner of the refund decision"
    signed_by: str = "morgan"
    signed_at: datetime
    source_owner: str = "carrier integration team"
    response: str = "Hold breaching reasons until the exact safety bound is below the ceiling."

    @model_validator(mode="after")
    def named_segments(self):
        if not self.segments or len(set(self.segments)) != len(self.segments):
            raise ValueError("segments must be distinct and named in advance")
        return self


class AuditObservation(Record):
    request_id: str
    order_id: str
    segment: str
    at: datetime
    outcome: Literal["approve", "deny", "refer"]
    reason: str
    held_proposal: bool = False
    owed: bool | None
    found_false_if_paid: bool | None
    label_mature_at: datetime

    @model_validator(mode="after")
    def proposal_is_denial(self):
        if self.held_proposal and (
            self.outcome != "refer"
            or self.reason
            not in {
                "amount_over_policy",
                "already_delivered",
                "outside_return_window",
            }
        ):
            raise ValueError("held proposal must retain a denial reason")
        return self


class SegmentAudit(Record):
    segment: str
    requests: int
    decided: int
    paid: int
    referred: int
    denials: int
    held_proposals: int
    denial_share: float | None
    paid_share: float | None
    referred_share: float | None
    read: int
    sample_ids: tuple[str, ...]
    wrongful: int
    wrongful_share: float | None
    wilson: tuple[float, float] | None
    found_false_share: float | None
    owed_payments: int
    estimate: float | None
    estimate_range: tuple[float, float] | None
    ceiling_count: int
    chance: float | None
    breach: bool
    held_reasons: tuple[str, ...]
    status: Literal["assessed", "no_denials", "pending_labels", "unassessable"]
    upper_wrongful_count: int
    upper_wrongful_share: float | None
    upper_owed_denial_share: float | None
    minimum_clean_release_sample: int | None
    release_allowed: bool


class AuditRecord(Record):
    period: Period
    rule: AuditRule
    seed: int
    at: datetime
    segments: tuple[SegmentAudit, ...]

    @property
    def fingerprint(self) -> str:
        return fingerprint(self.model_dump(mode="json"))


def _share(n: float, d: float) -> float | None:
    return n / d if d else None


def sample_ids(ids: tuple[str, ...], seed: int, n: int) -> tuple[str, ...]:
    ranked = sorted(
        ids,
        key=lambda rid: sha256(
            f"audits-explanations:{seed}:{rid}".encode()
        ).digest(),
    )
    return tuple(sorted(ranked[:n]))


def _segment(rows, segment, rule, seed, at):
    denied = [r for r in rows if r.outcome == "deny" or r.held_proposal]
    paid = [r for r in rows if r.outcome == "approve"]
    N, P = len(denied), len(paid)
    m = min(N, rule.read_per_segment)
    ids = sample_ids(tuple(r.request_id for r in denied), seed, m)
    chosen = set(ids)  # Freeze selection before inspecting any blind reading.
    sample = [r for r in denied if r.request_id in chosen]
    pending = any(
        r.label_mature_at > at or r.found_false_if_paid is None for r in paid
    )
    unresolved = any(r.owed is None for r in sample)
    k = sum(r.owed is True for r in sample)
    false = sum(r.found_false_if_paid is True for r in paid)
    owed = sum(
        r.label_mature_at <= at and r.found_false_if_paid is False
        for r in paid
    )
    K, chance = ceiling_test(k, N, owed, m, rule.ceiling)
    estimate = _share(N * k / m if m else 0, owed + (N * k / m if m else 0))
    interval = wilson_interval(k, m) if m else None
    conditional_range = None
    if interval:
        low = _share(N * interval[0], owed + N * interval[0])
        high = _share(N * interval[1], owed + N * interval[1])
        if low is not None and high is not None:
            conditional_range = (low, high)
    status = (
        "pending_labels"
        if pending
        else "unassessable"
        if unresolved or not (owed + k)
        else ("assessed" if N else "no_denials")
    )
    assessable = status in {"assessed", "no_denials"}
    upper = exact_upper_count(k, N, m)
    upper_rate = _share(upper, owed + upper) if assessable else None
    breach = assessable and breaches(
        chance, rule.alpha_family, len(rule.segments)
    )
    reasons = (
        tuple(sorted({r.reason for r in sample if r.owed is True}))
        if breach
        else ()
    )
    return SegmentAudit(
        segment=segment,
        requests=len(rows),
        decided=N + P,
        denials=N,
        paid=P,
        referred=sum(r.outcome == "refer" for r in rows),
        held_proposals=sum(r.held_proposal for r in rows),
        denial_share=_share(N, N + P),
        paid_share=_share(P, N + P),
        referred_share=_share(
            sum(r.outcome == "refer" for r in rows), len(rows)
        ),
        read=m,
        sample_ids=ids,
        wrongful=k,
        wrongful_share=_share(k, m),
        wilson=interval,
        found_false_share=_share(false, P) if not pending else None,
        owed_payments=owed,
        estimate=estimate if assessable else None,
        estimate_range=conditional_range if assessable else None,
        ceiling_count=K,
        chance=float(chance) if assessable else None,
        breach=breach,
        held_reasons=reasons,
        status=status,
        upper_wrongful_count=upper,
        upper_wrongful_share=_share(upper, N),
        upper_owed_denial_share=upper_rate,
        minimum_clean_release_sample=minimum_release_sample(
            N, owed, rule.ceiling
        )
        if assessable
        else None,
        release_allowed=upper_rate is not None and upper_rate < rule.ceiling,
    )


def audit_period(
    observations: tuple[AuditObservation, ...],
    period: Period,
    rule: AuditRule,
    seed: int,
    *,
    at: datetime,
) -> AuditRecord:
    period, rule = (
        Period.model_validate(period),
        AuditRule.model_validate(rule),
    )
    if rule.signed_at >= period.start:
        raise ValueError("audit rule must be signed before the period")
    if at < period.end:
        raise ValueError("audit cannot precede the period's close")
    if len({r.request_id for r in observations}) != len(observations):
        raise ValueError("duplicate audit request identifiers")
    if any(r.segment not in rule.segments for r in observations):
        raise ValueError("unnamed audit segment")
    rows = [
        AuditObservation.model_validate(r)
        for r in observations
        if period.start <= r.at < period.end
    ]
    return AuditRecord(
        period=period,
        rule=rule,
        seed=seed,
        at=at,
        segments=tuple(
            _segment([r for r in rows if r.segment == g], g, rule, seed, at)
            for g in rule.segments
        ),
    )


def power(
    rule: AuditRule, N: int, wrongful: int, owed: int, m: int
) -> Fraction:
    _counts(N, wrongful, m)
    for k in range(m + 1):
        _, chance = ceiling_test(k, N, owed, m, rule.ceiling)
        if breaches(chance, rule.alpha_family, len(rule.segments)):
            return hypergeom_tail(k, N, wrongful, m)
    return Fraction(0)


class ConfusionCounts(Record):
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0
    referred: int = 0


def confusion_by_segment(month, decisions) -> dict[str, ConfusionCounts]:
    """Oracle view only. The month truth is never passed to tools or a rule."""
    truths = {t.request_id: t for t in month.truth}
    requests = {r.request_id: r for r in month.data.requests}
    if len({d.request_id for d in decisions}) != len(decisions) or set(
        truths
    ) != {d.request_id for d in decisions}:
        raise ValueError("one oracle comparison is required for every request")
    counts = {
        g: {k: 0 for k in ("tp", "fp", "fn", "tn", "referred")}
        for g in ("A", "B", "C")
    }
    for d in decisions:
        segment = month.segments.region_of(
            requests[d.request_id].order_reference
        )
        outcome = d.outcome.value
        key = (
            "referred"
            if outcome == "refer"
            else ("fp" if truths[d.request_id].owed else "tp")
            if outcome == "deny"
            else ("tn" if truths[d.request_id].owed else "fn")
        )
        counts[segment][key] += 1
    return {g: ConfusionCounts(**row) for g, row in counts.items()}


def criteria(counts: ConfusionCounts) -> dict[str, float | None]:
    tp, fp, fn, tn = counts.tp, counts.fp, counts.fn, counts.tn
    return {
        "denial_share": _share(tp + fp, tp + fp + fn + tn),
        "tpr": _share(tp, tp + fn),
        "fpr": _share(fp, fp + tn),
        "ppv": _share(tp, tp + fp),
        "base_rate": _share(tp + fn, tp + fp + fn + tn),
    }
