"""Chapter 34's count mathematics and the signed, forecast-based allowance.

Ratios use exact decimal fractions. A recent burn rate becomes a lifetime only
after specifying actual eligible-event intensity and remaining allowance.
"""

import fcntl
import json
import math
from datetime import datetime, timedelta
from fractions import Fraction
from functools import lru_cache
from pathlib import Path

from pydantic import model_validator

from refund_agent import contracts as c
from refund_agent.release.candidate import PACKAGE
from refund_agent.selection import fingerprint


def fraction(value) -> Fraction:
    if isinstance(value, bool):
        raise ValueError("boolean is not a number")
    try:
        return Fraction(str(value))
    except (ValueError, ZeroDivisionError) as error:
        raise ValueError("finite rational number required") from error


def probability(value) -> Fraction:
    value = fraction(value)
    if not 0 <= value <= 1:
        raise ValueError("probability outside [0, 1]")
    return value


def budget(rho_slo, n_elig) -> Fraction:
    n = fraction(n_elig)
    if n < 0:
        raise ValueError("negative eligible count")
    return probability(rho_slo) * n


def burn_rate(n_bad, n_elig, rho_slo) -> Fraction:
    bad, n, rho = fraction(n_bad), fraction(n_elig), probability(rho_slo)
    if n <= 0 or rho == 0 or not 0 <= bad <= n:
        raise ValueError("positive sample and allowance required")
    return bad / n / rho


def alert_threshold(budget_share, t_slo, t_w) -> Fraction:
    total, window = fraction(t_slo), fraction(t_w)
    if total <= 0 or window <= 0 or window > total:
        raise ValueError("invalid alert window")
    return probability(budget_share) * total / window


def remaining_lifetime(remaining, eligible_per_hour, bad_share) -> Fraction:
    remaining, intensity = fraction(remaining), fraction(eligible_per_hour)
    share = probability(bad_share)
    if remaining < 0 or intensity <= 0 or share <= 0:
        raise ValueError(
            "remaining allowance and positive bad intensity required"
        )
    return remaining / (intensity * share)


@lru_cache(maxsize=256)
def _tails(n: int, p: Fraction) -> tuple[Fraction, ...]:
    if type(n) is not int or n < 0:
        raise ValueError("nonnegative integer count required")
    # Adjacent masses avoid factorial recomputation; O(n) exact operations.
    if p == 1:
        return (Fraction(1),) * (n + 1) + (Fraction(0),)
    mass = (1 - p) ** n
    cumulative, tails = Fraction(0), []
    for j in range(n + 1):
        tails.append(1 - cumulative)
        cumulative += mass
        if j < n:
            mass *= Fraction(n - j, j + 1) * p / (1 - p)
    return tuple(tails + [Fraction(0)])


def binom_tail(k: int, n: int, p) -> Fraction:
    if type(k) is not int:
        raise ValueError("integer threshold required")
    tails = _tails(n, probability(p))
    return Fraction(1) if k <= 0 else Fraction(0) if k > n else tails[k]


def min_count(n: int, p, p_tol) -> int:
    tolerance = probability(p_tol)
    if tolerance == 0:
        raise ValueError("positive false-page tolerance required")
    return next(
        k
        for k, tail in enumerate(_tails(n, probability(p)))
        if tail <= tolerance
    )


def within_budget(n: int, allowance, p) -> Fraction:
    allowed = fraction(allowance)
    if allowed < 0:
        raise ValueError("negative allowance")
    return 1 - binom_tail(math.floor(allowed) + 1, n, p)


def every_event_false_page(n: int, p, p_tol) -> Fraction:
    """Exact probability of ever crossing a recomputed prefix threshold.

    This is the deliberately wrong repeated-testing comparator, not a monitor.
    Dynamic programming keeps only paths that have not paged: O(n^2) work.
    """
    p = probability(p)
    surviving: dict[int, Fraction] = {0: Fraction(1)}
    for seen in range(1, n + 1):
        next_paths: dict[int, Fraction] = {}
        threshold = min_count(seen, p, p_tol)
        for bad, mass in surviving.items():
            for extra, chance in ((0, 1 - p), (1, p)):
                if bad + extra < threshold:
                    next_paths[bad + extra] = (
                        next_paths.get(bad + extra, Fraction(0))
                        + mass * chance
                    )
        surviving = next_paths
    return Fraction(1) - sum(surviving.values(), Fraction(0))


class Objective(c.Record):
    name: str
    definition: str
    target: str
    window_hours: int
    population: str
    allowance: str
    forecast_eligible: str
    read_on: str
    alert_window_hours: int
    tolerance: str
    basis: str

    @model_validator(mode="after")
    def valid_objective(self):
        if self.name not in ("corrections", "timeliness"):
            raise ValueError("unknown objective")
        if not all((self.definition, self.population, self.basis)):
            raise ValueError(
                "definition, population and threshold basis required"
            )
        if (
            not 0 < probability(self.target) < 1
            or not 0 < probability(self.tolerance) < 1
        ):
            raise ValueError("nonzero target and tolerance required")
        if not 0 < self.alert_window_hours <= self.window_hours:
            raise ValueError("invalid objective windows")
        if fraction(self.allowance) != budget(
            1 - fraction(self.target), self.forecast_eligible
        ):
            raise ValueError("allowance differs from signed forecast")
        expected = (
            ("settled", 24) if self.name == "corrections" else ("fast", 6)
        )
        if (self.read_on, self.alert_window_hours) != expected:
            raise ValueError("wrong objective view or count window")
        return self


class ErrorBudgetRecord(c.Record):
    version: str = "observability-reliability-budget-1"
    canary_id: str
    candidate_fingerprint: str
    start: datetime
    end: datetime
    objectives: tuple[Objective, ...]
    owner_role: str = "release_owner"
    breach_instruction: str
    signed_by: str
    signed_at: datetime

    @model_validator(mode="after")
    def signed_window(self):
        owners = json.loads((PACKAGE / "release/owners.json").read_text())
        signers = {
            o["id"] for o in owners["owners"] if "decision_owner" in o["roles"]
        }
        if self.signed_by not in signers or self.signed_at > self.start:
            raise ValueError(
                "budget requires decision-owner signature before start"
            )
        if (
            not self.canary_id
            or not self.candidate_fingerprint
            or not self.breach_instruction
        ):
            raise ValueError(
                "budget requires plan, candidate and standing instruction"
            )
        if self.end <= self.start or {o.name for o in self.objectives} != {
            "corrections",
            "timeliness",
        }:
            raise ValueError(
                "budget requires both objectives and a positive window"
            )
        if len(self.objectives) != 2 or self.owner_role != "release_owner":
            raise ValueError("duplicate objective or unknown release role")
        hours = (self.end - self.start).total_seconds() / 3600
        if any(o.window_hours != hours for o in self.objectives):
            raise ValueError("objective differs from named plan window")
        return self

    @property
    def fingerprint(self) -> str:
        return fingerprint(self.model_dump(mode="json"))

    def canary_start_record(self) -> dict:
        """Admission fields for chapter 32; spend retains exact rationals."""
        record = c._validated(ErrorBudgetRecord, self)
        value = record.model_dump(mode="json")
        value["definition_fingerprint"] = record.fingerprint
        value["definition"] = record.model_dump(mode="json")
        for objective in value["objectives"]:
            objective["window"] = {
                "start": record.start.isoformat(),
                "end": record.end.isoformat(),
            }
            objective["allowance"] = float(fraction(objective["allowance"]))
        return value

    def write(self, path: Path) -> None:
        record = c._validated(ErrorBudgetRecord, self)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as stream:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            existing = path.read_text(encoding="utf-8")
            if existing:
                if ErrorBudgetRecord.model_validate_json(existing) != record:
                    raise ValueError(
                        "signed budget definition cannot be replaced"
                    )
                return
            stream.write(record.model_dump_json() + "\n")
            stream.flush()


def signed_budget(
    canary_id: str,
    candidate_fingerprint: str,
    start: datetime,
    *,
    owner="morgan",
) -> ErrorBudgetRecord:
    specs = (
        (
            "corrections",
            "payment not later corrected by settled evidence",
            "0.99",
            "1120",
            "settled",
            24,
        ),
        (
            "timeliness",
            "request answered within its promise with a policy-valid answer",
            "0.98",
            "5600/3",
            "fast",
            6,
        ),
    )
    objectives = tuple(
        Objective(
            name=name,
            definition=definition,
            target=target,
            forecast_eligible=n,
            allowance=str(budget(1 - fraction(target), n)),
            window_hours=672,
            population="all tier 1 decisions; executed cohort and recorded shadow",
            read_on=view,
            alert_window_hours=window,
            tolerance="0.01",
            basis="closed disjoint windows from start; exact binomial tail <= 1%",
        )
        for name, definition, target, n, view, window in specs
    )
    return ErrorBudgetRecord(
        canary_id=canary_id,
        candidate_fingerprint=candidate_fingerprint,
        start=start,
        end=start + timedelta(days=28),
        objectives=objectives,
        breach_instruction="end at the first forbidden or stale payment; no suppression",
        signed_by=owner,
        signed_at=start - timedelta(seconds=1),
    )
