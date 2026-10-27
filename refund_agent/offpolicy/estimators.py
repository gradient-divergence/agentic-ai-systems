"""One-step estimates and paired, per-record uncertainty, using applied chances."""

from dataclasses import asdict, dataclass, field
from statistics import NormalDist

import numpy as np


@dataclass(frozen=True)
class Record:
    context: str
    action: str
    reward: float | None
    logged: dict[str, float]
    final: bool = True
    override: bool = False
    logger_version: str = "fixed-1"
    logger_fingerprint: str = ""
    logger_seed: int = 7
    decision_id: str = ""
    request_id: str = ""
    customer_id: str = ""
    decided_at: str = ""
    matures_at: str = ""
    observed_at: str = ""
    inputs: dict = field(default_factory=dict)
    eligible: tuple = ()

    def __post_init__(self):
        p = list(self.logged.values())
        if not p or any(not np.isfinite(x) or x < 0 for x in p) or not np.isclose(sum(p), 1):
            raise ValueError("full applied logging distribution required")
        if self.action not in self.logged or (self.logged[self.action] <= 0 and not self.override):
            raise ValueError("the logged action must have positive applied chance")
        if self.reward is not None and (not np.isfinite(self.reward) or not 0 <= self.reward <= 1):
            raise ValueError("this chapter's reward is in [0, 1]")

    def record(self):
        return asdict(self)


def ratios(target, logged):
    target, logged = np.asarray(target, dtype=float), np.asarray(logged, dtype=float)
    if (
        target.shape != logged.shape
        or np.any(~np.isfinite(target))
        or np.any(target < 0)
        or np.any(~np.isfinite(logged))
        or np.any(logged <= 0)
    ):
        raise ValueError("matching nonnegative target and positive logged chances required")
    return target / logged


def _arrays(weights, rewards):
    w, r = np.asarray(weights, dtype=float), np.asarray(rewards, dtype=float)
    if (
        w.ndim != 1
        or w.shape != r.shape
        or not len(w)
        or np.any(~np.isfinite(w))
        or np.any(w < 0)
        or np.any(~np.isfinite(r))
    ):
        raise ValueError("nonempty, finite paired ratios and rewards required")
    return w, r


def ips(weights, rewards):
    w, r = _arrays(weights, rewards)
    return float(np.mean(w * r))


def self_normalized_ips(weights, rewards):
    w, r = _arrays(weights, rewards)
    return float(w @ r / w.sum()) if w.sum() else None


def effective_sample_size(weights):
    w = np.asarray(weights, dtype=float)
    if w.ndim != 1 or np.any(~np.isfinite(w)) or np.any(w < 0):
        raise ValueError("finite nonnegative ratios required")
    return float(w.sum() ** 2 / (w @ w)) if np.any(w) else 0.0


def normal_interval(terms, level=0.95):
    values = np.asarray(terms, dtype=float)
    if values.ndim != 1 or not len(values) or np.any(~np.isfinite(values)) or not 0 < level < 1:
        raise ValueError("finite terms and a level in (0, 1) required")
    if len(values) < 2:
        return None
    radius = NormalDist().inv_cdf((1 + level) / 2) * values.std(ddof=1) / np.sqrt(len(values))
    return float(values.mean() - radius), float(values.mean() + radius)


def self_normalized_interval(weights, rewards, level=0.95):
    w, r = _arrays(weights, rewards)
    estimate = self_normalized_ips(w, r)
    if estimate is None or len(w) < 2:
        return None
    influence = w * (r - estimate) / w.mean()
    low, high = normal_interval(influence, level)
    return float(estimate + low), float(estimate + high)


def direct_method(policy, model, contexts):
    contexts = list(contexts)
    if not contexts:
        raise ValueError("at least one context required")
    return float(
        np.mean(
            [
                sum(p * model.predict(cell, a) for a, p in policy.probabilities(cell).items())
                for cell in contexts
            ]
        )
    )


def policy_ratios(policy, records):
    return ratios(
        [policy.probabilities(r.context).get(r.action, 0) for r in records],
        [r.logged[r.action] for r in records],
    )


def per_record_terms(policy, model, records, estimator="dr"):
    records = list(records)
    if not records or any(r.override or r.reward is None for r in records):
        raise ValueError("resolved, non-override records required")
    w = policy_ratios(policy, records)
    rewards = np.array([r.reward for r in records])
    prediction = np.array(
        [
            sum(p * model.predict(r.context, a) for a, p in policy.probabilities(r.context).items())
            for r in records
        ]
    )
    if estimator == "dm":
        return prediction
    if estimator == "ips":
        return w * rewards
    if estimator == "dr":
        return prediction + w * (
            rewards - np.array([model.predict(r.context, r.action) for r in records])
        )
    if estimator == "sn":
        return w * rewards / w.mean() if w.sum() else None
    raise ValueError("estimator must be ips, sn, dm or dr")


def doubly_robust(policy, model, records):
    return float(np.mean(per_record_terms(policy, model, records, "dr")))


def improvement_terms(candidate, baseline, model, records, estimator="dr"):
    return per_record_terms(candidate, model, records, estimator) - per_record_terms(
        baseline, model, records, estimator
    )
