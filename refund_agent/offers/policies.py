"""Per-context estimates and the probability each offer has before it is drawn."""

import hashlib
import json
import math
from dataclasses import dataclass

import numpy as np

from demos.exploration_bandits_demo import (
    capped_probabilities,
    epsilon_greedy_probabilities,
)
from demos.exploration_bandits_demo import (
    choose as choose,
)
from demos.exploration_bandits_demo import (
    incremental_mean as incremental_mean,
)
from refund_agent.offers.notices import BASELINE, OFFERS


def fingerprint(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def hoeffding_radius(n, alpha, counts=1):
    if n < 1 or counts < 1 or not 0 < alpha < 1:
        raise ValueError("positive counts and alpha in (0, 1) required")
    return math.sqrt(math.log(counts / alpha) / (2 * n))


class EstimateTable:
    def __init__(self, answers=(0, 0, 0), no_contacts=(0, 0, 0)):
        self.answers = np.array(answers, dtype=int)
        self.no_contacts = np.array(no_contacts, dtype=int)
        if (
            self.answers.shape != (3,)
            or self.no_contacts.shape != (3,)
            or np.any(self.answers < self.no_contacts)
            or np.any(self.no_contacts < 0)
        ):
            raise ValueError("three valid counts required")

    @property
    def means(self):
        return np.divide(self.no_contacts, self.answers, out=np.zeros(3), where=self.answers > 0)

    @property
    def fingerprint(self):
        return fingerprint(self.record())

    def record(self):
        return {"answers": self.answers.tolist(), "no_contacts": self.no_contacts.tolist()}

    def observe(self, answer, as_of):
        if answer.matures_at > as_of:
            return False
        index = OFFERS.index(answer.offer)
        self.answers[index] += 1
        self.no_contacts[index] += answer.reward
        return True


def _indices(offers):
    if not offers or len(set(offers)) != len(offers) or any(a not in OFFERS for a in offers):
        raise ValueError("a nonempty mask of distinct known offers is required")
    return [i for i, offer in enumerate(OFFERS) if offer in offers]


def _mask(probabilities, offers):
    indices = _indices(offers)
    result = np.zeros(3)
    result[indices] = probabilities
    return result


@dataclass(frozen=True)
class EpsilonGreedy:
    epsilon: float
    version: str = "epsilon-greedy-1"

    def __post_init__(self):
        if not 0 <= self.epsilon <= 1:
            raise ValueError("epsilon in [0, 1] required")

    @property
    def fingerprint(self):
        return fingerprint({"version": self.version, "epsilon": self.epsilon})

    def probabilities(self, table, t, offers=OFFERS):
        return _mask(
            epsilon_greedy_probabilities(table.means[_indices(offers)], self.epsilon),
            offers,
        )


@dataclass(frozen=True)
class UCB1:
    version: str = "ucb1-1"

    @property
    def fingerprint(self):
        return fingerprint({"version": self.version})

    def probabilities(self, table, t, offers=OFFERS):
        indices = _indices(offers)
        count = table.answers[indices]
        if np.any(count == 0):
            winner = int(np.flatnonzero(count == 0)[0])
        else:
            winner = int(np.argmax(table.means[indices] + np.sqrt(2 * math.log(max(t, 1)) / count)))
        result = np.zeros(len(offers))
        result[winner] = 1
        return _mask(result, offers)


def _beta_pdf(alpha, beta, grid):
    pdf = []
    for a, b in zip(alpha, beta):
        log_density = np.full(len(grid), math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b))
        if a > 1:
            with np.errstate(divide="ignore"):
                log_density += (a - 1) * np.log(grid)
        if b > 1:
            with np.errstate(divide="ignore"):
                log_density += (b - 1) * np.log1p(-grid)
        pdf.append(np.exp(log_density))
    return np.array(pdf)


def best_probabilities(no_contacts, answers):
    """Integrate f_a(x) times the other Beta CDFs on a shared refined grid.

    Log densities avoid overflow; local refinement resolves narrow posteriors.
    """
    k, n = np.asarray(no_contacts), np.asarray(answers)
    if k.ndim != 1 or k.shape != n.shape or not len(k) or np.any(k < 0) or np.any(n < k):
        raise ValueError("valid success and answer counts required")
    alpha, beta = k + 1, n - k + 1
    mean = alpha / (alpha + beta)
    sd = np.sqrt(alpha * beta / ((alpha + beta) ** 2 * (alpha + beta + 1)))
    grid = np.unique(
        np.clip(
            np.concatenate(
                [np.linspace(0, 1, 1001)]
                + [m + np.linspace(-10, 10, 401) * s for m, s in zip(mean, sd)]
            ),
            0,
            1,
        )
    )
    pdf = _beta_pdf(alpha, beta, grid)
    masses = (pdf[:, :-1] + pdf[:, 1:]) * np.diff(grid) / 2
    pdf /= masses.sum(axis=1)[:, None]
    cdf = np.column_stack((np.zeros(len(k)), np.cumsum(masses, axis=1)))
    cdf /= cdf[:, -1:]
    probabilities = []
    for index in range(len(k)):
        integrand = pdf[index] * np.prod(np.delete(cdf, index, axis=0), axis=0)
        probabilities.append(np.sum((integrand[:-1] + integrand[1:]) * np.diff(grid) / 2))
    probabilities = np.array(probabilities)
    return probabilities / probabilities.sum()


@dataclass(frozen=True)
class Thompson:
    version: str = "beta-thompson-1"

    @property
    def fingerprint(self):
        return fingerprint({"version": self.version, "prior": [1, 1]})

    def probabilities(self, table, t, offers=OFFERS):
        indices = _indices(offers)
        return _mask(best_probabilities(table.no_contacts[indices], table.answers[indices]), offers)


@dataclass(frozen=True)
class Capped:
    inner: EpsilonGreedy | UCB1 | Thompson
    cap: float
    baseline: str = BASELINE
    version: str = "capped-1"

    def __post_init__(self):
        if not 0 <= self.cap <= 1 or self.baseline not in OFFERS:
            raise ValueError("valid cap and baseline required")

    @property
    def fingerprint(self):
        return fingerprint(
            {
                "version": self.version,
                "inner": self.inner.fingerprint,
                "cap": self.cap,
                "baseline": self.baseline,
            }
        )

    def probabilities(self, table, t, offers=OFFERS):
        if self.baseline not in offers:
            raise ValueError("the cap requires an eligible baseline")
        return capped_probabilities(
            self.inner.probabilities(table, t, offers), self.cap, OFFERS.index(self.baseline)
        )
