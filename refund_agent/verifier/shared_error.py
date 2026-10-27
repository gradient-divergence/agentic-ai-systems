"""Binary joint errors; a shared cause is a stated model, not an empirical law."""

import itertools
import math
from types import MappingProxyType

import numpy as np

JOINT_A = MappingProxyType(
    {
        (0, 0, 0): 0.52,
        (1, 0, 0): 0.12,
        (0, 1, 0): 0.12,
        (0, 0, 1): 0.12,
        (1, 1, 0): 0.04,
        (1, 0, 1): 0.04,
        (0, 1, 1): 0.04,
        (1, 1, 1): 0.0,
    }
)
JOINT_B = MappingProxyType(
    {
        (0, 0, 0): 0.48,
        (1, 0, 0): 0.16,
        (0, 1, 0): 0.16,
        (0, 0, 1): 0.16,
        (1, 1, 0): 0.0,
        (1, 0, 1): 0.0,
        (0, 1, 1): 0.0,
        (1, 1, 1): 0.04,
    }
)


def _probability(value):
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("finite probability required")
    return value


def pass_through(rho_p, rho_v, cov):
    _probability(rho_p), _probability(rho_v)
    joint = rho_p * rho_v + cov
    lower, upper = max(0, rho_p + rho_v - 1), min(rho_p, rho_v)
    if not math.isfinite(joint) or not lower - 1e-12 <= joint <= upper + 1e-12:
        raise ValueError("impossible joint probability")
    return min(upper, max(lower, joint))


def catches(rho_p, rho_v, cov):
    return rho_p - pass_through(rho_p, rho_v, cov)


def majority_error(n_a, rho):
    """An odd vote with mutually independent binary errors at the same rate."""
    _probability(rho)
    if type(n_a) is not int or n_a < 1 or n_a % 2 == 0:
        raise ValueError("positive odd agent count required")
    return sum(
        math.comb(n_a, k) * rho**k * (1 - rho) ** (n_a - k) for k in range((n_a + 1) // 2, n_a + 1)
    )


def shared_cause_majority(rho_shared, rho_own, n_a=3):
    """A shared cause makes every agent err; otherwise errors are mutually independent at rho_own."""
    _probability(rho_shared)
    return rho_shared + (1 - rho_shared) * majority_error(n_a, rho_own)


def joint_law_summary(law):
    if not law:
        raise ValueError("nonempty joint law required")
    n_a = len(next(iter(law)))
    if n_a < 1 or n_a % 2 == 0:
        raise ValueError("an odd vote requires no unstated tie rule")
    for bits, weight in law.items():
        if len(bits) != n_a or any(bit not in (0, 1) for bit in bits):
            raise ValueError("equal-length binary error patterns required")
        _probability(weight)
    if not math.isclose(sum(law.values()), 1, abs_tol=1e-12):
        raise ValueError("joint probabilities must sum to one")
    rates = [sum(weight * bits[i] for bits, weight in law.items()) for i in range(n_a)]
    covariances = [
        sum(weight * bits[i] * bits[j] for bits, weight in law.items()) - rates[i] * rates[j]
        for i, j in itertools.combinations(range(n_a), 2)
    ]
    return {
        "rates": rates,
        "covariances": covariances,
        "majority": sum(weight for bits, weight in law.items() if sum(bits) > n_a // 2),
    }


def error_table(proposer, verifier):
    proposer, verifier = list(proposer), list(verifier)
    if (
        not proposer
        or len(proposer) != len(verifier)
        or any(x not in (0, 1) for x in proposer + verifier)
    ):
        raise ValueError("matched nonempty binary errors required")
    n = len(proposer)
    counts = {
        f"{i}{j}": sum(a == i and b == j for a, b in zip(proposer, verifier))
        for i, j in itertools.product((0, 1), repeat=2)
    }
    rho_p, rho_v = sum(proposer) / n, sum(verifier) / n
    covariance = counts["11"] / n - rho_p * rho_v
    denominator = math.sqrt(rho_p * (1 - rho_p) * rho_v * (1 - rho_v))
    return {
        "n": n,
        "counts": counts,
        "rho_p": rho_p,
        "rho_v": rho_v,
        "pass_through": counts["11"] / n,
        "catches": counts["10"] / n,
        "covariance": covariance,
        "correlation": covariance / denominator if denominator else None,
    }


def simulation(seed=7, n=200_000):
    if type(n) is not int or n < 1:
        raise ValueError("positive simulation size required")
    points = figure_data(seed, n)["simulation"]
    rows = []
    for name, observed, exact in (
        ("independent pair", points[0]["pair"], 0.04),
        ("independent majority", points[0]["majority"], 0.104),
        ("shared-cause majority", points[2]["majority"], 53 / 405),
        ("identical errors", points[4]["majority"], 0.2),
    ):
        rows.append(
            {
                "name": name,
                "n": n,
                "seed": seed,
                "exact": exact,
                "observed": observed,
                "standard_error": math.sqrt(exact * (1 - exact) / n),
            }
        )
    return rows


def figure_data(seed=7, n=200_000):
    rng = np.random.default_rng(seed)
    exact, points = [], []
    for share in np.linspace(0, 0.2, 101):
        share = float(share)
        own = (0.2 - share) / (1 - share)
        exact.append(
            {
                "shared": share,
                "own": own,
                "single": 0.2,
                "pair": share + (1 - share) * own**2,
                "majority": shared_cause_majority(share, own),
            }
        )
    for share in (0, 0.05, 0.1, 0.15, 0.2):
        own = (0.2 - share) / (1 - share)
        common = rng.random(n) < share
        errors = rng.random((n, 3)) < own
        pair = common | errors[:, :2].all(axis=1)
        majority = common | (errors.sum(axis=1) >= 2)
        points.append(
            {"shared": share, "pair": float(pair.mean()), "majority": float(majority.mean())}
        )
    return {"seed": seed, "n": n, "failure_rate": 0.2, "exact": exact, "simulation": points}
