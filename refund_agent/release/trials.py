"""Outcome-blind subset estimators and explicitly constructed trial curves."""

# Keep the printed excerpt within 79 columns.
# fmt: off
# region book:fnd.agent-evaluation.trials-01 output
from math import comb

import numpy as np


def _counts(n, n_pass, k):
    if (
        any(type(v) is not int for v in (n, n_pass, k))
        or not 0 <= n_pass <= n
        or not 1 <= k <= n
    ):
        raise ValueError(
            "integer counts with 0 <= passing <= trials "
            "and 1 <= k <= trials required"
        )


def pass_at_k(n, n_pass, k):
    _counts(n, n_pass, k)
    return 1 - (comb(n - n_pass, k) if n - n_pass >= k else 0) / comb(n, k)


def pass_hat_k(n, n_pass, k):
    _counts(n, n_pass, k)
    return (comb(n_pass, k) if n_pass >= k else 0) / comb(n, k)


def plug_at_k(n, n_pass, k):
    _counts(n, n_pass, k)
    return 1 - (1 - n_pass / n) ** k


def plug_hat_k(n, n_pass, k):
    _counts(n, n_pass, k)
    return (n_pass / n) ** k


if __name__ == "__main__":
    from demos.agent_evaluation_release_demo import print_trials

    print_trials()
# endregion book:fnd.agent-evaluation.trials-01
# fmt: on


def constructed_curves(seed=7, n=20):
    if type(n) is not int or n < 10:
        raise ValueError("at least ten trials required for the plotted k range")
    rng = np.random.default_rng(seed)
    suites = {"A": [1.0] * 38 + [0.0] * 2, "B": [0.95] * 40}
    records = {}
    for name, probabilities in suites.items():
        passes = [int((rng.random(n) < p).sum()) for p in probabilities]
        records[name] = {
            "probabilities": probabilities,
            "passing_trials": passes,
            "rows": [
                dict(
                    k=k,
                    exact_at=float(np.mean([1 - (1 - p) ** k for p in probabilities])),
                    exact_hat=float(np.mean([p**k for p in probabilities])),
                    estimate_at=float(np.mean([pass_at_k(n, c, k) for c in passes])),
                    estimate_hat=float(np.mean([pass_hat_k(n, c, k) for c in passes])),
                    pooled_at=1 - (1 - float(np.mean(probabilities))) ** k,
                    pooled_hat=float(np.mean(probabilities)) ** k,
                )
                for k in range(1, 11)
            ],
        }
    return {"seed": seed, "trials_per_scenario": n, "constructed": True, "suites": records}
