"""A pre-read rule, with the sharp-null randomization check beside the interval."""

import itertools
import math
from collections import Counter
from dataclasses import asdict, dataclass

import numpy as np

from refund_agent.offers.policies import fingerprint
from refund_agent.offpolicy.estimators import improvement_terms
from refund_agent.offpolicy.reward_model import CellModel
from refund_agent.offpolicy.support import support_table, supported


@dataclass(frozen=True)
class Rule:
    estimand: str = "candidate minus fixed offer rule; no repeat contact per eligible request"
    estimator: str = "dr"
    level: float = 0.95
    n_eff_floor: int = 400
    exact_level: float = 0.025
    support_required: bool = True
    outcomes_final: bool = True
    one_logging_policy: bool = True
    seed: int = 7
    draws: int = 100000

    def __post_init__(self):
        if (
            self.estimator not in ("ips", "dr")
            or not 0 < self.level < 1
            or self.n_eff_floor < 1
            or not 0 < self.exact_level < 1
            or self.draws < 1
        ):
            raise ValueError("a valid predeclared evaluation rule is required")

    @property
    def fingerprint(self):
        return fingerprint(asdict(self))


def _multinomial_states(n, p, scores):
    def counts(total, k):
        if k == 1:
            yield (total,)
        else:
            for first in range(total + 1):
                for rest in counts(total - first, k - 1):
                    yield (first, *rest)

    for c in counts(n, len(p)):
        mass = math.factorial(n)
        for count, probability in zip(c, p):
            mass = mass / math.factorial(count) * probability**count
        yield sum(count * score for count, score in zip(c, scores)), mass


def _row_scores(row, candidate, baseline, model, estimator):
    target, fixed = candidate.probabilities(row.context), baseline.probabilities(row.context)
    actions = tuple(a for a, p in row.logged.items() if p > 0)
    p = tuple(row.logged[a] for a in actions)
    predicted = (
        sum(
            (target.get(a, 0) - fixed.get(a, 0)) * model.predict(row.context, a)
            for a in set(target) | set(fixed)
        )
        if estimator == "dr"
        else 0
    )
    scores = tuple(
        (target.get(a, 0) - fixed.get(a, 0))
        / row.logged[a]
        * (row.reward - (model.predict(row.context, a) if estimator == "dr" else 0))
        for a in actions
    )
    return p, scores, predicted


def _redraw_groups(records, candidate, baseline, model, estimator):
    groups, constant = Counter(), 0.0
    for row in records:
        p, scores, predicted = _row_scores(row, candidate, baseline, model, estimator)
        constant += predicted
        if max(scores) - min(scores) < 1e-15:
            constant += scores[0]
        else:
            groups[(p, scores)] += 1
    return groups, constant


def _enumerate(groups, threshold, observed):
    states = [_multinomial_states(n, p, score) for (p, score), n in groups.items()]
    probability, assignments = 0.0, 0
    for combination in itertools.product(*states):
        assignments += 1
        if sum(value for value, mass in combination) >= threshold - 1e-10:
            probability += math.prod(mass for value, mass in combination)
    return dict(
        p_value=min(1.0, probability),
        method="enumeration",
        variable_records=sum(groups.values()),
        states=assignments,
        observed=observed,
    )


def _seeded_redraws(groups, threshold, observed, seed, draws):
    rng = np.random.default_rng(seed)
    statistics = np.zeros(draws)
    for (p, scores), n in groups.items():
        statistics += rng.multinomial(n, p, size=draws) @ np.asarray(scores)
    extreme = int(np.count_nonzero(statistics >= threshold - 1e-10))
    return dict(
        p_value=extreme / draws,
        method="seeded redraws",
        draws=draws,
        extreme=extreme,
        seed=seed,
        variable_records=sum(groups.values()),
        observed=observed,
        monte_carlo_se=math.sqrt((extreme / draws) * (1 - extreme / draws) / draws),
    )


def exact_check(records, candidate, baseline, seed=7, draws=100000, *, model=None, estimator="dr"):
    """Redraw actions under the sharp null, keeping the independent model fixed.

    Multinomial counts group identical records. This is the same probability law
    as redrawing every row, and uses much less memory than draws by log length.
    Exact enumeration counts every possible assignment when <=20 rows vary.
    Beyond that the reported check is a seeded Monte Carlo approximation.
    """
    records = list(records)
    model = CellModel() if model is None else model
    if draws < 1 or any(r.override or r.reward is None for r in records) or not records:
        return dict(p_value=None, method="undefined", reason="resolved randomized records required")
    if not all(supported(support_table(records, p)) for p in (candidate, baseline)):
        return dict(p_value=None, method="undefined", reason="missing action support")
    observed = float(np.mean(improvement_terms(candidate, baseline, model, records, estimator)))
    groups, constant = _redraw_groups(records, candidate, baseline, model, estimator)
    threshold = observed * len(records) - constant
    if sum(groups.values()) <= 20:
        return _enumerate(groups, threshold, observed)
    return _seeded_redraws(groups, threshold, observed, seed, draws)


def decide(record, rule):
    reasons = []
    if rule.support_required and not record["judged"]:
        reasons.append("missing action support; policy not judged")
    if record["n_eff"] < rule.n_eff_floor:
        reasons.append(f"candidate effective size below {rule.n_eff_floor}")
    if record["baseline_n_eff"] < rule.n_eff_floor:
        reasons.append(f"baseline effective size below {rule.n_eff_floor}")
    if rule.outcomes_final and not record["outcomes_final"]:
        reasons.append("outcomes not final")
    if rule.one_logging_policy and not record["one_logging_policy"]:
        reasons.append("more than one logging policy version, fingerprint or seed")
    if record.get("intact") is False:
        reasons.append("frozen window differs from manifest")
    if record["interval"] is None or record["interval"][0] <= 0:
        reasons.append("improvement interval is not above zero")
    p = record["exact"]["p_value"]
    if p is None or p > rule.exact_level:
        reasons.append(
            f"exact check exceeds {rule.exact_level:g}"
            if p is not None
            else "exact check undefined"
        )
    return ("hold" if reasons else "eligible for shadow"), reasons
