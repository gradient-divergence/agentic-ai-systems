"""Reference solution to the July-only scoring and negation exercise."""

import numpy as np

from demos.typed_decisions_demo import HISTORY_SEED, decomposition


def accept(rows, model):
    p = np.asarray(model.predict(rows))
    negated = np.asarray(model.predict(rows, negated=True))
    parts = decomposition([r.found_false_30d for r in rows], p)
    violations = int(np.sum(np.abs(p + negated - 1) > 1e-9))
    reasons = []
    if parts["reliability"] > 0.0005:
        reasons.append("reliability")
    if violations:
        reasons.append("negation")
    return dict(
        decision="refuse" if reasons else "accept",
        reasons=reasons,
        seed=HISTORY_SEED,
        requests=len(rows),
        negation_violations=violations,
        **{key: value for key, value in parts.items() if key != "bins"},
    )
