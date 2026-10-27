"""Reference solution for exercise 6 solution; never reads a frozen window."""

import numpy as np

from demos.supervised_learning_demo import boosted


def self_train(X, y, pool, factory=boosted, confidence=0.9, rounds=3):
    base_rate = float(np.mean(y))
    model = factory().fit(X, y)
    remaining, report, total = np.arange(len(pool)), [], 0
    for step in range(1, rounds + 1):
        if not len(remaining):
            break
        probabilities = model.predict_proba(pool[remaining])
        keep = probabilities.max(axis=1) >= confidence
        labels = model.classes_[probabilities[keep].argmax(axis=1)]
        accepted = int(keep.sum())
        share = float((labels == 1).mean()) if accepted else 0.0
        stopped = not (0.5 * base_rate <= share <= 1.5 * base_rate)
        guard = "stop" if stopped else "continue"
        print(
            f"round {step}: proposed {accepted}; share_one {share:.4f}; base_rate {base_rate:.4f}; guard {guard}"
        )
        if not stopped and accepted:
            X = np.concatenate([X, pool[remaining[keep]]])
            y = np.concatenate([y, labels])
            model = factory().fit(X, y)
            total += accepted
        report.append(
            {
                "round": step,
                "accepted": accepted,
                "share_one": share,
                "base_rate": base_rate,
                "guard": guard,
                "total": total,
            }
        )
        if stopped:
            break
        remaining = remaining[~keep]
    return model, report
