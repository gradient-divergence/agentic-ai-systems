"""Reproduce chapter 5's time-split baselines and descriptive reliability/cost tables."""

import argparse
import json
import platform
from collections.abc import Sequence
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

import numpy as np
import sklearn
from numpy.typing import NDArray
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from refund_agent.data.history import (
    DEFAULT_PATH,
    DRIFT_AT,
    FEATURE_COLUMNS,
    HistoryRow,
    load_history,
)

TEST_AT = datetime(2026, 4, 1, tzinfo=timezone.utc)
THRESHOLD = 0.080
REVIEW_COST = 4.0
WRONG_DENIAL_COST = 20.0


def feature_matrix(rows: Sequence[HistoryRow]) -> NDArray[np.float64]:
    """Encode only the decision-time allowlist; no planted probability or labels."""
    encoded = []
    for row in rows:
        values = []
        for name in FEATURE_COLUMNS:
            value = getattr(row, name)
            if name == "reason":
                # not_delivered is the reference category; no target encoding.
                values.extend([float(value == "damaged"), float(value == "changed_mind")])
            else:
                values.append(float(value) if value is not None else np.nan)
        encoded.append(values)
    return np.asarray(encoded, dtype=np.float64)


def time_split(
    rows: Sequence[HistoryRow],
) -> tuple[list[HistoryRow], list[HistoryRow], list[HistoryRow]]:
    train, gap, test = [], [], []
    for row in rows:
        if not row.paid:
            continue
        if row.at >= TEST_AT:
            test.append(row)
        elif row.label_final_at is not None and row.label_final_at <= TEST_AT:
            train.append(row)
        else:
            gap.append(row)
    return train, gap, test


def reliability_table(labels: Sequence[int], probabilities: Sequence[float]) -> list[dict]:
    """Ten quantile bins on test predictions; ties stay together, empty bins remain."""
    y = np.asarray(labels)
    p = np.asarray(probabilities)
    edges = np.quantile(p, np.linspace(0, 1, 11))
    assignments = np.searchsorted(edges[1:-1], p, side="right")
    table = []
    for index in range(10):
        selected = assignments == index
        count = int(selected.sum())
        table.append(
            {
                "bin": index + 1,
                "lower": float(edges[index]),
                "upper": float(edges[index + 1]),
                "count": count,
                "positives": int(y[selected].sum()),
                "mean_probability": float(p[selected].mean()) if count else None,
                "observed_rate": float(y[selected].mean()) if count else None,
            }
        )
    return table


def cost_summary(
    labels: Sequence[int],
    probabilities: Sequence[float],
    amounts_cents: Sequence[int],
    threshold: float = THRESHOLD,
) -> dict:
    """Pay below the tier threshold, review at/above it; a score never denies.

    Review is illustratively perfect and costs $4 for either class. A false claim
    paid costs its requested amount. A genuine claim denied would cost $20.
    """
    y, p = np.asarray(labels), np.asarray(probabilities)
    amounts = np.asarray(amounts_cents) / 100
    pay = p < threshold
    expected = np.where(pay, p * amounts, REVIEW_COST)
    realized = np.where(pay, y * amounts, REVIEW_COST)
    return {
        "threshold": threshold,
        "review_cost_usd": REVIEW_COST,
        "wrong_denial_cost_usd": WRONG_DENIAL_COST,
        "deny_edge": 1 - REVIEW_COST / WRONG_DENIAL_COST,
        "paid_count": int(pay.sum()),
        "review_count": int((~pay).sum()),
        "denied_count": 0,
        "wrong_denial_count": 0,
        "false_claims_paid": int(y[pay].sum()),
        "expected_total_usd": float(expected.sum()),
        "expected_per_request_usd": float(expected.mean()),
        "realized_total_usd": float(realized.sum()),
        "realized_per_request_usd": float(realized.mean()),
    }


def _counts(rows: Sequence[HistoryRow]) -> dict:
    paid = [row for row in rows if row.paid]
    positives = sum(row.found_false_30d == 1 for row in paid)
    return {
        "requests": len(rows),
        "paid": len(paid),
        "positives": positives,
        "base_rate": positives / len(paid) if paid else None,
    }


def _models(seed: int) -> dict:
    return {
        "logistic_regression": make_pipeline(
            SimpleImputer(strategy="constant", fill_value=0),
            StandardScaler(),
            LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs", random_state=seed),
        ),
        "gradient_boosted": make_pipeline(
            SimpleImputer(strategy="constant", fill_value=0),
            HistGradientBoostingClassifier(
                learning_rate=0.05,
                max_iter=100,
                max_leaf_nodes=7,
                min_samples_leaf=100,
                l2_regularization=1.0,
                early_stopping=False,
                random_state=seed,
            ),
        ),
    }


def _model_results(train: list[HistoryRow], test: list[HistoryRow], seed: int) -> dict:
    y_train = [int(row.found_false_30d) for row in train if row.found_false_30d is not None]
    y_test = [int(row.found_false_30d) for row in test if row.found_false_30d is not None]
    if not train or not test or len(set(y_train)) != 2 or len(set(y_test)) != 2:
        raise ValueError("both time windows need paid examples of both classes")
    x_train, x_test = feature_matrix(train), feature_matrix(test)
    results = {}
    for name, model in _models(seed).items():
        model.fit(x_train, y_train)
        probabilities = model.predict_proba(x_test)[:, 1].tolist()
        results[name] = {
            "auc": float(roc_auc_score(y_test, probabilities)),
            "mean_probability": float(np.mean(probabilities)),
            "reliability": reliability_table(y_test, probabilities),
            "cost": cost_summary(y_test, probabilities, [row.amount_cents for row in test]),
        }
    return results


def _history_counts(
    rows: Sequence[HistoryRow],
    train: list[HistoryRow],
    gap: list[HistoryRow],
    test: list[HistoryRow],
) -> dict:
    return {
        "all": _counts(rows),
        "first_12_months": _counts([row for row in rows if row.at < TEST_AT]),
        "last_6_months": _counts([row for row in rows if row.at >= TEST_AT]),
        "train": _counts(train),
        "maturity_gap": _counts(gap),
        "test": _counts(test),
        "before_drift": _counts([row for row in rows if row.at < DRIFT_AT]),
        "after_drift": _counts([row for row in rows if row.at >= DRIFT_AT]),
        "test_before_drift": _counts([row for row in test if row.at < DRIFT_AT]),
        "test_after_drift": _counts([row for row in test if row.at >= DRIFT_AT]),
    }


def evaluate_history(rows: Sequence[HistoryRow], seed: int = 7) -> dict:
    train, gap, test = time_split(rows)
    return {
        "seed": seed,
        "python": platform.python_version(),
        "sklearn": sklearn.__version__,
        "numpy": np.__version__,
        "feature_columns": list(FEATURE_COLUMNS),
        "test_at": TEST_AT.isoformat(),
        "drift_at": DRIFT_AT.isoformat(),
        "counts": _history_counts(rows, train, gap, test),
        "models": _model_results(train, test, seed),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_PATH)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", type=Path, default=Path("refund-history/baseline.json"))
    args = parser.parse_args(argv)
    result = evaluate_history(load_history(args.data), args.seed)
    result["data_sha256"] = sha256(args.data.read_bytes()).hexdigest()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
