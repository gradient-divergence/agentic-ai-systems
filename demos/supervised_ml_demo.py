#!/usr/bin/env python3
from __future__ import annotations

# region book:fnd.data-science-strategy.core-01
from pydantic import BaseModel, Field


class Hypothesis(BaseModel):
    statement: str
    metric: str
    expected_delta: float


class ExperimentPlan(BaseModel):
    hypothesis: Hypothesis
    success_criteria: str
    data_window_weeks: int = Field(..., gt=0)
# endregion book:fnd.data-science-strategy.core-01


# region book:fnd.data-science-strategy.example-01
plan = ExperimentPlan(
    hypothesis=Hypothesis(
        statement="Personalized ranking improves conversion",
        metric="conversion_rate",
        expected_delta=0.02,
    ),
    success_criteria="p < 0.05 and +2% conversion",
    data_window_weeks=8,
)
print(plan.model_dump())
# endregion book:fnd.data-science-strategy.example-01


# region book:fnd.ml-data-and-evaluation.sota-01
from sklearn.calibration import CalibratedClassifierCV
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, make_scorer
from sklearn.model_selection import StratifiedKFold, cross_validate


def calibrated_cv_scores(seed: int = 13) -> dict[str, float]:
    X, y = make_classification(
        n_samples=240,
        n_features=8,
        n_informative=5,
        random_state=seed,
    )
    base_model = LogisticRegression(max_iter=250)
    calibrated = CalibratedClassifierCV(base_model, cv=3)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    scores = cross_validate(
        calibrated,
        X,
        y,
        cv=cv,
        scoring={
            "roc_auc": "roc_auc",
            "brier": make_scorer(brier_score_loss, response_method="predict_proba", greater_is_better=False),
        },
        error_score="raise",
    )
    return {
        "roc_auc": float(scores["test_roc_auc"].mean()),
        "brier": -float(scores["test_brier"].mean()),
    }


print(calibrated_cv_scores())
# endregion book:fnd.ml-data-and-evaluation.sota-01


# region book:fnd.ml-data-and-evaluation.imbalance-01
import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split


def compare_class_weights(seed: int = 41) -> dict[str, float]:
    X, y = make_classification(
        n_samples=400,
        n_features=8,
        n_informative=5,
        weights=[0.9, 0.1],
        random_state=seed,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=seed, stratify=y
    )
    plain = LogisticRegression(max_iter=300)
    balanced = LogisticRegression(max_iter=300, class_weight="balanced")
    plain.fit(X_train, y_train)
    balanced.fit(X_train, y_train)
    plain_pred = plain.predict(X_test)
    balanced_pred = balanced.predict(X_test)
    return {
        "plain_bal_acc": float(balanced_accuracy_score(y_test, plain_pred)),
        "balanced_bal_acc": float(balanced_accuracy_score(y_test, balanced_pred)),
        "plain_f1": float(f1_score(y_test, plain_pred)),
        "balanced_f1": float(f1_score(y_test, balanced_pred)),
    }


print(compare_class_weights())
# endregion book:fnd.ml-data-and-evaluation.imbalance-01


# region book:fnd.ml-data-and-evaluation.leakage-01
from pydantic import BaseModel, Field


class SplitConfig(BaseModel):
    holdout_ratio: float = Field(0.2, gt=0.0, lt=1.0)
    allow_shuffle: bool = False


def temporal_split(indices: np.ndarray, cfg: SplitConfig) -> tuple[np.ndarray, np.ndarray]:
    if cfg.allow_shuffle:
        raise ValueError("Temporal split forbids shuffling to prevent leakage.")
    cutoff = int(len(indices) * (1 - cfg.holdout_ratio))
    return indices[:cutoff], indices[cutoff:]


timestamps = np.arange(100)
train_idx, test_idx = temporal_split(timestamps, SplitConfig(holdout_ratio=0.2))
print({"train_max": int(train_idx.max()), "test_min": int(test_idx.min())})
# endregion book:fnd.ml-data-and-evaluation.leakage-01


# region book:fnd.ml-data-and-evaluation.feature-eng-libs-01
import importlib
from types import ModuleType
from typing import Callable, cast

import pandas as pd


def _optional_import(name: str) -> ModuleType | None:
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError:
        return None


ft = _optional_import("featuretools")
autofeat = _optional_import("autofeat")
tsfresh = _optional_import("tsfresh")


def relational_features(entityset: object, target: str) -> pd.DataFrame:
    if ft is None:
        raise RuntimeError("featuretools not installed")
    dfs = cast(
        Callable[..., tuple[pd.DataFrame, object]],
        getattr(ft, "dfs"),
    )
    features, _ = dfs(entityset=entityset, target_dataframe_name=target, max_depth=2)
    return features


def automated_tabular_features(X: pd.DataFrame, y: pd.Series) -> object:
    if autofeat is None:
        raise RuntimeError("autofeat not installed")
    regressor_cls = cast(type[object], getattr(autofeat, "AutoFeatRegressor"))
    model = regressor_cls(verbose=0)
    model.fit(X, y)
    return model


def time_series_features(df: pd.DataFrame, entity: str, time: str) -> pd.DataFrame:
    if tsfresh is None:
        raise RuntimeError("tsfresh not installed")
    extract_features = cast(
        Callable[..., pd.DataFrame],
        getattr(tsfresh, "extract_features"),
    )
    return extract_features(df, column_id=entity, column_sort=time)


# endregion book:fnd.ml-data-and-evaluation.feature-eng-libs-01


# region book:fnd.supervised-learning.stacking-01
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.metrics import roc_auc_score


def stacking_auc(seed: int = 17) -> float:
    from sklearn.datasets import make_classification
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    X, y = make_classification(
        n_samples=260,
        n_features=9,
        n_informative=6,
        random_state=seed,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=seed, stratify=y
    )
    base_models = [
        ("lr", LogisticRegression(max_iter=300)),
        ("rf", RandomForestClassifier(n_estimators=120, random_state=seed)),
    ]
    meta = LogisticRegression(max_iter=300)
    model = StackingClassifier(estimators=base_models, final_estimator=meta, passthrough=True)
    model.fit(X_train, y_train)
    probs = model.predict_proba(X_test)[:, 1]
    return float(roc_auc_score(y_test, probs))


print({"stacking_auc": round(stacking_auc(), 4)})
# endregion book:fnd.supervised-learning.stacking-01


# region book:fnd.optimization-regularization.optuna-01
from typing import Any


def optuna_tune(seed: int = 23) -> dict[str, Any]:
    from sklearn.datasets import make_classification
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split

    try:
        import optuna
    except ModuleNotFoundError:
        return {"error": "optuna not installed", "best_params": {}}

    def objective(trial: optuna.Trial) -> float:
        c = trial.suggest_float("C", 1e-3, 10.0, log=True)
        l1_ratio = trial.suggest_categorical("l1_ratio", [0.0])
        X, y = make_classification(
            n_samples=260,
            n_features=8,
            n_informative=5,
            random_state=seed,
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=seed, stratify=y
        )
        model = LogisticRegression(max_iter=400, C=c, l1_ratio=l1_ratio)
        model.fit(X_train, y_train)
        probs = model.predict_proba(X_test)[:, 1]
        return roc_auc_score(y_test, probs)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=8)
    return {"best_params": study.best_params, "best_value": float(study.best_value)}


print(optuna_tune())
# endregion book:fnd.optimization-regularization.optuna-01
