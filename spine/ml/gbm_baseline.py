"""Gradient boosting baseline for Yelp spine."""

from __future__ import annotations

# region book:fnd.supervised-learning.core-01
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, Field, field_validator
from sklearn.metrics import accuracy_score, roc_auc_score

from spine.core.eval import Metric, write_metrics
from spine.data.yelp_features import build_feature_rows, time_based_split


class TrainConfig(BaseModel):
    subset_dir: Path = Field(...)
    output_dir: Path = Field(...)
    seed: int = Field(13, ge=0)
    test_fraction: float = Field(0.2, gt=0.0, lt=1.0)
    label_threshold: float = Field(4.0, ge=1.0, le=5.0)
    use_lightgbm: bool = True
    max_rows: int | None = Field(None, ge=1)

    @field_validator("subset_dir")
    @classmethod
    def _validate_subset_dir(cls, value: Path) -> Path:
        if not value.exists():
            raise ValueError(f"Subset directory does not exist: {value}")
        return value

    @field_validator("output_dir")
    @classmethod
    def _ensure_output_dir(cls, value: Path) -> Path:
        value.mkdir(parents=True, exist_ok=True)
        return value

    model_config = {"validate_assignment": True}


class ModelMetrics(BaseModel):
    accuracy: float
    roc_auc: float
    model_type: str


def _build_arrays(rows: list[Any]) -> tuple[np.ndarray, np.ndarray]:
    X = np.array([row.to_feature_vector() for row in rows], dtype=float)
    y = np.array([row.label for row in rows], dtype=int)
    return X, y


def train_baseline(cfg: TrainConfig) -> ModelMetrics:
    rows = build_feature_rows(
        cfg.subset_dir,
        label_threshold=cfg.label_threshold,
        max_rows=cfg.max_rows,
    )
    train_rows, test_rows = time_based_split(rows, test_fraction=cfg.test_fraction)
    X_train, y_train = _build_arrays(train_rows)
    X_test, y_test = _build_arrays(test_rows)

    model, model_type = _train_model(X_train, y_train, cfg)

    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= 0.5).astype(int)
    metrics = ModelMetrics(
        accuracy=float(accuracy_score(y_test, preds)),
        roc_auc=float(roc_auc_score(y_test, probs)),
        model_type=model_type,
    )

    metrics_path = cfg.output_dir / "metrics.csv"
    write_metrics(
        [
            Metric("accuracy", metrics.accuracy),
            Metric("roc_auc", metrics.roc_auc),
        ],
        metrics_path,
    )

    return metrics
# endregion


# region book:fnd.supervised-learning.sota-01



def _train_model(X_train: np.ndarray, y_train: np.ndarray, cfg: "TrainConfig") -> tuple[Any, str]:
    if cfg.use_lightgbm:
        try:
            from lightgbm import LGBMClassifier

            model = LGBMClassifier(
                n_estimators=200,
                learning_rate=0.06,
                random_state=cfg.seed,
            )
            model.fit(X_train, y_train)
            return model, "lightgbm"
        except ModuleNotFoundError:
            pass

    from sklearn.ensemble import HistGradientBoostingClassifier

    model = HistGradientBoostingClassifier(random_state=cfg.seed)
    model.fit(X_train, y_train)
    return model, "sklearn_hist_gbm"
# endregion


def run_training(cfg: TrainConfig) -> ModelMetrics:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    return train_baseline(cfg)


def describe_features() -> list[str]:
    from spine.data.yelp_features import FEATURE_COLUMNS

    return FEATURE_COLUMNS.copy()
