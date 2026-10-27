from __future__ import annotations

from typing import Annotated, List

from pydantic import BaseModel, Field


class TimeSeriesConfig(BaseModel):
    time_col: str = "ds"
    target_col: str = "y"
    lags: List[Annotated[int, Field(gt=0)]] = Field(default_factory=lambda: [1, 7, 14])
    rolling_window: int = Field(7, ge=1)
    train_fraction: float = Field(0.8, gt=0.0, lt=1.0)
    use_gpu: bool = False


def _make_features(df: pd.DataFrame, cfg: TimeSeriesConfig) -> Tuple[pd.DataFrame, pd.Series]:
    df = df.sort_values(cfg.time_col).copy()
    for lag in cfg.lags:
        df[f"{cfg.target_col}_lag_{lag}"] = df[cfg.target_col].shift(lag)
    df[f"{cfg.target_col}_roll_mean_{cfg.rolling_window}"] = (
        df[cfg.target_col].shift(1).rolling(cfg.rolling_window).mean()
    )
    df = df.dropna()
    X = df.drop(columns=[cfg.target_col, cfg.time_col])
    y = df[cfg.target_col]
    return X, y


# region book:fnd.supervised-learning.timeseries-lgbm-01
from typing import Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd


def train_lgbm_forecaster(df: pd.DataFrame, cfg: TimeSeriesConfig) -> Tuple[lgb.Booster, float]:
    X, y = _make_features(df, cfg)
    split = int(len(X) * cfg.train_fraction)
    X_train, X_valid = X.iloc[:split], X.iloc[split:]
    y_train, y_valid = y.iloc[:split], y.iloc[split:]

    params = {
        "objective": "regression",
        "metric": "mae",
        "learning_rate": 0.1,
        "num_leaves": 31,
        "feature_fraction": 0.9,
    }
    if cfg.use_gpu:
        params["device_type"] = "gpu"

    train_set = lgb.Dataset(X_train, label=y_train)
    valid_set = lgb.Dataset(X_valid, label=y_valid)
    model = lgb.train(
        params,
        train_set,
        num_boost_round=300,
        valid_sets=[valid_set],
        callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)],
    )
    preds = model.predict(X_valid, num_iteration=model.best_iteration)
    mae = float(np.mean(np.abs(preds - y_valid.to_numpy())))
    return model, mae


# endregion


def _example() -> None:
    rng = pd.date_range("2024-01-01", periods=180, freq="D")
    noise = np.random.default_rng(13).normal(size=len(rng))
    series = pd.DataFrame({"ds": rng, "y": np.sin(np.arange(len(rng)) / 10.0) + 0.1 * noise})
    cfg = TimeSeriesConfig(use_gpu=False)
    _, mae = train_lgbm_forecaster(series, cfg)
    print({"mae": mae})


if __name__ == "__main__":
    _example()
