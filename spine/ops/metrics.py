"""Ops metrics utilities for spine runs."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel


class RunMetrics(BaseModel):
    rows: int
    positive_rate: float
    model_accuracy: float | None = None
    model_roc_auc: float | None = None
    policy_accuracy: float | None = None
    average_reward: float | None = None


def write_run_metrics(output_dir: Path, metrics: RunMetrics) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "run_metrics.json"
    path.write_text(json.dumps(metrics.model_dump(mode="json"), indent=2), encoding="utf-8")
    return path
