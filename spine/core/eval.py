"""Minimal evaluation harness placeholder."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class Metric:
    name: str
    value: float


def write_metrics(metrics: Iterable[Metric], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{metric.name},{metric.value:.6f}" for metric in metrics]
    output_path.write_text("\n".join(lines) + "\n")
