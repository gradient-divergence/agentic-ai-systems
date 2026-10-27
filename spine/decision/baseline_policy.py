"""Baseline decision policy for the spine."""

from __future__ import annotations

# region book:fnd.agentic-loop.core-01
from dataclasses import dataclass


@dataclass(frozen=True)
class DecisionInput:
    review_stars: float
    user_average_stars: float
    business_stars: float


@dataclass(frozen=True)
class DecisionOutput:
    score: float
    threshold: float
    action: str


def _normalize_stars(value: float) -> float:
    if value <= 0:
        return 0.0
    return min(max(value / 5.0, 0.0), 1.0)


def score_review(input_row: DecisionInput) -> float:
    review = _normalize_stars(input_row.review_stars)
    user = _normalize_stars(input_row.user_average_stars)
    business = _normalize_stars(input_row.business_stars)
    return 0.4 * review + 0.3 * user + 0.3 * business
# endregion


def decide(input_row: DecisionInput, threshold: float = 0.8) -> DecisionOutput:
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("Score threshold must be between zero and one.")
    score = score_review(input_row)
    action = "promote" if score >= threshold else "monitor"
    return DecisionOutput(score=score, threshold=threshold, action=action)
