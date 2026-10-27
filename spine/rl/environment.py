"""Deterministic review environment for policy evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field, field_validator

from spine.data.yelp_features import build_feature_rows


class Action(str, Enum):
    PROMOTE = "promote"
    MONITOR = "monitor"


@dataclass(frozen=True)
class Observation:
    review_id: str
    decision_input: DecisionInput
    label: int


@dataclass(frozen=True)
class StepResult:
    observation: Observation | None
    reward: float
    done: bool


class EnvironmentConfig(BaseModel):
    subset_dir: Path = Field(...)
    label_threshold: float = Field(4.0, ge=1.0, le=5.0)
    max_steps: int = Field(1000, ge=1)

    @field_validator("subset_dir")
    @classmethod
    def _validate_subset_dir(cls, value: Path) -> Path:
        if not value.exists():
            raise ValueError(f"Subset directory does not exist: {value}")
        return value

    model_config = {"validate_assignment": True}


from spine.data.yelp_features import FeatureRow
from spine.decision.baseline_policy import DecisionInput


class ReviewEnvironment:
    def __init__(self, rows: list[FeatureRow], max_steps: int) -> None:
        self._rows = rows[:max_steps]
        self._cursor = 0

    @staticmethod
    def reward_for_action(action: Action, label: int) -> float:
        if action == Action.PROMOTE and label == 1:
            return 1.0
        if action == Action.PROMOTE and label == 0:
            return -1.0
        if action == Action.MONITOR and label == 0:
            return 0.2
        return -0.2

    def reset(self) -> Observation | None:
        self._cursor = 0
        return self._current_observation()

    def step(self, action: Action) -> StepResult:
        obs = self._current_observation()
        if obs is None:
            return StepResult(observation=None, reward=0.0, done=True)
        reward = self.reward_for_action(action, obs.label)
        self._cursor += 1
        done = self._cursor >= len(self._rows)
        next_obs = None if done else self._current_observation()
        return StepResult(observation=next_obs, reward=reward, done=done)

    def _current_observation(self) -> Observation | None:
        if self._cursor >= len(self._rows):
            return None
        row = self._rows[self._cursor]
        input_row = DecisionInput(
            review_stars=row.review_stars,
            user_average_stars=row.user_average_stars,
            business_stars=row.business_stars,
        )
        return Observation(
            review_id=row.review_id,
            decision_input=input_row,
            label=row.label,
        )


def run_one_step(env: ReviewEnvironment, action: Action) -> StepResult:
    obs = env.reset()
    if obs is None:
        return StepResult(observation=None, reward=0.0, done=True)
    return env.step(action)


def load_environment(cfg: EnvironmentConfig) -> ReviewEnvironment:
    rows = build_feature_rows(cfg.subset_dir, label_threshold=cfg.label_threshold)
    rows_sorted = sorted(rows, key=lambda row: row.review_date)
    return ReviewEnvironment(rows_sorted, max_steps=cfg.max_steps)
