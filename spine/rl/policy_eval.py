"""Policy evaluation utilities for the spine."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field, field_validator


class PolicyEvalConfig(BaseModel):
    subset_dir: Path = Field(...)
    output_dir: Path = Field(...)
    label_threshold: float = Field(4.0, ge=1.0, le=5.0)
    max_steps: int = Field(1000, ge=1)

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


class PolicyEvalResult(BaseModel):
    steps: int
    average_reward: float
    promote_rate: float
    accuracy: float
    majority_accuracy: float
    output_dir: Path


import json

from spine.decision.baseline_policy import decide
from spine.rl.environment import (
    Action,
    EnvironmentConfig,
    ReviewEnvironment,
    load_environment,
)


def evaluate_baseline_policy(cfg: PolicyEvalConfig) -> PolicyEvalResult:
    env_cfg = EnvironmentConfig(
        subset_dir=cfg.subset_dir,
        label_threshold=cfg.label_threshold,
        max_steps=cfg.max_steps,
    )
    env = load_environment(env_cfg)

    obs = env.reset()
    steps = 0
    rewards = 0.0
    promote_count = 0
    correct = 0
    positive_count = 0
    while obs is not None:
        decision = decide(obs.decision_input, threshold=cfg.label_threshold / 5.0)
        action = Action.PROMOTE if decision.action == "promote" else Action.MONITOR
        reward = ReviewEnvironment.reward_for_action(action, obs.label)
        rewards += reward
        steps += 1
        positive_count += obs.label
        if action == Action.PROMOTE:
            promote_count += 1
        predicted = 1 if action == Action.PROMOTE else 0
        if predicted == obs.label:
            correct += 1
        step_result = env.step(action)
        obs = step_result.observation

    result = PolicyEvalResult(
        steps=steps,
        average_reward=(rewards / steps) if steps else 0.0,
        promote_rate=(promote_count / steps) if steps else 0.0,
        accuracy=(correct / steps) if steps else 0.0,
        majority_accuracy=(max(positive_count, steps - positive_count) / steps) if steps else 0.0,
        output_dir=cfg.output_dir,
    )

    (cfg.output_dir / "policy_eval.json").write_text(
        json.dumps(result.model_dump(mode="json"), indent=2), encoding="utf-8"
    )
    return result
