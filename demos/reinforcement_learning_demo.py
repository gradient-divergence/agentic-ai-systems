#!/usr/bin/env python3
from __future__ import annotations











import numpy as np
from pydantic import BaseModel, Field


class PrioritizedReplayConfig(BaseModel):
    alpha: float = Field(0.6, ge=0.0)
    beta: float = Field(0.4, ge=0.0, le=1.0)


def prioritized_weights(td_errors: np.ndarray, config: PrioritizedReplayConfig) -> np.ndarray:
    priorities = np.power(np.abs(td_errors) + 1e-6, config.alpha)
    return priorities / priorities.sum()


def importance_sampling_weights(priorities: np.ndarray, config: PrioritizedReplayConfig) -> np.ndarray:
    weights = np.power(len(priorities) * priorities, -config.beta)
    return weights / weights.max()


errors = np.array([0.2, 1.1, 0.6, 0.4])
cfg = PrioritizedReplayConfig(alpha=0.7, beta=0.5)
priorities = prioritized_weights(errors, cfg)
is_weights = importance_sampling_weights(priorities, cfg)
print({"priorities": [round(w, 3) for w in priorities.tolist()], "is": [round(w, 3) for w in is_weights.tolist()]})
