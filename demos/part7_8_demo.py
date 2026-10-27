#!/usr/bin/env python3
from __future__ import annotations










# region book:fnd.appendix-math-foundations.core-01
import math


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or not left:
        raise ValueError("Vectors must be non-empty and same length")
    if not all(math.isfinite(value) for value in left + right):
        raise ValueError("Vectors must be finite")
    scale_left, scale_right = max(map(abs, left)), max(map(abs, right))
    if scale_left == 0 or scale_right == 0:
        raise ValueError("Vectors must have non-zero norms")
    left = [value / scale_left for value in left]
    right = [value / scale_right for value in right]
    norm_left, norm_right = math.hypot(*left), math.hypot(*right)
    return math.fsum(
        (a / norm_left) * (b / norm_right) for a, b in zip(left, right)
    )
# endregion book:fnd.appendix-math-foundations.core-01


# region book:fnd.appendix-math-foundations.example-01
print({"cosine": round(cosine_similarity([1.0, 0.0], [0.5, 0.5]), 3)})
# endregion book:fnd.appendix-math-foundations.example-01


# region book:fnd.appendix-engineering-practices.core-01
from pydantic import BaseModel


class ChecklistItem(BaseModel):
    name: str
    done: bool


def completion_rate(items: list[ChecklistItem]) -> float:
    if not items:
        return 0.0
    completed = sum(1 for item in items if item.done)
    return completed / len(items)
# endregion book:fnd.appendix-engineering-practices.core-01


# region book:fnd.appendix-engineering-practices.example-01
items = [
    ChecklistItem(name="threat_model", done=True),
    ChecklistItem(name="eval_baseline", done=False),
]
print({"completion": round(completion_rate(items), 2)})
# endregion book:fnd.appendix-engineering-practices.example-01


# region book:fnd.appendix-exercises.core-01
from pydantic import BaseModel, Field


class Exercise(BaseModel):
    title: str
    difficulty: int = Field(..., ge=1, le=5)
    tags: list[str]


def shortlist_exercises(
    exercises: list[Exercise], max_difficulty: int
) -> list[str]:
    return [
        exercise.title
        for exercise in exercises
        if exercise.difficulty <= max_difficulty
    ]
# endregion book:fnd.appendix-exercises.core-01


# region book:fnd.appendix-exercises.example-01
exercises = [
    Exercise(
        title="Design an eval harness", difficulty=3, tags=["evaluation"]
    ),
    Exercise(title="Instrument tool latency", difficulty=2, tags=["ops"]),
]
print({"shortlist": shortlist_exercises(exercises, max_difficulty=3)})
# endregion book:fnd.appendix-exercises.example-01
