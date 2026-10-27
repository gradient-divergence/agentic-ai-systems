"""Decision cost matrix utilities."""

from __future__ import annotations

from pydantic import BaseModel, Field


class DecisionCosts(BaseModel):
    false_positive: float = Field(1.0, ge=0.0)
    false_negative: float = Field(5.0, ge=0.0)

    def threshold(self) -> float:
        denom = self.false_positive + self.false_negative
        if denom == 0:
            return 0.5
        return self.false_positive / denom
