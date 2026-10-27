"""Minimal policy + tool registry primitives for the Foundations companion demos."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime

from pydantic import BaseModel, Field


class ActionRequest(BaseModel):
    tool_name: str
    payload: dict[str, object]
    estimated_cost: float | None = Field(default=None, ge=0)


class ActionResult(BaseModel):
    status: str
    output: dict[str, object]


class EvaluationRecord(BaseModel):
    trace_id: str
    observation: dict[str, object]
    action: ActionRequest
    result: ActionResult
    timestamp: datetime


class EvaluationRecorder:
    def __init__(self) -> None:
        self._records: list[EvaluationRecord] = []

    def record(self, record: EvaluationRecord) -> None:
        self._records.append(record)

    @property
    def records(self) -> list[EvaluationRecord]:
        return list(self._records)


class PolicyDecision(BaseModel):
    allowed: bool
    reasons: list[str]


class ToolSpec(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    name: str
    description: str
    input_model: type[BaseModel]
    output_model: type[BaseModel]
    handler: Callable[..., BaseModel]


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        self._tools[spec.name] = spec

    def validate_and_run(self, tool_name: str, payload: dict[str, object]) -> BaseModel:
        if tool_name not in self._tools:
            raise ValueError(f"Unknown tool: {tool_name}")
        spec = self._tools[tool_name]
        request = spec.input_model(**payload)
        result = spec.handler(request)
        if not isinstance(result, spec.output_model):
            result = spec.output_model(**result.model_dump())
        return result


class PolicyRule:
    def check(self, action: ActionRequest) -> str | None:
        raise NotImplementedError


class ToolAllowlistRule(PolicyRule):
    def __init__(self, allowed: set[str]) -> None:
        self._allowed = set(allowed)

    def check(self, action: ActionRequest) -> str | None:
        if action.tool_name not in self._allowed:
            return f"tool_not_allowed:{action.tool_name}"
        return None


class MaxCostRule(PolicyRule):
    def __init__(self, max_cost: float) -> None:
        self._max_cost = max_cost

    def check(self, action: ActionRequest) -> str | None:
        if action.estimated_cost is None:
            return None
        if action.estimated_cost > self._max_cost:
            return f"max_cost_exceeded:{action.estimated_cost}"
        return None


class PolicyEngine:
    def __init__(self, rules: list[PolicyRule]) -> None:
        self._rules = list(rules)

    def evaluate(self, action: ActionRequest) -> PolicyDecision:
        reasons: list[str] = []
        for rule in self._rules:
            reason = rule.check(action)
            if reason:
                reasons.append(reason)
        return PolicyDecision(allowed=not reasons, reasons=reasons)
