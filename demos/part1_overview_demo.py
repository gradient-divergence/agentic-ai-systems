#!/usr/bin/env python3
from __future__ import annotations

# region book:fnd.index.core-01
from enum import Enum

from pydantic import BaseModel, Field


class StackLayer(str, Enum):
    data = "data"
    model = "model"
    control = "control"
    delivery = "delivery"


class SystemGoal(BaseModel):
    outcome: str
    decision_cadence: str = Field(..., description="How often decisions are made")
    risk_tier: str


class SystemContext(BaseModel):
    goal: SystemGoal
    primary_layers: list[StackLayer]
    constraints: list[str]
# endregion book:fnd.index.core-01


# region book:fnd.index.example-01
from pydantic import BaseModel


class ExampleSummary(BaseModel):
    title: str
    goal: str
    layers: list[str]


summary = ExampleSummary(
    title="Foundations of Agentic AI Systems",
    goal="Make reliable decisions with traceable automation",
    layers=["data", "model", "control", "delivery"],
)
print(summary.model_dump())
# endregion book:fnd.index.example-01


# region book:fnd.part1-orientation.core-01
from pydantic import BaseModel, Field


class ScopeCheck(BaseModel):
    decision: str
    latency_ms: int = Field(..., gt=0)
    audit_required: bool

    def complexity_score(self) -> int:
        score = 1
        if self.latency_ms < 500:
            score += 1
        if self.audit_required:
            score += 1
        return score
# endregion book:fnd.part1-orientation.core-01


# region book:fnd.part1-orientation.example-01
scope = ScopeCheck(decision="approve_refund", latency_ms=250, audit_required=True)
print({"decision": scope.decision, "complexity": scope.complexity_score()})
# endregion book:fnd.part1-orientation.example-01


# region book:fnd.what-is-agentic-ai.core-01
from pydantic import BaseModel


class Observation(BaseModel):
    text: str


class AgentState(BaseModel):
    goal: str
    memory: list[str]
    observation: Observation


class ToolCall(BaseModel):
    tool_name: str
    payload: dict[str, str]


def decide_action(state: AgentState) -> ToolCall:
    return ToolCall(tool_name="summarize", payload={"text": state.observation.text})
# endregion book:fnd.what-is-agentic-ai.core-01


# region book:fnd.what-is-agentic-ai.example-01
state = AgentState(
    goal="summarize the alert",
    memory=["service latency spiked last hour"],
    observation=Observation(text="Error rate increased on checkout."),
)
action = decide_action(state)
print(action.model_dump())
# endregion book:fnd.what-is-agentic-ai.example-01


# region book:fnd.modern-ai-stack.core-01
from pydantic import BaseModel


class StackSpec(BaseModel):
    data_plane: list[str]
    model_plane: list[str]
    control_plane: list[str]
    ops_plane: list[str]


def minimal_stack() -> StackSpec:
    return StackSpec(
        data_plane=["ingest", "quality", "feature_store"],
        model_plane=["training", "evaluation", "serving"],
        control_plane=["policy", "safety", "tooling"],
        ops_plane=["observability", "deployment", "costs"],
    )
# endregion book:fnd.modern-ai-stack.core-01


# region book:fnd.modern-ai-stack.example-01
print(minimal_stack().model_dump())
# endregion book:fnd.modern-ai-stack.example-01
