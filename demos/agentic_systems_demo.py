#!/usr/bin/env python3
from __future__ import annotations

# region book:fnd.reasoning-planning.core-01
from pydantic import BaseModel


class PlanStep(BaseModel):
    action: str
    rationale: str


def draft_plan(issue: str) -> list[PlanStep]:
    return [
        PlanStep(action="triage", rationale=f"Identify root cause for {issue}"),
        PlanStep(action="mitigate", rationale="Apply mitigation and monitor"),
    ]
# endregion book:fnd.reasoning-planning.core-01


# region book:fnd.reasoning-planning.example-01
plan = draft_plan("payment failures")
print([step.model_dump() for step in plan])
# endregion book:fnd.reasoning-planning.example-01


from pydantic import BaseModel


class AgentMessage(BaseModel):
    sender: str
    content: str
    recipient: str | None = None


def route_message(message: AgentMessage, peers: list[str]) -> list[AgentMessage]:
    return [message.model_copy(update={"recipient": peer}) for peer in peers]


msg = AgentMessage(sender="planner", content="validate risk")
print([m.model_dump() for m in route_message(msg, ["executor", "auditor"])])


# region book:fnd.multi-agent-systems.sota-01 output
from pydantic import BaseModel, Field


class ConsensusConfig(BaseModel):
    quorum: int = Field(..., gt=0)
    min_weight: float = Field(..., ge=0.0)


class Vote(BaseModel):
    agent: str
    approve: bool
    weight: float = Field(..., gt=0.0)


def weighted_consensus(
    votes: list[Vote], config: ConsensusConfig
) -> dict[str, float | bool]:
    if len({vote.agent for vote in votes}) != len(votes):
        raise ValueError("Each agent may vote only once")
    approvals = [v for v in votes if v.approve]
    approvals_weight = sum(v.weight for v in approvals)
    approved = (
        len(approvals) >= config.quorum
        and approvals_weight >= config.min_weight
    )
    return {
        "approved": approved,
        "approvals": float(len(approvals)),
        "weight": approvals_weight,
    }


config = ConsensusConfig(quorum=2, min_weight=1.5)
votes = [
    Vote(agent="planner", approve=True, weight=0.9),
    Vote(agent="auditor", approve=True, weight=0.7),
    Vote(agent="executor", approve=False, weight=0.6),
]
print(weighted_consensus(votes, config))
# endregion book:fnd.multi-agent-systems.sota-01
