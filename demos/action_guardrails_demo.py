#!/usr/bin/env python3
from __future__ import annotations

# region book:fnd.action-guardrails.core-01
from pydantic import BaseModel, Field

from agents.raom_minimal import (
    ActionRequest,
    MaxCostRule,
    PolicyEngine,
    ToolAllowlistRule,
    ToolRegistry,
    ToolSpec,
)


class PriceQuoteRequest(BaseModel):
    item_id: str
    quantity: int = Field(..., gt=0)


class PriceQuoteResponse(BaseModel):
    item_id: str
    unit_price: float
    total: float


def quote_price(payload: PriceQuoteRequest) -> PriceQuoteResponse:
    unit_price = 19.5
    return PriceQuoteResponse(
        item_id=payload.item_id,
        unit_price=unit_price,
        total=unit_price * payload.quantity,
    )


def build_guardrails() -> tuple[ToolRegistry, PolicyEngine]:
    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name="quote_price",
            description="Return an item price quote",
            input_model=PriceQuoteRequest,
            output_model=PriceQuoteResponse,
            handler=quote_price,
        )
    )
    policy = PolicyEngine(
        rules=[
            ToolAllowlistRule({"quote_price"}),
            MaxCostRule(max_cost=25.0),
        ]
    )
    return registry, policy


# endregion book:fnd.action-guardrails.core-01


# region book:fnd.action-guardrails.example-01
def run_guarded_tool_call() -> None:
    registry, policy = build_guardrails()
    action = ActionRequest(
        tool_name="quote_price",
        payload={"item_id": "item-1001", "quantity": 2},
        estimated_cost=4.0,
    )
    decision = policy.evaluate(action)
    if not decision.allowed:
        print("blocked", decision.reasons)
        return
    result = registry.validate_and_run(action.tool_name, action.payload)
    print(result.model_dump())


if __name__ == "__main__":
    run_guarded_tool_call()
# endregion book:fnd.action-guardrails.example-01
