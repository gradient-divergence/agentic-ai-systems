#!/usr/bin/env python3
from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from pydantic import BaseModel, Field

from agents.raom_minimal import ActionRequest, ActionResult, EvaluationRecord, EvaluationRecorder


class EvalCase(BaseModel):
    trace_id: str
    expected_tool: str
    expected_status: str
    min_score: float = Field(default=0.8, ge=0, le=1)


def score_case(case: EvalCase, action: ActionRequest, result: ActionResult) -> float:
    score = 1.0
    if action.tool_name != case.expected_tool:
        score -= 0.5
    if result.status != case.expected_status:
        score -= 0.5
    return max(score, 0.0)


def record_case(
    recorder: EvaluationRecorder,
    case: EvalCase,
    action: ActionRequest,
    result: ActionResult,
) -> float:
    score = score_case(case, action, result)
    recorder.record(
        EvaluationRecord(
            trace_id=case.trace_id,
            observation={"case": case.model_dump(), "score": score},
            action=action,
            result=result,
            timestamp=datetime.now(timezone.utc),
        )
    )
    return score




def run_eval_demo() -> None:
    recorder = EvaluationRecorder()
    case = EvalCase(
        trace_id=f"trace_{uuid4().hex[:8]}",
        expected_tool="quote_price",
        expected_status="ok",
    )
    action = ActionRequest(
        tool_name="quote_price",
        payload={"item_id": "item-1001", "quantity": 2},
        estimated_cost=4.0,
    )
    result = ActionResult(status="ok", output={"total": 39.0})
    score = record_case(recorder, case, action, result)
    passed = score >= case.min_score
    print({"score": score, "passed": passed})


if __name__ == "__main__":
    run_eval_demo()
