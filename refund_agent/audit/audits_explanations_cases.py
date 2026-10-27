"""A fixed must-pass denial fixture, scheduled and graded by chapter 29."""

import json
from datetime import timedelta
from pathlib import Path

from refund_agent import contracts as c
from refund_agent.audit.audits_explanations_records import EvaluationRecord
from refund_agent.data.generate import Dataset
from refund_agent.release.suite import Case
from refund_agent.runtimes.swap import SwapCase

FIXTURE = (
    Path(__file__).resolve().parents[2]
    / "exercises/audits-explanations/two_clause_06.json"
)


def cases(*, settings):
    row = json.loads(FIXTURE.read_text())["current"]
    record = EvaluationRecord.model_validate_json(json.dumps(row))
    # A tier-1 two-clause denial keeps the must-pass slot independent of the
    # tier-routing component chapter 37 adds later.
    request = record.request.model_copy(update={"amount_cents": 2499})
    order = record.order.model_copy(
        update={
            "paid_cents": 2000,
            "items": [
                c.Item(name="physical lamp", quantity=1, unit_cents=2000)
            ],
        }
    )
    data = Dataset(
        requests=[request],
        orders=[order],
        shipments=[record.shipment],
        policies=[record.policy],
        retrievals=[
            c.PolicySnapshot(
                request_id=request.request_id,
                version=record.policy.version,
                observed_at=record.policy_observed_at,
            )
        ],
        correct=[],
    )
    fixture = SwapCase(
        "slot-audits-explanations", "audits-explanations", data, request, 0
    )
    return (
        Case(
            "slot-audits-explanations",
            "regression",
            "denial-explanation-completeness",
            1,
            fixture=fixture,
            handler="audits-explanations-replay",
            expected={
                "paid_cents": 0,
                "outcome": "deny",
                "reason": "amount_over_policy",
            },
        ),
    )


def run_case(*, case, candidate, settings, execution):
    if case.handler != "audits-explanations-replay":
        return None
    execution.run(
        at=case.fixture.data.orders[0].observed_at + timedelta(seconds=120)
    )
    return True
