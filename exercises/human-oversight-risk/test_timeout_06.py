"""Use HUMAN_OVERSIGHT_RISK_REFERENCE=1 to check the reference solution."""

import os
from datetime import datetime, timezone

from refund_agent import contracts as c
from refund_agent.oversight.human_oversight_risk_lifecycle import ReviewTicket

if os.environ.get("HUMAN_OVERSIGHT_RISK_REFERENCE") == "1":
    from reference.timeout_06 import on_deadline
else:
    from timeout_06 import on_deadline


def test_deadline_never_records_an_approval_or_closes_the_item():
    ticket = ReviewTicket(
        review_id="human-oversight-risk-exercise-06", request_id="request-06",
        requester_id="customer-06", work_class="tier2", tier=c.Tier.REVIEWER,
        opened_at=datetime(2026, 10, 2, 18, 20, tzinfo=timezone.utc),
        due_at=datetime(2026, 10, 5, 17, tzinfo=timezone.utc),
        assignee="alex", path=("alex", "morgan", "decision-owner"), queue_length=1,
    )
    result = on_deadline(ticket, ticket.due_at)
    assert not any(e.kind in ("approval", "denial_confirmation", "payment") for e in result.events), (
        "a deadline recorded an approval in place of a person's decision"
    )
    assert result.closed_at is None and result.outcome == "escalated"
    assert result.assignee == "morgan"
    assert result.due_at == datetime(2026, 10, 6, 13, tzinfo=timezone.utc)
    assert result.notices[0].due_at == result.due_at
    again = on_deadline(result, result.due_at)
    assert again.assignee == "decision-owner" and again.closed_at is None
    assert len(again.events) == len(again.notices) == 2
