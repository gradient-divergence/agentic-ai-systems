from datetime import timedelta

import pytest
from capstone_blueprint_fixtures import action, payment_action

from refund_agent.launch.capstone_blueprint_requirement import Period
from refund_agent.launch.capstone_blueprint_scan import LaunchLogs, scan
from refund_agent.launch.capstone_blueprint_tier3 import inspect_tier3


@pytest.mark.parametrize(
    "tier,outcome,reviewed,executed",
    [
        (2, "approve", False, True),
        (2, "deny", False, False),
        (3, "approve", False, True),
        (3, "deny", False, False),
    ],
)
def test_payment_and_denial_both_need_review(
    tier, outcome, reviewed, executed
):
    row, policy = action(
        tier, outcome=outcome, reviewed=reviewed, executed=executed
    )
    events = scan(LaunchLogs(decisions=(row,)), None, policy, 0)
    assert len(events) == 1
    assert events[0].condition_id == (
        "tier2_unbound_approval"
        if tier == 2
        else "tier3_without_named_approver"
    )


def test_deadline_handler_and_approval_without_receipt_are_found():
    row, policy = action()
    bad = row.model_copy(
        update={
            "review": row.review.model_copy(
                update={"actor_kind": "deadline_handler"}
            )
        }
    )
    assert (
        "deadline handler"
        in scan(LaunchLogs(decisions=(bad,)), None, policy, 0)[0].reason
    )
    paid = row.model_copy(
        update={"decision": row.decision.model_copy(update={"executed": True})}
    )
    assert (
        "no approval"
        in scan(LaunchLogs(decisions=(paid,)), None, policy, 0)[0].reason
    )


def test_bound_denial_by_named_approver_is_clean():
    row, policy = action(3, outcome="deny", actor="morgan")
    assert scan(LaunchLogs(decisions=(row,)), None, policy, 0) == ()
    result = inspect_tier3(
        LaunchLogs(decisions=(row,)), policy, Period(start=2, end=32)
    )
    assert result.requests == 1 and result.verdict == "ok"
    assert result.period.end - result.period.start == 30


def test_agent_denial_does_not_count_as_tier3_evidence():
    row, policy = action(3, outcome="deny", actor="agent", reviewed=False)
    result = inspect_tier3(
        LaunchLogs(decisions=(row,)), policy, Period(start=2, end=32)
    )
    assert result.verdict == "hold" and result.decided_otherwise == (
        row.request.request_id,
    )


@pytest.mark.parametrize("failure", ["order_not_found", "ambiguous_order"])
def test_core_request_only_review_is_pending_and_never_a_terminal_tier3_denial(failure):
    from test_human_oversight_risk_component import request_only_fixture

    from refund_agent import contracts as c
    from refund_agent.launch.capstone_blueprint_scan import EffectiveDecision

    data, tools, trace = request_only_fixture(failure=failure)
    request = data.requests[0]
    review, = tools._named_reviews().values()
    assert review.proposal is None and review.request == request
    assert not tools.proposals and not tools.payments
    row = EffectiveDecision(
        record_id="request-only-referral", day=2, requested_day=2,
        request=request, decision=trace.decision, at=request.at,
        proposal=None, actor="refund-agent",
    )
    logs = LaunchLogs(decisions=(row,))
    policy = data.policy_at(request.at)
    result = inspect_tier3(logs, policy, Period(start=2, end=32), request_ids=(request.request_id,))
    assert result.requests == 1
    assert result.verdict == "hold" and result.pending == (request.request_id,)
    assert result.decided_otherwise == () and scan(logs, None, policy, 0) == ()
    denied = row.model_copy(update={"decision": row.decision.model_copy(update={
        "outcome": c.Outcome.DENY, "policy_citation": f"refund_cap_v4@{policy.version}#eligibility",
    })})
    logs = LaunchLogs(decisions=(denied,))
    result = inspect_tier3(logs, policy, Period(start=2, end=32))
    assert result.verdict == "hold" and result.decided_otherwise == (request.request_id,)
    event, = scan(logs, None, policy, 0)
    assert event.condition_id == "tier3_without_named_approver"


def test_reviewer_cannot_approve_another_proposal():
    row, policy = action(outcome="deny")
    bad = row.model_copy(
        update={
            "review": row.review.model_copy(
                update={"proposal_fingerprint": "other"}
            )
        }
    )
    assert len(scan(LaunchLogs(decisions=(bad,)), None, policy, 0)) == 1


@pytest.mark.parametrize("tier", [2, 3])
def test_payment_with_receipt_bound_approval_is_clean(tier):
    row, policy = payment_action(tier)
    assert scan(LaunchLogs(decisions=(row,)), None, policy, 0) == ()


@pytest.mark.parametrize("tier", [2, 3])
@pytest.mark.parametrize(
    "fault", ["absent", "other_identity", "before_review"]
)
def test_actual_receipt_requires_its_approval_before_payment(tier, fault):
    row, policy = payment_action(tier)
    assert scan(LaunchLogs(decisions=(row,)), None, policy, 0) == ()
    if fault == "absent":
        receipt = row.receipt.model_copy(update={"approvals": ()})
    elif fault == "other_identity":
        other = row.approval.model_copy(update={"approval_id": "another"})
        receipt = row.receipt.model_copy(update={"approvals": (other,)})
    else:
        receipt = row.receipt.model_copy(
            update={"at": row.review.at - timedelta(seconds=1)}
        )
    bad = row.model_copy(update={"receipt": receipt})
    events = scan(LaunchLogs(decisions=(bad,)), None, policy, 0)
    assert len(events) == 1
    assert events[0].condition_id == (
        "tier2_unbound_approval"
        if tier == 2
        else "tier3_without_named_approver"
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("request_id", "another"),
        ("proposal_id", "another"),
        ("order_id", "another"),
        ("amount_cents", 5001),
        ("reason", "another"),
        ("policy_version", "another"),
        ("reviewer", "another"),
        ("approved_at", timedelta(seconds=1)),
    ],
)
def test_each_payment_approval_binding_is_checked(field, value):
    row, policy = payment_action()
    assert scan(LaunchLogs(decisions=(row,)), None, policy, 0) == ()
    if field == "approved_at":
        value = row.approval.approved_at + value
    approval = row.approval.model_copy(update={field: value})
    bad = row.model_copy(
        update={
            "approval": approval,
            "receipt": row.receipt.model_copy(
                update={"approvals": (approval,)}
            ),
        }
    )
    events = scan(LaunchLogs(decisions=(bad,)), None, policy, 0)
    assert len(events) == 1
    assert "no approval bound" in events[0].reason
