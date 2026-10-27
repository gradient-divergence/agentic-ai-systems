"""Chapter-37 side of the partner contracts; no partner implementation is shipped."""

from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from hmac import compare_digest, new

import pytest

from refund_agent import contracts as c
from refund_agent.agent import Agent
from refund_agent.data.generate import Dataset
from refund_agent.oversight.human_oversight_risk_deadline import on_deadline
from refund_agent.oversight.human_oversight_risk_desk import (
    ProposedExplanation,
    ReviewDesk,
    ReviewerContext,
)
from refund_agent.oversight.human_oversight_risk_lifecycle import expiry_risk, respond_by
from refund_agent.oversight.human_oversight_risk_partners import appeal_route
from refund_agent.tools import Tools

AT = datetime(2026, 10, 2, 18, 20, tzinfo=timezone.utc)
REVIEW_AT = datetime(2026, 10, 6, 10, 15, tzinfo=timezone.utc)


@dataclass(frozen=True)
class TestCredential:
    __test__ = False
    credential_id: str
    subject: str
    role: str
    not_before: datetime
    expires_at: datetime
    seal: str


def _seal(subject, role, before, expiry):
    payload = repr((subject, role, before, expiry)).encode()
    return new(b"human-oversight-risk-test-only", payload, sha256).hexdigest()


def issue(reviewer, at):
    role = "named_approver" if reviewer in ("morgan", "deputy") else "reviewer"
    expiry = at + timedelta(minutes=30)
    return TestCredential("human-oversight-risk-test-" + reviewer, reviewer, role, at, expiry,
                          _seal(reviewer, role, at, expiry))


def authenticate(credential, reviewer, at):
    if (not isinstance(credential, TestCredential) or not compare_digest(credential.seal, _seal(
            credential.subject, credential.role, credential.not_before, credential.expires_at))
            or not credential.not_before <= at < credential.expires_at
            or credential.subject != reviewer):
        raise ValueError("credential_reviewer")
    return ReviewerContext(credential_id=credential.credential_id, reviewer=credential.subject,
                           role=credential.role)


class AuthenticatedTestTools(Tools):
    def record_approval(self, approval, credential=None, at=None):
        authenticate(credential, approval.reviewer, at)
        return super().record_approval(approval)


def fixture(*, expired=False, denial=False, amount=12000, reviewer="alex"):
    policy = c.Policy(version="2026-09-01", effective_at=datetime(2026, 9, 1, tzinfo=timezone.utc),
                      return_days=30, cap_cents=100000, text="Constructed test policy")
    delivered = AT - timedelta(days=29 if expired else 1)
    order = c.Order(order_id="human-oversight-risk-order", reference="human-oversight-risk-order",
                    purchased_at=delivered - timedelta(days=2), observed_at=AT - timedelta(seconds=60),
                    items=[c.Item(name="physical lamp", quantity=1, unit_cents=150000)],
                    paid_cents=150000, payment_id="payment-fixture")
    request = c.Request(request_id="human-oversight-risk-request", requester_id="customer-fixture",
                        order_reference=order.order_id, at=AT, amount_cents=amount,
                        reason="not_delivered" if denial else "damaged")
    shipment = c.Shipment(order_id=order.order_id, observed_at=AT - timedelta(seconds=120),
                          events=[c.ShipmentEvent(at=delivered, status="delivered")])
    data = Dataset(orders=[order], requests=[request], shipments=[shipment], policies=[policy],
                   retrievals=[c.PolicySnapshot(request_id=request.request_id, version=policy.version,
                                                observed_at=AT - timedelta(hours=1))], correct=[],
                   reviewers=[c.Reviewer(reviewer_id="alex", name="Alex", role="reviewer"),
                              c.Reviewer(reviewer_id="riley", name="Riley", role="reviewer"),
                              c.Reviewer(reviewer_id="morgan", name="Morgan", role="named_approver"),
                              c.Reviewer(reviewer_id="deputy", name="Deputy", role="named_approver")])
    tools = AuthenticatedTestTools(data)
    trace = Agent(tools).run(request)
    state = trace.steps[-1].state.model_copy(deep=True)
    if not tools.proposals:
        assert tools.call(c.ToolCall(name="refund.propose", arguments=c.ToolArguments(
            amount_cents=amount, reason=request.reason)), state).ok
        assert tools.call(c.ToolCall(name="approval.request"), state).ok
    proposal = next(iter(tools.proposals.values()))
    from refund_agent.rules import draft

    proposed = draft(state)
    explanation = explain_fixture(state, proposed) if proposed.outcome == c.Outcome.DENY else None
    desk = ReviewDesk(tools, authenticate, decision_owner="decision-owner", denial_explainer=explain_fixture, mode="execute")
    ticket = desk.open(proposal, proposed, explanation, reviewer=reviewer)
    return data, tools, desk, ticket


def explain_fixture(state, proposed):
    return ProposedExplanation(request_id=state.request.request_id, policy_version=state.policy.version,
                                proposed=True, rendered="Constructed partner record: " + proposed.reason)


def freshen(tools, at, *, shipment_stale=False):
    data = tools.data
    tools.data = data.model_copy(update={
        "orders": [o.model_copy(update={"observed_at": at - timedelta(seconds=60)}) for o in data.orders],
        "shipments": [s.model_copy(update={"observed_at": at - timedelta(seconds=1801 if shipment_stale else 120)})
                      for s in data.shipments],
        "retrievals": [r.model_copy(update={"observed_at": at - timedelta(hours=1),
                                            "version": data.policy_at(at).version}) for r in data.retrievals],
    })


def test_deadline_clock_has_no_decision_or_receipt_effect():
    _, tools, desk, ticket = fixture()
    before = (tools.ledger, tools.audits, dict(tools._approvals))
    first = on_deadline(ticket, ticket.due_at)
    assert first.assignee == "morgan" and first.closed_at is None
    assert first.due_at == datetime(2026, 10, 6, 13, tzinfo=timezone.utc)
    assert first.notices[0].due_at == first.due_at
    assert on_deadline(first, ticket.due_at) == first
    second = on_deadline(first, first.due_at)
    assert second.assignee == "decision-owner" and second.closed_at is None
    assert second.due_at == datetime(2026, 10, 6, 17, tzinfo=timezone.utc)
    assert (tools.ledger, tools.audits, dict(tools._approvals)) == before
    assert desk.tickets[ticket.review_id] == ticket


def test_monday_at_0300_is_due_tuesday_not_monday():
    monday = datetime(2026, 10, 5, 3, tzinfo=timezone.utc)
    assert respond_by(monday) == datetime(2026, 10, 6, 17, tzinfo=timezone.utc)


def test_tier3_clock_escalates_only_to_owner_who_cannot_decide_it():
    _, tools, desk, ticket = fixture(amount=65000, denial=True)
    result = on_deadline(ticket, ticket.due_at)
    assert result.assignee == "decision-owner" and result.closed_at is None
    desk.tickets[ticket.review_id] = result
    freshen(tools, REVIEW_AT)
    before = (tools.ledger, tools.audits, dict(desk.tickets), dict(tools._approvals))
    with pytest.raises(ValueError, match="credential_reviewer|named_approver_required"):
        desk.approve(ticket.review_id, "decision-owner", issue("decision-owner", REVIEW_AT), REVIEW_AT)
    assert (tools.ledger, tools.audits, desk.tickets, dict(tools._approvals)) == before


@pytest.mark.parametrize("credential", [None, "another_person", "altered"])
def test_invalid_review_credentials_write_nothing(credential):
    _, tools, desk, ticket = fixture()
    if credential == "another_person":
        credential = issue("riley", REVIEW_AT)
    elif credential == "altered":
        credential = replace(issue("alex", REVIEW_AT), subject="riley")
    before = (tools.audits, tools.ledger, dict(desk.tickets), dict(tools._approvals))
    with pytest.raises(ValueError):
        desk.record_refusal(ticket.review_id, "alex", credential, "bad evidence", REVIEW_AT)
    assert (tools.audits, tools.ledger, desk.tickets, dict(tools._approvals)) == before


def test_approval_after_escalation_uses_fresh_gate_and_pays():
    _, tools, desk, ticket = fixture()
    desk.tickets[ticket.review_id] = on_deadline(ticket, ticket.due_at)
    freshen(tools, REVIEW_AT)
    result = desk.approve(ticket.review_id, "morgan", issue("morgan", REVIEW_AT), REVIEW_AT)
    assert result.request_state == "paid" and result.receipt_id
    assert len(tools.payments) == 1 and tools.payments[0].amount_cents == 12000
    assert tools.payments[0].evidence.freshness["shipment"] < 121


def test_stale_fresh_read_refuses_an_authorized_review():
    _, tools, desk, ticket = fixture()
    freshen(tools, REVIEW_AT, shipment_stale=True)
    result = desk.approve(ticket.review_id, "morgan", issue("morgan", REVIEW_AT), REVIEW_AT)
    assert result.request_state == "waiting" and result.reason == "stale_shipment"
    assert tools.payments == [] and tools._approvals
    assert desk.tickets[ticket.review_id].events[-1].reason == "execution_refused:stale_shipment"


def test_authorized_review_across_expiry_is_refused_and_escalation_is_visible():
    _, tools, desk, ticket = fixture(expired=True)
    proposal = desk.packets[ticket.review_id][0]
    risk = expiry_risk([proposal], ticket.due_at)
    assert risk[0]["expired"] and risk[0]["hours_left"] < 0
    desk.tickets[ticket.review_id] = on_deadline(ticket, ticket.due_at)
    freshen(tools, REVIEW_AT)
    result = desk.approve(ticket.review_id, "morgan", issue("morgan", REVIEW_AT), REVIEW_AT)
    assert result.request_state == "waiting" and result.reason == "outside_return_window"
    assert tools.payments == []
    assert desk.tickets[ticket.review_id].assignee == "decision-owner"
    assert desk.tickets[ticket.review_id].events[-1].reason == "execution_refused:outside_return_window"
    assert desk.waiting_expiry(REVIEW_AT)[0]["expired"]


def test_refusal_is_closed_and_its_followup_is_open_and_unpaid():
    _, tools, desk, ticket = fixture()
    result = desk.record_refusal(ticket.review_id, "alex", issue("alex", REVIEW_AT), "balance wrong", REVIEW_AT)
    assert desk.tickets[ticket.review_id].outcome == "refused"
    assert desk.tickets[result.new_review_id].assignee == "morgan"
    assert desk.tickets[result.new_review_id].closed_at is None and not tools.payments
    assert desk.work[-1]["minutes"] == 15


@pytest.mark.parametrize("reviewer", ["riley", "morgan"])
@pytest.mark.parametrize("action", ["record_refusal", "escalate"])
def test_staffed_reviewers_can_refuse_and_escalate_without_paying(reviewer, action):
    _, tools, desk, ticket = fixture(reviewer=reviewer)
    result = getattr(desk, action)(ticket.review_id, reviewer, issue(reviewer, REVIEW_AT),
                                 "read the balance", REVIEW_AT)
    followup = desk.tickets[result.new_review_id or result.review_id]
    assert followup.assignee == ("decision-owner" if reviewer == "morgan" else "morgan")
    assert len(followup.path) == len(set(followup.path))
    assert followup.closed_at is None and not tools.payments


def test_approving_a_proposed_denial_confirms_it_and_never_pays():
    _, tools, desk, ticket = fixture(denial=True)
    freshen(tools, REVIEW_AT)
    result = desk.approve(ticket.review_id, "alex", issue("alex", REVIEW_AT), REVIEW_AT)
    assert result.action == "approved" and result.request_state == "denial_confirmed"
    assert result.receipt_id is None and not tools.payments and not tools._approvals
    assert desk.tickets[ticket.review_id].events[-1].kind == "denial_confirmation"


@pytest.mark.parametrize("field,value", [("request_id", "another-request"), ("amount_cents", 12001), ("tier", c.Tier.AGENT)])
def test_mixed_review_packet_is_rejected_before_intake_writes(field, value):
    _, _, desk, ticket = fixture(denial=True)
    proposal, proposed, explanation = desk.packets[ticket.review_id]
    before = dict(desk.tickets)
    with pytest.raises(ValueError, match="packet_mismatch"):
        desk.open(proposal, proposed.model_copy(update={field: value}), explanation, reviewer="alex")
    assert desk.tickets == before


@pytest.mark.parametrize("reviewer", ["alex", "riley", "morgan"])
def test_modification_opens_a_new_unapproved_proposal_and_counts_both_jobs(reviewer):
    _, tools, desk, ticket = fixture(reviewer=reviewer)
    freshen(tools, REVIEW_AT)
    request = desk.packets[ticket.review_id][0].request.model_copy(update={
        "request_id": "human-oversight-risk-modified", "amount_cents": 10000, "at": REVIEW_AT})
    result = desk.modify(ticket.review_id, reviewer, issue(reviewer, REVIEW_AT), request, REVIEW_AT)
    assert result.request_state == "renewed_review"
    assert desk.packets[result.new_review_id][0].proposal_id != desk.packets[ticket.review_id][0].proposal_id
    assert sum(row["minutes"] for row in desk.work) == 20 and not tools.payments
    assert desk.tickets[result.new_review_id].closed_at is None
    renewed = desk.tickets[result.new_review_id]
    assert renewed.assignee == reviewer and len(renewed.path) == len(set(renewed.path))
    assert renewed.path[-1] == "decision-owner"
    escalated = on_deadline(renewed, renewed.due_at)
    assert escalated.assignee == ("decision-owner" if reviewer == "morgan" else "morgan")


def test_appeals_require_an_independent_decider_and_named_tier3_authority():
    data, _, _, _ = fixture()
    assert appeal_route(request_id="r", tier=c.Tier.REVIEWER, first_reviewer="alex",
                        reviewers=data.reviewers, policy=data.policies[0]) != "alex"
    with pytest.raises(ValueError):
        appeal_route(request_id="r", tier=c.Tier.NAMED_APPROVER, first_reviewer="morgan",
                     reviewers=data.reviewers, policy=data.policies[0], appeals_approver="deputy",
                     independent_assessment={"request_id": "r", "reviewer": "alex", "text": "read independently"})
    policy = data.policies[0].model_copy(update={"named_approver": "deputy"})
    assert appeal_route(request_id="r", tier=c.Tier.NAMED_APPROVER, first_reviewer="morgan",
                        reviewers=data.reviewers, policy=policy, appeals_approver="deputy",
                        independent_assessment={"request_id": "r", "reviewer": "alex", "text": "read independently"}) == "deputy"


@pytest.mark.parametrize("denial", [False, True])
def test_policy_version_during_absence_renews_each_tier3_proposal_before_deputy_decides(denial):
    data, tools, desk, ticket = fixture(amount=65000, denial=denial)
    policy = data.policies[0].model_copy(update={
        "version": "2026-10-05", "effective_at": datetime(2026, 10, 5, tzinfo=timezone.utc),
        "named_approver": "deputy",
    })
    tools.data = tools.data.model_copy(update={"policies": data.policies + [policy]})
    freshen(tools, REVIEW_AT)
    result = desk.approve(ticket.review_id, "deputy", issue("deputy", REVIEW_AT), REVIEW_AT)
    assert result.request_state == "renewed_review"
    assert result.new_review_id != ticket.review_id
    assert desk.packets[result.new_review_id][0].evidence.policy.version == policy.version
    assert not tools.payments
    decided = desk.approve(result.new_review_id, "deputy", issue("deputy", REVIEW_AT), REVIEW_AT)
    assert decided.request_state == ("denial_confirmed" if denial else "paid")
    assert len(tools.payments) == (0 if denial else 1)


def print_constructed_review_records():
    from demos.human_oversight_risk_demo import print_review_records, say

    say("TEST-CONSTRUCTED partner records; assembled execution PENDING.")
    say("Chapter 35: authenticate -> ReviewerContext, plus")
    say("ScopedTools.record_approval(approval, credential=..., at=...).")
    say("Chapter 36: generated/replayed DenialExplanation -> ProposedExplanation.")
    say("Test authentication/explanation functions are confined to tests.")
    for expired, stale in ((False, False), (False, True), (True, False)):
        _, tools, desk, ticket = fixture(expired=expired)
        desk.tickets[ticket.review_id] = on_deadline(ticket, ticket.due_at)
        freshen(tools, REVIEW_AT, shipment_stale=stale)
        result = desk.approve(ticket.review_id, "morgan", issue("morgan", REVIEW_AT), REVIEW_AT)
        say("Post-deadline authorized review:", "expiry" if expired else "stale" if stale else "fresh")
        print_review_records(desk, result, REVIEW_AT)
    for denial in (False, True):
        data, tools, desk, ticket = fixture(amount=65000, denial=denial)
        desk.tickets[ticket.review_id] = on_deadline(ticket, ticket.due_at)
        policy = data.policies[0].model_copy(update={
            "version": "2026-10-05", "effective_at": datetime(2026, 10, 5, tzinfo=timezone.utc),
            "named_approver": "deputy"})
        tools.data = tools.data.model_copy(update={"policies": data.policies + [policy]})
        freshen(tools, REVIEW_AT)
        renewed = desk.approve(ticket.review_id, "deputy", issue("deputy", REVIEW_AT), REVIEW_AT)
        say("Deputy during absence:", "denial" if denial else "payment")
        say("Old/new policy:", desk.packets[ticket.review_id][0].evidence.policy.version, policy.version)
        say("Old/new proposal:", desk.packets[ticket.review_id][0].proposal_id,
            desk.packets[renewed.new_review_id][0].proposal_id)
        print_review_records(desk, renewed, REVIEW_AT)
        result = desk.approve(renewed.new_review_id, "deputy", issue("deputy", REVIEW_AT), REVIEW_AT)
        print_review_records(desk, result, REVIEW_AT)


def print_constructed_screen():
    from demos.human_oversight_risk_demo import print_screen

    _, _, desk, ticket = fixture(denial=True, reviewer="riley")
    proposal, proposed, explanation = desk.packets[ticket.review_id]
    proof = authenticate(issue("riley", ticket.opened_at), "riley", ticket.opened_at)
    print_screen(ticket, proposal, proposed, explanation, proof)


# region book:fnd.human-oversight-risk.review-records-01 output
if __name__ == "__main__":
    print_constructed_review_records()
# endregion book:fnd.human-oversight-risk.review-records-01

# region book:fnd.human-oversight-risk.screen-01 output
if __name__ == "__main__":
    print_constructed_screen()
# endregion book:fnd.human-oversight-risk.screen-01
