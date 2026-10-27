from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from test_human_oversight_risk_lifecycle import (
    authenticate,
    explain_fixture,
    fixture,
    freshen,
    issue,
)

from refund_agent import contracts as c
from refund_agent.oversight.human_oversight_risk_component import SELECTOR
from refund_agent.oversight.human_oversight_risk_partners import observability_referrals
from refund_agent.oversight.human_oversight_risk_screen import screen
from refund_agent.release.candidate import build_candidate, build_tools, run_request
from refund_agent.release.suite import build_suite

BASE = build_candidate(build_suite())


def test_selected_component_is_fingerprinted_and_requires_both_partners():
    suite = build_suite(components=[SELECTOR])
    candidate = build_candidate(suite)
    part = candidate.tools_composition["components"][0]
    assert part["name"] == SELECTOR["name"] and "version" in part
    assert candidate.fingerprint != BASE.fingerprint
    data, _, _, _ = fixture()
    with pytest.raises(ValueError, match="authenticated review"):
        build_tools(data, candidate, "execute")


@pytest.mark.parametrize("reviewer", ["alex", "riley", "morgan"])
def test_tier2_denial_is_a_referral_with_its_packet_explanation_and_review(reviewer):
    data, _, _, _ = fixture(denial=True)
    candidate = build_candidate(build_suite(components=[SELECTOR]))
    tools = build_tools(data, candidate, "execute", reviewer_authenticate=authenticate,
                        denial_explainer=explain_fixture, decision_owner="decision-owner",
                        assigned_reviewer=reviewer)
    trace = run_request(candidate, tools, data.requests[0])
    assert trace.decision.outcome == c.Outcome.REFER and trace.decision.reason == "already_delivered"
    assert len(tools.review_desk.tickets) == 1 and not tools.payments
    proposed, explanation = next(iter(tools.review_desk.packets.values()))[1:]
    assert proposed.outcome == c.Outcome.DENY and explanation.proposed
    assert all(row.decision.outcome != c.Outcome.DENY for row in tools.audits.values())
    ticket = next(iter(tools.review_desk.tickets.values()))
    assert ticket.assignee == reviewer
    assert ticket.path == (("morgan", "decision-owner") if reviewer == "morgan" else
                           (reviewer, "morgan", "decision-owner"))
    freshen(tools, ticket.opened_at)
    result = tools.review_desk.approve(ticket.review_id, reviewer,
                                     issue(reviewer, ticket.opened_at), ticket.opened_at)
    assert result.request_state == "denial_confirmed" and not tools.payments


def test_unselected_candidate_keeps_its_denial():
    data, _, _, _ = fixture(denial=True)
    tools = build_tools(data, BASE, "execute")
    assert run_request(BASE, tools, data.requests[0]).decision.outcome == c.Outcome.DENY


def test_selected_shadow_routing_changes_no_live_review_or_payment_store():
    data, _, _, _ = fixture(denial=True)
    candidate = build_candidate(build_suite(components=[SELECTOR]))
    tools = build_tools(data, candidate, "shadow", reviewer_authenticate=authenticate,
                        denial_explainer=explain_fixture, decision_owner="decision-owner",
                        assigned_reviewer="alex")
    trace = run_request(candidate, tools, data.requests[0])
    assert trace.decision.outcome == c.Outcome.REFER
    assert tools.shadow_reviews and tools.review_desk.tickets
    assert not tools.reviews and not tools.payments and not tools.ledger
    ticket = next(iter(tools.review_desk.tickets.values()))
    with pytest.raises(ValueError, match="executed_lane"):
        tools.review_desk.approve(ticket.review_id, "alex", issue("alex", ticket.opened_at), ticket.opened_at)


def request_only_fixture(mode="execute", failure="order_not_found"):
    data, _, _, _ = fixture(amount=65000)
    request = data.requests[0]
    if failure == "order_not_found":
        request = request.model_copy(update={"order_reference": "unavailable-order"})
        data = data.model_copy(update={"requests": [request]})
    else:
        other = data.orders[0].model_copy(update={
            "order_id": "another-order", "reference": request.order_reference,
        })
        data = data.model_copy(update={"orders": [*data.orders, other]})
    candidate = build_candidate(build_suite(components=[SELECTOR]))
    tools = build_tools(
        data, candidate, mode, reviewer_authenticate=authenticate,
        denial_explainer=explain_fixture, decision_owner="decision-owner",
        assigned_reviewer="alex",
    )
    return data, tools, run_request(candidate, tools, request)


@pytest.mark.parametrize("mode", ["execute", "shadow"])
@pytest.mark.parametrize("failure", ["order_not_found", "ambiguous_order"])
def test_core_request_only_tier3_review_reaches_the_desk_without_payment(mode, failure):
    from datetime import timedelta

    from refund_agent.oversight.human_oversight_risk_deadline import on_deadline

    data, tools, trace = request_only_fixture(mode, failure)
    request = data.requests[0]
    assert trace.decision.outcome == c.Outcome.REFER and trace.decision.reason == failure
    review, = tools._named_reviews().values()
    assert review.proposal is None and review.request == request
    assert review.named_approver == "morgan" and not tools.proposals
    desk = tools.review_desk
    ticket, = desk.tickets.values()
    assert ticket.path == ("morgan", "decision-owner")
    assert desk.packets[ticket.review_id] == (review, trace.decision, None)
    with pytest.raises(ValueError, match="request_only_review_packet"):
        desk.open(review, trace.decision.model_copy(update={"outcome": c.Outcome.APPROVE}), reviewer="alex")
    with pytest.raises(ValueError, match="request_only_review_packet"):
        desk.open(review.model_copy(update={"review_id": "not-core-issued"}), trace.decision, reviewer="alex")
    escalated = on_deadline(ticket, ticket.due_at)
    assert escalated.assignee == "decision-owner" and escalated.closed_at is None
    assert not tools.payments and not tools.ledger and not tools._approvals
    at = ticket.opened_at + timedelta(minutes=1)
    if mode == "shadow":
        assert not tools.reviews and tools.shadow_reviews
        with pytest.raises(ValueError, match="executed_lane"):
            desk.approve(ticket.review_id, "morgan", issue("morgan", at), at)
    else:
        before = (tools.audits, dict(desk.tickets))
        with pytest.raises(ValueError, match="credential_reviewer"):
            desk.approve(ticket.review_id, "alex", issue("alex", at), at)
        assert (tools.audits, desk.tickets) == before
        result = desk.approve(ticket.review_id, "morgan", issue("morgan", at), at)
        assert result.request_state == "waiting" and result.reason == failure
        assert desk.tickets[ticket.review_id].closed_at is None and tools.reviews[review.review_id].open
        assert {key: tools.audits[key] for key in before[0]} == before[0]
        assert all(row.decision.outcome == c.Outcome.REFER and not row.money_moved for row in tools.audits.values())
        assert desk.waiting_expiry(at) == []
    assert not tools.payments and not tools.ledger and not tools._approvals


def test_request_only_review_needs_a_new_evidence_bound_review_after_lookup_recovers():
    from datetime import timedelta

    data, tools, _ = request_only_fixture()
    desk = tools.review_desk
    ticket, = desk.tickets.values()
    at = ticket.opened_at + timedelta(minutes=1)
    order = tools.data.orders[0].model_copy(update={"reference": data.requests[0].order_reference})
    tools.data = tools.data.model_copy(update={"orders": [order]})
    freshen(tools, at)
    result = desk.approve(ticket.review_id, "morgan", issue("morgan", at), at)
    assert result.action == "modified" and result.request_state == "renewed_review"
    renewed = desk.tickets[result.new_review_id]
    proposal, proposed, explanation = desk.packets[renewed.review_id]
    assert proposal.request == data.requests[0] and proposal.evidence.order == tools.data.orders[0]
    assert renewed.path == ("morgan", "decision-owner") and renewed.closed_at is None
    assert not tools.reviews[desk.packets[ticket.review_id][0].review_id].open
    assert tools.reviews["review-" + proposal.proposal_id].open
    assert [row["minutes"] for row in desk.work] == [5, 30]
    assert not tools.payments and not tools.ledger and not tools._approvals
    assert proposed.outcome == c.Outcome.APPROVE and explanation is None

    candidate = build_candidate(build_suite(components=[SELECTOR]))
    trace = run_request(candidate, tools, data.requests[0], at=at)
    assert trace.decision.outcome == c.Outcome.REFER and trace.receipt is None
    assert not tools.reviews[desk.packets[ticket.review_id][0].review_id].open
    assert tools.reviews["review-" + proposal.proposal_id].open
    assert desk.tickets[ticket.review_id].closed_at is not None
    assert desk.tickets[renewed.review_id].closed_at is None
    assert not tools.payments and not tools.ledger and not tools._approvals


def test_request_only_refusal_keeps_the_core_request_open_for_its_named_approver():
    from datetime import timedelta

    _, tools, _ = request_only_fixture()
    desk = tools.review_desk
    ticket, = desk.tickets.values()
    at = ticket.opened_at + timedelta(minutes=1)
    result = desk.record_refusal(ticket.review_id, "morgan", issue("morgan", at), "needs evidence", at)
    assert result.action == "refused" and result.request_state == "waiting"
    followup = desk.tickets[result.new_review_id]
    assert followup.closed_at is None and followup.assignee == "decision-owner"
    review = desk.packets[followup.review_id][0]
    assert tools.reviews[review.review_id].open and desk.waiting_expiry(at) == []
    assert not tools.payments and not tools.ledger and not tools._approvals


def test_screen_requires_the_explanation_and_the_reviewing_identity():
    _, _, desk, ticket = fixture(denial=True)
    proposal, proposed, explanation = desk.packets[ticket.review_id]
    proof = authenticate(issue("alex", ticket.opened_at), "alex", ticket.opened_at)
    result = screen(ticket, proposal, proposed, explanation, proof)
    assert len(result) == 10 and result[4].startswith("5 Explanation (proposed)")
    assert "no money moves" in result[5]
    for missing in (None, {}):
        with pytest.raises((ValueError, TypeError, AttributeError)):
            screen(ticket, proposal, proposed, missing, proof)
        with pytest.raises((ValueError, TypeError, AttributeError)):
            screen(ticket, proposal, proposed, explanation, missing)
    unassigned = authenticate(issue("riley", ticket.opened_at), "riley", ticket.opened_at)
    with pytest.raises(ValueError, match="own credential"):
        screen(ticket, proposal, proposed, explanation, unassigned)


def test_observability_join_deduplicates_runs_and_does_not_admit_pending_holds():
    data, _, _, _ = fixture()
    request = data.requests[0]
    at = datetime(2026, 10, 6, tzinfo=timezone.utc)
    base = dict(name="run", status="refused", reason="stale_shipment", trace_id="trace-" + request.request_id,
                requester_id=request.requester_id, end=at)
    spans = [SimpleNamespace(**base, run_id="run-1"), SimpleNamespace(**base, run_id="run-2")]
    result = observability_referrals(spans, [request], {"stale_shipment": "system"})
    assert len(result) == 1 and result[0].at == at
    spans[0].reason = "shipment_incomplete"
    assert not observability_referrals(spans[:1], [request], {"shipment_incomplete": "hold"})
    with pytest.raises(ValueError, match="unclassified"):
        observability_referrals(spans, [request], {})
