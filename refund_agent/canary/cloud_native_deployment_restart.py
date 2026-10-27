"""Reconcile the synchronous ledger and audit log; never execute a payment."""

from datetime import timedelta

from refund_agent import contracts as c
from refund_agent.canary.cloud_native_deployment_store import PersistentTools
from refund_agent.data.generate import Dataset
from refund_agent.rules import decision


def recorded_sources(record, at):
    evidence = record.evidence
    request = record.request
    if evidence is None or request is None:
        raise ValueError("restart requires recorded request and evidence")
    return Dataset(
        requests=[request], orders=[evidence.order], policies=[evidence.policy], correct=[],
        shipments=[c.Shipment(order_id=evidence.order.order_id, events=evidence.shipment_events,
                              observed_at=at - timedelta(seconds=evidence.freshness["shipment"]))],
        retrievals=[c.PolicySnapshot(request_id=request.request_id, version=evidence.policy.version,
                                     observed_at=at - timedelta(seconds=evidence.freshness["policy"]))],
    )


def reconcile(tools: PersistentTools):
    if tools.canary_store is None:
        raise ValueError("restart requires the file-backed store")
    payments = {p.request.request_id: p for p in tools.payments if p.request is not None}
    audits = tuple(tools.audits.values())
    confirmed = {a.receipt_id for a in audits if a.event == "confirmation"}
    closed = {a.decision.request_id for a in audits if a.failure_detail == "intent_without_receipt"}
    queued, confirmations = [], []
    for receipt in payments.values():
        if receipt.receipt_id not in confirmed:
            if receipt.request is None:
                raise ValueError("receipt lacks its request")
            tools.data = recorded_sources(receipt, receipt.at)
            state = tools.completed_state(receipt.request)
            if state is None:
                raise ValueError("receipt cannot reconstruct its completed state")
            state.failure_detail = "confirmed_at_restart"
            with tools._boundary(state, "run"):
                tools.record_decision(state)
            confirmations.append(receipt.receipt_id)
            if tools.canary_store.latest("reconciliation", receipt.receipt_id) is None:
                tools.canary_store.append("reconciliation", receipt.receipt_id, {
                    "payment_id": receipt.receipt_id, "reason": "confirmed_at_restart",
                    "for_person": True,
                })
    for intent in audits:
        rid = intent.decision.request_id
        if intent.event != "intent" or rid in payments or rid in closed:
            continue
        proposal = tools.proposals.get(intent.proposal_id or "")
        if proposal is None:
            raise ValueError("intent cannot be reconciled without its proposal")
        state = c.State(request=proposal.request, **proposal.evidence.model_dump(),
                        proposal_id=proposal.proposal_id, execution_at=intent.at,
                        policy_version=proposal.evidence.policy.version,
                        failure_detail="intent_without_receipt")
        state.decision = decision(state, c.Outcome.REFER, "intent_without_receipt")
        tools.data = recorded_sources(proposal, intent.at)
        with tools._boundary(state, "run"):
            tools.record_decision(state)
        queued.append(proposal.request)
        closed.add(rid)
    return {"confirmed": tuple(confirmations), "unpaid": tuple(queued)}
