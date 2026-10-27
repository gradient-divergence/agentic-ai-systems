"""Canary boundaries, append storage, persisted authority and arrival ordering."""

import json
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest

from refund_agent import contracts as c
from refund_agent.canary.cloud_native_deployment_binding import CanaryTools
from refund_agent.canary.cloud_native_deployment_component import STORE_PATH
from refund_agent.canary.cloud_native_deployment_plan import CanaryPlan, in_cohort, sign_plan
from refund_agent.canary.cloud_native_deployment_rehearsal import (
    fixture,
    paid_twice,
    prepare,
    rehearsal_session,
)
from refund_agent.canary.cloud_native_deployment_stop import (
    CanarySession,
    SettledEvidence,
    StopRule,
)
from refund_agent.canary.cloud_native_deployment_store import AppendStore
from refund_agent.release.candidate import run_request
from refund_agent.shadow.tools import ShadowTools
from refund_agent.tools import ReconciliationError

AT = datetime(2026, 10, 12, tzinfo=timezone.utc)


@pytest.fixture
def session():
    plan = sign_plan(CanaryPlan(canary_id="cloud-native-deployment-test", candidate_fingerprint="candidate",
                               suite_fingerprint="suite", rollback_target_fingerprint="target",
                               requirement_ref="window", rehearsal_ref="rehearsal"), "morgan", AT)
    store = AppendStore(STORE_PATH, "cloud-native-deployment-test-" + uuid4().hex)
    return rehearsal_session(plan, store, AT)


def tools_for(session, tag, at=AT, amount=4999, **updates):
    data, request = fixture(session.plan.canary_id, tag, at, amount=amount)
    request = request.model_copy(update=updates)
    data = data.model_copy(update={"requests": [request]})
    tools = CanaryTools(data, canary_session=session, shadow_type=ShadowTools, shadow_kwargs={})
    return tools, request, prepare(tools, request, at)


def execute(tools, request, state, amount=None):
    return tools.call(c.ToolCall(name="refund.execute", arguments=c.ToolArguments(
        amount_cents=request.amount_cents if amount is None else amount,
        idempotency_key=f"execute-{request.request_id}")), state)


def test_cohort_is_fixed_and_new_canary_changes_it():
    cohort = [in_cohort("cloud-native-deployment-a", f"person-{i}") for i in range(10000)]
    assert 400 <= sum(cohort) <= 600
    assert cohort == [in_cohort("cloud-native-deployment-a", f"person-{i}") for i in range(10000)]
    assert cohort != [in_cohort("cloud-native-deployment-b", f"person-{i}") for i in range(10000)]


def test_plan_cap_derived_and_signer_checked(session):
    assert session.plan.daily_cap_count == 15
    assert session.plan.daily_cap_cents == 74985
    with pytest.raises(ValueError, match="decision_owner"):
        sign_plan(session.plan, "alex", AT)
    with pytest.raises(ValueError, match="decided canary settings"):
        CanaryPlan.model_validate_json(session.plan.model_copy(update={"recheck_hours": 24}).model_dump_json())


@pytest.mark.parametrize("requested,called", [(5000, 5000), (4999, 5000), (50001, 1)])
def test_amount_cap_holds_even_when_gate_approves_everything(session, requested, called, monkeypatch):
    tools, request, state = tools_for(session, "cap", amount=requested)
    monkeypatch.setattr(tools, "_authorize", lambda *args: None)
    result = execute(tools, request, state, called)
    assert not result.ok and result.reason == "canary_amount"
    assert not tools.payments


def test_outside_cohort_is_shadow_and_external_ledger_unchanged(session):
    outside = next(f"outside-{i}" for i in range(100) if not in_cohort(session.plan.canary_id, f"outside-{i}"))
    tools, request, state = tools_for(session, "outside", requester_id=outside)
    before = json.dumps({k: v.model_dump(mode="json") for k, v in tools.ledger.items()}, sort_keys=True)
    assert execute(tools, request, state).reason == "canary_outside_cohort"
    assert tools.shadow_payments and request.request_id in tools.person_requests
    assert json.dumps({k: v.model_dump(mode="json") for k, v in tools.ledger.items()}, sort_keys=True) == before


def test_daily_cap_sixteenth_and_replay_do_not_pay_twice(session):
    for i in range(15):
        tools, request, state = tools_for(session, f"daily-{i}", AT + timedelta(seconds=i))
        assert execute(tools, request, state).ok
    tools, request, state = tools_for(session, "daily-16", AT + timedelta(seconds=16))
    assert execute(tools, request, state).reason == "canary_daily_cap"
    assert len(tools.payments) == 15 and sum(p.amount_cents for p in tools.payments) == 74985


def test_time_box_boundary_and_restart_cannot_renew_expiry(session):
    expiry = session.grant.expires_at
    tools, request, state = tools_for(session, "before-expiry", expiry - timedelta(seconds=1))
    assert execute(tools, request, state).ok
    tools, request, state = tools_for(session, "at-expiry", expiry)
    assert execute(tools, request, state).reason == "canary_ended"
    assert session.stops[-1].trigger == "time_box"
    restarted = CanarySession(session.plan, AppendStore(STORE_PATH, session.store.namespace))
    grant = restarted.grant
    assert restarted.outcome == "ended" and grant is not None and grant.expires_at == expiry
    with pytest.raises(ValueError, match="only a paused"):
        StopRule(restarted).resume("morgan", SettledEvidence(final=True, allowed_on_read=True), at=expiry)


@pytest.mark.parametrize("ended_from_pause", [False, True])
def test_budget_end_is_permanent_running_or_paused(session, ended_from_pause):
    rule = StopRule(session)
    if ended_from_pause:
        rule.trigger("owner", owner="morgan", at=AT)
    rule.end("budget_spent", at=AT + timedelta(seconds=1))
    assert session.outcome == "ended" and session.grant.budget_spent
    assert session.stops[-1].outcome == "ended"
    with pytest.raises(ValueError):
        rule.resume("morgan", SettledEvidence(final=True, allowed_on_read=True), at=AT + timedelta(seconds=2))


def test_overdue_stops_at_four_hours_and_all_rechecks_required(session):
    tools, request, state = tools_for(session, "overdue")
    assert execute(tools, request, state).ok
    receipt = tools.payments[0]
    session.tick(AT + timedelta(hours=4), tools.payments)
    assert session.outcome == "paused" and session.stops[-1].trigger == "recheck_overdue"
    rule = StopRule(session, tools.payments)
    settled = SettledEvidence(payment_id=receipt.receipt_id, final=True, allowed_on_read=True)
    with pytest.raises(ValueError, match="every overdue"):
        rule.resume("morgan", settled, at=AT + timedelta(hours=4))
    rule.rechecked(receipt.receipt_id, at=AT + timedelta(hours=4), owner="morgan")
    rule.resume("morgan", settled, at=AT + timedelta(hours=4))
    assert session.outcome == "running" and session.grant.expires_at == AT + timedelta(days=28)


def test_resume_waits_for_settled_and_residual_is_counted(session):
    tools, request, state = tools_for(session, "residual")
    assert execute(tools, request, state).ok
    payment = tools.payments[0]
    rule = StopRule(session, tools.payments)
    rule.trigger("correction", at=AT + timedelta(minutes=1), payment_id=payment.receipt_id)
    with pytest.raises(ValueError, match="settled evidence"):
        rule.resume("morgan", SettledEvidence(payment_id=payment.receipt_id, final=False,
                                              allowed_on_read=True), at=AT + timedelta(minutes=2))
    rule.resume("morgan", SettledEvidence(payment_id=payment.receipt_id, final=True,
                                          allowed_on_read=True, late_scan=True), at=AT + timedelta(minutes=2))
    assert session.grant.residuals == (payment.receipt_id,)
    assert session.store.latest("reconciliation", payment.receipt_id)["count_against_error_budget"]


@pytest.mark.parametrize("allowed,stale", [(False, False), (True, True)])
def test_confirmed_defect_ends_irreversibly(session, allowed, stale):
    rule = StopRule(session)
    rule.trigger("owner", owner="morgan", at=AT)
    with pytest.raises(ValueError, match="permanently ended"):
        rule.resume("morgan", SettledEvidence(final=True, allowed_on_read=allowed, stale_data=stale),
                    at=AT + timedelta(minutes=1))
    assert session.outcome == "ended"


def test_duplicate_payment_count_is_by_request_across_actors_and_keys():
    receipts = [
        {"request": {"request_id": "r-0412"}, "idempotency_key": "execute-r-0412"},
        {"request": {"request_id": "r-0412"}, "idempotency_key": "desk-r-0412"},
    ]
    assert paid_twice(receipts) == 1


def test_fresh_process_sees_exhaustion(session):
    StopRule(session).end("budget_spent", at=AT)
    script = """import json
from refund_agent.canary.cloud_native_deployment_plan import CanaryPlan
from refund_agent.canary.cloud_native_deployment_stop import CanarySession
from refund_agent.canary.cloud_native_deployment_store import AppendStore
from refund_agent.canary.cloud_native_deployment_component import STORE_PATH
from sys import stdin
p=json.load(stdin)
s=CanarySession(CanaryPlan.model_validate_json(json.dumps(p['plan'])), AppendStore(STORE_PATH,p['namespace']))
print(s.outcome, s.grant.expires_at.isoformat())
"""
    result = subprocess.run([sys.executable, "-c", script], input=json.dumps({
        "plan": session.plan.model_dump(mode="json"), "namespace": session.store.namespace,
    }), capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "ended " + session.grant.expires_at.isoformat()


@pytest.mark.parametrize("mode,outcome", [("shadow", "running"), ("executed", "paused")])
def test_only_matching_executed_correction_pauses(session, mode, outcome):
    from refund_agent.batch_stream.replay import CorrectionRecord

    tools, request, state = tools_for(session, "correction-mode")
    assert execute(tools, request, state).ok
    receipt = tools.payments[0]
    record = CorrectionRecord(receipt.receipt_id, mode, request.request_id, receipt.order_id,
                              "late-scan", AT, AT + timedelta(minutes=1), "shipped", "delivered")
    StopRule(session, tools.payments).correction(record)
    assert session.outcome == outcome


def test_correction_during_owner_pause_requires_its_own_settlement(session):
    from refund_agent.batch_stream.replay import CorrectionRecord

    tools, request, state = tools_for(session, "paused-correction")
    assert execute(tools, request, state).ok
    receipt = tools.payments[0]
    rule = StopRule(session, tools.payments)
    rule.trigger("owner", owner="morgan", at=AT + timedelta(minutes=1))
    record = CorrectionRecord(receipt.receipt_id, "executed", request.request_id, receipt.order_id,
                              "late-scan", AT, AT + timedelta(minutes=2), "shipped", "delivered")
    rule.correction(record)
    rule.correction(record)
    assert [s.trigger for s in session.stops] == ["owner", "correction"]
    with pytest.raises(ValueError, match="every pending"):
        rule.resume("morgan", SettledEvidence(final=True, allowed_on_read=True),
                    at=AT + timedelta(minutes=3))
    rule.resume("morgan", (
        SettledEvidence(final=True, allowed_on_read=True),
        SettledEvidence(payment_id=receipt.receipt_id, final=True, allowed_on_read=True),
    ), at=AT + timedelta(minutes=3))
    assert session.outcome == "running"
    assert session.grant.residuals == (receipt.receipt_id,)
    assert len(session.store.rows("reconciliation")) == 1


def test_resume_requires_all_distinct_corrected_payments(session):
    receipts = []
    for tag in ("correction-one", "correction-two"):
        tools, request, state = tools_for(session, tag)
        assert execute(tools, request, state).ok
        receipts = tools.payments
    rule = StopRule(session, receipts)
    for payment in receipts:
        rule.trigger("correction", at=AT + timedelta(minutes=1), payment_id=payment.receipt_id)
    settled = tuple(SettledEvidence(payment_id=p.receipt_id, final=True, allowed_on_read=True)
                    for p in receipts)
    with pytest.raises(ValueError, match="every pending"):
        rule.resume("morgan", settled[0], at=AT + timedelta(minutes=2))
    rule.resume("morgan", settled, at=AT + timedelta(minutes=2))
    assert set(session.grant.residuals) == {p.receipt_id for p in receipts}


def test_payment_pause_cannot_name_an_unrecorded_payment(session):
    with pytest.raises(ValueError, match="existing cohort payment"):
        StopRule(session).trigger("correction", at=AT, payment_id="invented")


def test_paused_cohort_denial_routes_to_person(session):
    from refund_agent.release.candidate import build_candidate
    from refund_agent.release.suite import build_suite

    StopRule(session).end("budget_spent", at=AT)
    data, request = fixture(session.plan.canary_id, "paused-denial", AT, forbidden=True)
    tools = CanaryTools(data, canary_session=session, shadow_type=ShadowTools, shadow_kwargs={})
    trace = run_request(build_candidate(build_suite()), tools, request)
    assert trace.decision.outcome == c.Outcome.REFER and not trace.decision.executed
    assert request.request_id in tools.person_requests
    assert not tools.payments


def test_store_returns_detached_rows_and_detects_external_tampering(tmp_path):
    # No selected component reads this deliberately corrupt isolated file.
    store = AppendStore(tmp_path / "cloud-native-deployment-store.jsonl", "isolated")
    store.append("example", "id", {"nested": {"value": 1}})
    row = store.rows()[0]
    row["value"]["nested"]["value"] = 99
    assert store.latest("example", "id") == {"nested": {"value": 1}}
    damaged = store.path.read_text().replace('"value": 1', '"value": 2')
    store.path.write_text(damaged)
    with pytest.raises(ValueError, match="append chain"):
        store.rows()


@pytest.mark.parametrize("damage", ["schema", "key", "chain", "json"])
def test_store_integrity_failure_stops_before_another_payment(tmp_path, session, damage):
    store = AppendStore(tmp_path / "integrity.jsonl", "isolated")
    session = rehearsal_session(session.plan, store, AT)
    tools, request, state = tools_for(session, "integrity")
    original = store.path.read_bytes()
    proposal = tools.proposals[state.proposal_id].model_dump(mode="json")
    if damage == "schema":
        store.append("proposals", state.proposal_id, {**proposal, "unexpected": True})
    elif damage == "key":
        store.append("proposals", "wrong-key", proposal)
    elif damage == "chain":
        store.path.write_bytes(original.replace(b'"previous": ""', b'"previous": "damaged"', 1))
    else:
        with store.path.open("ab") as writer:
            writer.write(b"{not-json}\n")
    damaged = store.path.read_bytes()
    with pytest.raises(ReconciliationError):
        execute(tools, request, state)
    assert session.outcome == "paused"
    assert session.stops[-1].trigger == "untrusted_record"
    if damage in ("chain", "json"):
        assert store.path.read_bytes() == damaged
        store.path.write_bytes(original)
        session.tick(AT, ())
    assert not store.records("ledger")
    restarted = CanarySession(session.plan, AppendStore(store.path, store.namespace))
    assert restarted.outcome == "paused"
    assert restarted.stops[-1].trigger == "untrusted_record"


def test_store_contract_failure_during_load_stops_canary(tmp_path, session):
    from pydantic import ValidationError

    store = AppendStore(tmp_path / "load-integrity.jsonl", "isolated")
    session = rehearsal_session(session.plan, store, AT)
    tools, request, state = tools_for(session, "load-integrity")
    proposal = tools.proposals[state.proposal_id].model_dump(mode="json")
    store.append("proposals", state.proposal_id, {**proposal, "unexpected": True})
    with pytest.raises(ValidationError):
        CanaryTools(tools.data, canary_session=session, shadow_type=ShadowTools, shadow_kwargs={})
    assert session.outcome == "paused"
    assert session.stops[-1].trigger == "untrusted_record"
    assert not store.records("ledger")


@pytest.mark.parametrize("boundary", ["intent", "receipt"])
def test_store_write_failure_stops_and_keeps_any_paid_receipt(tmp_path, session, boundary):
    class FailingStore(AppendStore):
        failed = False

        def append(self, kind, key, value):
            if not self.failed and boundary == "intent" and kind == "audits" and value["event"] == "intent":
                self.failed = True
                raise OSError("constructed intent write failure")
            super().append(kind, key, value)
            if not self.failed and boundary == "receipt" and kind == "ledger":
                self.failed = True
                raise ValueError("untrusted_record: constructed append read-back failure")

    store = FailingStore(tmp_path / "write-integrity.jsonl", "isolated")
    session = rehearsal_session(session.plan, store, AT)
    tools, request, state = tools_for(session, "write-integrity")
    with pytest.raises(ReconciliationError):
        execute(tools, request, state)
    assert session.outcome == "paused"
    assert session.stops[-1].trigger == "untrusted_record"
    assert len(store.records("ledger")) == (1 if boundary == "receipt" else 0)
    tools, request, state = tools_for(session, "after-write-integrity")
    assert execute(tools, request, state).reason == "canary_paused"
    assert len(store.records("ledger")) == (1 if boundary == "receipt" else 0)
