"""Unit admission records are constructed here; they are not release evidence."""

import json
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest

from refund_agent.release.candidate import build_candidate
from refund_agent.release.suite import build_suite

# isort: split
from refund_agent.canary.cloud_native_deployment_component import JOURNAL_PATH, SELECTOR, STORE_PATH
from refund_agent.canary.cloud_native_deployment_plan import CanaryPlan, sign_plan
from refund_agent.canary.cloud_native_deployment_start import start_canary
from refund_agent.canary.cloud_native_deployment_stop import (
    CanarySession,
    SettledEvidence,
    StopRule,
)
from refund_agent.canary.cloud_native_deployment_store import AppendJournal, AppendStore
from refund_agent.canary.cloud_native_deployment_window import ShadowWindowResult, sign_window

AT = datetime(2026, 10, 12, tzinfo=timezone.utc)


@pytest.fixture
def evidence():
    suite = build_suite(components=[SELECTOR])
    candidate = build_candidate(suite)
    namespace = "cloud-native-deployment-start-test-" + uuid4().hex
    ids = tuple(f"{namespace}-request-{i}" for i in range(299))
    rows = tuple({"request_id": rid, "request_at": AT.isoformat(), "terminal_at": AT.isoformat(),
                  "tier": 1, "person_saw_shadow": False, "agent": "approve", "baseline": "approve",
                  "person_paid": True, "payment_at": AT.isoformat(), "found_false_30d": 0,
                  "label_final_at": (AT + timedelta(days=30)).isoformat(), "forbidden_payment": False}
                 for rid in ids)
    window = sign_window(ShadowWindowResult(
        window_id=namespace + "-window", candidate_fingerprint=candidate.fingerprint,
        start=AT, completed_at=AT, labels_mature_at=AT + timedelta(days=30),
        generated_label_shortcut=True, required_requests=299, request_ids=ids, completion_order=ids,
        comparisons=rows, forbidden_payments=0, referrals=0, baseline_referrals=0,
        wrongful_denials=0, baseline_wrongful_denials=0,
        measure_verdicts={"forbidden_payments": True, "referral_share": True, "wrongful_denials": True},
        measures_pass=True,
    ), "morgan", AT)
    plan = sign_plan(CanaryPlan(canary_id=namespace, candidate_fingerprint=candidate.fingerprint,
                               suite_fingerprint=suite.fingerprint, rollback_target_fingerprint="target",
                               requirement_ref=window.window_id, rehearsal_ref="rehearsal",
                               error_budget_ref="budget"), "morgan", AT)
    journal = AppendJournal(JOURNAL_PATH)
    unsigned = {"decision": "pass", "signature": None, "record_id": namespace + "-release",
                "candidate_fingerprint": candidate.fingerprint, "software_fingerprint": candidate.fingerprint,
                "suite_fingerprint": suite.fingerprint, "unresolved_earlier_failures": [], "failure_dispositions": []}
    signature = {"owner": "morgan", "at": AT.isoformat()}
    with journal.transaction() as (history, stream):
        journal.write(history, stream, unsigned)
        journal.write(history, stream, {"event": "sign", "record_id": unsigned["record_id"], "signature": signature})
    release = {**unsigned, "signature": signature}
    rehearsal = {"verdict": "pass", "rehearsal_id": "rehearsal", "candidate_fingerprint": candidate.fingerprint,
                 "rollback_target_fingerprint": "target", "paid_twice": 0,
                 "unreadable_records": 0, "restart_fresh_process": True,
                 "checks": {name: True for name in (
                     "next_cohort_to_person", "receipts_kept", "intents_closed_unpaid",
                     "receipts_confirmed_at_restart", "reconcile_idempotent", "readable_records",
                     "post_restart", "components_restored", "no_request_paid_twice",
                 )}}
    budget = {"record_id": "budget", "signed_by": "morgan", "signed_at": AT.isoformat(),
              "candidate_fingerprint": candidate.fingerprint, "canary_id": plan.canary_id, "spent": False,
              "objectives": [{"definition": definition, "window": "28 days", "allowance": allowance,
                              "population": "all tier 1 decisions"}
                             for definition, allowance in (("not later corrected", 11), ("timely answer", 37))]}
    session = CanarySession(plan, AppendStore(STORE_PATH, namespace))
    return {"plan": plan, "release_record": release, "window": window, "rehearsal": rehearsal,
            "candidate": candidate, "suite": suite, "journal": journal, "session": session,
            "owner": "morgan", "at": AT + timedelta(minutes=1), "error_budget": budget}


def test_start_accepts_all_owned_records_and_preserves_original_expiry(evidence):
    first = start_canary(**evidence)
    assert first.accepted and len(first.checks) == 5
    again = start_canary(**{**evidence, "at": evidence["at"] + timedelta(days=1)})
    assert again.accepted
    assert again.started_at == first.started_at and again.expires_at == first.expires_at


@pytest.mark.parametrize("kind", ["missing", "failed", "unsigned", "candidate", "suite"])
def test_start_refuses_release_without_opening_binding(evidence, kind):
    record = evidence["release_record"].copy()
    if kind == "missing":
        record = None
    elif kind == "failed":
        record["decision"] = "fail"
    elif kind == "unsigned":
        record["signature"] = None
    else:
        record[kind + "_fingerprint"] = "other"
    result = start_canary(**{**evidence, "release_record": record})
    assert not result.accepted and not result.checks[0].passed
    assert evidence["session"].outcome == "shadow"


@pytest.mark.parametrize("kind", ["missing", "failed", "candidate", "target"])
def test_start_refuses_rehearsal(evidence, kind):
    record = evidence["rehearsal"].copy()
    if kind == "missing":
        record = None
    elif kind == "failed":
        record["verdict"] = "fail"
    else:
        record["candidate_fingerprint" if kind == "candidate" else "rollback_target_fingerprint"] = "other"
    result = start_canary(**{**evidence, "rehearsal": record})
    assert not result.accepted and not result.checks[4].passed


@pytest.mark.parametrize("missing", [
    "next_cohort_to_person", "receipts_kept", "intents_closed_unpaid",
    "receipts_confirmed_at_restart", "reconcile_idempotent", "readable_records",
    "post_restart", "components_restored", "no_request_paid_twice",
])
def test_start_requires_every_rehearsal_check(evidence, missing):
    record = json.loads(json.dumps(evidence["rehearsal"]))
    del record["checks"][missing]
    result = start_canary(**{**evidence, "rehearsal": record})
    assert not result.accepted and not result.checks[4].passed
    assert evidence["session"].outcome == "shadow"


@pytest.mark.parametrize("field,value", [("unreadable_records", 1), ("restart_fresh_process", False)])
def test_start_refuses_rehearsal_with_contradictory_evidence(evidence, field, value):
    record = {**evidence["rehearsal"], field: value}
    result = start_canary(**{**evidence, "rehearsal": record})
    assert not result.accepted and not result.checks[4].passed
    assert evidence["session"].outcome == "shadow"


@pytest.mark.parametrize("field", ["definition", "window", "allowance", "population"])
def test_start_requires_each_budget_objective_field(evidence, field):
    record = json.loads(json.dumps(evidence["error_budget"]))
    del record["objectives"][1][field]
    result = start_canary(**{**evidence, "error_budget": record})
    assert not result.accepted and not result.checks[3].passed


@pytest.mark.parametrize("trigger", ["budget_spent", "defect", "time_box"])
def test_ended_authority_cannot_be_restarted(evidence, trigger):
    first = start_canary(**evidence)
    rule = StopRule(evidence["session"])
    rule.end(trigger, at=evidence["at"] + timedelta(minutes=1))
    result = start_canary(**{**evidence, "at": evidence["at"] + timedelta(days=1)})
    assert not result.accepted and not result.checks[2].passed
    assert evidence["session"].grant.expires_at == first.expires_at


def test_resume_rereads_persisted_window_authority(evidence):
    assert start_canary(**evidence).accepted
    rule = StopRule(evidence["session"])
    rule.trigger("owner", owner="morgan", at=evidence["at"])
    evidence["session"].store.append("window", evidence["window"].window_id, None)
    with pytest.raises(ValueError, match="window authority"):
        rule.resume("morgan", SettledEvidence(final=True, allowed_on_read=True),
                    at=evidence["at"] + timedelta(minutes=1))


def test_repeated_start_rereads_persisted_window_authority(evidence):
    assert start_canary(**evidence).accepted
    evidence["session"].store.append("window", evidence["window"].window_id, None)
    result = start_canary(**{**evidence, "at": evidence["at"] + timedelta(minutes=1)})
    assert not result.accepted and not result.checks[2].passed
    assert evidence["session"].outcome == "paused"


@pytest.mark.parametrize("kind", ["unsigned", "candidate"])
def test_start_refuses_unsigned_or_other_candidate_window(evidence, kind):
    updates = {"signed_by": None, "signed_at": None} if kind == "unsigned" else {"candidate_fingerprint": "other"}
    result = start_canary(**{**evidence, "window": evidence["window"].model_copy(update=updates)})
    assert not result.accepted and not result.checks[1].passed
    assert evidence["session"].outcome == "shadow"


def test_start_refuses_unsigned_plan(evidence):
    plan = evidence["plan"].model_copy(update={"signed_by": None, "signed_at": None})
    session = CanarySession(plan, evidence["session"].store)
    result = start_canary(**{**evidence, "plan": plan, "session": session})
    assert not result.accepted and not result.checks[2].passed
    assert session.outcome == "shadow"


def test_same_window_cannot_start_another_plan(evidence):
    assert start_canary(**evidence).accepted
    changed = sign_plan(evidence["plan"].model_copy(update={"canary_id": "another-canary"}), "morgan", AT)
    session = CanarySession(changed, evidence["session"].store)
    result = start_canary(**{**evidence, "plan": changed, "session": session})
    assert not result.accepted and not result.checks[1].passed


@pytest.mark.parametrize("trigger", ["budget_spent", "defect", "time_box"])
def test_namespace_change_cannot_hide_spent_authority(evidence, trigger):
    first = start_canary(**evidence)
    StopRule(evidence["session"]).end(trigger, at=evidence["at"] + timedelta(minutes=1))
    other = CanarySession(evidence["plan"], AppendStore(STORE_PATH, "cloud-native-deployment-other-" + uuid4().hex))
    result = start_canary(**{**evidence, "session": other, "at": evidence["at"] + timedelta(days=1)})
    assert not result.accepted and not result.checks[1].passed
    assert other.grant is None
    assert evidence["session"].grant.expires_at == first.expires_at


def test_caller_boolean_cannot_bypass_canary_binding(evidence):
    from refund_agent.canary.cloud_native_deployment_rehearsal import fixture
    from refund_agent.release.candidate import build_tools

    data, _ = fixture(evidence["plan"].canary_id, "bypass", AT)
    with pytest.raises(ValueError, match="canary_session_required"):
        build_tools(data, evidence["candidate"], "execute", cloud_native_deployment_fixture=True)


def test_public_fixture_hook_cannot_mint_canary_authority(evidence):
    from refund_agent.canary.cloud_native_deployment_rehearsal import fixture
    from refund_agent.release import extensions
    from refund_agent.release.candidate import build_tools

    data, _ = fixture(evidence["plan"].canary_id, "hook-bypass", AT)
    context = extensions.apply(evidence["candidate"], "fixture_context", value={}, data=data,
                               mode="execute", purpose="review")
    with pytest.raises(ValueError, match="canary_session_required"):
        build_tools(data, evidence["candidate"], "execute", **context)


def test_resigning_spent_window_does_not_make_a_new_window(evidence):
    assert start_canary(**evidence).accepted
    window = sign_window(evidence["window"], "morgan", AT + timedelta(hours=1))
    plan = sign_plan(evidence["plan"].model_copy(update={"canary_id": "resigned-" + uuid4().hex}), "morgan", AT)
    session = CanarySession(plan, AppendStore(STORE_PATH, "cloud-native-deployment-resigned-" + uuid4().hex))
    result = start_canary(**{**evidence, "window": window, "plan": plan, "session": session,
                             "at": AT + timedelta(hours=2)})
    assert not result.accepted and not result.checks[1].passed


@pytest.mark.parametrize("shortcut", [False, True])
def test_unresolved_wrongful_denial_labels_cannot_pass_or_sign(evidence, shortcut):
    raw = evidence["window"].model_dump(mode="json")
    raw.update(generated_label_shortcut=shortcut, signed_by=None, signed_at=None)
    for row in raw["comparisons"]:
        row.update(agent="deny", found_false_30d=None, label_final_at=None)
    with pytest.raises(ValueError, match="verdicts differ"):
        ShadowWindowResult.model_validate_json(json.dumps(raw))
