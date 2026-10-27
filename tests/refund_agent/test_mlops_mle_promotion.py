# ruff: noqa: E402
"""Chapter 33 tests. Constructed unit records never stand for release evidence."""

import json
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

import pytest

from refund_agent.release.candidate import build_candidate, build_tools, run_request
from refund_agent.release.extensions import resource_guard
from refund_agent.release.suite import build_suite

CORE = build_candidate(build_suite())

# isort: split

from refund_agent import contracts as c
from refund_agent.canary.cloud_native_deployment_component import JOURNAL_PATH as CJ
from refund_agent.canary.cloud_native_deployment_component import SELECTOR as CANARY
from refund_agent.canary.cloud_native_deployment_component import STORE_PATH
from refund_agent.canary.cloud_native_deployment_plan import CanaryPlan, sign_plan
from refund_agent.canary.cloud_native_deployment_start import start_canary
from refund_agent.canary.cloud_native_deployment_stop import CanarySession, StopRecord, StopRule
from refund_agent.canary.cloud_native_deployment_store import AppendJournal, AppendStore
from refund_agent.canary.cloud_native_deployment_window import ShadowWindowResult, sign_window
from refund_agent.data.generate import _policies
from refund_agent.memory.register import PolicyRegister
from refund_agent.promotion.mlops_mle_bundle import Component, VersionBundle, from_candidate
from refund_agent.promotion.mlops_mle_cases import policy_fixture
from refund_agent.promotion.mlops_mle_component import JOURNAL_PATH, REGISTRY_PATH, selector
from refund_agent.promotion.mlops_mle_diff import diff
from refund_agent.promotion.mlops_mle_record import decide, sign
from refund_agent.promotion.mlops_mle_registry import Registry
from refund_agent.promotion.mlops_mle_requirements import REQUIREMENTS, required
from refund_agent.sources.data_systems_release import SELECTOR as DATA

AT = datetime(2026, 10, 12, tzinfo=timezone.utc)


def initial_snapshot():
    register = PolicyRegister()
    for policy in _policies():
        register.announce(policy, policy.effective_at)
    return register.snapshot(AT)


@pytest.fixture(scope="module")
def built():
    snapshot = initial_snapshot()
    suite = build_suite(components=[DATA, selector(snapshot, [p.version for p in _policies()])])
    candidate = build_candidate(suite)
    return suite, candidate, from_candidate(candidate), snapshot


def changed(bundle, part, **items):
    before = next(p for p in bundle.components if p.name == part)
    return bundle.model_copy(update={part: Component(name=part, items={**before.items, **items})})


def admission(bundle, *, stage="shadow", incumbent=None, refs=None):
    from refund_agent.promotion.mlops_mle_record import PromotionRecord

    return sign(
        PromotionRecord(
            candidate_fingerprint=bundle.fingerprint,
            incumbent_fingerprint=incumbent.fingerprint if incumbent else None,
            from_stage="shadow" if incumbent else "offline",
            to_stage=stage,
            changes=(),
            requirements={},
            status="promoted",
            unmet=(),
            rollback_target=incumbent.fingerprint if incumbent else None,
            refs=refs or {},
        ),
        "morgan",
        AT,
    )


def registry():
    return Registry(REGISTRY_PATH, "mlops-mle-unit-" + uuid4().hex)


@pytest.mark.parametrize("part", ["model", "prompt", "tools", "policy", "index"])
def test_any_item_changes_bundle_identity(built, part):
    bundle = built[2]
    assert changed(bundle, part, test_item="changed byte").fingerprint != bundle.fingerprint


def test_manifest_only_change_leaves_both_identities_unchanged(built):
    suite, candidate, bundle, _ = built
    another = replace(suite, version=suite.version + "-another-manifest")
    current = build_candidate(another)
    new_bundle = from_candidate(current)
    assert current.suite != candidate.suite
    assert current.fingerprint == candidate.fingerprint
    assert new_bundle.fingerprint == bundle.fingerprint
    assert new_bundle.refs["suite"] != bundle.refs["suite"]


@pytest.mark.parametrize(
    "relative,changes_identity",
    [
        ("typed.py", False),
        ("rules.py", True),
        ("release/graders.py", True),
        ("promotion/mlops_mle_coverage.py", True),
    ],
)
def test_fixed_core_and_selected_code_only(built, monkeypatch, relative, changes_identity):
    suite, candidate, bundle, _ = built
    from refund_agent.release.candidate import PACKAGE

    target = PACKAGE / relative
    original = Path.read_bytes

    def altered(path):
        content = original(path)
        return content + b"\n# mlops-mle identity mutation\n" if path == target else content

    monkeypatch.setattr(Path, "read_bytes", altered)
    current = build_candidate(suite)
    assert (current.fingerprint != candidate.fingerprint) == changes_identity
    assert (from_candidate(current).fingerprint != bundle.fingerprint) == changes_identity
    if changes_identity:
        with pytest.raises(ValueError, match="differs|changed"):
            # A stored bundle cannot be mounted under edited selected code.
            build_tools(policy_fixture(_policies()[-1], 0)[0], candidate, "shadow")


def test_candidate_component_fingerprints_are_kept_exactly(built):
    _, candidate, bundle, snapshot = built
    assert (
        from_candidate(
            candidate, register_snapshot=snapshot, versions=[p.version for p in _policies()]
        ).fingerprint
        == bundle.fingerprint
    )
    assert bundle.model.items["scripted_model"] == candidate.model
    assert bundle.tools.items["agent"] == candidate.agent
    assert bundle.tools.items["runtime"] == candidate.runtime
    assert bundle.tools.items["adapter"] == candidate.adapter
    assert bundle.tools.items["core"] == candidate.tools_composition["fingerprint"]


@pytest.mark.parametrize(
    "path,part,item",
    [("adaptation/prompt.py", "prompt", "code"), ("contracts.py", "tools", "tool_contracts")],
)
def test_code_diff_names_prompt_and_contracts(built, path, part, item):
    bundle = built[2]
    altered = changed(
        bundle, "tools", agent_code_files={**bundle.tools.items["agent_code_files"], path: "edited"}
    )
    changes = diff(bundle, altered)
    assert {"component": part, "item": item} in changes
    assert {"component": "tools", "item": "agent_code", "path": path} in changes


def test_each_requirement_keeps_release_and_owner():
    for row in REQUIREMENTS:
        needs = required([{"component": row.component, "item": row.items[0]}])
        assert {"release", "release_owner", "policy_coverage"} <= set(needs)
        assert set(row.evidence) <= set(needs)
    assert "runtime_swap" in required([{"component": "tools", "item": "adapter"}])


def test_missing_evidence_holds_and_cannot_be_signed(built):
    bundle = built[2]
    held = decide(
        changed(bundle, "tools", adapter={"version": "unjudged"}),
        bundle,
        from_stage="shadow",
        evidence={},
    )
    assert held.status == "held" and {"release", "runtime_swap"} <= set(held.unmet)
    with pytest.raises(ValueError, match="held"):
        sign(held, "morgan", AT)
    with pytest.raises(ValueError, match="owner"):
        sign(
            admission(bundle).model_copy(update={"signed_by": None, "signed_at": None}),
            "unlisted",
            AT,
        )


def test_bundle_json_detects_a_tampered_component(built):
    bundle = built[2]
    assert VersionBundle.from_record(bundle.record()) == bundle
    row = bundle.record()
    row["model"]["items"]["risk_threshold"] = 0.99
    with pytest.raises(ValueError, match="fingerprint"):
        VersionBundle.from_record(row)


def test_registry_never_overwrites_and_held_record_grants_nothing(built):
    bundle = built[2]
    store = registry()
    store.append(bundle)
    before = REGISTRY_PATH.read_bytes()
    with pytest.raises(ValueError, match="overwrite"):
        store.append(bundle)
    assert REGISTRY_PATH.read_bytes() == before
    store.append(decide(bundle, evidence={}))
    assert not store.admitted(bundle.fingerprint, "shadow")
    store.append(admission(bundle))
    assert store.admitted(bundle.fingerprint, "shadow")


def test_whole_bundle_rollback_and_mixed_state(built):
    b0 = built[2]
    b1 = changed(b0, "policy", register_snapshot="new")
    store = registry()
    store.append(b0)
    store.append(b1)
    store.append(admission(b0))
    replacement = admission(b1, incumbent=b0)
    store.append(replacement)
    assert store.admitted(b1.fingerprint, "shadow") and not store.admitted(b0.fingerprint, "shadow")
    with pytest.raises(ValueError, match="rollback target"):
        store.rollback(replacement, b1, owner="morgan", at=AT)
    store.rollback(replacement, b0, owner="morgan", at=AT)
    assert store.admitted(b0.fingerprint, "shadow") and not store.admitted(b1.fingerprint, "shadow")
    mixed = changed(b1, "index", sha256="unadmitted")
    assert not store.admitted(mixed.fingerprint, "shadow")
    assert not store.admitted(mixed.fingerprint, "canary")


@pytest.mark.parametrize(
    "outcome,trigger",
    [("paused", "owner"), ("ended", "defect"), ("ended", "budget_spent"), ("ended", "time_box")],
)
def test_only_an_ended_stop_demotes(built, outcome, trigger):
    from refund_agent.canary.cloud_native_deployment_stop import StopRecord

    bundle = built[2]
    store = registry()
    store.append(bundle)
    store.append(admission(bundle, stage="canary", refs={"canary_id": "mlops-mle-test"}))
    row = StopRecord(
        stop_id="mlops-mle-stop",
        canary_id="mlops-mle-test",
        at=AT,
        trigger=trigger,
        payments_after_trigger=0,
        cents_after_trigger=0,
        outcome=outcome,
    )
    demoted = store.demote(row, bundle, canary_id=row.canary_id)
    assert store.admitted(bundle.fingerprint, "canary") == (outcome == "paused")
    assert (demoted is not None) == (outcome == "ended")
    if demoted is not None:
        assert store.demote(row, bundle, canary_id=row.canary_id) == demoted
        assert store.admitted(bundle.fingerprint, "shadow")


def test_successive_canaries_each_demote_on_their_own_stop_1(built):
    bundle = built[2]
    store = registry()
    store.append(bundle)
    demotions = []
    for canary_id in ("mlops-mle-canary-a", "mlops-mle-canary-b"):
        store.append(admission(bundle, stage="canary", refs={"canary_id": canary_id}))
        assert store.admitted(bundle.fingerprint, "canary")
        stop = StopRecord(
            stop_id="stop-1",
            canary_id=canary_id,
            at=AT,
            trigger="defect",
            payments_after_trigger=0,
            cents_after_trigger=0,
            outcome="ended",
        )
        demoted = store.demote(stop, bundle, canary_id=canary_id)
        assert demoted.refs == {"stop": "stop-1", "canary_id": canary_id}
        assert not store.admitted(bundle.fingerprint, "canary")
        assert store.demote(stop, bundle, canary_id=canary_id) == demoted
        demotions.append(demoted)
    assert demotions[0] != demotions[1]


@pytest.mark.parametrize("newer_stage", ["canary", "shadow"])
def test_delayed_end_preserves_newer_bundle_admission(built, newer_stage):
    older = built[2]
    newer = changed(older, "index", sha256="mlops-mle-newer-bundle")
    store = registry()
    for bundle in (older, newer):
        store.append(bundle)
    store.append(admission(older, stage="canary", refs={"canary_id": "older-canary"}))
    store.append(
        admission(newer, stage=newer_stage, refs={"canary_id": "newer-canary"}).model_copy(
            update={"signed_at": AT + timedelta(days=2)}
        )
    )
    assert store.admitted(newer.fingerprint, newer_stage)
    stop = StopRecord(
        stop_id="stop-1",
        canary_id="older-canary",
        at=AT + timedelta(days=1),
        trigger="defect",
        payments_after_trigger=0,
        cents_after_trigger=0,
        outcome="ended",
    )
    demoted = store.demote(stop, older, canary_id=stop.canary_id)
    assert demoted in store.transitions()
    assert demoted.refs == {"stop": "stop-1", "canary_id": "older-canary"}
    before_duplicate = store.path.read_bytes()
    assert store.demote(stop, older, canary_id=stop.canary_id) == demoted
    assert store.path.read_bytes() == before_duplicate
    reopened = Registry(store.path, store.namespace)
    assert reopened.admitted(newer.fingerprint, newer_stage)
    assert not reopened.admitted(older.fingerprint, "canary")
    assert not reopened.admitted(older.fingerprint, "shadow")
    assert reopened.admitted(older.fingerprint, "canary", at=AT)
    assert reopened.admitted(older.fingerprint, "shadow", at=stop.at)


def test_delayed_end_preserves_new_canary_on_same_bundle(built):
    bundle = built[2]
    store = registry()
    store.append(bundle)
    store.append(admission(bundle, stage="canary", refs={"canary_id": "older-canary"}))
    store.append(
        admission(bundle, stage="canary", refs={"canary_id": "newer-canary"}).model_copy(
            update={"signed_at": AT + timedelta(days=2)}
        )
    )
    stop = StopRecord(
        stop_id="stop-1",
        canary_id="older-canary",
        at=AT + timedelta(days=1),
        trigger="defect",
        payments_after_trigger=0,
        cents_after_trigger=0,
        outcome="ended",
    )
    demoted = store.demote(stop, bundle, canary_id=stop.canary_id)
    assert demoted in store.transitions()
    before_duplicate = store.path.read_bytes()
    assert store.demote(stop, bundle, canary_id=stop.canary_id) == demoted
    assert store.path.read_bytes() == before_duplicate
    assert store.admitted(bundle.fingerprint, "canary")
    assert not store.admitted(bundle.fingerprint, "shadow")


@pytest.mark.parametrize("amount", [3499, 1200000, 65000])
def test_uncovered_policy_refers_payments_and_denials_and_named_approver(built, amount):
    _, candidate, bundle, _ = built
    newer = _policies()[-1].model_copy(
        update={"version": "2026-09-02", "effective_at": datetime(2026, 9, 2, tzinfo=timezone.utc)}
    )
    data, request = policy_fixture(newer, 0)
    request = request.model_copy(update={"amount_cents": amount})
    tools = build_tools(data, candidate, "shadow")
    trace = run_request(candidate, tools, request)
    assert trace.decision.outcome == c.Outcome.REFER
    assert trace.decision.reason == "policy_not_in_bundle"
    assert not tools.payments and not tools.shadow_payments
    assert all(row.bundle == bundle.fingerprint for row in tools.lineage)


def test_covered_policy_stamps_all_lineage_without_changing_decision(built):
    _, candidate, bundle, _ = built
    data, request = policy_fixture(_policies()[-1], 0)
    tools = build_tools(data, candidate, "shadow")
    trace = run_request(candidate, tools, request)
    assert trace.decision.outcome == c.Outcome.APPROVE
    assert len(tools.shadow_payments) == 1
    assert tools.lineage and all(row.bundle == bundle.fingerprint for row in tools.lineage)
    assert all(row.code_version == candidate.fingerprint for row in tools.lineage)


def test_secret_item_refused_and_environment_never_reads_env(built, monkeypatch):
    with pytest.raises(ValueError, match="secret"):
        changed(built[2], "tools", signing_key={"secret": True, "value": "never-hash-me"})
    monkeypatch.setenv("MLOPS_MLE_SECRET", "value-must-never-enter-identity")
    assert from_candidate(built[1]).fingerprint == built[2].fingerprint


@resource_guard(1)
def unit_release(suite, candidate, bundle, *, journal_path=JOURNAL_PATH, omit_version=None):
    # Assumed signed evidence in a unit scenario, not measured release evidence.
    journal = AppendJournal(journal_path)
    record_id = "mlops-mle-unit-release-" + uuid4().hex
    rows = [
        {
            "case_id": "mlops-mle-policy-change-" + version,
            "trials": [
                {
                    "failed": [],
                    "decision": {"policy_citation": "refund_cap_v4@" + version + "#eligibility"},
                }
            ],
        }
        for version in bundle.policy.items["versions"]
        if version != omit_version
    ]
    record = {
        "decision": "pass",
        "signature": None,
        "record_id": record_id,
        "candidate_fingerprint": candidate.fingerprint,
        "software_fingerprint": candidate.fingerprint,
        "suite_fingerprint": suite.fingerprint,
        "candidate": candidate.record(),
        "manifest": suite.manifest(),
        "unresolved_earlier_failures": [],
        "failure_dispositions": [],
        "case_results": rows,
    }
    signature = {"owner": "morgan", "at": AT.isoformat()}
    with journal.transaction() as (history, stream):
        journal.write(history, stream, record)
        journal.write(
            history, stream, {"event": "sign", "record_id": record_id, "signature": signature}
        )
    return {**record, "signature": signature}, journal


def test_declared_version_without_passing_case_holds_admission(built):
    suite, candidate, bundle, _ = built
    release, journal = unit_release(
        suite, candidate, bundle, omit_version=bundle.policy.items["versions"][0]
    )
    record = decide(bundle, evidence={"release": release}, journal=journal)
    assert record.status == "held" and record.unmet == ("policy_coverage",)


def test_signed_experiment_decline_with_owned_assumption_is_accepted(built):
    from refund_agent.promotion.mlops_mle_record import _simple_evidence

    bundle = built[2]
    evidence = {
        "kind": "threshold_experiment",
        "bundle_fingerprint": bundle.fingerprint,
        "passed": True,
        "outcome": "declined",
        "signed_by": "morgan",
        "signed_at": AT.isoformat(),
        "assumption": "the cost-derived threshold applies",
    }
    assert _simple_evidence("threshold_experiment", evidence, bundle) == evidence
    for field in ("signed_by", "signed_at", "assumption"):
        with pytest.raises(ValueError):
            _simple_evidence("threshold_experiment", {**evidence, field: None}, bundle)


@pytest.mark.parametrize("transition", ["replacement", "demotion"])
def test_shadow_hook_refuses_unadmitted_demoted_or_mixed_bundle(built, transition):
    from refund_agent.release import extensions

    suite, candidate, bundle, _ = built
    release, journal = unit_release(suite, candidate, bundle)
    path = REGISTRY_PATH.parent / ("mlops-mle-shadow-" + uuid4().hex) / "release.json"
    with pytest.raises(ValueError, match="not admitted"):
        extensions.notify(candidate, "admit_shadow", record=release, journal=journal, path=path)
    store = Registry(REGISTRY_PATH, path.parent.name)
    store.append(bundle)
    store.append(admission(bundle))
    extensions.notify(candidate, "admit_shadow", record=release, journal=journal, path=path)
    mixed = changed(bundle, "index", sha256="not-judged")
    store.append(mixed)
    if transition == "replacement":
        store.append(admission(mixed, incumbent=bundle))
    else:
        canary_id = path.parent.name
        store.append(admission(mixed, stage="canary", refs={"canary_id": canary_id}))
        store.demote(
            StopRecord(
                stop_id="stop-1",
                canary_id=canary_id,
                at=AT,
                trigger="defect",
                payments_after_trigger=0,
                cents_after_trigger=0,
                outcome="ended",
            ),
            mixed,
            canary_id=canary_id,
        )
        assert not store.admitted(mixed.fingerprint, "canary")
    assert store.admitted(mixed.fingerprint, "shadow")
    assert not store.admitted(bundle.fingerprint, "shadow")
    with pytest.raises(ValueError, match="not admitted"):
        extensions.notify(candidate, "admit_shadow", record=release, journal=journal, path=path)


def canary_unit_evidence(namespace):
    # Chapter 34's interface: records made here, never a budget implementation.
    snapshot = initial_snapshot()
    suite = build_suite(
        components=[DATA, CANARY, selector(snapshot, [p.version for p in _policies()])]
    )
    candidate = build_candidate(suite)
    bundle = from_candidate(candidate)
    release, journal = unit_release(suite, candidate, bundle, journal_path=CJ)
    ids = tuple(f"{namespace}-request-{i}" for i in range(299))
    comparisons = tuple(
        {
            "request_id": rid,
            "request_at": AT.isoformat(),
            "terminal_at": AT.isoformat(),
            "tier": 1,
            "person_saw_shadow": False,
            "agent": "approve",
            "baseline": "approve",
            "person_paid": True,
            "payment_at": AT.isoformat(),
            "found_false_30d": 0,
            "label_final_at": (AT + timedelta(days=30)).isoformat(),
            "forbidden_payment": False,
        }
        for rid in ids
    )
    window = sign_window(
        ShadowWindowResult(
            window_id=namespace + "-window",
            candidate_fingerprint=candidate.fingerprint,
            bundle_fingerprint=bundle.fingerprint,
            start=AT,
            completed_at=AT,
            labels_mature_at=AT + timedelta(days=30),
            generated_label_shortcut=True,
            required_requests=299,
            request_ids=ids,
            completion_order=ids,
            comparisons=comparisons,
            forbidden_payments=0,
            referrals=0,
            baseline_referrals=0,
            wrongful_denials=0,
            baseline_wrongful_denials=0,
            measure_verdicts={
                "forbidden_payments": True,
                "referral_share": True,
                "wrongful_denials": True,
            },
            measures_pass=True,
        ),
        "morgan",
        AT,
    )
    plan = sign_plan(
        CanaryPlan(
            canary_id=namespace,
            candidate_fingerprint=candidate.fingerprint,
            suite_fingerprint=suite.fingerprint,
            rollback_target_fingerprint="unit-target",
            requirement_ref=window.window_id,
            rehearsal_ref=namespace + "-rehearsal",
            error_budget_ref=namespace + "-budget",
        ),
        "morgan",
        AT,
    )
    rehearsal = {
        "verdict": "pass",
        "rehearsal_id": plan.rehearsal_ref,
        "candidate_fingerprint": candidate.fingerprint,
        "rollback_target_fingerprint": plan.rollback_target_fingerprint,
        "paid_twice": 0,
        "unreadable_records": 0,
        "restart_fresh_process": True,
        "checks": {
            name: True
            for name in (
                "next_cohort_to_person",
                "receipts_kept",
                "intents_closed_unpaid",
                "receipts_confirmed_at_restart",
                "reconcile_idempotent",
                "readable_records",
                "post_restart",
                "components_restored",
                "no_request_paid_twice",
            )
        },
    }
    budget = {
        "record_id": plan.error_budget_ref,
        "signed_by": "morgan",
        "signed_at": AT.isoformat(),
        "canary_id": plan.canary_id,
        "window_id": window.window_id,
        "candidate_fingerprint": candidate.fingerprint,
        "spent": False,
        "objectives": [
            {
                "definition": definition,
                "window": "28 days",
                "population": "all tier 1",
                "allowance": allowance,
            }
            for definition, allowance in (("not later corrected", 11.2), ("timely answers", 37.3))
        ],
    }
    store = Registry(REGISTRY_PATH, namespace)
    store.append(bundle)
    transition = admission(
        bundle,
        stage="canary",
        refs={
            "canary_id": plan.canary_id,
            "window": window.window_id,
            "candidate": candidate.fingerprint,
        },
    )
    store.append(transition)
    session = CanarySession(plan, AppendStore(STORE_PATH, namespace))
    return (
        {
            "plan": plan,
            "release_record": release,
            "window": window,
            "rehearsal": rehearsal,
            "candidate": candidate,
            "suite": suite,
            "journal": journal,
            "session": session,
            "owner": "morgan",
            "at": AT + timedelta(minutes=1),
            "error_budget": budget,
            "records": {
                "mlops-mle": {
                    "bundle": bundle,
                    "namespace": namespace,
                    "current_policy": _policies()[-1],
                }
            },
        },
        store,
        bundle,
    )


@pytest.mark.parametrize("stage", ["shadow", "canary"])
@pytest.mark.parametrize("end", ["running", "spent", "ended", "expired"])
def test_bundle_rollback_never_revives_canary_authority(end, stage):
    evidence, store, b0 = canary_unit_evidence("mlops-mle-grant-" + uuid4().hex)
    first = start_canary(**evidence)
    assert first.accepted and len(first.checks) == 6
    session = evidence["session"]
    if end == "spent":
        end_session(evidence["candidate"], session, "budget_spent", evidence["at"])
    elif end == "ended":
        end_session(evidence["candidate"], session, "defect", evidence["at"])
    elif end == "expired":
        evidence["at"] = first.expires_at
    grant_before = read_grant(evidence["candidate"], session)
    if stage == "shadow":
        store.append(admission(b0))
    target_admission = store.transitions()[-1]
    b1 = changed(b0, "index", sha256="another-admitted-bundle")
    store.append(b1)
    replacement = admission(b1, incumbent=b0, stage=stage, refs={"canary_id": "another-plan"})
    store.append(replacement)
    rollback = store.rollback(replacement, b0, owner="morgan", at=evidence["at"])
    assert store.admitted(b0.fingerprint, stage)
    assert rollback.refs == {**target_admission.refs, "reverses": replacement.fingerprint}
    assert read_grant(evidence["candidate"], session) == grant_before
    # New session object must reconstruct the original spent state and expiry.
    evidence["session"] = CanarySession(
        evidence["plan"], AppendStore(STORE_PATH, session.store.namespace)
    )
    result = start_canary(**evidence)
    assert result.accepted == (end == "running")
    assert next(c for c in result.checks if c.name == "mlops-mle.promotion").passed
    assert next(c for c in result.checks if c.name == "plan").passed == (end == "running")
    grant_after = read_grant(evidence["candidate"], evidence["session"])
    assert grant_after.started_at == first.started_at
    assert grant_after.expires_at == first.expires_at


@resource_guard(0)
def end_session(candidate, session, trigger, at):
    return StopRule(session).end(trigger, at=at)


@resource_guard(0)
def read_grant(candidate, session):
    return session.grant


@pytest.mark.parametrize("fault", ["missing", "mixed", "other_namespace", "uncovered", "demoted"])
def test_start_refuses_bad_current_bundle_records(fault):
    evidence, registry, bundle = canary_unit_evidence("mlops-mle-start-" + uuid4().hex)
    supplied = evidence["records"]["mlops-mle"]
    if fault == "missing":
        evidence["records"] = {}
    elif fault == "mixed":
        supplied["bundle"] = changed(bundle, "index", sha256="mixed")
    elif fault == "other_namespace":
        supplied["namespace"] = "absent-namespace"
    elif fault == "uncovered":
        supplied["current_policy"] = _policies()[-1].model_copy(
            update={
                "version": "2026-10-01",
                "effective_at": datetime(2026, 10, 1, tzinfo=timezone.utc),
            }
        )
    else:
        from refund_agent.canary.cloud_native_deployment_stop import StopRecord

        registry.demote(
            StopRecord(
                stop_id="mlops-mle-end",
                canary_id=evidence["plan"].canary_id,
                at=AT,
                trigger="defect",
                payments_after_trigger=0,
                cents_after_trigger=0,
                outcome="ended",
            ),
            bundle,
            canary_id=evidence["plan"].canary_id,
        )
    result = start_canary(**evidence)
    assert not result.accepted
    assert not next(c for c in result.checks if c.name == "mlops-mle.promotion").passed


def test_payment_reread_checks_external_policy_again(built):
    from refund_agent.tools import Tools

    _, candidate, bundle, _ = built
    old = _policies()[-1]
    later = old.model_copy(
        update={"version": "2026-09-02", "effective_at": old.effective_at + timedelta(days=1)}
    )
    data, request = policy_fixture(later, 0)

    def continue_payment(tools):
        # A resumed/direct execution has a valid proposal under the new policy.
        # It must check coverage independently of assessment and final audit.
        state = c.State(request=request, trace_id="mlops-mle-continuation")
        for name in ("order.lookup", "shipment.status", "policy.retrieve"):
            assert tools.call(c.ToolCall(name=name), state).ok
        state.decision = Tools.assess(tools, state)
        assert state.decision.outcome == c.Outcome.APPROVE
        assert tools.call(
            c.ToolCall(
                name="refund.propose",
                arguments=c.ToolArguments(amount_cents=request.amount_cents, reason=request.reason),
            ),
            state,
        ).ok
        state.phase = c.Phase.EXECUTE
        result = tools.call(
            c.ToolCall(
                name="refund.execute",
                arguments=c.ToolArguments(
                    amount_cents=request.amount_cents, idempotency_key="mlops-mle-continuation"
                ),
            ),
            state,
        )
        return result, state

    baseline = Tools(data)
    control, control_state = continue_payment(baseline)
    assert control.ok and control_state.decision.executed and len(baseline.payments) == 1
    tools = build_tools(data, candidate, "execute")
    _, state = continue_payment(tools)
    assert state.decision.outcome == c.Outcome.REFER
    assert state.decision.reason == "policy_not_in_bundle"
    assert state.receipt is None and not tools.payments
    assert all(row.bundle == bundle.fingerprint for row in tools.lineage)


def test_fresh_process_cannot_restart_spent_bundle_grant():
    import subprocess
    import sys

    evidence, _, bundle = canary_unit_evidence("mlops-mle-process-" + uuid4().hex)
    first = start_canary(**evidence)
    assert first.accepted
    end_session(evidence["candidate"], evidence["session"], "budget_spent", evidence["at"])
    payload = {
        "candidate": evidence["candidate"].record(),
        "release": evidence["release_record"],
        "window": evidence["window"].model_dump(mode="json"),
        "plan": evidence["plan"].model_dump(mode="json"),
        "rehearsal": evidence["rehearsal"],
        "budget": evidence["error_budget"],
        "bundle": bundle.record(),
        "namespace": evidence["session"].store.namespace,
        "at": evidence["at"].isoformat(),
    }
    runner = r"""import json,sys
from datetime import datetime
from refund_agent.release.candidate import build_candidate
from refund_agent.release.suite import build_suite
build_candidate(build_suite())
from refund_agent.canary.cloud_native_deployment_component import JOURNAL_PATH,STORE_PATH
from refund_agent.canary.cloud_native_deployment_store import AppendJournal,AppendStore
from refund_agent.canary.cloud_native_deployment_plan import CanaryPlan
from refund_agent.canary.cloud_native_deployment_window import ShadowWindowResult
from refund_agent.canary.cloud_native_deployment_stop import CanarySession
from refund_agent.canary.cloud_native_deployment_start import start_canary
from refund_agent.promotion.mlops_mle_bundle import VersionBundle
from refund_agent.data.generate import _policies
p=json.load(sys.stdin)
suite=build_suite(components=p["candidate"]["tools_composition"]["components"])
candidate=build_candidate(suite)
plan=CanaryPlan.model_validate_json(json.dumps(p["plan"]))
window=ShadowWindowResult.model_validate_json(json.dumps(p["window"]))
bundle=VersionBundle.from_record(p["bundle"])
result=start_canary(plan,p["release"],window,p["rehearsal"],candidate=candidate,suite=suite,
    journal=AppendJournal(JOURNAL_PATH),session=CanarySession(plan,AppendStore(STORE_PATH,p["namespace"])),
    owner="morgan",at=datetime.fromisoformat(p["at"]),error_budget=p["budget"],
    records={"mlops-mle":{"bundle":bundle,"namespace":p["namespace"],"current_policy":_policies()[-1]}})
print(json.dumps({"accepted":result.accepted,"plan":next(c.passed for c in result.checks if c.name=="plan")}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", runner],
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        timeout=120,
        check=True,
    )
    assert json.loads(completed.stdout) == {"accepted": False, "plan": False}


def test_release_for_old_component_records_cannot_cover_a_mixed_bundle(built):
    suite, candidate, bundle, _ = built
    release, journal = unit_release(suite, candidate, bundle)
    mixed = changed(bundle, "prompt", template="another-unexecuted-prompt")
    verdict = decide(
        mixed, bundle, from_stage="shadow", evidence={"release": release}, journal=journal
    )
    assert "release" in verdict.unmet


def test_shadow_admission_is_not_limited_to_release_signature_time(built):
    from refund_agent.release import extensions

    suite, candidate, bundle, _ = built
    release, journal = unit_release(suite, candidate, bundle)
    path = REGISTRY_PATH.parent / ("mlops-mle-after-release-" + uuid4().hex) / "release.json"
    store = Registry(REGISTRY_PATH, path.parent.name)
    store.append(bundle)
    later = admission(bundle).model_copy(update={"signed_at": AT + timedelta(minutes=1)})
    store.append(later)
    assert later.signed_at > datetime.fromisoformat(release["signature"]["at"])
    extensions.notify(candidate, "admit_shadow", record=release, journal=journal, path=path)


def test_check_cli_reads_the_declared_append_journal(built, tmp_path, capsys):
    from refund_agent.promotion.__main__ import main

    suite, candidate, bundle, _ = built
    release, journal = unit_release(suite, candidate, bundle)
    bundle_path = tmp_path / "mlops-mle-cli-bundle.json"
    bundle_path.write_text(json.dumps(bundle.record()))
    evidence_path = journal.path.parent / "release.json"
    original = evidence_path.read_bytes() if evidence_path.exists() else None
    try:
        evidence_path.write_text(json.dumps(release))
        status = main(
            [
                "check",
                "--candidate",
                str(bundle_path),
                "--evidence",
                str(journal.path.parent),
            ]
        )
        output = capsys.readouterr().out
        assert status == 0, output
        record = json.loads(output)
        assert record["status"] == "promoted"
        assert record["signed_by"] is None  # Checking alone issues no admission.
    finally:
        if original is None:
            evidence_path.unlink()
        else:
            evidence_path.write_bytes(original)


def test_cold_selection_after_reading_canary_transport():
    import subprocess
    import sys

    runner = r"""from datetime import datetime,timezone
from refund_agent.release.candidate import build_candidate
from refund_agent.release.suite import build_suite
build_candidate(build_suite())
from refund_agent.canary.cloud_native_deployment_component import STORE_PATH,SELECTOR
from refund_agent.canary.cloud_native_deployment_store import AppendStore
AppendStore(STORE_PATH,"mlops-mle-cold-transport-read").rows()
from refund_agent.batch_stream.component import selector as event_selector
from refund_agent.promotion.mlops_mle_component import selector
from refund_agent.sources.data_systems_release import SELECTOR as DATA
from refund_agent.data.generate import _policies
from refund_agent.memory.register import PolicyRegister
at=datetime(2026,10,1,tzinfo=timezone.utc)
register=PolicyRegister()
for policy in _policies(): register.announce(policy,policy.effective_at)
suite=build_suite(components=[DATA,event_selector(),SELECTOR,
    selector(register.snapshot(at),[p.version for p in _policies()])])
build_candidate(suite)
print("cold selection passed")
"""
    completed = subprocess.run(
        [sys.executable, "-c", runner],
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "cold selection passed"


def test_registered_unselected_component_edits_preserve_identity(built, monkeypatch):
    from refund_agent.release import extensions
    from refund_agent.release.candidate import PACKAGE

    suite, candidate, bundle, _ = built
    # Register an actual component over an existing module, but never select it.
    extensions.register(
        extensions.Component(
            name="mlops-mle.unselected-test",
            version="mlops-mle.unselected-test@1",
            module="refund_agent.typed",
        )
    )
    original = Path.read_bytes
    target = PACKAGE / "typed.py"

    def edited(path):
        content = original(path)
        return content + b"\n# unselected component mutation\n" if path == target else content

    monkeypatch.setattr(Path, "read_bytes", edited)
    rebuilt = build_candidate(suite)
    assert rebuilt.fingerprint == candidate.fingerprint
    assert from_candidate(rebuilt).fingerprint == bundle.fingerprint


def test_real_shadow_entry_point_reads_current_registry(built, tmp_path):
    from refund_agent.data.live import generate_live
    from refund_agent.shadow.__main__ import start

    suite, candidate, bundle, _ = built
    release, journal = unit_release(suite, candidate, bundle)
    path = tmp_path / ("mlops-mle-shadow-" + uuid4().hex) / "release.json"
    path.parent.mkdir()
    path.write_text(json.dumps(release))
    args = dict(
        candidate=candidate, suite=suite, journal=journal, stream=generate_live(7, AT, 1, 1)
    )
    with pytest.raises(ValueError, match="not admitted to shadow"):
        start(path, **args)
    store = Registry(REGISTRY_PATH, path.parent.name)
    store.append(bundle)
    store.append(admission(bundle))
    assert start(path, **args)["release_record_id"] == release["record_id"]
    mixed = changed(bundle, "index", sha256="unadmitted-mixed-state")
    store.append(mixed)
    store.append(admission(mixed, incumbent=bundle))
    with pytest.raises(ValueError, match="not admitted to shadow"):
        start(path, **args)
