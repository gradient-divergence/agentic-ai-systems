"""Real predecessor composition; constructed records for unmerged partners."""

import json
from uuid import uuid4

import pytest

from refund_agent import contracts as c
from refund_agent.batch_stream.component import selector as event_selector
from refund_agent.batch_stream.generate import snapshot_log
from refund_agent.canary.cloud_native_deployment_component import SELECTOR as CANARY
from refund_agent.canary.cloud_native_deployment_component import STORE_PATH
from refund_agent.canary.cloud_native_deployment_plan import CanaryPlan
from refund_agent.canary.cloud_native_deployment_stop import CanarySession
from refund_agent.canary.cloud_native_deployment_store import AppendStore
from refund_agent.data.generate import generate
from refund_agent.release import extensions
from refund_agent.release.candidate import build_candidate, build_tools, run_request
from refund_agent.release.suite import build_suite, run_case
from refund_agent.security.security_privacy_attacks import resend
from refund_agent.security.security_privacy_component import SELECTOR, scope_start_check
from refund_agent.security.security_privacy_experiment import runtime
from refund_agent.security.security_privacy_scan import observations
from refund_agent.sources.data_systems_release import SELECTOR as CONTRACTS


@pytest.fixture(scope="module")
def candidates():
    suite = build_suite(trials=1, components=(CONTRACTS, event_selector(), CANARY, SELECTOR))
    return suite, build_candidate(suite, verifier="independent")


def security_privacy_refusal_records(candidates, reference, reason):
    suite, candidate = candidates
    trusted = generate(7, orders=40, requests=40)
    issuer, security = runtime(trusted)
    attack = resend(trusted)
    request = attack.request.model_copy(update={"order_reference": reference})
    foreign = next(o for o in attack.data.orders if o.order_id == "o-0007")
    foreign = foreign.model_copy(
        update={"items": [c.Item(name="FOREIGN_PRIVATE_PAYLOAD", quantity=1, unit_cents=2499)]}
    )
    data = attack.data.model_copy(
        update={
            "orders": [
                foreign if o.order_id == foreign.order_id else o for o in attack.data.orders
            ],
            "requests": [request],
        }
    )
    STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    plan = CanaryPlan(
        canary_id="security-privacy-a1",
        candidate_fingerprint=candidate.fingerprint,
        suite_fingerprint=suite.fingerprint,
        rollback_target_fingerprint=candidate.fingerprint,
        requirement_ref="security-privacy-window",
        rehearsal_ref="security-privacy-rehearsal",
    )
    session = CanarySession(plan, AppendStore(STORE_PATH, "security-privacy-a1-" + uuid4().hex))
    tools = build_tools(
        data,
        candidate,
        "execute",
        security_privacy=security,
        security_credential=issuer.issue_request(request, security.accounts, request.at),
        event_log=snapshot_log(data),
        canary_session=session,
    )
    invocations = []
    original = tools._order

    def instrument(call, state):
        invocations.append(call.name)
        return original(call, state)

    tools._order = instrument
    trace = run_request(candidate, tools, request)
    assert trace.decision.reason == reason and not trace.decision.executed
    assert invocations == [], "inner order.lookup was invoked"
    assert all(step.state.order is None and step.result.order is None for step in trace.steps)
    serialized = json.dumps(
        [str(o) for o in observations(trace, tools, security.log)] + [str(tools.lineage)]
    )
    assert "FOREIGN_PRIVATE_PAYLOAD" not in serialized
    assert all(
        "FOREIGN_PRIVATE_PAYLOAD" not in json.dumps(snapshot, default=str)
        for snapshot in tools._snapshots.values()
    )
    assert all(
        entry.reason == reason for entry in security.log.entries if entry.tool == "order.lookup"
    )
    return (
        trace.decision,
        tuple(step.result for step in trace.steps),
        security.log.entries,
        tuple(tools.audits.values()),
    )


@pytest.mark.parametrize(
    "reference,reason",
    [
        ("o-0007", "order_not_found"),
        ("o-9999", "order_not_found"),
        ("ambiguous", "order_not_found"),
    ],
)
def test_no_foreign_payload_in_all_available_components(candidates, reference, reason):
    security_privacy_refusal_records(candidates, reference, reason)


def test_foreign_and_missing_orders_have_identical_refusal_records(candidates):
    foreign = security_privacy_refusal_records(candidates, "o-0007", "order_not_found")
    missing = security_privacy_refusal_records(candidates, "o-9999", "order_not_found")
    shared_foreign = security_privacy_refusal_records(candidates, "ambiguous", "order_not_found")
    assert foreign == missing == shared_foreign


def test_a1_detects_inner_lookup_even_without_a_fetched_payload(candidates, monkeypatch):
    from refund_agent.security.security_privacy_accounts import ScopeResolver

    monkeypatch.setattr(ScopeResolver, "resolve", lambda *args: (None, None))
    with pytest.raises(AssertionError, match="inner order.lookup was invoked"):
        security_privacy_refusal_records(candidates, "o-0007", "order_not_found")


def test_component_hooks_supply_versioned_records(candidates):
    _, candidate = candidates
    bundle = extensions.contributions(candidate, "bundle_items")[SELECTOR["name"]]
    assert bundle == {
        "tools": {
            "scopes_version": "scopes@1",
            "scoped_tools_version": "security-privacy.scoped@1",
        },
    }
    assert "key" not in json.dumps(bundle)
    requirements = extensions.contributions(candidate, "promotion_requirements")[SELECTOR["name"]]
    assert len(requirements) == 1 and requirements[0]["component"] == "tools"
    reasons = extensions.contributions(candidate, "reasons")[SELECTOR["name"]]
    assert reasons["credential_expired"] == reasons["credential_revoked"] == "system"
    assert reasons["order_not_found"] == reasons["credential_missing"] == "policy"
    checks = extensions.contributions(candidate, "start_checks", records={})[SELECTOR["name"]]
    assert checks == ({"name": "scope", "passed": True, "detail": "pass"},)


def test_scope_check_rejects_unselected_component():
    suite = build_suite(trials=1)
    assert not scope_start_check(build_candidate(suite))["passed"]


def test_partner_run_record_joins_credential_log():
    data = generate(7, orders=40, requests=40)
    request = next(r for r in data.requests if r.request_id == "r-7-0007")
    issuer, security = runtime(data)
    from refund_agent.security.security_privacy_tools import ScopedTools

    tools = ScopedTools(
        data,
        security_privacy=security,
        security_credential=issuer.issue_request(request, security.accounts, request.at),
    )
    record = {
        "name": "run",
        "trace_id": "trace-r-7-0007",
        "run_id": "run-r-7-0007-2",
        "requester_id": request.requester_id,
    }
    tools.bind_run_record(record)
    from refund_agent.agent import Agent

    Agent(tools).run(request)
    assert security.log.trace(record["trace_id"], record["run_id"])
    assert all(e.run_id == record["run_id"] for e in security.log.entries)
    with pytest.raises(ValueError, match="context_differs"):
        tools.bind_run_record({**record, "requester_id": "other"})


@pytest.mark.parametrize(
    "kind",
    [
        "unauthorized",
        "missing",
        "policy_denied",
    ],
)
def test_every_request_over_500_opens_named_review_without_payment(kind):
    from refund_agent.agent import Agent
    from refund_agent.security.security_privacy_experiment import fresh_tools

    source = generate(7, orders=40, requests=40)
    honest = next(r for r in source.requests if r.request_id == "r-7-0007")
    changes = {"amount_cents": 65000}
    if kind == "unauthorized":
        changes["requester_id"] = "customer-7-0012"
    elif kind == "missing":
        changes["order_reference"] = "o-9999"
    else:
        changes["reason"] = "not_delivered"
    request = honest.model_copy(update=changes)
    foreign = next(o for o in source.orders if o.order_id == "o-0007")
    marked = foreign.model_copy(
        update={
            "paid_cents": 200000,
            "items": [c.Item(name="FOREIGN_PRIVATE_PAYLOAD", quantity=1, unit_cents=200000)],
        }
    )
    data = source.model_copy(
        update={
            "orders": [marked if o.order_id == marked.order_id else o for o in source.orders],
            "requests": [request],
        }
    )
    tools, _, security = fresh_tools(data, request, trusted_data=source)
    invoked = []
    original = tools._order

    def record(call, state):
        invoked.append(call.name)
        return original(call, state)

    tools._order = record
    try:
        trace = Agent(tools).run(request)
    except RuntimeError as error:
        assert str(error) == "named_review_unavailable:order_not_found"
        raise
    assert trace.decision.outcome == c.Outcome.REFER and not trace.decision.executed
    assert not tools.payments
    reviews = list(tools.reviews.values())
    assert len(reviews) == 1 and reviews[0].open
    assert reviews[0].required_role == "named_approver"
    review = reviews[0]
    if kind != "policy_denied":
        assert review.policy_reason == "order_not_found"
        packet = review.model_dump(mode="json")
        assert packet["request"] == request.model_dump(mode="json")
        assert packet["named_approver"] == source.policy_at(request.at).named_approver
        assert invoked == []
        assert "FOREIGN_PRIVATE_PAYLOAD" not in json.dumps(
            [packet, *[str(row) for row in observations(trace, tools, security.log)]]
        )
        assert all(step.state.order is None and step.result.order is None for step in trace.steps)
    else:
        assert review.policy_reason == "already_delivered"
        assert review.proposal.request == request
        assert (
            review.proposal.evidence.policy.named_approver
            == source.policy_at(request.at).named_approver
        )
    assert all(not a.money_moved and a.paid_amount_cents == 0 for a in tools.audits.values())


def test_core_fixture_gap_is_explicit_and_does_not_issue_from_attack():
    suite = build_suite(trials=1, components=(SELECTOR,))
    candidate = build_candidate(suite)
    case = next(case for case in suite.cases if case.part == "injection")
    with pytest.raises(ValueError, match="authenticated_runtime_required"):
        run_case(case, candidate)


def test_release_slot_registers_the_full_effect_suite():
    suite = build_suite(trials=1, components=(SELECTOR,))
    cases = [case for case in suite.cases if case.part == "injection"]
    assert len(cases) == 5
    complete = next(case for case in cases if case.failure_class == "full-effects")
    assert complete.must_pass and complete.handler == "security-privacy-injection"


def test_release_handler_runs_and_enforces_full_constructed_effect_checks(monkeypatch):
    from refund_agent.release import suite as core
    from refund_agent.release.rule import _trial
    from refund_agent.security import security_privacy_release_checks as checks
    from refund_agent.security.security_privacy_component import run_case as handle

    suite = build_suite(trials=1, components=(SELECTOR,))
    candidate = build_candidate(suite)
    case = next(case for case in suite.cases if case.failure_class == "full-effects")
    from refund_agent.agent import Agent
    from refund_agent.security.security_privacy_experiment import fresh_tools

    traces = []

    def schedule(**kwargs):
        tools, _, _ = fresh_tools(case.fixture.data, case.fixture.request)
        traces.append(Agent(tools).run(case.fixture.request))

    execution = core.CaseExecution(schedule, None, lambda: len(traces))
    report = checks.injection_checks()
    assert set(report) == {
        "attacks",
        "altered_role",
        "carrier",
        "reply_grid",
        "tier3_grid",
        "leakage",
        "scope_removed",
        "scope_skips_order",
        "absolute_positive",
        "relative_positive",
        "relative_public",
    }
    assert all(report.values())
    monkeypatch.setattr(checks, "injection_checks", lambda: report)
    assert handle(case=case, candidate=candidate, settings={}, execution=execution)
    assert execution.runs == 1 and traces[-1].receipt.amount_cents == 2499

    def constructed(case, candidate):
        return handle(case=case, candidate=candidate, settings={}, execution=execution)

    monkeypatch.setattr(core, "run_case", constructed)
    for name in report:
        broken = {**report, name: False}
        monkeypatch.setattr(checks, "injection_checks", lambda: broken)
        with pytest.raises(ValueError, match="security_privacy_injection_failed:" + name):
            handle(case=case, candidate=candidate, settings={}, execution=execution)
        assert _trial(case, candidate)["failed"] == ["trial_aborted"]


@pytest.mark.parametrize("secret", ["FIXTURE_KEY", "HASH_KEY", "credential_seal"])
@pytest.mark.parametrize("encoding", ["raw", "hexadecimal", "base64"])
def test_release_trial_aborts_on_secret_leakage(monkeypatch, secret, encoding):
    from base64 import b64encode

    from refund_agent.agent import Agent
    from refund_agent.release import suite as core
    from refund_agent.release.rule import _trial
    from refund_agent.security import security_privacy_experiment as fixtures
    from refund_agent.security import security_privacy_release_checks as checks
    from refund_agent.security.security_privacy_component import run_case as handle

    suite = build_suite(trials=1, components=(SELECTOR,))
    candidate = build_candidate(suite)
    case = next(case for case in suite.cases if case.failure_class == "full-effects")
    original = checks.attack_runs

    def leaking(**kwargs):
        results, rows, seals = original(**kwargs)
        value = seals[0] if secret == "credential_seal" else getattr(fixtures, secret)
        leaked = {
            "raw": value,
            "hexadecimal": value.hex(),
            "base64": b64encode(value).decode(),
        }[encoding]
        return results, (*rows, leaked), seals

    monkeypatch.setattr(checks, "attack_runs", leaking)
    traces = []

    def schedule(**kwargs):
        tools, _, _ = fixtures.fresh_tools(case.fixture.data, case.fixture.request)
        traces.append(Agent(tools).run(case.fixture.request))

    execution = core.CaseExecution(schedule, None, lambda: len(traces))
    with pytest.raises(ValueError, match="^security_privacy_injection_failed:leakage$"):
        handle(case=case, candidate=candidate, settings={}, execution=execution)
    assert execution.runs == 1 and traces[-1].receipt.amount_cents == 2499

    def constructed(case, candidate):
        return handle(case=case, candidate=candidate, settings={}, execution=execution)

    monkeypatch.setattr(core, "run_case", constructed)
    assert _trial(case, candidate) == {
        "failed": ["trial_aborted"],
        "status": "aborted",
        "error_type": "ValueError",
    }


@pytest.mark.parametrize("attack", ["replay", "expired", "revoked"])
def test_release_effect_checks_refuse_payments_under_invalid_credentials(monkeypatch, attack):
    from dataclasses import replace

    from refund_agent.security import security_privacy_release_checks as checks

    original = checks.attack_runs

    def paid_invalid(**kwargs):
        results, observations, seals = original(**kwargs)
        if not kwargs:
            results = tuple(
                replace(row, paid_cents=2499, outcome="approve", reason="eligible")
                if row.name == attack
                else row
                for row in results
            )
        return results, observations, seals

    monkeypatch.setattr(checks, "attack_runs", paid_invalid)
    assert not checks.injection_checks()["attacks"]


def test_unselected_security_never_enters_stage4_identity():
    base = build_candidate(build_suite(trials=1))
    selected = build_candidate(build_suite(trials=1, components=(SELECTOR,)))
    assert "components" not in base.tools_composition
    module = selected.tools_composition["components"][0]
    assert any("security_privacy_tools" in name for name in module["modules"])
    assert base.fingerprint != selected.fingerprint


def test_selected_code_edit_changes_only_selected_identity(monkeypatch):
    from pathlib import Path

    base_suite = build_suite(trials=1)
    scope_suite = build_suite(trials=1, components=(SELECTOR,))
    base, scoped = build_candidate(base_suite), build_candidate(scope_suite)
    read = Path.read_bytes

    def altered(path):
        value = read(path)
        return (
            value + b"\n# security-privacy identity mutation\n"
            if path.name == "security_privacy_tools.py"
            else value
        )

    monkeypatch.setattr(Path, "read_bytes", altered)
    assert build_candidate(base_suite).fingerprint == base.fingerprint
    assert build_candidate(scope_suite).fingerprint != scoped.fingerprint


def test_real_contract_and_event_composition_preserves_five_positive_payments():
    from refund_agent.batch_stream.generate import read_sources
    from refund_agent.batch_stream.view import first_complete
    from refund_agent.evaluation import _positive_data
    from refund_agent.security.security_privacy_attacks import approval_for
    from refund_agent.security.security_privacy_experiment import fixture_log, issue_run

    data = _positive_data()
    suite = build_suite(trials=1, components=(CONTRACTS, event_selector(), SELECTOR))
    candidate = build_candidate(suite)
    paid = []
    for request in data.requests:
        issuer, security = runtime(data)
        tools = build_tools(
            data, candidate, "execute", event_log=fixture_log(data), security_privacy=security
        )
        at = request.at
        issue_run(tools, issuer, request, at)
        trace = run_request(candidate, tools, request, at=at)
        if trace.decision.reason == "shipment_incomplete":
            at = first_complete(tools.event_log, request.at + tools.allowed_lateness, tools.margin)
            tools.data = read_sources(data, at)
            issue_run(tools, issuer, request, at)
            trace = run_request(candidate, tools, request, at=at)
        if request.amount_cents >= 5000:
            approval = approval_for(tools, trace)
            reviewer = next(r for r in data.reviewers if r.reviewer_id == "alex")
            assert tools.record_approval(
                approval, credential=issuer.issue_reviewer(reviewer, approval.approved_at)
            ).ok
            at = tools.run_time(request)
            issue_run(tools, issuer, request, at)
            trace = run_request(candidate, tools, request, at=at)
        assert trace.decision.executed and tools.lineage
        paid.append(trace.receipt.amount_cents)
    assert len(paid) == 5 and sum(paid) == 62598
