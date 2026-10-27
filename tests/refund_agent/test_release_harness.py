"""A rehearsal restart capability cannot authorize component process spawning."""

import subprocess
import sys
from types import ModuleType

import pytest

from refund_agent.release import agent_evaluation_resources as resources


@pytest.fixture
def component(tmp_path, monkeypatch):
    # Copied module identities must not contaminate later real-component selection.
    for name in ("READS", "WRITES", "IMPORTS"):
        monkeypatch.setattr(
            resources, name, {k: set(v) for k, v in getattr(resources, name).items()}
        )
    module = ModuleType("refund_agent._restart_component")
    module.__file__ = str(tmp_path / "component.py")
    module.subprocess, module.executable = subprocess, sys.executable
    return module


def define(module, source):
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    return module.run


def test_component_direct_restart_is_refused(component):
    run = define(component, "def run():\n    return subprocess.run([executable, '-c', 'pass'])\n")
    with pytest.raises(ValueError, match="component capability refused: process spawning"):
        run()


@pytest.mark.parametrize("copied_identity", [False, True])
def test_component_cannot_use_core_issued_rehearsal_restart(component, copied_identity):
    from refund_agent.canary.cloud_native_deployment_rehearsal import run_rehearsal

    component.restart = resources.rehearsal_restart()
    if copied_identity:
        component.__name__ = run_rehearsal.__module__
        component.__file__ = run_rehearsal.__wrapped__.__code__.co_filename
    source = "def run_rehearsal():\n    return restart([executable, '-c', 'pass'])\n"
    exec(compile(source, component.__file__, "exec"), component.__dict__)
    with pytest.raises(ValueError, match="rehearsal restart capability requires its driver"):
        component.run_rehearsal()


@pytest.fixture
def canary_candidate():
    from refund_agent.release.candidate import build_candidate
    from refund_agent.release.suite import build_suite

    build_candidate(build_suite(trials=1))
    from refund_agent.canary.cloud_native_deployment_component import SELECTOR

    suite = build_suite(trials=1, components=(SELECTOR,))
    return suite, build_candidate(suite)


def test_core_fixture_constructs_selected_canary_without_operational_authority(canary_candidate):
    from refund_agent.canary.cloud_native_deployment_binding import CanaryTools
    from refund_agent.release.candidate import build_tools
    from refund_agent.release.suite import _CaseRun

    suite, candidate = canary_candidate
    case = next(c for c in suite.cases if c.part == "public" and c.expected["paid_cents"])
    execution = _CaseRun(case, candidate, "execute")
    assert not isinstance(execution._tools, CanaryTools)
    assert not hasattr(execution._tools, "canary_session")
    execution.run()
    assert execution._artifacts().trace.decision.executed
    with pytest.raises(ValueError, match="canary_session_required"):
        build_tools(
            case.fixture.data, candidate, "execute", release_fixture={"authenticated": True}
        )


@pytest.mark.parametrize("copied_identity", [False, True])
def test_component_cannot_issue_core_fixture(component, canary_candidate, copied_identity):
    from refund_agent.release import suite as core

    suite, candidate = canary_candidate
    component.issue = core._fixture_tools
    component.candidate = candidate
    component.data = next(c.fixture.data for c in suite.cases if c.part == "public")
    if copied_identity:
        component.__name__ = core.__name__
        component.__file__ = core.__file__
    run = define(
        component, "def run():\n    return issue(data, candidate, 'execute', purpose='runtime')\n"
    )
    with pytest.raises(ValueError, match="fixture capability requires its core driver"):
        run()


def test_fixture_capability_is_bound_and_expires_after_construction(canary_candidate, monkeypatch):
    from dataclasses import replace

    from refund_agent.canary import cloud_native_deployment_component as canary
    from refund_agent.release import extensions
    from refund_agent.release.candidate import build_tools
    from refund_agent.release.suite import _CaseRun

    suite, candidate = canary_candidate
    case = next(c for c in suite.cases if c.part == "public" and c.expected["paid_cents"])
    captured = {}

    def observe(**kwargs):
        capability = kwargs["context"].get("release_fixture")
        if not resources.fixture_active(capability, candidate, "execute"):
            return canary.compose(**kwargs)
        captured["capability"] = capability
        assert resources.fixture_active(capability, candidate, "execute")
        assert not resources.fixture_active(object(), candidate, "execute")
        assert not resources.fixture_active(capability, replace(candidate), "execute")
        assert not resources.fixture_active(capability, candidate, "shadow")
        # Even a live token cannot be borrowed for another construction/data object.
        with pytest.raises(ValueError, match="canary_session_required"):
            build_tools(
                case.fixture.data.model_copy(deep=True), candidate, "execute", **kwargs["context"]
            )
        return canary.compose(**kwargs)

    monkeypatch.setitem(extensions._REGISTRY[canary.NAME].hooks, "compose", observe)
    _CaseRun(case, candidate, "execute")
    monkeypatch.undo()
    assert not resources.fixture_active(captured["capability"], candidate, "execute")
    with pytest.raises(ValueError, match="canary_session_required"):
        build_tools(case.fixture.data, candidate, "execute", release_fixture=captured["capability"])


def constructed_case(canary_candidate):
    suite, candidate = canary_candidate
    case = next(c for c in suite.cases if c.part == "public" and c.expected["paid_cents"])
    return case, candidate


def test_constructed_run_issues_core_capability_and_returns_detached_evidence(canary_candidate):
    case, candidate = constructed_case(canary_candidate)
    assert hasattr(resources, "run_constructed_request"), "core constructed-run API is absent"
    data = resources.ConstructedDataset(case.fixture.data, source="generated")
    result = resources.run_constructed_request(data, candidate, case.fixture.request)
    assert result.trace.decision.executed
    assert sum(p.amount_cents for p in result.payments) == case.expected["paid_cents"]
    assert not hasattr(result, "tools") and not hasattr(result, "canary_session")
    assert result.explanations == {} and result.held_requests == frozenset()
    assert case.fixture.data.correct  # Oracle labels were not consumed or mutated.


@pytest.mark.parametrize("source", ["real", None, "", True])
def test_constructed_run_refuses_real_or_unmarked_data(canary_candidate, source):
    case, candidate = constructed_case(canary_candidate)
    assert hasattr(resources, "run_constructed_request"), "core constructed-run API is absent"
    with pytest.raises(ValueError, match="generated dataset"):
        resources.run_constructed_request(
            resources.ConstructedDataset(case.fixture.data, source=source),
            candidate, case.fixture.request,
        )
    with pytest.raises(ValueError, match="generated dataset"):
        resources.run_constructed_request(case.fixture.data, candidate, case.fixture.request)


def test_constructed_capability_binding_borrowing_and_expiry(canary_candidate, monkeypatch):
    from dataclasses import replace

    from refund_agent.canary import cloud_native_deployment_component as canary
    from refund_agent.release import extensions
    from refund_agent.release.candidate import build_tools

    case, candidate = constructed_case(canary_candidate)
    assert hasattr(resources, "run_constructed_request"), "core constructed-run API is absent"
    captured = {}

    def observe(**kwargs):
        capability = kwargs["context"].get("release_fixture")
        if not resources.fixture_active(capability, candidate, "execute"):
            return canary.compose(**kwargs)
        assert "capability" not in captured, "a nested construction borrowed the token"
        captured["capability"] = capability
        bound = resources._FIXTURE.get()
        bound_data = bound[2]
        assert resources.fixture_active(capability, candidate, "execute")
        assert not resources.fixture_active(object(), candidate, "execute")
        assert not resources.fixture_active(capability, replace(candidate), "execute")
        assert not resources.fixture_active(capability, candidate, "shadow")
        token = resources._FIXTURE.set((*bound[:2], bound_data.model_copy(deep=True), *bound[3:]))
        try:
            assert not resources.fixture_active(capability, candidate, "execute")
        finally:
            resources._FIXTURE.reset(token)
        for data in (bound_data, bound_data.model_copy(deep=True)):
            with pytest.raises(ValueError, match="canary_session_required"):
                build_tools(data, candidate, "execute", **kwargs["context"])
        return canary.compose(**kwargs)

    monkeypatch.setitem(extensions._REGISTRY[canary.NAME].hooks, "compose", observe)
    result = resources.run_constructed_request(
        resources.ConstructedDataset(case.fixture.data, source="generated"),
        candidate, case.fixture.request,
    )
    monkeypatch.undo()
    assert result.trace.decision.executed
    assert resources._FIXTURE.get() is None
    assert not resources.fixture_active(captured["capability"], candidate, "execute")
    with pytest.raises(ValueError, match="canary_session_required"):
        build_tools(case.fixture.data, candidate, "execute", release_fixture=captured["capability"])


def test_constructed_run_capability_expires_before_agent_execution(canary_candidate, monkeypatch):
    from refund_agent.canary import cloud_native_deployment_component as canary
    from refund_agent.release import extensions

    case, candidate = constructed_case(canary_candidate)
    assert hasattr(resources, "run_constructed_request"), "core constructed-run API is absent"
    original = extensions._REGISTRY[canary.NAME].hooks["before_run"]

    def observe(**kwargs):
        assert resources._FIXTURE.get() is None
        return original(**kwargs)

    monkeypatch.setitem(extensions._REGISTRY[canary.NAME].hooks, "before_run", observe)
    result = resources.run_constructed_request(
        resources.ConstructedDataset(case.fixture.data, source="generated"),
        candidate, case.fixture.request,
    )
    assert result.trace.decision.executed


@pytest.mark.parametrize("key", ["canary_session", "release_fixture"])
def test_constructed_run_refuses_operational_authority_context(canary_candidate, key):
    case, candidate = constructed_case(canary_candidate)
    assert hasattr(resources, "run_constructed_request"), "core constructed-run API is absent"
    with pytest.raises(ValueError, match="operational authority"):
        resources.run_constructed_request(
            resources.ConstructedDataset(case.fixture.data, source="generated"),
            candidate, case.fixture.request, **{key: object()},
        )


def test_constructed_run_does_not_grant_named_payment_approval(canary_candidate):
    suite, candidate = canary_candidate
    assert hasattr(resources, "run_constructed_request"), "core constructed-run API is absent"
    case = next(c for c in suite.cases if c.part == "public" and c.fixture.request.amount_cents > 50000)
    result = resources.run_constructed_request(
        resources.ConstructedDataset(case.fixture.data, source="generated"),
        candidate, case.fixture.request,
    )
    assert not result.trace.decision.executed
    assert result.payments == ()
    assert not result.trace.steps[-1].state.approvals


def test_constructed_run_refuses_request_outside_generated_data(canary_candidate):
    case, candidate = constructed_case(canary_candidate)
    assert hasattr(resources, "run_constructed_request"), "core constructed-run API is absent"
    with pytest.raises(ValueError, match="request.*generated dataset"):
        resources.run_constructed_request(
            resources.ConstructedDataset(case.fixture.data, source="generated"),
            candidate, case.fixture.request.model_copy(update={"request_id": "outside"}),
        )


@pytest.mark.parametrize("failure,escape", [(False, "write"), (True, "write"), (True, "run")])
def test_constructed_constructor_retention_cannot_pay(
    canary_candidate, monkeypatch, failure, escape
):
    from types import SimpleNamespace

    from refund_agent.agent import Agent
    from refund_agent.canary import cloud_native_deployment_component as canary
    from refund_agent.release import extensions
    from refund_agent.release.candidate import run_request
    from refund_agent.tools import Tools

    case, candidate = constructed_case(canary_candidate)
    hooks = extensions._REGISTRY[canary.NAME].hooks
    compose = hooks["compose"]
    retained = []

    def compose_retaining(**kwargs):
        base, options = compose(**kwargs)

        class RetainingTools(base):
            def __init__(self, *args, **kwargs):
                write = self._write
                super().__init__(*args, **kwargs)
                # Capture the writer before the composed constructor returns to the driver.
                retained.append(SimpleNamespace(tools=self, write=write))
                if failure:
                    raise RuntimeError("constructed constructor failure")

        return RetainingTools, options

    ordinary = Tools(case.fixture.data)
    assert Agent(ordinary).run(case.fixture.request).decision.executed
    receipt = ordinary.payments[0]
    assert receipt.amount_cents == 2499
    monkeypatch.setitem(hooks, "compose", compose_retaining)
    dataset = resources.ConstructedDataset(case.fixture.data, source="generated")
    if failure:
        with pytest.raises(RuntimeError, match="constructed constructor failure"):
            resources.run_constructed_request(dataset, candidate, case.fixture.request)
    else:
        assert resources.run_constructed_request(
            dataset, candidate, case.fixture.request
        ).trace.decision.executed
    saved = retained[0]
    before = saved.tools.payments
    assert len(before) == (0 if failure else 1)
    assert resources._FIXTURE.get() is None
    try:
        with pytest.raises(ValueError, match="constructed run"):
            if escape == "write":
                saved.write("ledger", "constructor-escaped-payment", receipt)
            else:
                run_request(candidate, saved.tools, case.fixture.request)
    finally:
        assert saved.tools.payments == before


@pytest.mark.parametrize("failure", [None, "before_run", "after_run"])
@pytest.mark.parametrize("escape", ["run_request", "original_request", "commit", "write"])
def test_constructed_retained_tools_cannot_pay_after_return(
    canary_candidate,
    monkeypatch,
    failure,
    escape,
):
    from datetime import datetime, timezone
    from types import SimpleNamespace

    from refund_agent.agent import Agent
    from refund_agent.canary import cloud_native_deployment_component as canary
    from refund_agent.canary.cloud_native_deployment_rehearsal import fixture, prepare
    from refund_agent.release import extensions
    from refund_agent.release.candidate import build_tools, run_request
    from refund_agent.tools import Tools

    case, candidate = constructed_case(canary_candidate)
    hooks = extensions._REGISTRY[canary.NAME].hooks
    original = hooks["before_run"]
    original_after = hooks.get("after_run")
    retained = []

    def retain(**kwargs):
        tools = kwargs["tools"]
        retained.append(
            SimpleNamespace(
                tools=tools,
                commit=tools._commit,
                write=tools._write,
                run=lambda request: run_request(candidate, tools, request),
            )
        )
        if failure == "before_run":
            raise RuntimeError("constructed hook failure")
        return original(**kwargs)

    def fail_after(**kwargs):
        raise RuntimeError("constructed hook failure")

    monkeypatch.setitem(hooks, "before_run", retain)
    if failure == "after_run":
        monkeypatch.setitem(hooks, "after_run", fail_after)
    dataset = resources.ConstructedDataset(case.fixture.data, source="generated")
    if failure:
        with pytest.raises(RuntimeError, match="constructed hook failure"):
            resources.run_constructed_request(dataset, candidate, case.fixture.request)
    else:
        assert resources.run_constructed_request(
            dataset,
            candidate,
            case.fixture.request,
        ).trace.decision.executed
    monkeypatch.setitem(hooks, "before_run", original)
    if original_after is None:
        monkeypatch.delitem(hooks, "after_run", raising=False)
    else:
        monkeypatch.setitem(hooks, "after_run", original_after)
    saved = retained[0]
    tools = saved.tools
    assert resources._FIXTURE.get() is None and not hasattr(tools, "canary_session")
    before = tools.payments
    data, request = fixture(
        "retained-tools",
        "escape",
        datetime(2026, 10, 12, tzinfo=timezone.utc),
        amount=2499,
    )
    if escape != "original_request":
        tools.data = data
    with pytest.raises(ValueError, match="canary_session_required"):
        build_tools(data, candidate, "execute")
    # Real eligible state/receipt; the retained commit/write must enforce scope themselves.
    ordinary = Tools(data)
    state = prepare(ordinary, request, request.at)
    Agent(ordinary).run(request)
    receipt = ordinary.payments[0]
    assert receipt.amount_cents == 2499
    try:
        with pytest.raises(ValueError, match="constructed run"):
            if escape == "run_request":
                saved.run(request)  # Closure and attribute both retain the live object.
            elif escape == "original_request":
                saved.run(case.fixture.request)
            elif escape == "commit":
                saved.commit(state, "escaped-payment", 2499, None)
            else:
                saved.write("ledger", "escaped-payment", receipt)
    finally:
        assert tools.payments == before


@pytest.mark.parametrize("swap", ["dataset", "in_place", "identical_copy", "request", "same_id"])
def test_constructed_run_refuses_swapped_snapshot_in_scope(canary_candidate, monkeypatch, swap):
    from refund_agent.canary import cloud_native_deployment_component as canary
    from refund_agent.release import extensions
    from refund_agent.release.candidate import run_request

    case, candidate = constructed_case(canary_candidate)
    request = case.fixture.request
    other = request.model_copy(update={"request_id": request.request_id + "-swapped"})
    data = case.fixture.data.model_copy(deep=True)
    data.requests.append(other)
    data.retrievals.extend(
        r.model_copy(update={"request_id": other.request_id})
        for r in tuple(data.retrievals)
        if r.request_id == request.request_id
    )
    hooks = extensions._REGISTRY[canary.NAME].hooks
    original = hooks["before_run"]
    retained = []

    def swap_snapshot(**kwargs):
        tools = kwargs["tools"]
        retained.append(tools)
        monkeypatch.setitem(hooks, "before_run", original)
        if swap == "request":
            run_request(candidate, tools, other)
        elif swap == "same_id":
            run_request(candidate, tools, request.model_copy(update={"requester_id": "swapped"}))
        elif swap == "in_place":
            tools._data.public_scenarios.append("swapped")
        else:
            replacement = tools.data
            if swap == "dataset":
                replacement.public_scenarios.append("swapped")
            tools.data = replacement
        return original(**kwargs)

    monkeypatch.setitem(hooks, "before_run", swap_snapshot)
    try:
        with pytest.raises(ValueError, match="constructed run"):
            resources.run_constructed_request(
                resources.ConstructedDataset(data, source="generated"),
                candidate,
                request,
            )
    finally:
        assert retained[0].payments == []


@pytest.mark.parametrize("mutation", ["replacement", "in_place"])
def test_constructed_run_refuses_final_hook_snapshot_change(
    canary_candidate, monkeypatch, mutation
):
    from refund_agent.canary import cloud_native_deployment_component as canary
    from refund_agent.release import extensions

    case, candidate = constructed_case(canary_candidate)
    retained = []

    def change_after_run(*, value, tools, **kwargs):
        retained.append(tools)
        if mutation == "replacement":
            tools.data = tools.data
        else:
            tools._data.public_scenarios.append("changed-after-run")
        return value

    monkeypatch.setitem(extensions._REGISTRY[canary.NAME].hooks, "after_run", change_after_run)
    with pytest.raises(ValueError, match="constructed run data snapshot differs"):
        resources.run_constructed_request(
            resources.ConstructedDataset(case.fixture.data, source="generated"),
            candidate,
            case.fixture.request,
        )
    assert len(retained) == 1
    assert resources._CONSTRUCTED_EXECUTION.get() is None
    assert not retained[0]._constructed_execution.active


def test_constructed_payment_rechecks_snapshot_after_intent(canary_candidate, monkeypatch):
    from refund_agent.canary import cloud_native_deployment_component as canary
    from refund_agent.release import extensions

    case, candidate = constructed_case(canary_candidate)
    hooks = extensions._REGISTRY[canary.NAME].hooks
    original = hooks["before_run"]
    retained = []

    def change_after_intent(**kwargs):
        tools = kwargs["tools"]
        retained.append(tools)
        append = tools._checked_append

        def changing_append(record):
            result = append(record)
            if record.event == "intent":
                tools._data.public_scenarios.append("changed-after-intent")
            return result

        monkeypatch.setattr(tools, "_checked_append", changing_append)
        return original(**kwargs)

    monkeypatch.setitem(hooks, "before_run", change_after_intent)
    try:
        with pytest.raises(ValueError, match="constructed run"):
            resources.run_constructed_request(
                resources.ConstructedDataset(case.fixture.data, source="generated"),
                candidate,
                case.fixture.request,
            )
    finally:
        assert retained[0].payments == []


def test_constructed_returned_closure_cannot_pay(canary_candidate, monkeypatch):
    from datetime import datetime, timezone
    from types import SimpleNamespace

    from refund_agent.canary import cloud_native_deployment_component as canary
    from refund_agent.canary.cloud_native_deployment_rehearsal import fixture
    from refund_agent.release import extensions
    from refund_agent.release.candidate import run_request

    case, candidate = constructed_case(canary_candidate)
    retained = []

    def return_closure(*, value, tools, **kwargs):
        retained.append(tools)

        def resume():
            data, request = fixture(
                "returned-tools",
                "escape",
                datetime(2026, 10, 12, tzinfo=timezone.utc),
                amount=2499,
            )
            tools.data = data
            return run_request(candidate, tools, request)

        return SimpleNamespace(decision=value.decision, steps=value.steps, resume=resume)

    monkeypatch.setitem(extensions._REGISTRY[canary.NAME].hooks, "after_run", return_closure)
    result = resources.run_constructed_request(
        resources.ConstructedDataset(case.fixture.data, source="generated"),
        candidate,
        case.fixture.request,
    )
    before = retained[0].payments
    try:
        with pytest.raises(ValueError, match="constructed run"):
            result.trace.resume()
    finally:
        assert retained[0].payments == before


def test_constructed_scope_cannot_be_borrowed_by_another_stack(canary_candidate, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from contextvars import copy_context

    from refund_agent.canary import cloud_native_deployment_component as canary
    from refund_agent.release import extensions
    from refund_agent.release.candidate import run_request

    case, candidate = constructed_case(canary_candidate)
    hooks = extensions._REGISTRY[canary.NAME].hooks
    original = hooks["before_run"]

    def borrow(**kwargs):
        tools = kwargs["tools"]
        monkeypatch.setitem(hooks, "before_run", original)
        with ThreadPoolExecutor(max_workers=1) as worker:
            result = worker.submit(
                copy_context().run, run_request, candidate, tools, kwargs["request"]
            )
            try:
                with pytest.raises(ValueError, match="constructed run"):
                    result.result(timeout=30)
            finally:
                assert tools.payments == []
        return original(**kwargs)

    monkeypatch.setitem(hooks, "before_run", borrow)
    result = resources.run_constructed_request(
        resources.ConstructedDataset(case.fixture.data, source="generated"),
        candidate,
        case.fixture.request,
    )
    assert result.trace.decision.executed


@pytest.mark.parametrize("swap_log", [False, True])
def test_constructed_event_time_view_is_bound_to_snapshotted_log(
    canary_candidate,
    monkeypatch,
    swap_log,
):
    from refund_agent.batch_stream.component import selector
    from refund_agent.batch_stream.generate import snapshot_log
    from refund_agent.canary import cloud_native_deployment_component as canary
    from refund_agent.release import extensions
    from refund_agent.release.candidate import build_candidate
    from refund_agent.release.suite import build_suite

    case, _ = constructed_case(canary_candidate)
    candidate = build_candidate(build_suite(trials=1, components=(selector(), canary.SELECTOR)))
    hooks = extensions._REGISTRY[canary.NAME].hooks
    original = hooks["before_run"]
    retained = []

    def swap(**kwargs):
        retained.append(kwargs["tools"])
        kwargs["tools"].event_log = snapshot_log(case.fixture.data)
        return original(**kwargs)

    if swap_log:
        monkeypatch.setitem(hooks, "before_run", swap)
    dataset = resources.ConstructedDataset(case.fixture.data, source="generated")
    if swap_log:
        try:
            with pytest.raises(ValueError, match="constructed run"):
                resources.run_constructed_request(
                    dataset,
                    candidate,
                    case.fixture.request,
                    event_log=snapshot_log(case.fixture.data),
                )
        finally:
            assert retained[0].payments == []
    else:
        result = resources.run_constructed_request(
            dataset,
            candidate,
            case.fixture.request,
            event_log=snapshot_log(case.fixture.data),
        )
        assert result.trace.decision.executed
        assert sum(p.amount_cents for p in result.payments) == 2499


@pytest.mark.parametrize("mutation", ["replacement", "in_place"])
def test_constructed_scoped_view_checks_changes_before_restoring_data(
    canary_candidate, monkeypatch, mutation,
):
    from contextlib import contextmanager

    from refund_agent.canary import cloud_native_deployment_component as canary
    from refund_agent.release.candidate import build_candidate
    from refund_agent.release.suite import build_suite
    from refund_agent.security.security_privacy_component import SELECTOR
    from refund_agent.security.security_privacy_experiment import runtime
    from refund_agent.tools import Tools

    case, _ = constructed_case(canary_candidate)
    candidate = build_candidate(build_suite(trials=1, components=(canary.SELECTOR, SELECTOR)))
    issuer, security = runtime(case.fixture.data)
    boundary = Tools._boundary
    changed = []

    @contextmanager
    def change_view(tools, state, operation):
        with boundary(tools, state, operation):
            yield
            if operation == "completed_state":
                changed.append(tools)
                if mutation == "replacement":
                    tools.data = tools.data
                else:
                    tools._data.public_scenarios.append("changed-scoped-view")

    monkeypatch.setattr(Tools, "_boundary", change_view)
    with pytest.raises(ValueError, match="constructed run data snapshot differs"):
        resources.run_constructed_request(
            resources.ConstructedDataset(case.fixture.data, source="generated"),
            candidate, case.fixture.request,
            security_privacy=security,
            security_credential=issuer.issue_request(
                case.fixture.request, security.accounts, case.fixture.request.at,
            ),
        )
    assert changed and changed[0].payments == []
