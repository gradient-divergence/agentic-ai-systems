"""Restart proof and immediate budget stop on independently constructed records."""

import json
import subprocess
import sys
from types import SimpleNamespace

import pytest

from refund_agent.renewal.future_directions_watch import BudgetStatus, WatchStore, watch


class Referred:
    def __init__(self, ordinal):
        self.ordinal, self.request_id = ordinal, f"request-{ordinal}"
        self.paid, self.prediction = False, 0.04

    @property
    def found_false_30d(self):
        raise AssertionError("referred outcome read")


def rows(first, count, paid, wrong, final):
    return [
        SimpleNamespace(
            ordinal=first + i,
            request_id=f"request-{first + i}",
            paid=True,
            prediction=0.04,
            paid_day=final - 30,
            label_final_day=final,
            found_false_30d=int(i < wrong),
        )
        if i < paid
        else Referred(first + i)
        for i in range(count)
    ]


def setup(tmp_path):
    req = SimpleNamespace(fingerprint="requirement")
    adoption = {
        "verdict": "adopted",
        "stage": 6,
        "fingerprint": "adoption",
        "incumbent_bundle_fingerprint": "B6-bundle",
        "incumbent_candidate_fingerprint": "B6-candidate",
        "plan_fingerprint": "plan",
        "window_fingerprint": "window",
    }
    return (
        req,
        adoption,
        WatchStore(tmp_path / "future-directions-watch.jsonl", "adoption", "requirement"),
    )


def rollback(log):
    def call(**kwargs):
        assert kwargs["target"] == "B6-bundle"
        assert kwargs["target_candidate_fingerprint"] == "B6-candidate"
        log.append("rollback")
        return {
            "target_fingerprint": "B6-bundle",
            "target_candidate_fingerprint": "B6-candidate",
            "stage": 6,
            "verdict": "restored",
        }

    return call


def withdraw(log):
    def call(**kwargs):
        log.append("withdraw")
        return {"tier1_authority": "shadow", "decider": "person"}

    return call


def test_failure_survives_restart_with_separate_boundaries(tmp_path):
    req, adoption, store = setup(tmp_path)
    result = watch(rows(1, 1200, 200, 36, 60), adoption, req, 60, store, period="month-01")
    assert result.verdict == "pending"
    assert (
        result.state.failed_month == "month-01"
        and result.state.first == 1201
        and result.state.last == 3200
    )
    fresh = WatchStore(store.path, "adoption", "requirement")
    assert fresh.load() == result.state
    code = (
        "import sys; from refund_agent.renewal.future_directions_watch import WatchStore; "
        "print(WatchStore(sys.argv[1], 'adoption', 'requirement').load().model_dump_json())"
    )
    restored = json.loads(
        subprocess.check_output([sys.executable, "-c", code, str(store.path)], text=True)
    )
    assert restored["first"] == 1201 and restored["last"] == 3200
    assert restored["failed_month"] == "month-01" and restored["status"] == "pending"
    with pytest.raises(ValueError, match="cohort"):
        watch(rows(1, 2000, 500, 90, 91), adoption, req, 91, fresh, period="confirmation")


@pytest.mark.parametrize("wrong,status", [(20, "clear"), (90, "reverse")])
def test_confirmation_clears_or_reverses_only_matured_paid(tmp_path, wrong, status):
    req, adoption, store = setup(tmp_path)
    watch(rows(1, 1200, 200, 36, 60), adoption, req, 60, store, period="month-01")
    log = []
    result = watch(
        rows(1201, 2000, 500, wrong, 91),
        adoption,
        req,
        91,
        store,
        period="confirmation",
        rollback=rollback(log),
    )
    assert result.verdict == status
    assert log == (["rollback"] if status == "reverse" else [])
    assert store.load().grant_withdrawn is False


@pytest.mark.parametrize("field", ["target_fingerprint", "target_candidate_fingerprint"])
def test_rollback_must_restore_both_explicit_identities(tmp_path, field):
    from refund_agent.renewal.future_directions_watch import restore

    _, adoption, _ = setup(tmp_path)
    target = {
        "target_fingerprint": "B6-bundle",
        "target_candidate_fingerprint": "B6-candidate",
        "stage": 6,
        "verdict": "restored",
    }
    with pytest.raises(ValueError, match="admitted incumbent"):
        restore(adoption, rollback=lambda **k: {**target, field: "another-identity"}, day=91)


def test_immature_and_incomplete_confirmation_stay_pending(tmp_path):
    req, adoption, store = setup(tmp_path)
    watch(rows(1, 1200, 200, 36, 60), adoption, req, 60, store, period="month-01")
    for partial, day in ((rows(1201, 2000, 500, 90, 91), 90), (rows(1201, 1999, 500, 90, 91), 91)):
        result = watch(partial, adoption, req, day, store, period="confirmation")
        assert (
            result.state.status == "pending"
            and result.state.first == 1201
            and result.state.last == 3200
        )


@pytest.mark.parametrize("pending", [False, True])
def test_budget_exhaustion_withdraws_before_rollback_and_never_restores_grant(tmp_path, pending):
    from pathlib import Path

    # A runtime owns its resource history from a fresh interpreter, like captures.
    code = """
import runpy, sys
from pathlib import Path
source = Path(sys.argv[1])
sys.path.insert(0, str(source.parent))
runpy.run_path(str(source))['_budget_fixture_case'](Path(sys.argv[2]), sys.argv[3] == 'True')
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(Path(__file__).resolve()), str(tmp_path), str(pending)],
        text=True, capture_output=True, timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def _budget_fixture_case(tmp_path, pending):
    from test_future_directions_risk import component

    from refund_agent.canary.cloud_native_deployment_component import selector as canary_selector
    from refund_agent.canary.cloud_native_deployment_rehearsal import fixture
    from refund_agent.canary.cloud_native_deployment_store import AppendStore, PersistentTools
    from refund_agent.data.history import END_AT
    from refund_agent.future_directions_risk import mount
    from refund_agent.future_directions_risk import selector as risk_selector
    from refund_agent.release.candidate import build_candidate, run_request
    from refund_agent.release.extensions import RuntimeStore, resource_guard
    from refund_agent.release.suite import build_suite

    req, adoption, store = setup(tmp_path)
    if pending:
        watch(rows(1, 1200, 200, 36, 60), adoption, req, 60, store, period="month-01")
    data, request = fixture("future-directions-test", "watch-ledger", END_AT)
    ledger = AppendStore(tmp_path / "payments.jsonl", "runtime")
    mounted, kwargs = mount(
        PersistentTools, component(-10), policy_version=data.policies[0].version
    )
    selected = build_candidate(
        build_suite(
            components=(
                risk_selector(
                    kwargs["future_directions_component"], policy_version=data.policies[0].version
                ),
                canary_selector(
                    store=RuntimeStore(str(ledger.path), "morgan", "append"),
                    journal=RuntimeStore(str(tmp_path / "journal.jsonl"), "morgan", "append"),
                ),
            )
        )
    )

    @resource_guard(0)
    def initialize_and_pay(candidate):
        # Unit composition still declares every store through the public guard.
        runtime = SimpleNamespace(
            tools=mounted(data, canary_store=ledger, **kwargs), authority="stage6"
        )
        trace = run_request(build_candidate(build_suite()), runtime.tools, request)
        return runtime, trace

    runtime, trace = initialize_and_pay(selected)
    assert trace.decision.executed and len(runtime.tools.payments) == 1
    original = ledger.path.read_bytes()
    paid = tuple(runtime.tools.payments)
    log = []

    def connected_withdraw(**kwargs):
        runtime.authority = "shadow"
        return withdraw(log)(**kwargs)

    def connected_rollback(**kwargs):
        runtime.tools = PersistentTools(data, canary_store=ledger)
        return rollback(log)(**kwargs)

    spent = BudgetStatus(
        plan_fingerprint="plan",
        window_fingerprint="window",
        spent=True,
        status_fingerprint="budget",
    )
    result = watch(
        [],
        adoption,
        req,
        61,
        store,
        period="budget",
        budget=spent,
        withdraw=connected_withdraw,
        rollback=connected_rollback,
    )
    assert result.verdict == "withdrawn" and result.state.grant_withdrawn
    assert log == ["withdraw", "rollback"]
    assert runtime.authority == "shadow"
    assert not hasattr(runtime.tools, "future_directions_component")
    assert tuple(runtime.tools.payments) == paid
    fresh = WatchStore(store.path, "adoption", "requirement")
    assert fresh.load().grant_withdrawn
    watch(
        [],
        adoption,
        req,
        61,
        fresh,
        period="budget",
        budget=spent,
        withdraw=connected_withdraw,
        rollback=connected_rollback,
    )
    assert log == ["withdraw", "rollback"]
    assert watch([], adoption, req, 62, fresh, period="month").verdict == "withdrawn"
    assert ledger.path.read_bytes() == original
    assert tuple(PersistentTools(data, canary_store=ledger).payments) == paid
    assert runtime.authority == "shadow"


def test_budget_record_identity_is_not_optional(tmp_path):
    req, adoption, store = setup(tmp_path)
    other = BudgetStatus(
        plan_fingerprint="other",
        window_fingerprint="window",
        spent=True,
        status_fingerprint="budget",
    )
    with pytest.raises(ValueError, match="plan/window"):
        watch([], adoption, req, 1, store, period="budget", budget=other)
    assert not store.load().grant_withdrawn


def test_failed_withdrawal_leaves_durable_stop_intent(tmp_path):
    req, adoption, store = setup(tmp_path)
    spent = BudgetStatus(
        plan_fingerprint="plan",
        window_fingerprint="window",
        spent=True,
        status_fingerprint="budget",
    )

    def unavailable(**kwargs):
        raise OSError("withdrawal client failed")

    with pytest.raises(OSError):
        watch(
            [],
            adoption,
            req,
            1,
            store,
            period="budget",
            budget=spent,
            withdraw=unavailable,
            rollback=rollback([]),
        )
    assert WatchStore(store.path, "adoption", "requirement").load().grant_withdrawn


def test_absent_rollback_does_not_delay_budget_withdrawal(tmp_path):
    req, adoption, store = setup(tmp_path)
    spent = BudgetStatus(
        plan_fingerprint="plan",
        window_fingerprint="window",
        spent=True,
        status_fingerprint="budget",
    )
    log = []
    result = watch(
        [], adoption, req, 1, store, period="budget", budget=spent, withdraw=withdraw(log)
    )
    assert log == ["withdraw"] and result.state.grant_withdrawn
    assert result.state.rollback_record is None
    assert "awaits chapter 33" in result.detail


def test_journal_cannot_rebind_or_hide_tampering(tmp_path):
    req, adoption, store = setup(tmp_path)
    watch(rows(1, 1200, 200, 36, 60), adoption, req, 60, store, period="month")
    with pytest.raises(ValueError, match="another adoption"):
        WatchStore(store.path, "other", "requirement").load()
    store.path.write_text(store.path.read_text().replace('"first": 1201', '"first": 1'))
    with pytest.raises(ValueError, match="integrity"):
        store.load()
