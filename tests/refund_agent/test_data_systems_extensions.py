"""Chapter 30 selection, code-closure admission and ledger-key mutations."""

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from refund_agent import contracts as c
from refund_agent.evaluation import _positive_data
from refund_agent.model import ScriptedModel
from refund_agent.release import extensions
from refund_agent.release.candidate import PACKAGE, build_candidate, build_tools, run_request
from refund_agent.release.journal import Journal
from refund_agent.release.rule import evaluate_release, sign
from refund_agent.release.suite import CaseExecution, build_suite, run_case
from refund_agent.runtimes.faults import FaultTools
from refund_agent.runtimes.swap import _project
from refund_agent.shadow.__main__ import start
from refund_agent.sources import data_systems_release, data_systems_tools
from refund_agent.sources.data_systems_check import contract_problem
from refund_agent.sources.data_systems_release import COMPONENT, SELECTOR


def test_data_systems_contract_case_composes_with_event_time_ingress():
    from refund_agent.batch_stream.component import selector

    suite = build_suite(trials=1, components=[SELECTOR, selector()])
    candidate = build_candidate(suite)
    case = next(case for case in suite.cases if case.case_id == "slot-data-systems")
    result = run_case(case, candidate)
    assert not result["failed"]
    assert result["payments"] == 0
    assert result["decision"]["outcome"] == "refer"
    assert result["decision"]["reason"] == "contract_invariant:order.paid_equals_items"


def test_data_systems_handler_schedules_for_core_grading():
    suite = build_suite(trials=1, components=[SELECTOR])
    candidate = build_candidate(suite)
    case = next(case for case in suite.cases if case.case_id == "slot-data-systems")
    scheduled = []
    execution = CaseExecution(
        lambda **kwargs: scheduled.append(kwargs),
        lambda **kwargs: pytest.fail("contract referral must not schedule approval"),
        lambda: len(scheduled),
    )
    assert data_systems_release.run_case(
        case=case, candidate=candidate, settings={}, execution=execution
    ) is True
    assert scheduled == [{"at": None, "max_steps": None}]
    assert execution.runs == 1
    assert case.fixture.request.request_id == "r-7-0029"
    assert case.expected == {
        "paid_cents": 0,
        "outcome": "refer",
        "reason": "contract_invariant:order.paid_equals_items",
    }
    assert not run_case(case, candidate)["failed"]


def test_data_systems_core_catches_handler_without_order_guard(monkeypatch):
    suite = build_suite(trials=1, components=[SELECTOR])
    candidate = build_candidate(suite)
    case = next(case for case in suite.cases if case.case_id == "slot-data-systems")
    original = data_systems_tools.contract_problem
    # Keep the mutation in the component module so import refusal does not mask grading.
    namespace = {"__name__": data_systems_tools.__name__, "original": original}
    exec(
        "def contract_problem(*args, **kwargs):\n"
        "    return None if args[0] == 'order' else original(*args, **kwargs)\n",
        namespace,
    )
    monkeypatch.setattr(
        data_systems_tools,
        "contract_problem",
        namespace["contract_problem"],
    )
    result = run_case(case, candidate)
    assert result["payments"] == 0
    assert result["decision"]["outcome"] == "deny"
    assert result["decision"]["reason"] == "already_delivered"
    assert {"decision_outcome", "decision_reason"} <= set(
        result["failed"]
    )


def test_data_systems_foreign_guard_alias_is_refused(monkeypatch):
    suite = build_suite(trials=1, components=[SELECTOR])
    candidate = build_candidate(suite)
    case = next(case for case in suite.cases if case.case_id == "slot-data-systems")
    monkeypatch.setattr(data_systems_tools, "contract_problem", lambda *args, **kwargs: None)
    with pytest.raises(ValueError, match="first-party dependency outside refund_agent"):
        run_case(case, candidate)


@pytest.mark.parametrize("runtime", ["own", "second"])
@pytest.mark.parametrize("mode", ["execute", "shadow"])
@pytest.mark.parametrize("path", ["eligible", "denial", "contract", "model", "budget", "tool"])
def test_data_systems_named_approver_on_composed_paths(runtime, mode, path):
    data = _positive_data()
    request = data.requests[0].model_copy(update={"amount_cents": 65000})
    data = _project(data, request)
    if path == "denial":
        request = request.model_copy(update={"amount_cents": 1200000})
    elif path == "contract":
        shipment = data.shipments[0]
        data = data.model_copy(
            update={
                "shipments": [
                    shipment.model_copy(
                        update={
                            "events": [
                                event.model_copy(update={"status": "delivered_to_door"})
                                for event in shipment.events
                            ]
                        }
                    )
                ]
            }
        )

    class UnavailableModel(ScriptedModel):
        def reply(self, state):
            raise TimeoutError()

    candidate = build_candidate(build_suite(trials=1, components=[SELECTOR]), runtime=runtime)
    tools = build_tools(data, candidate, mode)
    trace = run_request(
        candidate,
        FaultTools(tools, "order.lookup") if path == "tool" else tools,
        request,
        model=UnavailableModel() if path == "model" else None,
        max_steps=1 if path == "budget" else None,
    )
    assert trace.decision.outcome == c.Outcome.REFER and not trace.decision.executed
    assert not tools.payments and not tools._approvals
    reviews = tools.shadow_reviews if mode == "shadow" else tools.reviews
    assert len(reviews) == 1
    review = next(iter(reviews.values()))
    assert review.open and review.required_role == "named_approver"
    assert review.proposal.request == request
    assert review.proposal.evidence.policy.named_approver == "morgan"
    assert (
        trace.decision.reason
        == {
            "eligible": "approval_required",
            "denial": "amount_over_policy",
            "contract": "contract_vocabulary:shipment.status",
            "model": "model_error",
            "budget": "step_budget_exhausted",
            "tool": "tool_error",
        }[path]
    )


def test_data_systems_unselected_registration_and_bytes_change_no_identity(monkeypatch):
    monkeypatch.delitem(extensions._REGISTRY, COMPONENT.name)
    before = build_candidate(build_suite(trials=1))
    extensions.register(COMPONENT)
    original = Path.read_bytes
    module = Path(data_systems_tools.__file__)
    monkeypatch.setattr(
        Path,
        "read_bytes",
        lambda path: original(path) + (b"\n" if path == module else b""),
    )
    after = build_candidate(build_suite(trials=1))
    assert after.record() == before.record()
    assert after.fingerprint == before.fingerprint


@pytest.mark.parametrize("changed", ["registration", "component", "core"])
def test_data_systems_byte_change_refuses_shadow(tmp_path, monkeypatch, changed):
    full = build_suite(trials=1, components=[SELECTOR])
    suite = replace(
        full,
        cases=tuple(case for case in full.cases if case.part in ("public", "data-systems")),
        generated=(),
    )
    candidate = build_candidate(suite)
    journal = Journal(tmp_path / "journal.jsonl")
    passed = evaluate_release(candidate, suite, journal)
    record = sign(passed, journal, "morgan", "2026-10-12T12:01:00+00:00")
    release = tmp_path / "release.json"
    release.write_text(json.dumps(record))
    target = {
        "registration": Path(data_systems_release.__file__),
        "component": Path(data_systems_tools.__file__),
        "core": PACKAGE / "tools.py",
    }[changed]
    original = Path.read_bytes
    monkeypatch.setattr(
        Path,
        "read_bytes",
        lambda path: original(path) + (b"\n" if path == target else b""),
    )
    rebuilt = build_candidate(suite)
    assert rebuilt.fingerprint != candidate.fingerprint
    assert suite.fingerprint != passed["suite_fingerprint"]
    with pytest.raises(ValueError, match="fingerprint"):
        start(release, suite=suite, journal=journal, output=tmp_path / "shadow.json")
    assert not (tmp_path / "shadow.json").exists()


def test_data_systems_required_contract_case_cannot_be_removed():
    suite = build_suite(trials=1, components=[SELECTOR])
    with pytest.raises(ValueError, match="must-pass"):
        suite.without("slot-data-systems")


def test_data_systems_duplicate_ledger_key_is_refused():
    receipt = c.Receipt.model_construct(amount_cents=100, idempotency_key="same")
    assert contract_problem("ledger", (receipt, receipt), "ledger@1", 0) == (
        "contract_invariant:ledger.unique_idempotency_key"
    )


@pytest.mark.parametrize("removed", ["vocabulary", "ledger-key"])
def test_data_systems_key_guard_mutation_is_caught(tmp_path, removed):
    from refund_agent.sources import data_systems_check

    source = Path(data_systems_check.__file__).read_text()
    guards = {
        "vocabulary": (
            '    if source == "shipment":\n'
            "        for event in record.events:\n"
            '            if event.status not in contract.vocabularies["status"]:\n'
            '                return "contract_vocabulary:shipment.status"\n'
        ),
        "ledger-key": (
            '        if len(keys) != len(set(keys)):\n            return "unique_idempotency_key"\n'
        ),
    }
    guard = guards[removed]
    test = (
        "rename_refers_instead_of_paying"
        if removed == "vocabulary"
        else "duplicate_ledger_key_is_refused"
    )
    test_file = (
        Path(__file__).with_name("test_data_systems.py")
        if removed == "vocabulary"
        else Path(__file__)
    )
    assert source.count(guard) == 1
    mutant = tmp_path / "check_without_key_guard.py"
    mutant.write_text(source.replace(guard, ""))
    runner = tmp_path / "key_guard_mutation.py"
    runner.write_text(
        "import sys\nfrom pathlib import Path\nimport pytest\n"
        "from refund_agent.release.suite import build_suite\n"
        "from refund_agent.sources import data_systems_check\n"
        "exec(compile(Path(sys.argv[1]).read_text(), sys.argv[1], 'exec'), "
        "data_systems_check.__dict__)\n"
        "from refund_agent.sources import data_systems_tools\n"
        "data_systems_tools.contract_problem = data_systems_check.contract_problem\n"
        "raise SystemExit(pytest.main([sys.argv[2], '-q', '-k', "
        "sys.argv[3], '-p', 'no:cacheprovider']))\n"
    )
    result = subprocess.run(
        [sys.executable, str(runner), str(mutant), str(test_file), test],
        capture_output=True,
        text=True,
        timeout=60,
    )
    (tmp_path / "key_guard_mutation.txt").write_text(result.stdout + result.stderr)
    assert result.returncode == 1, result.stdout + result.stderr
    assert "FAILED" in result.stdout and test in result.stdout
