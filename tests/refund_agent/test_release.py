"""Effects, independent comparisons and release authority on offline cases."""

import json
import subprocess
import sys
from dataclasses import replace
from fractions import Fraction
from itertools import combinations, product
from math import comb
from pathlib import Path

import pytest


@pytest.mark.parametrize("name", ["as-built", "resume-saved", "deny-all", "gate-removed"])
def test_release_command_persists_the_actual_verdict(tmp_path, name):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "refund_agent.release",
            "--candidate",
            name,
            "--trials",
            "1",
            "--run-dir",
            str(tmp_path),
            "--at",
            "2026-10-12T12:00:00+00:00",
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == int(name != "as-built"), result.stdout + result.stderr
    record = json.loads((tmp_path / "release.json").read_text())
    assert record["decision"] == ("pass" if name == "as-built" else "fail")
    assert bool(record["failed_cases"]) == (name != "as-built")
    assert all(case in result.stdout for case in record["failed_cases"])
    journaled = json.loads((tmp_path / "journal.jsonl").read_text())
    assert {k: v for k, v in journaled.items() if k != "entry"} == record


def test_estimators_match_outcome_blind_subset_enumeration():
    from refund_agent.release.trials import pass_at_k, pass_hat_k

    for n in range(1, 7):
        for outcomes in product((0, 1), repeat=n):
            for k in range(1, n + 1):
                subsets = [tuple(outcomes[i] for i in s) for s in combinations(range(n), k)]
                assert pass_at_k(n, sum(outcomes), k) == pytest.approx(
                    sum(any(s) for s in subsets) / comb(n, k)
                )
                assert pass_hat_k(n, sum(outcomes), k) == pytest.approx(
                    sum(all(s) for s in subsets) / comb(n, k)
                )
    assert Fraction(1) - Fraction(comb(3, 3), comb(12, 3)) == Fraction(219, 220)


@pytest.mark.parametrize("values", [(0, 0, 1), (5, 6, 1), (5, -1, 1), (5, 2, 6), (True, 0, 1)])
def test_invalid_trial_counts_are_refused(values):
    from refund_agent.release.trials import pass_at_k

    with pytest.raises(ValueError):
        pass_at_k(*values)


def test_judge_counts_and_undefined_kappa():
    from refund_agent.release.judges import agreement

    card = agreement(14, 6, 5, 175)
    assert card["agreement"] == 0.945 and card["chance"] == 0.824
    assert card["kappa"] == pytest.approx(0.6875) and card["miss_rate"] == 0.3
    assert agreement(0, 20, 0, 180)["kappa"] == 0
    assert agreement(0, 0, 0, 10)["kappa"] is None


@pytest.mark.parametrize("runtime", ["own", "second"])
@pytest.mark.parametrize(
    "name,expected",
    [
        ("as-built", "pass"),
        ("resume-saved", "fail"),
        ("deny-all", "fail"),
        ("gate-removed", "fail"),
    ],
)
def test_release_arms_and_baseline_floor(tmp_path, runtime, name, expected):
    from refund_agent.release.candidate import build_candidate
    from refund_agent.release.journal import Journal
    from refund_agent.release.rule import evaluate_release
    from refund_agent.release.suite import build_suite

    suite = build_suite(trials=1)
    candidate = build_candidate(suite, name=name, runtime=runtime)
    record = evaluate_release(candidate, suite, Journal(tmp_path / "journal.jsonl"), trials=1)
    assert record["decision"] == expected, record["failed_cases"]
    if name == "resume-saved":
        assert set(record["failed_cases"]) == {
            "continue-order",
            "continue-shipment",
            "continue-policy",
        }
        assert {r["part"] for r in record["failures"]} == {"state-continuation"}
    if name == "deny-all":
        assert record["floor"]["pass"]
        assert any(r["part"] == "positive" for r in record["failures"])


def test_failed_fingerprint_is_refused_across_invocations_and_new_candidate_keeps_history(tmp_path):
    from refund_agent.release.candidate import build_candidate
    from refund_agent.release.journal import Journal
    from refund_agent.release.rule import evaluate_release
    from refund_agent.release.suite import build_suite

    suite = replace(
        build_suite(trials=1),
        cases=tuple(c for c in build_suite(trials=1).cases if c.part == "public"),
    )
    failed = build_candidate(suite, name="deny-all")
    record = evaluate_release(failed, suite, Journal(tmp_path / "journal.jsonl"), trials=1)
    again = evaluate_release(failed, suite, Journal(tmp_path / "journal.jsonl"), trials=1)
    assert (
        again["decision"] == "fail" and again["status"] == "unchanged_candidate_previously_failed"
    )
    assert again["trials_run"] == 0
    changed = evaluate_release(
        build_candidate(suite),
        suite,
        Journal(tmp_path / "journal.jsonl"),
        trials=1,
        dispositions=[
            {
                "kind": "fixed",
                "record_id": record["record_id"],
                "case_id": case,
                "software_fingerprint": build_candidate(suite).fingerprint,
            }
            for case in record["failed_cases"]
        ],
    )
    assert changed["decision"] == "pass"
    assert changed["earlier_failures"][0]["record_id"] == record["record_id"]
    assert changed["cases_run"] == len(suite.cases)


def test_signature_and_override_require_a_listed_owner(tmp_path):
    from refund_agent.release.candidate import build_candidate
    from refund_agent.release.journal import Journal
    from refund_agent.release.rule import evaluate_release, sign
    from refund_agent.release.suite import build_suite

    suite = replace(
        build_suite(trials=1),
        cases=tuple(c for c in build_suite(trials=1).cases if c.part == "public"),
    )
    journal = Journal(tmp_path / "journal.jsonl")
    candidate = build_candidate(suite)
    with pytest.raises(ValueError, match="signer"):
        evaluate_release(
            candidate,
            suite,
            journal,
            trials=1,
            overrides=[{"case_id": "positive-2499", "reason": "waive"}],
        )
    record = evaluate_release(candidate, suite, journal, trials=1)
    with pytest.raises(ValueError, match="owner"):
        sign(record, journal, "refund-agent", "2026-10-12T12:00:00+00:00")
    signed = sign(record, journal, "morgan", "2026-10-12T12:00:00+00:00")
    assert signed["signature"]["owner"] == "morgan"


def test_soft_pass_does_not_clear_a_hard_failure(tmp_path):
    from refund_agent.release.candidate import build_candidate
    from refund_agent.release.journal import Journal
    from refund_agent.release.rule import evaluate_release
    from refund_agent.release.suite import build_suite

    suite = build_suite(trials=1)
    record = evaluate_release(
        build_candidate(suite, name="deny-all"),
        suite,
        Journal(tmp_path / "journal.jsonl"),
        trials=1,
        soft={"positive-2499": "pass"},
    )
    assert record["decision"] == "fail" and "positive-2499" in record["failed_cases"]


def test_candidate_fingerprint_binds_software_and_record_names_suite():
    from refund_agent.release.candidate import build_candidate
    from refund_agent.release.suite import build_suite

    suite = build_suite(trials=1)
    first = build_candidate(suite)
    for other in (
        build_candidate(suite, runtime="second"),
        build_candidate(suite, verifier="independent"),
        replace(first, model={**first.model, "version": "changed"}),
    ):
        assert first.fingerprint != other.fingerprint
    assert set(first.record()) == {
        "name",
        "agent",
        "model",
        "runtime",
        "adapter",
        "tools_composition",
        "offer_policy",
        "suite",
    }


def test_release_job_does_not_consult_golden_or_instructor_rows(monkeypatch, tmp_path):
    from refund_agent import split
    from refund_agent.release.candidate import build_candidate
    from refund_agent.release.journal import Journal
    from refund_agent.release.rule import evaluate_release
    from refund_agent.release.suite import build_suite

    def forbidden(*args, **kwargs):
        raise AssertionError("new frozen or held-out consultation")

    monkeypatch.setattr(split.GoldenSet, "consult", forbidden)
    monkeypatch.setattr(split, "freeze", forbidden)
    build_suite.cache_clear()
    suite = build_suite(trials=1)
    assert not any("instructor-" in case.case_id for case in suite.cases)
    result = evaluate_release(
        build_candidate(suite), suite, Journal(tmp_path / "journal.jsonl"), trials=1
    )
    assert result["decision"] == "pass"
    assert not result["failed_cases"]
    assert all("spent" in a["purpose"] or "record" in a["purpose"] for a in result["accesses"])


def test_fixture_cannot_raise_the_selected_candidate_step_budget():
    from refund_agent.release.candidate import build_candidate
    from refund_agent.release.suite import build_suite, run_case
    from refund_agent.runtimes.adapter import AdapterSettings

    suite = build_suite(trials=1)
    candidate = build_candidate(suite, settings=AdapterSettings(max_steps=1))
    case = next(c for c in suite.cases if c.case_id == "positive-2499")
    result = run_case(case, candidate)
    assert result["failed"] and not result["decision"]["executed"]
    assert result["decision"]["reason"] == "step_budget_exhausted"


@pytest.mark.parametrize(
    "relative,software_change",
    [
        ("baseline.py", True),
        ("memory/support.py", True),
        ("offpolicy/policies.py", True),
        ("runtimes/runner.py", True),
        ("evaluation.py", True),
    ],
)
def test_dependency_change_invalidates_the_appropriate_identity(
    relative, software_change, monkeypatch
):
    from refund_agent.release.candidate import PACKAGE, build_candidate
    from refund_agent.release.suite import build_suite

    frozen = build_suite(trials=1).requirement
    before = build_candidate(build_suite(trials=1))
    original = Path.read_bytes
    changed = PACKAGE / relative

    def modified(path):
        content = original(path)
        return content + b"\n# changed implementation\n" if path == changed else content

    monkeypatch.setattr(Path, "read_bytes", modified)
    renewed = {
        **frozen,
        "signature": {**frozen["signature"], "at": "2026-10-12T12:00:00+00:00"},
    }
    after = build_candidate(build_suite(trials=1, requirement=renewed))
    assert after.suite["fingerprint"] != before.suite["fingerprint"]
    assert (after.fingerprint != before.fingerprint) == software_change


def test_trial_count_is_fixed_in_the_suite_before_run(tmp_path):
    from refund_agent.release.candidate import build_candidate
    from refund_agent.release.journal import Journal
    from refund_agent.release.rule import evaluate_release
    from refund_agent.release.suite import build_suite

    suite = build_suite()
    with pytest.raises(ValueError, match="trial"):
        evaluate_release(
            build_candidate(suite), suite, Journal(tmp_path / "journal.jsonl"), trials=1
        )


@pytest.mark.parametrize(
    "change,reason",
    [
        ("order", "amount_over_policy"),
        ("shipment", "delivery_unknown"),
        ("policy", "amount_over_policy"),
    ],
)
@pytest.mark.parametrize("verifier", ["off", "independent"])
def test_resume_fault_and_unchanged_tool_on_each_live_source(change, reason, verifier):
    from refund_agent.release.candidate import build_candidate
    from refund_agent.release.suite import build_suite, run_case

    suite = build_suite(trials=1)
    case = next(c for c in suite.cases if c.case_id == f"continue-{change}")
    control = run_case(case, build_candidate(suite, verifier=verifier))
    fault = run_case(case, build_candidate(suite, name="resume-saved", verifier=verifier))
    assert control["paid_cents"] == 0 and control["reason"] == reason and not control["failed"]
    assert fault["paid_cents"] == 5000 and fault["failed"] == ["no_payment_on_changed_evidence"]


@pytest.mark.parametrize("mutation", ["omitted", "duplicate"])
def test_public_floor_requires_all_expected_unique_cases(tmp_path, mutation):
    from refund_agent.release.candidate import build_candidate
    from refund_agent.release.journal import Journal
    from refund_agent.release.rule import evaluate_release
    from refund_agent.release.suite import build_suite

    full = build_suite(trials=1)
    public = tuple(c for c in full.cases if c.part == "public")
    duplicate = next(c for c in public if c.tier == 1 and c.expected["outcome"] == "deny")
    cases = public[1:] if mutation == "omitted" else public + (duplicate,)
    suite = replace(full, cases=cases, generated=())
    record = evaluate_release(build_candidate(suite), suite, Journal(tmp_path / "journal"))
    assert record["decision"] == "fail"
    assert not record["floor"]["pass"]
    assert "drafts-to-shadow-floor" in record["failed_cases"]


def test_public_floor_rejects_incomplete_trials(tmp_path):
    from refund_agent.release.candidate import build_candidate
    from refund_agent.release.journal import Journal
    from refund_agent.release.rule import _floor, evaluate_release
    from refund_agent.release.suite import build_suite

    full = build_suite(trials=1)
    suite = replace(full, cases=tuple(c for c in full.cases if c.part == "public"), generated=())
    record = evaluate_release(build_candidate(suite), suite, Journal(tmp_path / "journal"))
    assert record["floor"]["pass"]
    record["case_results"][0]["trials"] = []
    assert not _floor(record["case_results"], 1, suite.requirement)["pass"]


def test_trial_error_is_journaled_and_retry_cannot_erase_it(tmp_path, monkeypatch):
    from refund_agent.release import suite as checks
    from refund_agent.release.candidate import build_candidate
    from refund_agent.release.journal import Journal
    from refund_agent.release.rule import evaluate_release

    full = checks.build_suite(trials=1)
    suite = replace(full, cases=tuple(c for c in full.cases if c.part == "public"), generated=())
    candidate, journal = build_candidate(suite), Journal(tmp_path / "journal")
    original = checks.run_case

    def raised(case, candidate):
        raise RuntimeError("trial unavailable")

    monkeypatch.setattr(checks, "run_case", raised)
    record = evaluate_release(candidate, suite, journal)
    assert record["decision"] == "fail"
    failure = journal.read()[0]["failures"][0]
    assert failure["case_id"] == suite.cases[0].case_id
    assert failure["trials"] == [1]
    assert record["case_results"][0]["trials"][0]["error_type"] == "RuntimeError"
    monkeypatch.setattr(checks, "run_case", original)
    retry = evaluate_release(candidate, suite, journal)
    assert retry["decision"] == "fail" and retry["cases_run"] == 0


def test_generated_trial_error_is_also_journaled_as_a_block(tmp_path, monkeypatch):
    from refund_agent.release import suite as checks
    from refund_agent.release.candidate import build_candidate
    from refund_agent.release.journal import Journal
    from refund_agent.release.rule import evaluate_release

    full = checks.build_suite(trials=1)
    suite = replace(full, cases=tuple(c for c in full.cases if c.part == "public"))
    original = checks._CaseRun.run

    def raised(execution, *args, **kwargs):
        if execution._case.part == "generated":
            raise RuntimeError("generated trial unavailable")
        return original(execution, *args, **kwargs)

    monkeypatch.setattr(checks._CaseRun, "run", raised)
    journal = Journal(tmp_path / "journal")
    record = evaluate_release(build_candidate(suite), suite, journal)
    assert record["decision"] == "fail"
    failure = journal.read()[0]["failures"][0]
    assert failure["part"] == "generated" and failure["trials"] == [1]
    assert failure["failed_assertions"] == ["trial_aborted"]


@pytest.mark.parametrize("runtime", ["own", "second"])
def test_mandatory_finalization_calls_are_traced_and_counted(runtime):
    from refund_agent.evaluation import _positive_data
    from refund_agent.release.candidate import build_candidate, build_tools, run_request
    from refund_agent.release.graders import process_grade
    from refund_agent.release.suite import build_suite
    from refund_agent.runtimes.adapter import AdapterSettings
    from refund_agent.runtimes.faults import FaultTools
    from refund_agent.runtimes.swap import _project

    data = _positive_data()
    request = data.requests[0].model_copy(update={"amount_cents": 65000})
    candidate = build_candidate(
        build_suite(trials=1), runtime=runtime, settings=AdapterSettings(max_steps=1)
    )
    tools = build_tools(_project(data, request), candidate, "execute")
    recorded = FaultTools(tools)
    trace = run_request(candidate, recorded, request)
    calls = [s.call.model_dump(mode="json") for s in trace.steps if s.call]
    assert calls == recorded.calls and len(calls) == 6
    assert len(tools.reviews) == 1 and not tools.payments
    assert process_grade(trace, max_steps=1) is not None


def test_named_review_keeps_assessed_reason_separate_and_grades_it():
    from refund_agent.evaluation import _positive_data
    from refund_agent.release.candidate import build_candidate, build_tools, run_request
    from refund_agent.release.graders import effects
    from refund_agent.release.suite import build_suite
    from refund_agent.runtimes.swap import _project

    data = _positive_data()
    request = data.requests[0].model_copy(update={"amount_cents": 65000, "reason": "unknown"})
    tools = build_tools(_project(data, request), build_candidate(build_suite(trials=1)), "execute")
    trace = run_request(build_candidate(build_suite(trials=1)), tools, request)
    review = next(iter(tools.reviews.values()))
    assert review.proposal.reason == "unknown" and review.policy_reason == "unknown_reason"
    assert not effects(
        trace, tools, paid_cents=0, named_reason="unknown_reason", policy_reason="unknown_reason"
    )
    tools._write(
        "reviews", review.review_id, review.model_copy(update={"policy_reason": "unknown"})
    )
    assert "named_approver_packet_no_payment" in effects(
        trace, tools, paid_cents=0, named_reason="unknown_reason", policy_reason="unknown_reason"
    )


@pytest.mark.parametrize("paid_cents", [0, 4999])
def test_fixed_scenario_wiring_refuses_contradictory_submission(tmp_path, paid_cents):
    from refund_agent.release.suite import add_scenario, build_suite

    row = {
        "case_id": "submitted",
        "part": "approval",
        "failure_class": "approval-reuse",
        "tier": 2,
        "handler": "approval-reuse",
        "must_pass": True,
        "expected": {
            "stores": ["ledger", "audits", "reviews", "decision"],
            "paid_cents": paid_cents,
        },
    }
    path = tmp_path / "scenario.json"
    path.write_text(json.dumps(row))
    with pytest.raises(ValueError, match="fixed"):
        add_scenario(build_suite(trials=1), path)


def test_signed_adjudication_is_only_for_named_soft_holds(tmp_path):
    from refund_agent.release.candidate import build_candidate
    from refund_agent.release.journal import Journal
    from refund_agent.release.rule import evaluate_release
    from refund_agent.release.suite import build_suite

    full = build_suite(trials=1)
    suite = replace(full, cases=tuple(c for c in full.cases if c.part == "public"), generated=())
    candidate = build_candidate(suite, name="deny-all")
    waiver = {
        "case_id": suite.cases[0].case_id,
        "software_fingerprint": candidate.fingerprint,
        "signer": "morgan",
        "reason": "reviewed",
        "at": "2026-10-12T12:00:00+00:00",
    }
    with pytest.raises(ValueError, match="soft"):
        evaluate_release(
            candidate,
            suite,
            Journal(tmp_path / "hard"),
            overrides=[waiver],
        )
    hard = evaluate_release(
        candidate,
        suite,
        Journal(tmp_path / "both"),
        overrides=[waiver],
        soft={waiver["case_id"]: "hold"},
    )
    assert hard["decision"] == "fail" and hard["failed_cases"]
    waiver["case_id"] = "letter-explanation"
    waiver["software_fingerprint"] = build_candidate(suite).fingerprint
    record = evaluate_release(
        build_candidate(suite),
        suite,
        Journal(tmp_path / "soft"),
        overrides=[waiver],
        soft={"letter-explanation": "hold"},
    )
    assert record["decision"] == "pass"
    assert record["overrides"][0]["kind"] == "soft-hold-adjudication"


def test_suite_change_preserves_software_failure_until_signed_retirement(tmp_path, monkeypatch):
    from refund_agent.release import suite as checks
    from refund_agent.release.candidate import build_candidate
    from refund_agent.release.journal import Journal
    from refund_agent.release.rule import evaluate_release, sign

    full = checks.build_suite(trials=1)
    public = tuple(c for c in full.cases if c.part == "public")
    extra = next(c for c in full.cases if c.case_id == "continue-order")
    before = replace(full, cases=public + (extra,), generated=())
    after = replace(full, cases=public, generated=())
    first, second = build_candidate(before), build_candidate(after)
    journal = Journal(tmp_path / "journal")
    original = checks.run_case

    def planted(case, candidate):
        return (
            {"failed": ["planted"]} if case.case_id == extra.case_id else original(case, candidate)
        )

    monkeypatch.setattr(checks, "run_case", planted)
    failed = evaluate_release(first, before, journal)
    monkeypatch.setattr(checks, "run_case", original)
    assert first.fingerprint == second.fingerprint
    assert first.suite["fingerprint"] != second.suite["fingerprint"]
    blocked = evaluate_release(second, after, journal)
    assert blocked["decision"] == "fail" and blocked["cases_run"] == 0
    disposition = {
        "record_id": failed["record_id"],
        "case_id": extra.case_id,
        "software_fingerprint": second.fingerprint,
        "kind": "retired",
        "reason": "redundant case retired by owner",
        "signer": "morgan",
        "at": "2026-10-12T12:00:00+00:00",
    }
    record = evaluate_release(second, after, journal, dispositions=[disposition])
    assert record["decision"] == "pass" and record["failure_dispositions"] == [disposition]
    assert sign(record, journal, "morgan", "2026-10-12T12:01:00+00:00")["signature"]
    from refund_agent.shadow.__main__ import validate_release

    validate_release(
        sign(record, journal, "morgan", "2026-10-12T12:02:00+00:00"), second, after, journal
    )


@pytest.fixture
def continuation_suite():
    from refund_agent.release.suite import build_suite

    full = build_suite(trials=1)
    return replace(
        full,
        cases=tuple(c for c in full.cases if c.part == "public" or c.case_id == "continue-order"),
        generated=(),
    )


def test_fixed_candidate_cannot_admit_original_software_with_failure_omitted(
    tmp_path, continuation_suite
):
    from refund_agent.release.candidate import build_candidate
    from refund_agent.release.journal import Journal
    from refund_agent.release.rule import evaluate_release, sign
    from refund_agent.shadow.__main__ import start, validate_release

    suite = continuation_suite
    journal = Journal(tmp_path / "journal.jsonl")
    original = build_candidate(suite, name="resume-saved")
    failed = evaluate_release(original, suite, journal)
    assert failed["failed_cases"] == ["continue-order"]
    repaired = build_candidate(suite)
    disposition = {
        "kind": "fixed",
        "record_id": failed["record_id"],
        "case_id": "continue-order",
        "software_fingerprint": repaired.fingerprint,
    }
    passed = evaluate_release(repaired, suite, journal, dispositions=[disposition])
    assert passed["decision"] == "pass"
    signed = sign(passed, journal, "morgan", "2026-10-12T12:01:00+00:00")
    validate_release(signed, repaired, suite, journal)
    reduced = suite.without("continue-order")
    returned = build_candidate(reduced, name="resume-saved")
    assert returned.fingerprint == original.fingerprint
    blocked = evaluate_release(returned, reduced, journal)
    assert blocked["decision"] == "fail"
    assert blocked["status"] == "unchanged_candidate_previously_failed"
    assert blocked["cases_run"] == 0 and blocked["failed_cases"] == ["continue-order"]
    assert journal.failures(original.fingerprint) == [
        {
            "record_id": failed["record_id"],
            "case_id": "continue-order",
            "software_fingerprint": original.fingerprint,
        }
    ]
    with pytest.raises(ValueError, match="passing candidate"):
        sign(blocked, journal, "morgan", "2026-10-12T12:02:00+00:00")
    path, output = tmp_path / "release.json", tmp_path / "shadow.json"
    path.write_text(json.dumps(blocked))
    with pytest.raises(ValueError, match="not a pass"):
        start(path, candidate=returned, suite=reduced, journal=journal, output=output)
    assert not output.exists()


def test_fix_for_version_a_does_not_clear_version_b(tmp_path, continuation_suite):
    from refund_agent.release.candidate import build_candidate
    from refund_agent.release.journal import Journal
    from refund_agent.release.rule import evaluate_release

    suite = continuation_suite
    journal = Journal(tmp_path / "journal.jsonl")
    failed = evaluate_release(build_candidate(suite, name="resume-saved"), suite, journal)
    version_a = build_candidate(suite)
    version_b = build_candidate(suite, runtime="second")
    assert version_a.fingerprint != version_b.fingerprint
    disposition = {
        "kind": "fixed",
        "record_id": failed["record_id"],
        "case_id": "continue-order",
        "software_fingerprint": version_a.fingerprint,
    }
    assert (
        evaluate_release(version_a, suite, journal, dispositions=[disposition])["decision"]
        == "pass"
    )
    blocked = evaluate_release(version_b, suite, journal)
    assert not blocked["failed_cases"]  # Its own trials pass; version A's evidence cannot cover B.
    assert blocked["decision"] == "fail"
    assert blocked["unresolved_earlier_failures"][0]["record_id"] == failed["record_id"]
    disposition = {**disposition, "software_fingerprint": version_b.fingerprint}
    assert (
        evaluate_release(version_b, suite, journal, dispositions=[disposition])["decision"]
        == "pass"
    )
    assert evaluate_release(version_a, suite, journal)["decision"] == "pass"


@pytest.mark.parametrize("target", [None, "another-software"])
def test_retirement_requires_the_software_it_covers(tmp_path, continuation_suite, target):
    from refund_agent.release.candidate import build_candidate
    from refund_agent.release.journal import Journal
    from refund_agent.release.rule import evaluate_release

    suite = continuation_suite
    journal = Journal(tmp_path / "journal.jsonl")
    failed = evaluate_release(build_candidate(suite, name="resume-saved"), suite, journal)
    reduced = suite.without("continue-order")
    disposition = {
        "kind": "retired",
        "record_id": failed["record_id"],
        "case_id": "continue-order",
        "signer": "morgan",
        "reason": "case retired by owner",
        "at": "2026-10-12T12:01:00+00:00",
    }
    if target is not None:
        disposition["software_fingerprint"] = target
    with pytest.raises(ValueError, match="software fingerprint"):
        evaluate_release(
            build_candidate(reduced, name="resume-saved"),
            reduced,
            journal,
            dispositions=[disposition],
        )


def test_signed_retirement_covers_only_named_software(tmp_path, continuation_suite):
    from refund_agent.release.candidate import build_candidate
    from refund_agent.release.journal import Journal
    from refund_agent.release.rule import evaluate_release, sign

    suite = continuation_suite
    journal = Journal(tmp_path / "journal.jsonl")
    original = build_candidate(suite, name="resume-saved")
    failed = evaluate_release(original, suite, journal)
    reduced = suite.without("continue-order")
    other = build_candidate(reduced)
    retirement = {
        "kind": "retired",
        "record_id": failed["record_id"],
        "case_id": "continue-order",
        "software_fingerprint": other.fingerprint,
        "signer": "morgan",
        "reason": "case retired by owner",
        "at": "2026-10-12T12:01:00+00:00",
    }
    passed = evaluate_release(other, reduced, journal, dispositions=[retirement])
    assert sign(passed, journal, "morgan", "2026-10-12T12:02:00+00:00")["signature"]
    returned = build_candidate(reduced, name="resume-saved")
    assert (
        evaluate_release(returned, reduced, journal)["status"]
        == "unchanged_candidate_previously_failed"
    )
    retirement = {**retirement, "software_fingerprint": returned.fingerprint}
    passed = evaluate_release(returned, reduced, journal, dispositions=[retirement])
    assert passed["decision"] == "pass" and not journal.failures(returned.fingerprint)
    assert sign(passed, journal, "morgan", "2026-10-12T12:03:00+00:00")["signature"]


def test_new_software_requires_explicit_fix_dispositions(tmp_path):
    from refund_agent.release.candidate import build_candidate
    from refund_agent.release.journal import Journal
    from refund_agent.release.rule import evaluate_release
    from refund_agent.release.suite import build_suite

    full = build_suite(trials=1)
    suite = replace(full, cases=tuple(c for c in full.cases if c.part == "public"), generated=())
    journal = Journal(tmp_path / "journal")
    failed = evaluate_release(build_candidate(suite, name="deny-all"), suite, journal)
    candidate = build_candidate(suite)
    blocked = evaluate_release(candidate, suite, journal)
    assert blocked["decision"] == "fail" and blocked["unresolved_earlier_failures"]
    dispositions = [
        {
            "record_id": failed["record_id"],
            "case_id": case,
            "kind": "fixed",
            "software_fingerprint": candidate.fingerprint,
        }
        for case in failed["failed_cases"]
    ]
    record = evaluate_release(candidate, suite, journal, dispositions=dispositions)
    assert record["decision"] == "pass"
    assert record["failure_dispositions"] == dispositions


def test_offer_evaluation_evidence_is_not_software_identity():
    from refund_agent.release.candidate import build_candidate
    from refund_agent.release.suite import build_suite

    candidate = build_candidate(build_suite(trials=1))
    evidence = replace(
        candidate,
        offer_policy={
            **candidate.offer_policy,
            "evaluation_fingerprint": "new-evaluation",
            "limits": ["new evaluation limit"],
        },
    )
    assert candidate.record() != evidence.record()
    assert candidate.fingerprint == evidence.fingerprint


def test_referral_ceiling_belongs_to_signed_requirement_not_software(tmp_path):
    from refund_agent.release.candidate import build_candidate
    from refund_agent.release.journal import Journal
    from refund_agent.release.rule import evaluate_release
    from refund_agent.release.suite import build_suite

    full = build_suite(trials=1)
    suite = replace(full, cases=tuple(c for c in full.cases if c.part == "public"), generated=())
    stricter = replace(
        suite,
        requirement={
            **suite.requirement,
            "max_referral_share": 0,
            "signature": {**suite.requirement["signature"], "at": "2026-10-12T13:00:00+00:00"},
        },
    )
    assert build_candidate(suite).fingerprint == build_candidate(stricter).fingerprint
    assert suite.fingerprint != stricter.fingerprint
    record = evaluate_release(build_candidate(stricter), stricter, Journal(tmp_path / "strict"))
    assert record["decision"] == "fail" and not record["floor"]["pass"]
    relaxed = replace(
        suite,
        requirement={
            **suite.requirement,
            "max_referral_share": 1,
            "signature": {**suite.requirement["signature"], "at": "2026-10-12T14:00:00+00:00"},
        },
    )
    deny = evaluate_release(
        build_candidate(relaxed, name="deny-all"), relaxed, Journal(tmp_path / "deny")
    )
    assert deny["floor"]["pass"] and deny["decision"] == "fail"
    assert any(case.startswith("public-") for case in deny["failed_cases"])


@pytest.mark.parametrize("mutation", ["unsigned", "negative", "above-one"])
def test_requirement_must_be_signed_with_a_valid_share(tmp_path, mutation):
    from refund_agent.release.candidate import build_candidate
    from refund_agent.release.journal import Journal
    from refund_agent.release.rule import evaluate_release
    from refund_agent.release.suite import build_suite

    suite = build_suite(trials=1)
    requirement = {**suite.requirement}
    requirement.update(
        {"signature": None}
        if mutation == "unsigned"
        else {"max_referral_share": -1 if mutation == "negative" else 2}
    )
    suite = replace(suite, requirement=requirement)
    with pytest.raises(ValueError, match="requirement"):
        evaluate_release(build_candidate(suite), suite, Journal(tmp_path / "journal"))


@pytest.mark.parametrize("mutation", ["unsigned-retirement", "unchanged-fix", "failed-fix"])
def test_disposition_cannot_hide_an_unfixed_failure(tmp_path, mutation):
    from refund_agent.release.candidate import build_candidate
    from refund_agent.release.journal import Journal
    from refund_agent.release.rule import evaluate_release
    from refund_agent.release.suite import build_suite

    full = build_suite(trials=1)
    suite = replace(full, cases=tuple(c for c in full.cases if c.part == "public"), generated=())
    journal = Journal(tmp_path / "journal")
    failed_candidate = build_candidate(suite, name="deny-all")
    failed = evaluate_release(failed_candidate, suite, journal)
    case = failed["failed_cases"][0]
    if mutation == "unsigned-retirement":
        suite = suite.without(case)
        disposition = {
            "kind": "retired",
            "record_id": failed["record_id"],
            "case_id": case,
            "reason": "remove",
        }
        with pytest.raises(ValueError, match="signer"):
            evaluate_release(build_candidate(suite), suite, journal, dispositions=[disposition])
    else:
        candidate = (
            failed_candidate
            if mutation == "unchanged-fix"
            else replace(failed_candidate, model={**failed_candidate.model, "version": "new"})
        )
        disposition = {
            "kind": "fixed",
            "record_id": failed["record_id"],
            "case_id": case,
            "software_fingerprint": candidate.fingerprint,
        }
        if mutation == "unchanged-fix":
            with pytest.raises(ValueError, match="new software"):
                evaluate_release(candidate, suite, journal, dispositions=[disposition])
        else:
            record = evaluate_release(candidate, suite, journal, dispositions=[disposition])
            assert record["decision"] == "fail" and not record["failure_dispositions"]


def test_release_floor_cannot_be_retired_as_a_scenario(tmp_path):
    from refund_agent.release.candidate import build_candidate
    from refund_agent.release.journal import Journal
    from refund_agent.release.rule import evaluate_release
    from refund_agent.release.suite import build_suite

    full = build_suite(trials=1)
    public = replace(full, cases=tuple(c for c in full.cases if c.part == "public"), generated=())
    strict = replace(
        public,
        requirement={
            **public.requirement,
            "max_referral_share": 0,
            "signature": {**public.requirement["signature"], "at": "2026-10-12T13:00:00+00:00"},
        },
    )
    journal = Journal(tmp_path / "journal")
    failed = evaluate_release(build_candidate(strict), strict, journal)
    disposition = {
        "record_id": failed["record_id"],
        "case_id": "drafts-to-shadow-floor",
        "kind": "retired",
        "signer": "morgan",
        "reason": "loosen the floor",
        "at": "2026-10-12T12:00:00+00:00",
    }
    with pytest.raises(ValueError, match="requirement"):
        evaluate_release(build_candidate(public), public, journal, dispositions=[disposition])


def test_new_candidate_can_explicitly_fix_trial_abort_and_incomplete_floor(tmp_path, monkeypatch):
    from refund_agent.release import suite as checks
    from refund_agent.release.candidate import build_candidate
    from refund_agent.release.journal import Journal
    from refund_agent.release.rule import evaluate_release
    from refund_agent.runtimes.adapter import AdapterSettings

    full = checks.build_suite(trials=1)
    suite = replace(full, cases=tuple(c for c in full.cases if c.part == "public"), generated=())
    original = checks.run_case
    journal = Journal(tmp_path / "journal")

    def raised(case, candidate):
        if case.case_id == suite.cases[0].case_id:
            raise RuntimeError("trial unavailable")
        return original(case, candidate)

    monkeypatch.setattr(checks, "run_case", raised)
    failed = evaluate_release(build_candidate(suite), suite, journal)
    monkeypatch.setattr(checks, "run_case", original)
    candidate = build_candidate(suite, settings=AdapterSettings(max_steps=11))
    dispositions = [
        {
            "kind": "fixed",
            "record_id": failed["record_id"],
            "case_id": case,
            "software_fingerprint": candidate.fingerprint,
        }
        for case in failed["failed_cases"]
    ]
    record = evaluate_release(candidate, suite, journal, dispositions=dispositions)
    assert record["decision"] == "pass" and record["failure_dispositions"] == dispositions
