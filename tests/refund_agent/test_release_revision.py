"""The blind reviews' persistence, signed ceiling and freshness regressions."""

from dataclasses import replace
from datetime import timedelta

import pytest

from refund_agent import contracts as c
from refund_agent.release.candidate import build_candidate
from refund_agent.release.journal import Journal
from refund_agent.release.rule import evaluate_release, sign
from refund_agent.release.suite import build_suite, run_case


@pytest.mark.parametrize("second", ["omitted", "pass", "adjudicated"])
def test_soft_hold_survives_a_second_invocation(tmp_path, second):
    full = build_suite(trials=1)
    suite = replace(full, cases=tuple(c for c in full.cases if c.part == "public"), generated=())
    candidate = build_candidate(suite)
    first = evaluate_release(
        candidate, suite, Journal(tmp_path / "journal"), soft={"clarity": "hold"}
    )
    assert first["failed_cases"] == ["soft-hold:clarity"]
    disposition = {
        "kind": "soft-hold-adjudication",
        "record_id": first["record_id"],
        "case_id": "soft-hold:clarity",
        "software_fingerprint": candidate.fingerprint,
        "signer": "morgan",
        "reason": "Reviewed the held case against the recorded evidence",
        "at": "2026-10-12T12:01:00+00:00",
    }
    journal = Journal(tmp_path / "journal")
    again = evaluate_release(
        candidate,
        suite,
        journal,
        soft={"clarity": "pass"} if second == "pass" else {},
        dispositions=[disposition] if second == "adjudicated" else (),
    )
    if second == "adjudicated":
        assert again["decision"] == "pass" and not journal.failures(candidate.fingerprint)
        assert sign(again, journal, "morgan", "2026-10-12T12:02:00+00:00")["signature"]
    else:
        assert again["status"] == "unchanged_candidate_previously_failed"
        assert again["cases_run"] == 0 and again["failed_cases"] == ["soft-hold:clarity"]
        with pytest.raises(ValueError, match="passing candidate"):
            sign(again, journal, "morgan", "2026-10-12T12:02:00+00:00")


def test_default_ceiling_refuses_changed_baseline_but_accepts_new_signoff(monkeypatch):
    from refund_agent.release import suite as checks

    original = checks.baseline
    frozen = build_suite(trials=1).requirement

    def all_referred(data, requests=None):
        return [d.model_copy(update={"outcome": c.Outcome.REFER}) for d in original(data, requests)]

    monkeypatch.setattr(checks, "baseline", all_referred)
    with pytest.raises(ValueError, match="baseline"):
        checks.build_suite(trials=1)
    newly_signed = {
        **frozen,
        "max_referral_share": 1,
        "signature": {**frozen["signature"], "at": "2026-10-12T13:00:00+00:00"},
    }
    assert checks.build_suite(trials=1, requirement=newly_signed).requirement == newly_signed


def test_at_limit_freshness_requires_the_audited_payment(monkeypatch):
    from refund_agent.release import candidate as builders
    from refund_agent.rules import _FRESHNESS_LIMITS
    from refund_agent.tools import Tools

    suite = build_suite(trials=1)
    candidate = build_candidate(suite)
    cases = [c for c in suite.cases if c.failure_class == "freshness-boundary"]
    assert len(cases) == 6
    assert all(not run_case(case, candidate)["failed"] for case in cases)

    class RejectAtLimit(Tools):
        def _authorize(self, state, amount, approval):
            records = {"order": state.order, "shipment": state.shipment, "policy": state.retrieval}
            if any(
                row
                and state.request.at - row.observed_at == timedelta(seconds=_FRESHNESS_LIMITS[name])
                for name, row in records.items()
            ):
                return "constructed_boundary_refusal"
            return super()._authorize(state, amount, approval)

    monkeypatch.setattr(builders, "build_tools", lambda data, candidate, mode, **context: RejectAtLimit(data))
    assert [case.case_id for case in cases if run_case(case, candidate)["failed"]] == [
        f"freshness-{source}-at" for source in ("order", "shipment", "policy")
    ]


@pytest.mark.parametrize("mutation", ["software", "signer", "reason", "time", "case", "retired"])
def test_open_hold_needs_its_own_signed_adjudication(tmp_path, mutation):
    full = build_suite(trials=1)
    suite = replace(full, cases=tuple(c for c in full.cases if c.part == "public"), generated=())
    candidate, journal = build_candidate(suite), Journal(tmp_path / "journal")
    first = evaluate_release(candidate, suite, journal, soft={"clarity": "hold"})
    disposition = {
        "kind": "soft-hold-adjudication",
        "record_id": first["record_id"],
        "case_id": "soft-hold:clarity",
        "software_fingerprint": candidate.fingerprint,
        "signer": "morgan",
        "reason": "Reviewed this case",
        "at": "2026-10-12T12:01:00+00:00",
    }
    key, value = {
        "software": ("software_fingerprint", "another-software"),
        "signer": ("signer", "unlisted"),
        "reason": ("reason", ""),
        "time": ("at", "2026-10-12"),
        "case": ("case_id", "soft-hold:another-case"),
        "retired": ("kind", "retired"),
    }[mutation]
    disposition[key] = value
    with pytest.raises(ValueError):
        evaluate_release(candidate, suite, journal, dispositions=[disposition])
    assert journal.failures(candidate.fingerprint)[0]["case_id"] == "soft-hold:clarity"


def test_changed_ceiling_cannot_reuse_old_signoff():
    frozen = build_suite(trials=1).requirement
    with pytest.raises(ValueError, match="new owner sign-off"):
        build_suite(trials=1, requirement={**frozen, "max_referral_share": 1})


def test_baseline_byte_change_with_same_share_refuses_default(monkeypatch):
    from pathlib import Path

    from refund_agent.release.candidate import PACKAGE

    original = Path.read_bytes
    monkeypatch.setattr(
        Path, "read_bytes", lambda p: original(p) + (b"\n" if p == PACKAGE / "baseline.py" else b"")
    )
    with pytest.raises(ValueError, match="baseline"):
        build_suite(trials=1)


def test_settled_hold_does_not_remain_in_new_software_history(tmp_path):
    full = build_suite(trials=1)
    suite = replace(full, cases=tuple(c for c in full.cases if c.part == "public"), generated=())
    original, journal = build_candidate(suite), Journal(tmp_path / "journal")
    held = evaluate_release(original, suite, journal, soft={"clarity": "hold"})
    adjudication = {
        "kind": "soft-hold-adjudication",
        "record_id": held["record_id"],
        "case_id": "soft-hold:clarity",
        "software_fingerprint": original.fingerprint,
        "signer": "morgan",
        "reason": "This recorded hold is settled",
        "at": "2026-10-12T12:01:00+00:00",
    }
    changed = build_candidate(suite, runtime="second")
    assert evaluate_release(changed, suite, journal)["decision"] == "fail"
    assert (
        evaluate_release(original, suite, journal, dispositions=[adjudication])["decision"]
        == "pass"
    )
    passed = evaluate_release(changed, suite, journal)
    assert passed["decision"] == "pass" and not passed["unresolved_earlier_failures"]
    assert sign(passed, journal, "morgan", "2026-10-12T12:02:00+00:00")["signature"]


def test_explicit_old_default_cannot_bypass_baseline_check(monkeypatch):
    from refund_agent.release import suite as checks

    original = checks.baseline
    frozen = build_suite(trials=1).requirement
    monkeypatch.setattr(
        checks,
        "baseline",
        lambda data, requests=None: [
            d.model_copy(update={"outcome": c.Outcome.REFER}) for d in original(data, requests)
        ],
    )
    with pytest.raises(ValueError, match="baseline"):
        build_suite(trials=1, requirement=frozen)


def test_same_run_adjudication_rejects_wrong_software(tmp_path):
    full = build_suite(trials=1)
    suite = replace(full, cases=tuple(c for c in full.cases if c.part == "public"), generated=())
    candidate, journal = build_candidate(suite), Journal(tmp_path / "journal")
    row = {
        "case_id": "clarity",
        "software_fingerprint": "other-software",
        "signer": "morgan",
        "reason": "Reviewed this hold",
        "at": "2026-10-12T12:01:00+00:00",
    }
    with pytest.raises(ValueError, match="software"):
        evaluate_release(candidate, suite, journal, soft={"clarity": "hold"}, overrides=[row])
    assert not journal.path.exists()
    row["software_fingerprint"] = candidate.fingerprint
    assert (
        evaluate_release(candidate, suite, journal, soft={"clarity": "hold"}, overrides=[row])[
            "decision"
        ]
        == "pass"
    )


def test_public_suite_replacement_cannot_reuse_requirement_signoff(tmp_path):
    suite = build_suite(trials=1)
    changed = replace(suite, requirement={**suite.requirement, "max_referral_share": 1})
    with pytest.raises(ValueError, match="new owner sign-off"):
        evaluate_release(build_candidate(changed), changed, Journal(tmp_path / "journal"))


@pytest.mark.parametrize("nested", [False, True])
def test_requirement_is_deeply_immutable(nested):
    suite = build_suite(trials=1)
    with pytest.raises(TypeError):
        if nested:
            suite.requirement["signature"]["at"] = "2026-10-12T13:00:00+00:00"
        else:
            suite.requirement["max_referral_share"] = 1


@pytest.mark.parametrize("mutation", ["audit-only", "executed"])
def test_freshness_grading_reads_audit_and_executed_effects(monkeypatch, mutation):
    from refund_agent.release import candidate as builders
    from refund_agent.tools import Tools

    class BrokenEffects(Tools):
        def _checked_append(self, record):
            if mutation == "audit-only":
                return c.ToolResult(ok=True, record_id="unwritten")
            return super()._checked_append(record)

    monkeypatch.setattr(
        builders, "build_tools", lambda data, candidate, mode, **context: BrokenEffects(data)
    )
    if mutation == "executed":
        original_run = builders.run_request

        def wrong_flag(*args, **kwargs):
            trace = original_run(*args, **kwargs)
            return trace.model_copy(
                update={"decision": trace.decision.model_copy(update={"executed": False})}
            )

        monkeypatch.setattr(builders, "run_request", wrong_flag)
    suite = build_suite(trials=1)
    candidate = build_candidate(suite)
    case = next(c for c in suite.cases if c.case_id == "freshness-shipment-at")
    failures = run_case(case, candidate)["failed"]
    assert (
        "audit_matches_effects" if mutation == "audit-only" else "executed_matches_ledger"
    ) in failures


@pytest.mark.parametrize("changed_suite", ["unchanged", "trials", "version"])
def test_mixed_hard_failure_preserves_valid_adjudication(tmp_path, changed_suite):
    from refund_agent.runtimes.adapter import AdapterSettings

    full = build_suite(trials=1)
    suite = replace(full, cases=tuple(c for c in full.cases if c.part == "public"), generated=())
    broken = build_candidate(suite, settings=AdapterSettings(max_steps=13))
    journal = Journal(tmp_path / "journal")
    failed = evaluate_release(broken, suite, journal, soft={"clarity": "hold"})
    assert set(failed["failed_cases"]) == {"context-and-step-budget", "soft-hold:clarity"}
    adjudication = {
        "kind": "soft-hold-adjudication",
        "record_id": failed["record_id"],
        "case_id": "soft-hold:clarity",
        "software_fingerprint": broken.fingerprint,
        "signer": "morgan",
        "reason": "Reviewed the held clarity evidence",
        "at": "2026-10-12T12:01:00+00:00",
    }
    held = evaluate_release(broken, suite, journal, dispositions=[adjudication])
    assert held["decision"] == "fail"
    with pytest.raises(ValueError, match="passing candidate"):
        sign(held, journal, "morgan", "2026-10-12T12:02:00+00:00")
    reopened = Journal(journal.path)
    assert [f["case_id"] for f in reopened.failures(broken.fingerprint)] == [
        "context-and-step-budget"
    ]
    target = suite
    if changed_suite == "trials":
        target = replace(suite, trials=2)
    elif changed_suite == "version":
        target = replace(suite, version=suite.version + "-revised")
    corrected = build_candidate(target, settings=AdapterSettings(max_steps=12))
    without_fix = evaluate_release(corrected, target, reopened)
    assert without_fix["decision"] == "fail"
    fix = {
        "kind": "fixed",
        "record_id": failed["record_id"],
        "case_id": "context-and-step-budget",
        "software_fingerprint": corrected.fingerprint,
    }
    again = evaluate_release(corrected, target, reopened, dispositions=[fix])
    if changed_suite != "unchanged":
        assert again["decision"] == "fail"
        assert [f["case_id"] for f in again["unresolved_earlier_failures"]] == ["soft-hold:clarity"]
        # Explicit signed coverage of the earlier hold for this evaluation manifest.
        again = evaluate_release(
            corrected,
            target,
            reopened,
            dispositions=[fix, {**adjudication, "suite_fingerprint": target.fingerprint}],
        )
    assert again["decision"] == "pass" and not again["unresolved_earlier_failures"]
    assert sign(again, reopened, "morgan", "2026-10-12T12:03:00+00:00")["signature"]


@pytest.mark.parametrize("change", ["trials", "version"])
def test_changed_suite_needs_new_coverage_of_an_already_fixed_failure(tmp_path, change):
    from refund_agent.runtimes.adapter import AdapterSettings

    full = build_suite(trials=1)
    suite = replace(full, cases=tuple(c for c in full.cases if c.part == "public"), generated=())
    broken = build_candidate(suite, settings=AdapterSettings(max_steps=13))
    journal = Journal(tmp_path / "journal")
    failed = evaluate_release(broken, suite, journal)
    corrected = build_candidate(suite)
    fix = {
        "kind": "fixed",
        "record_id": failed["record_id"],
        "case_id": "context-and-step-budget",
        "software_fingerprint": corrected.fingerprint,
    }
    assert evaluate_release(corrected, suite, journal, dispositions=[fix])["decision"] == "pass"
    changed = (
        replace(suite, trials=2)
        if change == "trials"
        else replace(suite, version=suite.version + "-revised")
    )
    same_software = build_candidate(changed)
    assert same_software.fingerprint == corrected.fingerprint
    assert evaluate_release(same_software, changed, journal)["decision"] == "fail"
    assert (
        evaluate_release(same_software, changed, journal, dispositions=[fix])["decision"] == "pass"
    )


def test_new_adjudication_refuses_a_removed_owner_but_retains_historical_resolution(
    tmp_path, monkeypatch
):
    from refund_agent.release import rule

    full = build_suite(trials=1)
    suite = replace(full, cases=tuple(c for c in full.cases if c.part == "public"), generated=())
    candidate, journal = build_candidate(suite), Journal(tmp_path / "journal")
    failed = evaluate_release(candidate, suite, journal, soft={"clarity": "hold"})
    adjudication = {
        "kind": "soft-hold-adjudication",
        "record_id": failed["record_id"],
        "case_id": "soft-hold:clarity",
        "software_fingerprint": candidate.fingerprint,
        "signer": "morgan",
        "reason": "Reviewed the held evidence",
        "at": "2026-10-12T12:01:00+00:00",
    }
    assert (
        evaluate_release(candidate, suite, journal, dispositions=[adjudication])["decision"]
        == "pass"
    )
    remaining = rule.owner_ids() - {"morgan"}
    monkeypatch.setattr(rule, "owner_ids", lambda: remaining)
    with pytest.raises(ValueError, match="owner|signer"):
        rule._adjudication(
            adjudication, case_id="soft-hold:clarity", software=candidate.fingerprint
        )
    assert not Journal(journal.path).failures(candidate.fingerprint)


def test_validated_same_run_adjudication_is_detached_from_caller(tmp_path, monkeypatch):
    from refund_agent.release import rule

    full = build_suite(trials=1)
    suite = replace(full, cases=tuple(c for c in full.cases if c.part == "public"), generated=())
    candidate = build_candidate(suite)
    adjudication = {
        "case_id": "clarity",
        "software_fingerprint": candidate.fingerprint,
        "signer": "morgan",
        "reason": "Reviewed the recorded clarity evidence",
        "at": "2026-10-12T12:01:00+00:00",
    }
    original = rule._trials

    def mutate(*args):
        adjudication["signer"] = "unlisted"
        return original(*args)

    monkeypatch.setattr(rule, "_trials", mutate)
    record = evaluate_release(
        candidate,
        suite,
        Journal(tmp_path / "journal"),
        soft={"clarity": "hold"},
        overrides=[adjudication],
    )
    assert record["decision"] == "pass"
    assert adjudication["signer"] == "unlisted"
    assert record["overrides"][0]["signer"] == "morgan"
