"""The signed Chapter 3 floor plus a whole-suite decision on every trial."""

import json
from collections import Counter
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path

from refund_agent import contracts as c
from refund_agent.baseline import baseline
from refund_agent.data.generate import generate
from refund_agent.release import suite as checks
from refund_agent.release.journal import unresolved_failures
from refund_agent.selection import fingerprint

RULE_VERSION = "drafts-to-shadow-3"
OWNERS = Path(__file__).with_name("owners.json")


def owner_ids():
    return {row["id"] for row in json.loads(OWNERS.read_text())["owners"]}


def _time(value):
    if type(value) is not str:
        raise ValueError("a timezone-aware sign-off time is required")
    at = datetime.fromisoformat(value)
    if at.tzinfo is None or at.utcoffset() is None:
        raise ValueError("a timezone-aware sign-off time is required")
    return at.isoformat()


# region chapter29:adjudication-validation
def _adjudication(row, *, case_id, software):
    if row.get("signer") not in owner_ids():
        raise ValueError("override requires a listed signer")
    if (
        not isinstance(case_id, str)
        or not case_id.strip()
        or row.get("case_id") != case_id
        or not row.get("reason", "").strip()
    ):
        raise ValueError("signed adjudication requires the exact named soft hold and a reason")
    if row.get("software_fingerprint") != software:
        raise ValueError("adjudication requires the held software fingerprint")
    _time(row.get("at"))


def _overrides(overrides, soft, software):
    seen = set()
    for i, original in enumerate(overrides):
        row = overrides[i] = dict(original)
        case_id = row.get("case_id")
        held = case_id if soft.get(case_id) == "hold" else None
        _adjudication(row, case_id=held, software=software)
        if case_id in seen:
            raise ValueError("signed adjudication requires a unique named soft hold")
        seen.add(case_id)
        row["kind"] = "soft-hold-adjudication"


# endregion chapter29:adjudication-validation


def _floor(rows, n, requirement):
    data = generate(7, orders=40, requests=40)
    tier1 = [r for r in data.requests if r.amount_cents < 5000]
    base = baseline(data, tier1)
    expected = {d.request_id: d for d in data.correct}
    expected_ids = {f"public-{r.request_id}" for r in tier1}
    public = [r for r in rows if r["case_id"] in expected_ids]
    complete = (
        Counter(r["case_id"] for r in public) == Counter(expected_ids)
        and len({r["case_id"] for r in rows}) == len(rows)
        and all(len(r["trials"]) == n and all("decision" in t for t in r["trials"]) for r in public)
    )
    forbidden, referrals = _floor_counts(public, expected)
    baseline_referrals = sum(d.outcome == c.Outcome.REFER for d in base)
    return {
        "cases": len(tier1),
        "forbidden_payments": forbidden,
        "referrals_per_trial": referrals / n,
        "baseline_referrals": baseline_referrals,
        "complete": complete,
        "max_referral_share": requirement["max_referral_share"],
        "pass": complete
        and forbidden == 0
        and referrals / (len(tier1) * n) <= requirement["max_referral_share"],
    }


def _floor_counts(public, expected):
    forbidden, referrals = 0, 0
    for row in public:
        for trial in row["trials"]:
            if "decision" not in trial:
                continue
            d = trial["decision"]
            forbidden += d["executed"] and expected[d["request_id"]].outcome != c.Outcome.APPROVE
            referrals += d["outcome"] == "refer"
    return forbidden, referrals


def _trials(candidate, suite, n):
    rows, failures = [], []
    for case in suite.cases:
        if not checks.selected(case, candidate):
            continue
        trials = [_trial(case, candidate) for _ in range(n)]
        row = {
            "case_id": case.case_id,
            "part": case.part,
            "tier": case.tier,
            "must_pass": case.must_pass,
            "trials": trials,
        }
        rows.append(row)
        bad = [i for i, trial in enumerate(trials, 1) if trial["failed"]]
        if bad:
            failures.append(
                {
                    "case_id": case.case_id,
                    "part": case.part,
                    "trials": bad,
                    "failed_assertions": sorted({a for t in trials for a in t["failed"]}),
                }
            )
    return rows, failures


def _trial(case, candidate):
    try:
        return checks.run_case(case, candidate)
    except Exception as error:
        return {
            "failed": ["trial_aborted"],
            "status": "aborted",
            "error_type": type(error).__name__,
        }


def _generated(candidate, suite):
    reports = []
    for scenarios in suite.generated:
        cases = checks._scenario_cases(scenarios, "generated")
        results = [_trial(case, candidate) for case in cases]
        reports.append(
            {
                "fingerprint": scenarios.fingerprint,
                "tau": scenarios.manifest["tau"],
                "cases": len(cases),
                "passing": sum(not row["failed"] for row in results),
                "passing_case_ids": [
                    f"{scenarios.fingerprint}:{case.case_id}"
                    for case, result in zip(cases, results)
                    if not result["failed"]
                ],
                "aborted_trials": [
                    _aborted_generated(scenarios, case, result)
                    for case, result in zip(cases, results)
                    if result.get("status") == "aborted"
                ],
                "release_authority": False,
                "human_review": scenarios.manifest["human_review"],
            }
        )
    return reports


def _aborted_generated(scenarios, case, result):
    return {
        "case_id": f"{scenarios.fingerprint}:{case.case_id}",
        "part": "generated",
        "trials": [1],
        "failed_assertions": result["failed"],
        "error_type": result["error_type"],
    }


def _decision_record(
    candidate, suite, rows, failures, history, n, overrides, soft, at, dispositions, generated
):
    floor, budget, failed_cases = _release_blocks(
        candidate, suite, rows, failures, overrides, soft, n
    )
    aborted = [failure for report in generated for failure in report["aborted_trials"]]
    failures = failures + aborted
    failed_cases = sorted(set(failed_cases) | {f["case_id"] for f in aborted})
    evidence = _run_evidence(
        candidate.fingerprint, suite, rows, history, overrides, soft, dispositions, generated
    )
    record = _record_identity(candidate, suite, at)
    record.update(
        manifest=suite.manifest(),
        decision="fail" if failed_cases or evidence["unresolved_earlier_failures"] else "pass",
        status="evaluated",
        floor=floor,
        budget=budget,
        failed_cases=failed_cases,
        failures=failures,
        cases_run=len(rows),
        trials_run=n,
        case_results=rows,
    )
    record.update(evidence)
    record["record_id"] = fingerprint(record)
    return record


def _run_evidence(software, suite, rows, history, overrides, soft, dispositions, generated):
    return dict(
        counts_by_part=dict(Counter(r["part"] for r in rows)),
        earlier_failures=_earlier_failures(history),
        overrides=overrides,
        soft_grades=soft,
        accesses=list(suite.accesses),
        generated_apart=generated,
        failure_dispositions=dispositions,
        unresolved_earlier_failures=unresolved_failures(
            history,
            dispositions=dispositions,
            target_software=software,
            target_suite=suite.fingerprint,
        ),
    )


# Keep the printed excerpt within 79 columns.
# fmt: off
# region book:fnd.agent-evaluation.rule-01
if __name__ == "__main__":
    from refund_agent.release.rule import (
        _blocked_record,
        _declared_run,
        _dispositions,
        _evaluated_record,
        _floor,
        unresolved_failures,
    )


def evaluate_release(
    candidate,
    suite,
    journal,
    trials=None,
    *,
    at="2026-10-12T12:00:00+00:00",
    overrides=(),
    soft=None,
    dispositions=(),
):
    soft, overrides = dict(soft or {}), list(overrides)
    trials, at = _declared_run(
        candidate, suite, trials, overrides, soft, at
    )
    with journal.transaction() as (history, stream):
        dispositions = _dispositions(dispositions, history, candidate, suite)
        prior = unresolved_failures(
            history, candidate.fingerprint, dispositions
        )  # (1)
        if prior:
            record = _blocked_record(candidate, suite, prior, at, dispositions)
        else:
            record = _evaluated_record(
                candidate, suite, history, trials, overrides, soft, at,
                dispositions,
            )
        journal.write(history, stream, record)
    return record


def _release_blocks(candidate, suite, rows, failures, overrides, soft, n):
    floor = _floor(rows, n, suite.requirement)  # (2)
    adjudicated = {r["case_id"] for r in overrides}
    hard = {case.case_id for case in suite.cases if case.must_pass}
    blocked = {  # (3)
        f["case_id"]
        for f in failures
        if f["case_id"] in hard or "trial_aborted" in f["failed_assertions"]
    }
    budget = suite.manifest()["budget"]
    budget_pass = (  # (4)
        sum(budget["shares"]) <= budget["window"]
        and candidate.adapter["settings"]["max_steps"] <= budget["max_steps"]
    )
    if not floor["pass"]:
        blocked.add("drafts-to-shadow-floor")
    if not budget_pass:
        blocked.add("context-and-step-budget")
    blocked.update(  # (5)
        f"soft-hold:{name}"
        for name, verdict in soft.items()
        if verdict == "hold" and name not in adjudicated
    )
    return floor, {"pass": budget_pass, **budget}, sorted(blocked)
# endregion book:fnd.agent-evaluation.rule-01
# fmt: on


@checks.manifest_guard
def _evaluated_record(candidate, suite, history, trials, overrides, soft, at, dispositions):
    rows, failures = _trials(candidate, suite, trials)
    generated = _generated(candidate, suite)
    dispositions = _fixed_dispositions(dispositions, rows, generated, candidate, suite)
    return _decision_record(
        candidate,
        suite,
        rows,
        failures,
        history,
        trials,
        overrides,
        soft,
        at,
        dispositions,
        generated,
    )


def sign(record, journal, owner, at):
    if owner not in owner_ids():
        raise ValueError("sign-off requires a listed release owner")
    at = _time(at)
    with journal.transaction() as (rows, stream):
        unsigned = {**record, "signature": None}
        matches = any({k: v for k, v in row.items() if k != "entry"} == unsigned for row in rows)
        failed = unresolved_failures(
            rows,
            record["software_fingerprint"],
            record["failure_dispositions"],
            target_suite=record["suite_fingerprint"],
        )
        if record["decision"] != "pass" or not matches or failed:
            raise ValueError(
                "only a journaled passing candidate without an unchanged failure may be signed"
            )
        signature = {"owner": owner, "at": at, "kind": "recorded sign-off; not cryptographic"}
        signed = {**record, "signature": signature}
        journal.write(
            rows,
            stream,
            {
                "event": "sign",
                "record_id": record["record_id"],
                "candidate_fingerprint": record["candidate_fingerprint"],
                "software_fingerprint": record["software_fingerprint"],
                "suite_fingerprint": record["suite_fingerprint"],
                "signature": signature,
            },
        )
    return signed


def _earlier_failures(history):
    fields = (
        "record_id",
        "software_fingerprint",
        "candidate_fingerprint",
        "suite_fingerprint",
        "failed_cases",
    )
    return [{key: row[key] for key in fields} for row in history if row.get("decision") == "fail"]


def _record_identity(candidate, suite, at):
    return dict(
        version="refund-release-record-2",
        rule_version=RULE_VERSION,
        candidate=candidate.record(),
        candidate_fingerprint=candidate.fingerprint,
        software_fingerprint=candidate.fingerprint,
        suite_fingerprint=suite.fingerprint,
        at=at,
        signature=None,
    )


def _blocked_record(candidate, suite, prior, at, dispositions):
    record = _record_identity(candidate, suite, at)
    record.update(
        decision="fail",
        status="unchanged_candidate_previously_failed",
        failed_cases=sorted({r["case_id"] for r in prior}),
        cases_run=0,
        trials_run=0,
        failures=[],
        earlier_failures=prior,
        failure_dispositions=dispositions,
        unresolved_earlier_failures=prior,
    )
    record["record_id"] = fingerprint(record)
    return record


def _dispositions(dispositions, history, candidate, suite):
    failures = {
        (r["record_id"], r["case_id"]): r
        for r in unresolved_failures(
            history, target_software=candidate.fingerprint, target_suite=suite.fingerprint
        )
    }
    declared, seen = [dict(row) for row in dispositions], set()
    for row in declared:
        key = (row.get("record_id"), row.get("case_id"))
        if key not in failures or key in seen:
            raise ValueError("disposition requires a unique earlier unresolved failure")
        seen.add(key)
        if row.get("kind") == "soft-hold-adjudication":
            _soft_disposition(row, history, failures[key], suite)
        elif row.get("kind") == "retired":
            if row["case_id"].startswith("soft-hold:"):
                raise ValueError("a soft hold requires a signed adjudication")
            _retired_case(row, history, suite)
            if row.get("software_fingerprint") != candidate.fingerprint:
                raise ValueError("retirement requires the covered software fingerprint")
        elif row.get("kind") == "fixed":
            if (
                row.get("software_fingerprint") != candidate.fingerprint
                or candidate.fingerprint == failures[key]["software_fingerprint"]
            ):
                raise ValueError("a fix requires a new software fingerprint")
        else:
            raise ValueError("disposition is fixed or retired")
    return declared


def _soft_disposition(row, history, failure, suite):
    held = next(r for r in history if r.get("record_id") == row["record_id"])
    if (
        not row["case_id"].startswith("soft-hold:")
        or row.get("suite_fingerprint", suite.fingerprint) != suite.fingerprint
        or held.get("soft_grades", {}).get(row["case_id"].removeprefix("soft-hold:")) != "hold"
        or any(c["case_id"] == row["case_id"] and c["must_pass"] for c in held["manifest"]["cases"])
    ):
        raise ValueError("adjudication requires an exact soft hold and evaluation suite")
    _adjudication(row, case_id=failure["case_id"], software=failure["software_fingerprint"])


def _suite_case_ids(suite):
    return {c.case_id for c in suite.cases} | {
        f"{s.fingerprint}:generated-{row.draw}" for s in suite.generated for row in s.scenarios
    }


def _retired_case(row, history, suite):
    old = next(r for r in history if r.get("record_id") == row["record_id"])
    old_cases = {c["case_id"] for c in old["manifest"]["cases"]}
    old_cases.update(f["case_id"] for g in old["generated_apart"] for f in g["aborted_trials"])
    if row["case_id"] not in old_cases:
        raise ValueError("a release requirement cannot be retired as a scenario")
    if row["case_id"] in _suite_case_ids(suite):
        raise ValueError("a retired case must be absent from the new suite")
    if row.get("signer") not in owner_ids() or not row.get("reason", "").strip():
        raise ValueError("retirement requires a listed signer and a reason")
    _time(row.get("at"))


def _fixed_dispositions(dispositions, rows, generated, candidate, suite):
    passing = {
        r["case_id"] for r in rows if r["trials"] and all(not t["failed"] for t in r["trials"])
    }
    passing.update(case_id for report in generated for case_id in report["passing_case_ids"])
    floor, budget, _ = _release_blocks(candidate, suite, rows, [], [], {}, suite.trials)
    passing.update(
        check
        for check, result in (
            ("drafts-to-shadow-floor", floor),
            ("context-and-step-budget", budget),
        )
        if result["pass"]
    )
    return [
        d
        for d in dispositions
        if d["kind"] in ("retired", "soft-hold-adjudication") or d["case_id"] in passing
    ]


# region chapter29:requirement-validation
def _requirement(requirement):
    frozen = json.loads(Path(__file__).with_name("requirement.json").read_text())
    if not isinstance(requirement, Mapping) or set(requirement) != set(frozen):
        raise ValueError("complete signed referral requirement required")
    signature = requirement.get("signature") or {}
    if not isinstance(signature, Mapping) or set(signature) != set(frozen["signature"]):
        raise ValueError("complete requirement owner sign-off required")
    if requirement != frozen and all(
        signature.get(key) == frozen["signature"][key] for key in ("owner", "at")
    ):
        raise ValueError("changed requirement needs a new owner sign-off")
    if (
        requirement["version"] != frozen["version"]
        or signature["kind"] != frozen["signature"]["kind"]
        or type(requirement["baseline_seed"]) is not int
        or type(requirement["baseline_requests"]) is not int
        or requirement["baseline_requests"] < 1
        or type(requirement["baseline_referrals"]) is not int
        or not 0 <= requirement["baseline_referrals"] <= requirement["baseline_requests"]
        or not isinstance(requirement["baseline_identity"], str)
        or len(requirement["baseline_identity"]) != 64
        or any(char not in "0123456789abcdef" for char in requirement["baseline_identity"])
    ):
        raise ValueError("invalid complete baseline requirement")
    if requirement == frozen:
        checks._signed_requirement()
    share, signature = requirement.get("max_referral_share"), requirement.get("signature")
    if type(share) not in (float, int) or not 0 <= share <= 1:
        raise ValueError("signed requirement needs a referral share in [0, 1]")
    if not signature or signature.get("owner") not in owner_ids():
        raise ValueError("requirement needs a listed owner sign-off")
    _time(signature.get("at"))


# endregion chapter29:requirement-validation


def _declared_run(candidate, suite, trials, overrides, soft, at):
    _requirement(suite.requirement)
    trials = suite.trials if trials is None else trials
    if type(trials) is not int or trials < 1 or trials != suite.trials:
        raise ValueError("trial count differs from the fixed suite manifest")
    if candidate.suite["fingerprint"] != suite.fingerprint:
        raise ValueError("candidate names a different suite")
    if any(v not in ("pass", "hold") for v in soft.values()):
        raise ValueError("soft grades are pass or hold; no payment authority")
    _overrides(overrides, soft, candidate.fingerprint)
    return trials, _time(at)
