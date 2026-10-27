"""A decision cites evidence for this exact bundle; only its owner admits it."""

import json
from datetime import datetime
from typing import Literal

from refund_agent import contracts as c
from refund_agent.canary.cloud_native_deployment_plan import CanaryPlan, has_role
from refund_agent.canary.cloud_native_deployment_window import ShadowWindowResult
from refund_agent.promotion.mlops_mle_bundle import from_candidate
from refund_agent.promotion.mlops_mle_diff import diff
from refund_agent.promotion.mlops_mle_requirements import REQUIREMENTS, required
from refund_agent.release import extensions
from refund_agent.release.candidate import Candidate, build_candidate
from refund_agent.release.suite import build_suite
from refund_agent.selection import fingerprint
from refund_agent.shadow.__main__ import validate_release

REVERSAL = {
    "demotion": ["defect", "budget_spent", "time_box"],
    "rollback": ["release owner restores the whole rollback target"],
    "pause": "does not change bundle admission",
}


class PromotionRecord(c.Record):
    kind: Literal["promotion", "rollback", "demotion"] = "promotion"
    candidate_fingerprint: str
    incumbent_fingerprint: str | None
    from_stage: Literal["offline", "shadow", "canary", "all"]
    to_stage: Literal["shadow", "canary", "all"]
    changes: tuple[dict, ...]
    requirements: dict[str, str | None]
    status: Literal["promoted", "held"]
    unmet: tuple[str, ...]
    signed_by: str | None = None
    signed_at: datetime | None = None
    rollback_target: str | None
    reversal: dict = REVERSAL
    refs: dict[str, str] = {}

    @property
    def fingerprint(self):
        return fingerprint(self.model_dump(mode="json"))

    def record(self):
        return {**self.model_dump(mode="json"), "fingerprint": self.fingerprint}

    @classmethod
    def from_record(cls, row):
        row = dict(row)
        expected = row.pop("fingerprint")
        record = cls.model_validate_json(json.dumps(row))
        if record.fingerprint != expected:
            raise ValueError("promotion record fingerprint differs")
        return record


def passing_versions(record):
    versions, change_cases = set(), set()
    for row in record.get("case_results", ()):
        trials = row.get("trials", ())
        if not trials or any(t.get("failed") or "decision" not in t for t in trials):
            continue
        for trial in trials:
            citation = trial["decision"].get("policy_citation")
            if citation:
                versions.add(citation.split("@", 1)[1].split("#", 1)[0])
                if row["case_id"].startswith("mlops-mle-policy-change-"):
                    change_cases.add(citation.split("@", 1)[1].split("#", 1)[0])
    return versions, change_cases


@extensions.resource_guard(0)
def _validate_current(candidate, release, suite, journal):
    validate_release(release, candidate, suite, journal)


def check_release(bundle, release, journal):
    if release is None or journal is None:
        raise ValueError("signed journaled release unavailable")
    recorded = Candidate(**release["candidate"])
    manifest = release["manifest"]
    suite = build_suite(
        trials=manifest["trials"],
        requirement=manifest["requirement"],
        components=recorded.tools_composition.get("components", ()),
    )
    current = build_candidate(
        suite, recorded.name, recorded.runtime["identity"], recorded.tools_composition["verifier"]
    )
    if (
        current.record() != recorded.record()
        or current.fingerprint != bundle.refs["candidate"]
        or (suite.fingerprint != bundle.refs["suite"])
    ):
        raise ValueError("released software or evaluation manifest differs from bundle refs")
    if from_candidate(current).fingerprint != bundle.fingerprint:
        raise ValueError("release did not execute this bundle's component records")
    _validate_current(current, release, suite, journal)
    return release


def _simple_evidence(name, record, bundle):
    if (
        record is None
        or record.get("kind") != name
        or record.get("bundle_fingerprint") != bundle.fingerprint
    ):
        raise ValueError("evidence absent or belongs to another bundle")
    if record.get("passed") is not True:
        raise ValueError("evidence did not pass")
    if name == "signed_cost_matrix" and not has_role(record.get("signed_by"), "decision_owner"):
        raise ValueError("cost matrix requires decision owner")
    if name == "calibration" and not set(record["required_months"]) <= set(record["months"]):
        raise ValueError("calibration does not cover operating months")
    if name == "threshold_experiment":
        if record.get("outcome") not in ("run", "declined"):
            raise ValueError("threshold experiment neither run nor declined")
        if record["outcome"] == "declined" and (
            not has_role(record.get("signed_by"), "decision_owner")
            or not record.get("assumption")
            or not record.get("signed_at")
        ):
            raise ValueError("decline requires decision owner's dated assumption")
    if name == "rebuild" and record.get("receipt") != record.get("rebuilt_receipt"):
        raise ValueError("rebuild differs from the sealed selection receipt")
    if name == "policy_notice" and record.get("notice_days", 0) < 7:
        raise ValueError("policy owner must give seven days' notice")
    if name == "frozen_questions" and record.get("lost_clauses") != 0:
        raise ValueError("a frozen question lost its clause")
    if name == "runtime_swap" and "test_runtime_swap.py" not in record.get("checks", ()):
        raise ValueError("swap record lacks runtime swap tests")
    return record


def decide(
    bundle,
    incumbent=None,
    *,
    from_stage="offline",
    to_stage="shadow",
    evidence,
    journal=None,
    rows=REQUIREMENTS,
    candidate=None,
):
    changes = [] if incumbent is None else diff(incumbent, bundle)
    needs = required(changes, rows=rows, candidate=candidate, to_stage=to_stage)
    met, unmet = {}, []
    release = evidence.get("release")
    release_ok = False
    try:
        check_release(bundle, release, journal)
        release_ok = True
    except (ValueError, KeyError, TypeError, FileNotFoundError):
        pass
    added = set(bundle.policy.items["versions"]) - (
        set(incumbent.policy.items["versions"]) if incumbent is not None else set()
    )
    versions, cases = passing_versions(release or {})
    refs = {"candidate": bundle.refs["candidate"], "suite": bundle.refs["suite"]}
    if release is not None:
        refs["release"] = release["record_id"]
    for name in needs:
        try:
            if name == "release_owner":
                met[name] = None  # sign() supplies this; unsigned records grant nothing.
                continue
            if name == "release":
                if not release_ok:
                    raise ValueError("release not admitted")
                value = release
            elif name in ("policy_coverage", "policy_change_cases"):
                if not release_ok:
                    met[name] = None  # The unmet release blocks its dependent coverage too.
                    continue
                if not set(bundle.policy.items["versions"]) <= versions or (
                    name == "policy_change_cases" and not added <= cases
                ):
                    raise ValueError("declared version lacks passing policy cases")
                value = {
                    "release": release["record_id"],
                    "versions": sorted(versions),
                    "policy_change_cases": sorted(cases),
                }
            elif name == "shadow_window":
                raw = evidence[name]
                window = ShadowWindowResult.model_validate_json(json.dumps(raw))
                if (
                    not window.measures_pass
                    or not window.signed_by
                    or window.candidate_fingerprint != bundle.refs["candidate"]
                ):
                    raise ValueError("window unsigned, failed or for another candidate")
                if window.bundle_fingerprint != bundle.fingerprint:
                    raise ValueError("window did not run on this frozen bundle")
                value = raw
                refs["window"] = window.window_id
            elif name == "canary_plan":
                plan = CanaryPlan.model_validate_json(json.dumps(evidence[name]))
                if not plan.signed_by or (plan.candidate_fingerprint, plan.suite_fingerprint) != (
                    bundle.refs["candidate"],
                    bundle.refs["suite"],
                ):
                    raise ValueError("plan unsigned or for another candidate")
                window = evidence["shadow_window"]
                if (
                    plan.requirement_ref != window["window_id"]
                    or plan.rehearsal_ref != evidence["rehearsal"]["rehearsal_id"]
                ):
                    raise ValueError("plan names another window or rehearsal")
                value = evidence[name]
                refs["canary_id"] = plan.canary_id
            elif name == "rehearsal":
                value = evidence[name]
                if (
                    value["verdict"] != "pass"
                    or value["candidate_fingerprint"] != bundle.refs["candidate"]
                    or (
                        value["rollback_target_fingerprint"]
                        != evidence["canary_plan"]["rollback_target_fingerprint"]
                    )
                    or value["paid_twice"] != 0
                    or not value["checks"]
                    or not all(flag is True for flag in value["checks"].values())
                ):
                    raise ValueError("rehearsal failed or for another bundle")
                refs["rehearsal_id"] = value["rehearsal_id"]
            else:
                value = _simple_evidence(name, evidence.get(name), bundle)
            met[name] = fingerprint(value)
        except (ValueError, KeyError, TypeError, FileNotFoundError):
            met[name] = None
            unmet.append(name)
    return PromotionRecord(
        candidate_fingerprint=bundle.fingerprint,
        incumbent_fingerprint=incumbent.fingerprint if incumbent else None,
        from_stage=from_stage,
        to_stage=to_stage,
        changes=tuple(changes),
        requirements=met,
        status="held" if unmet else "promoted",
        unmet=tuple(unmet),
        rollback_target=incumbent.fingerprint if incumbent else None,
        refs=refs,
    )


def sign(record, owner, at):
    record = c._validated(PromotionRecord, record)
    if record.status == "held" or record.unmet:
        raise ValueError("a held promotion cannot be signed")
    if record.signed_by is not None or not has_role(owner, "release_owner"):
        raise ValueError("a listed release owner must sign an unsigned record")
    at = c._utc_time(at)
    return record.model_copy(
        update={
            "signed_by": owner,
            "signed_at": at,
            "requirements": {
                **record.requirements,
                "release_owner": fingerprint([owner, at.isoformat()]),
            },
        }
    )
