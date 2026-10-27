"""Five owned checks, then every selected component's required start checks."""

import json
from datetime import datetime, timedelta

from refund_agent import contracts as c
from refund_agent.canary.cloud_native_deployment_plan import CanaryPlan, has_role
from refund_agent.canary.cloud_native_deployment_stop import CanarySession, Grant
from refund_agent.canary.cloud_native_deployment_window import ShadowWindowResult
from refund_agent.release import extensions
from refund_agent.release.candidate import build_candidate
from refund_agent.selection import fingerprint


class StartCheck(c.Record):
    name: str
    passed: bool
    detail: str


class CanaryStart(c.Record):
    canary_id: str
    candidate_fingerprint: str
    plan_fingerprint: str
    window_fingerprint: str
    checks: tuple[StartCheck, ...]
    accepted: bool
    started_at: datetime | None
    expires_at: datetime | None
    attempted_at: datetime
    release_owner: str


def _budget(record, plan, at):
    # Chapter 34 owns the definitions. This reader only checks admission fields.
    if not plan.error_budget_ref or record is None:
        raise ValueError("error budget not named")
    if not has_role(record.get("signed_by"), "decision_owner"):
        raise ValueError("error budget requires decision_owner signature")
    signed = c._utc_time(datetime.fromisoformat(record["signed_at"]))
    if signed >= at:
        raise ValueError("error budget must be signed before start")
    objectives = record.get("objectives")
    if not isinstance(objectives, list) or len(objectives) != 2:
        raise ValueError("error budget requires two defined objectives")
    for objective in objectives:
        if any(objective.get(k) in (None, "") for k in ("definition", "window", "allowance", "population")):
            raise ValueError("error budget objective lacks definition, window, allowance or population")
        if type(objective["allowance"]) not in (float, int) or objective["allowance"] < 0:
            raise ValueError("error budget allowance must be nonnegative")


def _check(name, function):
    try:
        function()
    except (ValueError, KeyError, TypeError, FileNotFoundError) as error:
        return StartCheck(name=name, passed=False, detail=str(error))
    return StartCheck(name=name, passed=True, detail="pass")


@extensions.resource_guard(4)
def start_canary(plan, release_record, window, rehearsal, *, candidate, suite, journal,
                 session: CanarySession, owner: str, at: datetime, error_budget=None,
                 records=None) -> CanaryStart:
    at = c._utc_time(at)
    plan = c._validated(CanaryPlan, plan)
    window = c._validated(ShadowWindowResult, window)
    if session.plan != plan:
        raise ValueError("binding and start use different plans")

    def release():
        from refund_agent.shadow.__main__ import validate_release

        if release_record is None:
            raise ValueError("release record unavailable; predecessor composition failed")
        if not has_role(owner, "release_owner"):
            raise ValueError("start requires release_owner")
        current = build_candidate(suite, candidate.name, candidate.runtime["identity"],
                                  candidate.tools_composition["verifier"])
        if current.record() != candidate.record() or plan.candidate_fingerprint != current.fingerprint:
            raise ValueError("loaded candidate differs from the signed plan")
        if plan.suite_fingerprint != suite.fingerprint:
            raise ValueError("plan suite differs")
        validate_release(release_record, current, suite, journal)
        signed_at = c._utc_time(datetime.fromisoformat(release_record["signature"]["at"]))
        if signed_at > at:
            raise ValueError("release sign-off is after start")

    def requirement():
        if window.candidate_fingerprint != candidate.fingerprint or (
            plan.requirement_ref != window.window_id
        ):
            raise ValueError("window belongs to another candidate")
        if not window.measures_pass or window.signed_by is None or window.signed_at is None or window.signed_at > at:
            raise ValueError("window is unsigned, failed or signed after start")
        if not window.generated_label_shortcut and at < window.labels_mature_at:
            raise ValueError("deployed labels have not matured")
        windows = {fingerprint(r["value"]): r["value"]
                   for r in session.store.rows("window", all_namespaces=True) if r["value"] is not None}
        for row in session.store.rows("authority", all_namespaces=True):
            used = row["value"]
            if used["lane"] != "canary":
                continue
            if used["canary_id"] == plan.canary_id and row["namespace"] != session.store.namespace:
                raise ValueError("canary authority is bound to its original namespace")
            original = windows.get(used["window_fingerprint"])
            if original is None:
                raise ValueError("persisted window authority is absent")
            same = original["window_id"] == window.window_id or tuple(original["request_ids"]) == window.request_ids
            if same and used["canary_id"] != plan.canary_id:
                raise ValueError("window authority has already been spent by another canary")

    def signed_plan():
        if plan.signed_by is None or plan.signed_at is None or plan.signed_at > at:
            raise ValueError("plan must be signed before start")
        prior = session.grant
        if prior is not None:
            session.verify_window_authority()
        if prior is not None and (
            prior.outcome != "running" or prior.budget_spent or at >= prior.expires_at
            or prior.window_fingerprint != window.fingerprint or prior.lane != "canary"
            or prior.budget_fingerprint != fingerprint(error_budget)
        ):
            raise ValueError("persisted grant is paused, spent, ended, expired or mismatched")

    def rehearsed():
        if rehearsal is None or rehearsal.get("verdict") != "pass":
            raise ValueError("passing rollback rehearsal required")
        if (rehearsal.get("rehearsal_id"), rehearsal.get("candidate_fingerprint"),
            rehearsal.get("rollback_target_fingerprint")) != (
            plan.rehearsal_ref, candidate.fingerprint, plan.rollback_target_fingerprint
        ):
            raise ValueError("rehearsal candidate or rollback target differs")
        checks = rehearsal.get("checks")
        required = {
            "next_cohort_to_person", "receipts_kept", "intents_closed_unpaid",
            "receipts_confirmed_at_restart", "reconcile_idempotent", "readable_records",
            "post_restart", "components_restored", "no_request_paid_twice",
        }
        if (not isinstance(checks, dict) or not required <= checks.keys()
            or any(value is not True for value in checks.values())
            or rehearsal.get("paid_twice") != 0 or rehearsal.get("unreadable_records") != 0
            or rehearsal.get("restart_fresh_process") is not True):
            raise ValueError("rehearsal checks did not all pass")

    with session.store.transaction():
        checks = [
            _check("release", release), _check("window", requirement),
            _check("plan", signed_plan), _check("error_budget", lambda: _budget(error_budget, plan, at)),
            _check("rehearsal", rehearsed),
        ]
        # Chapters 33/35 register here from their own modules. Never cache checks.
        for component, contribution in extensions.contributions(
            candidate, "start_checks", plan=plan, at=at,
            records={**(records or {}), "release": release_record, "window": window,
                     "rehearsal": rehearsal, "error_budget": error_budget},
        ).items():
            if not contribution:
                checks.append(StartCheck(name=component, passed=False, detail="required start check absent"))
            else:
                checks.extend(StartCheck.model_validate_json(json.dumps(row)) for row in contribution)
        accepted = all(check.passed for check in checks)
        prior = session.grant
        started = prior.started_at if prior is not None else at
        expires = prior.expires_at if prior is not None else at + timedelta(days=plan.time_box_days)
        result = CanaryStart(
            canary_id=plan.canary_id, candidate_fingerprint=candidate.fingerprint,
            plan_fingerprint=plan.fingerprint, window_fingerprint=window.fingerprint,
            checks=tuple(checks), accepted=accepted, attempted_at=at, release_owner=owner,
            started_at=started if accepted else None, expires_at=expires if accepted else None,
        )
        session.store.append("start", str(len(session.store.rows("start")) + 1), result.model_dump(mode="json"))
        if accepted and prior is None:
            session.store.append("window", window.window_id, window.model_dump(mode="json"))
            session.store.append("budget", plan.error_budget_ref, error_budget)
            session.save(Grant(
                canary_id=plan.canary_id, plan_fingerprint=plan.fingerprint,
                window_fingerprint=window.fingerprint, started_at=started, expires_at=expires,
                outcome="running", lane="canary", budget_fingerprint=fingerprint(error_budget),
            ))
        elif not accepted and prior is not None and prior.outcome == "running":
            from refund_agent.canary.cloud_native_deployment_stop import StopRule

            if at >= prior.expires_at:
                StopRule(session).end("time_box", at=at)
            else:
                StopRule(session).trigger("untrusted_record", at=at, detail="start admission refused")
        return result
