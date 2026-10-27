"""Signed settings and evidence references; the signature precedes the window."""

from datetime import datetime, timedelta

from pydantic import Field

from refund_agent import contracts as c
from refund_agent.canary.cloud_native_deployment_plan import has_role
from refund_agent.rollout.org_adoption_records import (
    EvidenceRecord,
    EvidenceType,
    Window,
)
from refund_agent.selection import fingerprint

ROLES = (
    "decision_owner",
    "release_owner",
    "reconciler",
    "reviewer",
    "absence_cover",
    "named_approver",
)
EVIDENCE_TYPES: tuple[EvidenceType, ...] = (
    "budget",
    "approvals",
    "audit",
    "queue",
    "promotion",
)


class EvidenceRef(c.Record):
    record_type: EvidenceType
    record_id: c.Identity
    period_start: datetime | None = None
    period_end: datetime | None = None


class Owner(c.Record):
    role: c.Identity
    person: c.Identity
    work: str = Field(min_length=1)


class Step(c.Record):
    step_id: c.Identity
    requester_share: float = Field(ge=0.05, le=1, allow_inf_nan=False)
    tiers: tuple[int, ...]
    window: Window
    required: tuple[EvidenceRef, ...]
    governed_basis_points: int = Field(strict=True, ge=0, le=10000)
    around_limit: c.Nonnegative
    signed_at: datetime
    owners: tuple[Owner, ...]
    manual_route: str = Field(min_length=1)


class RolloutPlan(c.Record):
    plan_id: c.Identity
    canary_id: c.Identity
    candidate_fingerprint: c.Identity
    signed_by: c.Identity
    signed_at: datetime
    steps: tuple[Step, ...]
    widen_rule: str = Field(min_length=1)
    hold_rule: str = Field(min_length=1)
    reverse_rule: str = Field(min_length=1)

    @property
    def fingerprint(self):
        return fingerprint(self.model_dump(mode="json"))


def validate_structure(plan):
    plan = c._validated(RolloutPlan, plan)
    if not has_role(plan.signed_by, "decision_owner") or not plan.steps:
        raise ValueError("plan needs listed decision owner and steps")
    if len({s.step_id for s in plan.steps}) != len(plan.steps):
        raise ValueError("duplicate step identity")
    if len({s.window.window_id for s in plan.steps}) != len(plan.steps):
        raise ValueError("duplicate window identity")
    share, previous_end = 0.05, None
    for step in plan.steps:
        roles = {owner.role: owner for owner in step.owners}
        if set(roles) != set(ROLES) or len(roles) != len(step.owners):
            raise ValueError(
                "every role requires one named owner and its work"
            )
        for role in ("decision_owner", "release_owner"):
            if not has_role(roles[role].person, role):
                raise ValueError("plan role is not authorized by owners.json")
        if roles["decision_owner"].person != plan.signed_by:
            raise ValueError("step and plan name different decision owners")
        if step.tiers != (1, 2) or step.around_limit != 0:
            raise ValueError(
                "stage 6 grants tiers 1/2 and permits no paid workaround"
            )
        if step.governed_basis_points != 9900:
            raise ValueError(
                "chapter's signed governed-share threshold is 99 percent"
            )
        if step.window.end - step.window.start != timedelta(days=14):
            raise ValueError("step needs the named fourteen-day window")
        if not plan.signed_at <= step.signed_at < step.window.start:
            raise ValueError("settings must be signed before the window")
        if step.requester_share < share:
            raise ValueError(
                "step must include previous requesters under the same hash"
            )
        if previous_end is not None and step.window.start < previous_end:
            raise ValueError("step windows overlap")
        kinds = [ref.record_type for ref in step.required]
        if set(kinds) != set(EVIDENCE_TYPES) or len(kinds) != len(
            EVIDENCE_TYPES
        ):
            raise ValueError("step needs each predecessor evidence type once")
        share, previous_end = step.requester_share, step.window.end
    return plan


def resolve_step(plan, step, evidence):
    resolved = {
        (row.record_type, row.record_id): c._validated(EvidenceRecord, row)
        for row in evidence
    }
    if len(resolved) != len(evidence):
        raise ValueError("duplicate evidence identity")
    for ref in step.required:
        row = resolved.get((ref.record_type, ref.record_id))
        if row is None:
            raise ValueError("missing required record: " + ref.record_id)
        if (
            row.signed_by != plan.signed_by
            or not row.signed_at < step.window.start
        ):
            raise ValueError(
                "evidence must be signed by decision owner before window"
            )
        if row.candidate_fingerprint != plan.candidate_fingerprint:
            raise ValueError("evidence covers another candidate")
        if row.status != "pass":
            raise ValueError(
                "required record has not passed: " + ref.record_id
            )
        if ref.record_type == "audit" and (
            (row.period_start, row.period_end)
            != (ref.period_start, ref.period_end)
            or row.period_end is None
            or row.period_end > step.window.start
            or row.labels_mature_at is None
            or row.labels_mature_at >= step.window.start
        ):
            raise ValueError(
                "audit period or maturity does not cover prerequisite"
            )


def validate(plan, evidence):
    plan = validate_structure(plan)
    for step in plan.steps:
        resolve_step(plan, step, evidence)
    return plan
