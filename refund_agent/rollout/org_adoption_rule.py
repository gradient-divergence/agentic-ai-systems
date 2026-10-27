"""Paid workarounds hold; unpaid gaps count against the intake denominator."""

from datetime import datetime, timedelta
from typing import Literal

from pydantic import Field

from refund_agent import contracts as c
from refund_agent.rollout.org_adoption_plan import (
    resolve_step,
    validate_structure,
)
from refund_agent.rollout.org_adoption_records import (
    Cause,
    EvidenceRecord,
    Window,
    WindowCount,
)


class StepDecision(c.Record):
    plan_id: c.Identity
    step_id: c.Identity
    window_id: c.Identity
    at: datetime
    outcome: str
    reasons: tuple[str, ...]


class Change(c.Record):
    change_id: c.Identity
    description: str = Field(min_length=1)
    owner: c.Identity
    at: datetime
    held_window_id: c.Identity


class Retry(c.Record):
    kind: Literal["org-adoption-retry"] = "org-adoption-retry"
    plan_fingerprint: c.Identity
    held: StepDecision
    change: Change
    window: Window
    by: c.Identity
    at: datetime


def validate_retry(plan, step, retry):
    retry = c._validated(Retry, retry)
    if (
        retry.plan_fingerprint != plan.fingerprint
        or retry.held.plan_id != plan.plan_id
        or retry.held.step_id != step.step_id
        or retry.by != plan.signed_by
    ):
        raise ValueError("retry must name the held plan, step and owner")
    next_window(retry.held, retry.change, retry.window)
    if not retry.change.at <= retry.at < retry.window.start:
        raise ValueError("retry must be recorded after change before window")
    return retry


def count_reasons(step, count):
    count = c._validated(WindowCount, count)
    reasons = []
    if count.around:
        reasons.append("paid_around_path")
    if (
        not count.received
        or count.governed * 10000 < count.received * step.governed_basis_points
    ):
        reasons.append("governed_share_below_threshold")
    if count.pending:
        reasons.append("pending_decisions")
    if count.exceptions:
        reasons.append("unmatched_record_exceptions")
    return reasons


def decide(
    plan, step, count, evidence, *, reversals=(), causes=(), retry=None
):
    plan = validate_structure(plan)
    count = c._validated(WindowCount, count)
    if step not in plan.steps:
        raise ValueError("count must cover the plan's exact step window")
    if retry is not None:
        retry = validate_retry(plan, step, retry)
        step = step.model_copy(update={"window": retry.window})
    if count.window != step.window:
        raise ValueError("count must cover the plan's exact step window")
    allowed = {"forbidden_payment", "budget_spent", "tier2_unbound_approval"}
    recorded = set(reversals)
    if recorded - allowed:
        raise ValueError("unknown reversal condition")
    budget_ids = {
        ref.record_id for ref in step.required if ref.record_type == "budget"
    }
    for row in evidence:
        row = c._validated(EvidenceRecord, row)
        if (
            row.record_type == "budget"
            and row.record_id in budget_ids
            and row.candidate_fingerprint == plan.candidate_fingerprint
            and row.signed_by == plan.signed_by
            and row.signed_at < step.window.start
            and row.status == "spent"
        ):
            recorded.add("budget_spent")
    if recorded:
        outcome, reasons = "move_back", tuple(sorted(recorded))
    else:
        reasons = count_reasons(step, count)
        try:
            resolve_step(plan, step, evidence)
        except ValueError as error:
            reasons.append(str(error))
        # Exceptions are reported before a reconciler assigns causes.
        known = {p.payment_id for p in count.around} | set(count.missing)
        checked = [c._validated(Cause, row) for row in causes]
        if len({row.exception_id for row in checked}) != len(checked):
            raise ValueError("duplicate cause identity")
        if any(row.exception_id not in known for row in checked):
            raise ValueError("cause does not name a reconciled exception")
        if known - {row.exception_id for row in checked}:
            reasons.append("workaround_disposition_missing")
        outcome = "hold" if reasons else "widen"
    return StepDecision(
        plan_id=plan.plan_id,
        step_id=step.step_id,
        window_id=count.window.window_id,
        at=count.window.closed_at,
        outcome=outcome,
        reasons=tuple(reasons),
    )


def next_window(held, change, window):
    held = c._validated(StepDecision, held)
    window = c._validated(Window, window)
    if held.outcome != "hold":
        raise ValueError("only a held window needs a repair record")
    if change is None or change.held_window_id != held.window_id:
        raise ValueError("next window must name the held window's change")
    change = c._validated(Change, change)
    if not held.at < change.at < window.start:
        raise ValueError(
            "named change must follow hold and precede next window"
        )
    if (
        window.window_id == held.window_id
        or window.end - window.start != timedelta(days=14)
    ):
        raise ValueError("retry needs a new named fourteen-day window")
    return window
