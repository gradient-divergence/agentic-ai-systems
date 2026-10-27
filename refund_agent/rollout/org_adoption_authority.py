"""Latest signed step controls decisions and execution; withdrawals append only."""

from datetime import datetime, timedelta
from pathlib import Path
from threading import RLock
from typing import Literal

from refund_agent import contracts as c
from refund_agent.canary.cloud_native_deployment_plan import (
    has_role,
    in_cohort,
)
from refund_agent.canary.cloud_native_deployment_stop import CanarySession
from refund_agent.rollout.org_adoption_plan import (
    resolve_step,
    validate_structure,
)
from refund_agent.rollout.org_adoption_rule import (
    Retry,
    StepDecision,
    validate_retry,
)
from refund_agent.rules import decision, execution_time, tier_for
from refund_agent.tools import Tools

WithdrawalScope = Literal["tier1", "tier2", "all"]


class StepGrant(c.Record):
    kind: Literal["org-adoption-grant"] = "org-adoption-grant"
    plan_fingerprint: c.Identity
    step_id: c.Identity
    by: c.Identity
    at: datetime


class Withdrawal(c.Record):
    kind: Literal["org-adoption-withdrawal"] = "org-adoption-withdrawal"
    plan_fingerprint: c.Identity
    condition: Literal[
        "forbidden_payment", "budget_spent", "tier2_unbound_approval", "owner"
    ]
    scope: WithdrawalScope
    by: c.Identity
    at: datetime
    record_id: c.Identity


class BudgetExhaustion(c.Record):
    """The monitor callback's chapter-38 boundary, bound to plan and window."""

    budget_id: c.Identity
    plan_id: c.Identity
    window_id: c.Identity
    at: datetime
    spent: Literal[True]


class StageAuthority:
    version = "org-adoption.authority@1"

    def __init__(self, plan, path: Path, *, canary_session=None):
        self.plan = validate_structure(plan)
        self.path = Path(path)
        self.canary_session = canary_session
        if canary_session is not None and (
            not isinstance(canary_session, CanarySession)
            or canary_session.plan.canary_id != plan.canary_id
        ):
            raise ValueError("authority must name its actual canary session")
        self._lock = RLock()
        self.events()  # Reload validates the whole history before it can authorize.

    def events(self):
        result = []
        if not self.path.exists():
            return ()
        for line in self.path.read_text().splitlines():
            # Both declared shapes are closed. No unknown event is ignored.
            import json

            value = json.loads(line)
            kind = value["kind"]
            model = {
                "org-adoption-grant": StepGrant,
                "org-adoption-withdrawal": Withdrawal,
                "org-adoption-retry": Retry,
            }[kind]
            row = model.model_validate_json(line)
            if (
                row.plan_fingerprint != self.plan.fingerprint
                or not has_role(row.by, "release_owner")
                and not has_role(row.by, "decision_owner")
            ):
                raise ValueError("authority event has wrong plan or owner")
            if result and row.at < result[-1].at:
                raise ValueError("authority events are out of order")
            if isinstance(row, StepGrant):
                step = next(
                    (s for s in self.plan.steps if s.step_id == row.step_id),
                    None,
                )
                if (
                    step is None
                    or row.by != self.plan.signed_by
                    or row.at >= step.window.start
                ):
                    raise ValueError(
                        "step grant has wrong signer, step or signature time"
                    )
            elif isinstance(row, Retry):
                self._check_retry(row, result)
            result.append(row)
        return tuple(result)

    def _append(self, row):
        with self._lock:
            previous = self.events()
            if previous and row.at < previous[-1].at:
                raise ValueError("authority clock moved backwards")
            if not self.path.parent.is_dir():
                raise ValueError(
                    "authority store parent must be provisioned before composition"
                )
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(row.model_dump_json() + "\n")
            if self.events() != previous + (row,):
                raise ValueError("authority append was not read back")
        return row

    def _check_retry(self, retry, history):
        grants = [row for row in history if isinstance(row, StepGrant)]
        if not grants or retry.held.step_id != grants[-1].step_id:
            raise ValueError("retry needs the currently granted held step")
        index = next(
            i
            for i, step in enumerate(self.plan.steps)
            if step.step_id == retry.held.step_id
        )
        step = self.plan.steps[index]
        validate_retry(self.plan, step, retry)
        prior = [row for row in history if isinstance(row, Retry)]
        step_retries = [
            row for row in prior if row.held.step_id == step.step_id
        ]
        held_window = step_retries[-1].window if step_retries else step.window
        if (
            retry.held.window_id != held_window.window_id
            or retry.held.at != held_window.closed_at
            or retry.window.window_id
            in {s.window.window_id for s in self.plan.steps}
            | {row.window.window_id for row in prior}
        ):
            raise ValueError("retry must preserve the latest held window")
        if index + 1 < len(self.plan.steps) and (
            retry.window.end > self.plan.steps[index + 1].window.start
        ):
            raise ValueError("retry overlaps the next planned step")
        if any(isinstance(row, Withdrawal) for row in history):
            raise ValueError("retry cannot restore withdrawn authority")

    def retry_window(self, held, change, window, *, by, at):
        retry = Retry(
            plan_fingerprint=self.plan.fingerprint,
            held=held,
            change=change,
            window=window,
            by=by,
            at=at,
        )
        with self._lock:
            self._check_retry(retry, self.events())
            return self._append(retry)

    def sign_step(self, step_id, *, by, at, evidence, previous=None):
        at = c._utc_time(at)
        if by != self.plan.signed_by or not has_role(by, "decision_owner"):
            raise ValueError("only the plan's decision owner signs a step")
        index = next(
            i for i, s in enumerate(self.plan.steps) if s.step_id == step_id
        )
        step = self.plan.steps[index]
        if not self.plan.signed_at <= at < step.window.start:
            raise ValueError("step signature must precede its window")
        if any(
            row.signed_at > at
            for row in evidence
            if row.record_id in {r.record_id for r in step.required}
        ):
            raise ValueError("step signature precedes its evidence")
        resolve_step(self.plan, step, evidence)
        history = self.events()
        grants = [row for row in history if isinstance(row, StepGrant)]
        if len(grants) != index:
            raise ValueError("step grants must follow plan order exactly once")
        if index:
            prior_window = self.plan.steps[index - 1].window
            retries = [
                row
                for row in history
                if isinstance(row, Retry)
                and row.held.step_id == self.plan.steps[index - 1].step_id
            ]
            if retries:
                prior_window = retries[-1].window
            if previous is not None:
                previous = c._validated(StepDecision, previous)
            if (
                previous is None
                or previous.outcome != "widen"
                or previous.plan_id != self.plan.plan_id
                or previous.step_id != self.plan.steps[index - 1].step_id
                or previous.window_id != prior_window.window_id
                or previous.at != prior_window.closed_at
                or previous.at > at
            ):
                raise ValueError(
                    "next step needs previous window's widening evidence"
                )
        if any(isinstance(row, Withdrawal) for row in history):
            raise ValueError("a later step cannot restore withdrawn authority")
        return self._append(
            StepGrant(
                plan_fingerprint=self.plan.fingerprint,
                step_id=step_id,
                by=by,
                at=at,
            )
        )

    def reason(self, request, at):
        request, at = c._validated(c.Request, request), c._utc_time(at)
        rows = [row for row in self.events() if row.at <= at]
        tier = tier_for(request.amount_cents)
        closed = {"tier1": c.Tier.AGENT, "tier2": c.Tier.REVIEWER}
        if any(
            isinstance(row, Withdrawal)
            and (row.scope == "all" or closed.get(row.scope) == tier)
            for row in rows
        ):
            return "org_adoption_withdrawn"
        grants = [row for row in rows if isinstance(row, StepGrant)]
        if not grants:
            session = self.canary_session
            if (
                session is not None
                and session.outcome == "running"
                and session.grant is not None
                and session.grant.started_at <= at
                and at
                < session.grant.started_at
                + timedelta(days=session.plan.time_box_days)
                and tier == c.Tier.AGENT
                and in_cohort(self.plan.canary_id, request.requester_id)
            ):
                return None
            return "org_adoption_no_signed_step"
        step = next(
            s for s in self.plan.steps if s.step_id == grants[-1].step_id
        )
        if at < step.window.start:
            return "org_adoption_step_not_started"
        if int(tier) not in step.tiers:
            return "org_adoption_person_required"
        if step.requester_share < 1 and not in_cohort(
            self.plan.canary_id,
            request.requester_id,
            step.requester_share,
        ):
            return "org_adoption_outside_step"
        return None

    def route(self, request, at):
        return "agent" if self.reason(request, at) is None else "person"

    def withdraw(
        self,
        condition,
        *,
        by,
        at,
        record_id,
        scope: WithdrawalScope | None = None,
    ):
        if not has_role(by, "release_owner") and not has_role(
            by, "decision_owner"
        ):
            raise ValueError(
                "withdrawal needs a listed release or decision owner"
            )
        required: dict[str, WithdrawalScope] = {
            "forbidden_payment": "tier1",
            "budget_spent": "tier1",
            "tier2_unbound_approval": "tier2",
        }
        if condition == "owner":
            if scope is None:
                raise ValueError("owner withdrawal must name its scope")
        elif (
            condition not in required
            or scope is not None
            and scope != required[condition]
        ):
            raise ValueError(
                "withdrawal scope differs from its reversal condition"
            )
        else:
            scope = required[condition]
        for row in self.events():
            if isinstance(row, Withdrawal) and row.record_id == record_id:
                if (row.condition, row.scope, row.by, row.at) != (
                    condition,
                    scope,
                    by,
                    at,
                ):
                    raise ValueError("withdrawal identity conflict")
                return row
        return self._append(
            Withdrawal(
                plan_fingerprint=self.plan.fingerprint,
                condition=condition,
                scope=scope,
                by=by,
                at=at,
                record_id=record_id,
            )
        )

    def on_budget_spent(self, status):
        status = c._validated(BudgetExhaustion, status)
        if status.plan_id != self.plan.plan_id:
            raise ValueError("budget exhaustion names another rollout plan")
        windows = {s.window.window_id: s for s in self.plan.steps}
        for row in self.events():
            if isinstance(row, Retry):
                windows[row.window.window_id] = next(
                    s for s in self.plan.steps if s.step_id == row.held.step_id
                )
        step = windows.get(status.window_id)
        if step is None or status.budget_id not in {
            r.record_id for r in step.required if r.record_type == "budget"
        }:
            raise ValueError("budget exhaustion names another budget/window")
        return self.withdraw(
            "budget_spent",
            by=self.plan.signed_by,
            at=status.at,
            record_id="org-adoption-budget-" + status.budget_id,
        )


class AuthorityTools(Tools):
    """Mixin over the supplied tools: the core gate and credential checks remain."""

    def __init__(
        self,
        data,
        *,
        stage_authority,
        org_adoption_candidate,
        org_adoption_context,
        **kwargs,
    ):
        from refund_agent.release.candidate import build_tools

        self.stage_authority = stage_authority
        self.org_adoption_candidate = org_adoption_candidate
        self.shadow_recorder = build_tools(
            data, org_adoption_candidate, "shadow", **org_adoption_context
        )
        self._shadow_traces = {}
        super().__init__(data, **kwargs)

    @property
    def shadow_traces(self):
        return c._detach(tuple(self._shadow_traces.values()))

    def record_shadow(self, request, at):
        from refund_agent.release.candidate import run_request

        key = (request.request_id, at)
        if key not in self._shadow_traces:
            self.shadow_recorder.data = self.data
            self._shadow_traces[key] = run_request(
                self.org_adoption_candidate,
                self.shadow_recorder,
                request,
                at=at,
            )

    def assess(self, state):
        at = state.execution_at or state.request.at
        reason = self.stage_authority.reason(state.request, at)
        if reason:
            self.record_shadow(state.request, at)
            if (
                reason == "org_adoption_person_required"
                and tier_for(state.request.amount_cents) == c.Tier.NAMED_APPROVER
            ):
                return super().assess(state)
            return decision(state, c.Outcome.REFER, reason)
        return super().assess(state)

    def _execute(self, call, state):
        # The core resolves and rechecks this approval again inside refund.execute.
        at = execution_time(state, self.approval_for(state))
        reason = self.stage_authority.reason(state.request, at)
        if reason:
            self.record_shadow(state.request, at)
            return c.ToolResult(ok=False, reason=reason)
        return super()._execute(call, state)

    def _decision(self, state, reason=None):
        blocked = self.stage_authority.reason(
            state.request, state.execution_at or state.request.at
        )
        # A routing restriction must retain the proposed tier-3 denial's reason.
        if (
            blocked == "org_adoption_person_required"
            and tier_for(state.request.amount_cents) == c.Tier.NAMED_APPROVER
        ):
            blocked = None
        return super()._decision(state, reason=blocked or reason)
