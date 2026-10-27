"""Persisted paid-only confirmation; budget withdrawal has no confirmation delay."""

import fcntl
import json
from contextlib import contextmanager
from math import isfinite
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from refund_agent.contracts import Record
from refund_agent.renewal.future_directions_math import calibrated
from refund_agent.selection import fingerprint


def check_month(rows, state, *, period, day, confirmation_size=2000):
    """Pure exercise/reference kernel. Ordinals count every scored request."""
    if state["status"] not in {"clear", "pending", "reverse"}:
        raise ValueError("unknown confirmation state")
    if state["status"] == "reverse":
        return dict(state)
    if not rows:
        return dict(state)
    ordinals = [r.ordinal for r in rows]
    if ordinals != list(range(ordinals[0], ordinals[-1] + 1)) or (
        len({r.request_id for r in rows}) != len(rows)
    ):
        raise ValueError("cohort needs ordered consecutive unique requests")
    pending = state["status"] == "pending"
    if pending:
        if ordinals[0] != state["first"] or ordinals[-1] > state["last"]:
            raise ValueError("confirmation cohort differs from persisted boundaries")
        if ordinals[-1] < state["last"]:
            return dict(state)
    paid = [r for r in rows if r.paid]
    if not paid:
        return dict(state)
    if any(not isfinite(r.prediction) or not 0 <= r.prediction <= 1 for r in paid):
        raise ValueError("paid predictions must be finite probabilities")
    # Inspect maturity before reading any outcome, including outcomes already final.
    if any(r.label_final_day is None or r.label_final_day > day for r in paid):
        return dict(state)
    if any(r.label_final_day != r.paid_day + 30 for r in paid):
        raise ValueError("paid outcome must mature thirty days after payment")
    outcomes = [r.found_false_30d for r in paid]
    if any(type(y) is not int or y not in (0, 1) for y in outcomes):
        raise ValueError("matured paid outcomes required")
    prediction = sum(r.prediction for r in paid) / len(paid)
    if calibrated(prediction, sum(outcomes), len(paid)):
        return {"status": "clear"}
    if pending:
        return {**state, "status": "reverse"}
    return {
        "status": "pending",
        "failed_month": period,
        "first": ordinals[-1] + 1,
        "last": ordinals[-1] + confirmation_size,
    }


class WatchState(Record):
    adoption_fingerprint: str = Field(min_length=1)
    requirement_fingerprint: str = Field(min_length=1)
    status: Literal["clear", "pending", "reverse"] = "clear"
    failed_month: str | None = None
    first: int | None = None
    last: int | None = None
    grant_withdrawn: bool = False
    withdrawal_record: str | None = None
    rollback_record: str | None = None
    last_month_end: int = 0
    checked_day: float = 0

    @model_validator(mode="after")
    def boundaries(self):
        if self.status == "pending" and (
            not self.failed_month or self.first is None or self.last != self.first + 1999
        ):
            raise ValueError("pending confirmation requires next-2000 cohort boundaries")
        return self


class BudgetStatus(Record):
    """Chapter 41's reader contract for chapter 34's reloadable budget status."""

    plan_fingerprint: str = Field(min_length=1)
    window_fingerprint: str = Field(min_length=1)
    spent: bool = Field(strict=True)
    status_fingerprint: str = Field(min_length=1)


class WatchResult(Record):
    verdict: Literal["clear", "pending", "reverse", "withdrawn"]
    state: WatchState
    detail: str


class WatchStore:
    """Append-only local state journal, identity bound and locked across processes."""

    def __init__(self, path, adoption_fingerprint, requirement_fingerprint):
        self.path = Path(path)
        self.adoption = adoption_fingerprint
        self.requirement = requirement_fingerprint

    @contextmanager
    def transaction(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as stream:
            fcntl.flock(stream, fcntl.LOCK_EX)
            rows = [json.loads(line) for line in self.path.read_text().splitlines()]
            previous = ""
            state = WatchState(
                adoption_fingerprint=self.adoption, requirement_fingerprint=self.requirement
            )
            for row in rows:
                if row["previous"] != previous or row["fingerprint"] != fingerprint(
                    {k: v for k, v in row.items() if k != "fingerprint"}
                ):
                    raise ValueError("watch journal integrity failure")
                state = WatchState.model_validate(row["state"])
                if (state.adoption_fingerprint, state.requirement_fingerprint) != (
                    self.adoption,
                    self.requirement,
                ):
                    raise ValueError("watch journal belongs to another adoption")
                previous = row["fingerprint"]

            def save(value):
                nonlocal state, previous
                value = WatchState.model_validate(value.model_dump())
                if (
                    (value.adoption_fingerprint, value.requirement_fingerprint)
                    != (self.adoption, self.requirement)
                    or state.grant_withdrawn
                    and not value.grant_withdrawn
                ):
                    raise ValueError("cannot change identity or restore a withdrawn grant")
                record = {"previous": previous, "state": value.model_dump(mode="json")}
                record["fingerprint"] = fingerprint(record)
                stream.write(json.dumps(record, sort_keys=True) + "\n")
                stream.flush()
                import os

                os.fsync(stream.fileno())
                previous, state = record["fingerprint"], value

            yield state, save
            fcntl.flock(stream, fcntl.LOCK_UN)

    def load(self):
        with self.transaction() as (state, _):
            return state


def restore(adoption, *, rollback, day):
    """Call chapter 33's rollback command; its client never receives a ledger."""
    record = rollback(
        adoption=adoption,
        target=adoption["incumbent_bundle_fingerprint"],
        target_candidate_fingerprint=adoption["incumbent_candidate_fingerprint"],
        day=day,
    )
    if record["target_fingerprint"] != adoption["incumbent_bundle_fingerprint"] or (
        record["target_candidate_fingerprint"] != adoption["incumbent_candidate_fingerprint"]
        or record["stage"] != 6
        or record["verdict"] != "restored"
    ):
        raise ValueError("rollback did not restore the admitted incumbent at stage 6")
    return record


def budget_spent(store, adoption, *, budget, withdraw, rollback=None, day):
    """Registered monitor callback. Withdrawal precedes score restoration."""
    if (budget.plan_fingerprint, budget.window_fingerprint) != (
        adoption["plan_fingerprint"],
        adoption["window_fingerprint"],
    ) or not budget.spent:
        raise ValueError("budget status is unspent or belongs to another plan/window")
    with store.transaction() as (state, save):
        if state.adoption_fingerprint != adoption["fingerprint"]:
            raise ValueError("adoption differs from the watch journal")
        # Durable intent prevents a restart from reviving authority if a callback fails.
        state = state.model_copy(update={"grant_withdrawn": True, "checked_day": day})
        save(state)
        if state.withdrawal_record is None:
            if withdraw is None:
                raise ValueError("withdrawal client missing; authority stop intent persisted")
            record = withdraw(adoption=adoption, reason="budget_spent", day=day)
            if record["tier1_authority"] != "shadow" or record["decider"] != "person":
                raise ValueError("budget exhaustion must withdraw tier 1 to a person")
            state = state.model_copy(update={"withdrawal_record": fingerprint(record)})
            save(state)
        if state.rollback_record is None and rollback is not None:
            record = restore(adoption, rollback=rollback, day=day)
            state = state.model_copy(update={"rollback_record": fingerprint(record)})
            save(state)
        detail = "budget_spent; no confirmation"
        if state.rollback_record is None:
            detail += "; score rollback awaits chapter 33 client"
        return WatchResult(verdict="withdrawn", state=state, detail=detail)


def watch(rows, adoption, req, day, store, *, period, budget=None, withdraw=None, rollback=None):
    if (
        adoption["verdict"] not in {"adopted", "adopted-launch-pending"}
        or adoption["stage"] != 6
        or (store.requirement != req.fingerprint or store.adoption != adoption["fingerprint"])
    ):
        raise ValueError("watch requires an admitted stage-6 adoption and its requirement")
    if budget is not None:
        if (budget.plan_fingerprint, budget.window_fingerprint) != (
            adoption["plan_fingerprint"],
            adoption["window_fingerprint"],
        ):
            raise ValueError("budget status belongs to another plan/window")
        if budget.spent:
            return budget_spent(
                store, adoption, budget=budget, withdraw=withdraw, rollback=rollback, day=day
            )
    with store.transaction() as (state, save):
        if state.grant_withdrawn:
            return WatchResult(verdict="withdrawn", state=state, detail="grant remains withdrawn")
        if day < state.checked_day:
            raise ValueError("watch cannot move its clock backwards")
        if state.status == "clear" and rows and rows[0].ordinal <= state.last_month_end:
            raise ValueError("month overlaps evidence already read")
        kernel = state.model_dump(
            include={"status", "failed_month", "first", "last"}, exclude_none=True
        )
        result = check_month(rows, kernel, period=period, day=day)
        changes = {
            "status": result["status"],
            "failed_month": result.get("failed_month"),
            "first": result.get("first"),
            "last": result.get("last"),
            "checked_day": day,
        }
        paid = [r for r in rows if r.paid]
        mature = bool(paid) and all(
            r.label_final_day is not None and r.label_final_day <= day for r in paid
        )
        if mature and rows:
            changes["last_month_end"] = rows[-1].ordinal
        state = WatchState.model_validate({**state.model_dump(), **changes})
        if result["status"] == "reverse":
            save(state)
            if state.rollback_record is None:
                if rollback is None:
                    raise ValueError("confirmed reversal requires the rollback client")
                record = restore(adoption, rollback=rollback, day=day)
                state = state.model_copy(update={"rollback_record": fingerprint(record)})
        save(state)
        verdict = state.status if mature else "pending"
        return WatchResult(
            verdict=verdict,
            state=state,
            detail="matured paid-only check" if mature else "awaiting paid outcomes",
        )
