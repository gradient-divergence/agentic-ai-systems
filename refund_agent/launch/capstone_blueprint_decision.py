"""The two launch signatures and scope-specific withdrawals, never a payment."""

from typing import Literal, Protocol

from pydantic import JsonValue

from refund_agent.canary.cloud_native_deployment_plan import has_role
from refund_agent.launch.capstone_blueprint_gate import LaunchVerdict, check
from refund_agent.launch.capstone_blueprint_requirement import (
    Day,
    LaunchRecord,
    LaunchRequirement,
    ReversalCondition,
    Text,
)
from refund_agent.launch.capstone_blueprint_scan import ReversalEvent


class Admission(LaunchRecord):
    record_id: Text
    candidate_bundle_fingerprint: Text
    from_stage: Literal[6] = 6
    to_stage: Literal[7] = 7
    gate_fingerprint: Text
    signed_by: Text
    signed_day: Day


class StandingControl(LaunchRecord):
    name: Text
    owner: Text
    period: Text


class OpenItem(LaunchRecord):
    item: Text
    owner: Text
    accepted_by: Text | None = None


class HeldDecision(LaunchRecord):
    verdict: Literal["hold"] = "hold"
    bundle: Text
    packet_fingerprint: Text
    gate_fingerprint: Text
    held_rows: dict[str, tuple[str, ...]]
    passed_rows: tuple[str, ...]
    stage6_authority_records: tuple[dict[str, JsonValue], ...]
    authority_available: bool
    reversals: tuple[ReversalCondition, ...]
    controls: tuple[StandingControl, ...]
    open_items: tuple[OpenItem, ...]
    clears: dict[str, str]
    day: Day
    signed_by: Literal[None] = None


class LaunchDecision(LaunchRecord):
    verdict: Literal["launch"] = "launch"
    bundle: Text
    packet_fingerprint: Text
    gate_fingerprint: Text
    admission_id: Text
    signed_by: Text
    signed_day: Day
    reversals: tuple[ReversalCondition, ...]
    controls: tuple[StandingControl, ...]
    open_items: tuple[OpenItem, ...]


class AdmissionPort(Protocol):
    def admit_launch(
        self, *, bundle: str, gate: LaunchVerdict, owner: str, day: float
    ) -> Admission: ...


def admit(
    verdict: LaunchVerdict,
    release_owner: str,
    day: float,
    registry: AdmissionPort | None,
) -> Admission:
    verdict = LaunchVerdict.model_validate(verdict.model_dump(mode="json"))
    if (
        verdict.verdict != "launch"
        or verdict.failures
        or not all(all(row.checks.values()) for row in verdict.rows)
    ):
        raise ValueError("a hold cannot be admitted")
    if not has_role(release_owner, "release_owner") or day < verdict.day:
        raise ValueError("admission needs release_owner after the verdict")
    if registry is None:
        raise ValueError("admission needs the real promotion registry")
    admission = registry.admit_launch(
        bundle=verdict.launched_bundle,
        gate=verdict,
        owner=release_owner,
        day=day,
    )
    admission = Admission.model_validate(admission.model_dump(mode="json"))
    if (
        admission.candidate_bundle_fingerprint,
        admission.gate_fingerprint,
        admission.signed_by,
        admission.signed_day,
    ) != (verdict.launched_bundle, verdict.fingerprint, release_owner, day):
        raise ValueError("promotion differs from the judged launch")
    return admission


def _conditions(req):
    distinct = {}
    for row in req.rows:
        for condition in row.reversals:
            previous = distinct.setdefault(condition.condition_id, condition)
            if previous != condition:
                raise ValueError(
                    "reversal condition has conflicting definitions"
                )
    return tuple(distinct.values())


def sign(
    verdict: LaunchVerdict,
    admission: Admission | None,
    decision_owner: str,
    day: float,
    *,
    requirement: LaunchRequirement,
    controls: tuple[StandingControl, ...],
    open_items: tuple[OpenItem, ...],
) -> LaunchDecision:
    verdict = LaunchVerdict.model_validate(verdict.model_dump(mode="json"))
    if verdict.verdict != "launch" or verdict.failures or admission is None:
        raise ValueError("launch signature requires launch and admission")
    admission = Admission.model_validate(admission.model_dump(mode="json"))
    if (
        not has_role(decision_owner, "decision_owner")
        or not has_role(admission.signed_by, "release_owner")
        or day < max(verdict.day, admission.signed_day)
        or admission.signed_day < verdict.day
    ):
        raise ValueError("launch signature needs the correct role and order")
    if (
        admission.candidate_bundle_fingerprint != verdict.launched_bundle
        or admission.gate_fingerprint != verdict.fingerprint
        or requirement.fingerprint != verdict.requirement_fingerprint
    ):
        raise ValueError("signature, admission and gate differ")
    if not controls or any(
        item.accepted_by != decision_owner for item in open_items
    ):
        raise ValueError(
            "standing controls and named acceptance of open items required"
        )
    return LaunchDecision(
        bundle=verdict.launched_bundle,
        packet_fingerprint=verdict.packet_fingerprint,
        gate_fingerprint=verdict.fingerprint,
        admission_id=admission.record_id,
        signed_by=decision_owner,
        signed_day=day,
        reversals=_conditions(requirement),
        controls=controls,
        open_items=open_items,
    )


def hold_record(
    verdict: LaunchVerdict,
    requirement: LaunchRequirement,
    registry,
    authority,
    day: float,
    *,
    controls=(),
    open_items=(),
) -> HeldDecision:
    if (
        verdict.verdict != "hold"
        or day < verdict.day
        or (verdict.requirement_fingerprint != requirement.fingerprint)
    ):
        raise ValueError(
            "held decision needs its held gate and signed requirement"
        )
    held = {}
    for failure in verdict.failures:
        held.setdefault(failure.row, []).append(
            f"{failure.check}/{failure.kind}: {failure.reason}"
        )
    clears = {
        row: (
            "a replacement model shown better than the rules where they differ; a new replacement decision"
            if row == "1"
            else "supply the required record or repair the named check under a new requirement"
        )
        for row in held
    }
    snapshot = (
        ()
        if authority is None
        else tuple(
            event.model_dump(mode="json") for event in authority.events()
        )
    )
    return HeldDecision(
        bundle=verdict.launched_bundle,
        packet_fingerprint=verdict.packet_fingerprint,
        gate_fingerprint=verdict.fingerprint,
        held_rows={k: tuple(v) for k, v in held.items()},
        passed_rows=tuple(
            row.row for row in verdict.rows if all(row.checks.values())
        ),
        stage6_authority_records=snapshot,
        authority_available=authority is not None,
        reversals=_conditions(requirement),
        controls=controls,
        open_items=open_items,
        clears=clears,
        day=day,
    )


def reverse(
    event: ReversalEvent,
    *,
    requirement: LaunchRequirement,
    authority,
    registry,
    by: str,
):
    event = ReversalEvent.model_validate(event.model_dump(mode="json"))
    condition = next(
        c
        for c in _conditions(requirement)
        if c.condition_id == event.condition_id
    )
    if event.planned_time_box:
        raise ValueError("planned canary end is not a launch reversal")
    if not all(has_role(by, role) for role in condition.owner_roles):
        raise ValueError("reversal needs its named owner role")
    if condition.withdraws == "launch":
        return registry.withdraw_launch(event=event, by=by, to_stage=6)
    # Chapter 38 owns all stage-6 authority. Never call the ended canary binding.
    history = authority.events()
    recorded = next(
        (
            row
            for row in history
            if row.kind == "org-adoption-withdrawal" and row.record_id == event.record_id
        ),
        None,
    )
    at = recorded.at if recorded is not None else max((event.at, *(row.at for row in history)))
    withdrawal = authority.withdraw(
        "owner" if condition.withdraws == "all" else event.condition_id,
        by=by,
        at=at,
        record_id=event.record_id,
        scope=condition.withdraws,
    )
    registry.record_withdrawal(event=event, withdrawal=withdrawal)
    return withdrawal


def on_promotion(state, promotion) -> LaunchVerdict:
    """The merged registry supplies the new packet/requirement, not old evidence."""
    if (
        state["requirement"].launched_bundle
        != promotion["candidate_bundle_fingerprint"]
    ):
        raise ValueError(
            "promotion needs a requirement and packet for its candidate"
        )
    verdict = check(
        state["packet"],
        state["requirement"],
        state["registry"],
        state["logs"],
        state["owners"],
        state["day"],
    )
    state["registry"].record_gate(promotion=promotion, verdict=verdict)
    if state["decision"].verdict == "launch" and verdict.verdict == "hold":
        event = ReversalEvent(
            record_id="capstone-blueprint-promotion-" + verdict.fingerprint,
            condition_id="launch_gate_held",
            day=verdict.day,
            at=state["at"],
            reason="promotion's launch gate held",
        )
        reverse(
            event,
            requirement=state["requirement"],
            authority=state["authority"],
            registry=state["registry"],
            by=state["owner"],
        )
    return verdict
