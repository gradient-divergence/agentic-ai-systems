"""Adoption coordinates real partner commands; absent commands hold the step."""

from dataclasses import dataclass
from typing import Callable

from refund_agent.canary.cloud_native_deployment_plan import has_role
from refund_agent.canary.cloud_native_deployment_rehearsal import RehearsalRecord
from refund_agent.contracts import Record
from refund_agent.launch.capstone_blueprint_gate import LaunchVerdict
from refund_agent.selection import fingerprint

RECORDS = ("selection", "calibration", "signed_cost", "signed_decline", "rebuild_check")


class InvalidLaunchVerdict(ValueError):
    """Chapter 39's launch callback returned neither hold nor launch."""


@dataclass(frozen=True)
class AdoptionPorts:
    """Join clients supplied by the merged program, never substitute implementations.

    check_requirements: chapter 33, including its acceptance of signed decline.
    release: chapter 29's signed-record validation against its journal and suite.
    rehearse: chapter 32 run_rehearsal, wrapped with the real plan/stream/store.
    promote/on_promotion: chapters 33/39; register_watch: chapters 34/38.
    on_promotion returns chapter 39's LaunchVerdict model.
    """

    check_requirements: Callable
    release: Callable
    rehearse: Callable
    promote: Callable
    on_promotion: Callable
    register_watch: Callable


class Adoption(Record):
    verdict: str
    stage: int = 6
    decision_fingerprint: str
    requirement_fingerprint: str
    candidate_bundle_fingerprint: str | None
    candidate_fingerprint: str | None
    incumbent_bundle_fingerprint: str | None
    incumbent_candidate_fingerprint: str
    day: float
    reason: str
    records: tuple[str, ...] = ()
    promotion_fingerprint: str | None = None
    launch_verdict: str | None = None
    watch_id: str | None = None
    plan_fingerprint: str | None = None
    window_fingerprint: str | None = None
    release_record: dict | None = None
    rehearsal_record: dict | None = None
    promotion_record: dict | None = None
    launch_record: dict | None = None

    @property
    def fingerprint(self):
        return fingerprint(self.model_dump(mode="json"))


def adopt(decision, req, registry, owners, day, *, records, ports=None, launch_state=None):
    base = dict(
        decision_fingerprint=decision.fingerprint,
        requirement_fingerprint=req.fingerprint,
        candidate_bundle_fingerprint=decision.candidate_bundle_fingerprint,
        candidate_fingerprint=decision.candidate_fingerprint,
        incumbent_bundle_fingerprint=req.incumbent_bundle_fingerprint,
        incumbent_candidate_fingerprint=req.incumbent_candidate_fingerprint,
        day=day,
    )

    def hold(reason):
        return Adoption(verdict="held", reason=reason, **base)

    if decision.verdict != "replace":
        return hold("decision is " + decision.verdict + "; adoption requires replace")
    if (
        decision.candidate_fingerprint is None
        or decision.candidate_bundle_fingerprint is None
        or req.incumbent_bundle_fingerprint is None
    ):
        return hold("bundle and candidate identities required")
    if decision.requirement_fingerprint != req.fingerprint or day <= decision.day:
        return hold("adoption precedes or differs from its replacement decision")
    if not has_role(owners, "release_owner"):
        return hold("adoption requires release_owner")
    for name in RECORDS:
        record = records.get(name)
        if not record:
            return hold("missing " + name)
        if (
            not isinstance(record, dict)
            or record.get("requirement_fingerprint") != req.fingerprint
            or record.get("passed") is not True
        ):
            return hold("failed or mismatched " + name)
    if ports is None:
        return hold("partner commands unavailable: chapters 33, 37, 38 and 39; no promotion issued")
    promotion = None
    try:
        result = ports.check_requirements(
            registry=registry, records=records, req=req, decision=decision
        )
        if result["passed"] is not True or result["signed_decline_accepted"] is not True:
            return hold("chapter 33 requirements did not accept the signed decline")
        release = ports.release(candidate_fingerprint=decision.candidate_fingerprint)
        if (
            release["decision"] != "pass"
            or release["signature"]["owner"] not in {owners}
            or (release["software_fingerprint"] != decision.candidate_fingerprint)
        ):
            return hold("signed passing release record missing or mismatched")
        rehearsal = RehearsalRecord.model_validate(
            ports.rehearse(
                candidate_fingerprint=decision.candidate_fingerprint,
                rollback_target_fingerprint=req.incumbent_candidate_fingerprint,
            )
        ).model_dump(mode="json")
        if (
            rehearsal["verdict"] != "pass"
            or rehearsal["rollback_target_fingerprint"] != req.incumbent_candidate_fingerprint
            or (rehearsal["candidate_fingerprint"] != decision.candidate_fingerprint)
        ):
            return hold("rollback rehearsal failed or names another target")
        # Wiring is a gate before the promotion can make the score active.
        wiring = ports.register_watch(
            owner=req.reversal_owner,
            target=req.incumbent_bundle_fingerprint,
            target_candidate_fingerprint=req.incumbent_candidate_fingerprint,
            requirement=req,
            decision=decision,
        )
        if (
            wiring["budget_withdrawal_registered"] is not True
            or wiring["confirmation_persisted"] is not True
            or any(
                not isinstance(wiring.get(k), str) or not wiring[k]
                for k in ("watch_id", "plan_fingerprint", "window_fingerprint")
            )
        ):
            return hold("reversal and immediate withdrawal are not wired")
        promotion = ports.promote(
            registry=registry,
            owner=owners,
            stage=6,
            decision=decision,
            candidate_bundle_fingerprint=decision.candidate_bundle_fingerprint,
            rollback_target_fingerprint=req.incumbent_bundle_fingerprint,
            rollback_target_candidate_fingerprint=req.incumbent_candidate_fingerprint,
            rehearsal_fingerprint=fingerprint(rehearsal),
            day=day,
        )
        if (
            promotion["stage"] != 6
            or promotion["verdict"] != "admitted"
            or (
                promotion["rollback_target_fingerprint"] != req.incumbent_bundle_fingerprint
                or promotion["rollback_target_candidate_fingerprint"]
                != req.incumbent_candidate_fingerprint
                or promotion["candidate_bundle_fingerprint"]
                != decision.candidate_bundle_fingerprint
                or promotion["candidate_fingerprint"] != decision.candidate_fingerprint
                or promotion["decision_fingerprint"] != decision.fingerprint
            )
        ):
            raise ValueError("promotion refused or differs from the stage-6 replacement")
        try:
            gate = ports.on_promotion(state=launch_state, promotion=promotion).model_dump(
                mode="json"
            )
            if gate["verdict"] not in ("hold", "launch"):
                raise InvalidLaunchVerdict(f"{gate['verdict']!r}; expected 'hold' or 'launch'")
            gate = LaunchVerdict.model_validate(gate).model_dump(mode="json")
            if gate["launched_bundle"] != decision.candidate_bundle_fingerprint:
                raise InvalidLaunchVerdict("launch verdict names another candidate bundle")
        except Exception as error:
            # Promotion is durable; every callback failure must retain monitoring.
            raise ValueError(f"{type(error).__name__}: {error}") from error
        return Adoption(
            verdict="adopted",
            reason="admitted at stage 6; launch verdict recorded",
            records=RECORDS + ("release", "rehearsal", "promotion", "launch", "watch"),
            promotion_fingerprint=fingerprint(promotion),
            launch_verdict=gate["verdict"],
            watch_id=wiring["watch_id"],
            plan_fingerprint=wiring["plan_fingerprint"],
            window_fingerprint=wiring["window_fingerprint"],
            release_record=release,
            rehearsal_record=rehearsal,
            promotion_record=promotion,
            launch_record=gate,
            **base,
        )
    except (ValueError, KeyError, TypeError) as error:
        if promotion is not None and promotion.get("verdict") == "admitted":
            return Adoption(
                verdict="adopted-launch-pending",
                reason="stage-6 promotion recorded; partner reconciliation pending: " + str(error),
                records=RECORDS + ("release", "rehearsal", "promotion", "watch"),
                promotion_fingerprint=fingerprint(promotion),
                watch_id=wiring["watch_id"],
                plan_fingerprint=wiring["plan_fingerprint"],
                window_fingerprint=wiring["window_fingerprint"],
                release_record=release,
                rehearsal_record=rehearsal,
                promotion_record=promotion,
                **base,
            )
        return hold("partner step refused: " + str(error))
