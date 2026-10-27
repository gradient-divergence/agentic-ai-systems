"""Four checks per row. A missing partner parser or registry holds the row."""

import json
from typing import Literal, Protocol

from pydantic import Field, JsonValue, model_validator

from refund_agent.launch.capstone_blueprint_packet import (
    EvidenceClaims,
    EvidenceFile,
    ManifestEntry,
    Packet,
    within_run,
)
from refund_agent.launch.capstone_blueprint_requirement import (
    Day,
    LaunchRecord,
    RecordNeed,
    Text,
    holds_role,
)
from refund_agent.selection import fingerprint

CheckName = Literal["complete", "consistent", "signed_in_order", "reversible"]


class RegistryPort(Protocol):
    """Join to 33's registry/required(diff), schemas and launch commands.

    validate_record must parse the producer's actual type and compare every
    indexed claim (including governing signature and window) with that record.
    A metadata-only parser is insufficient. No local promotion table is used.
    """

    def validate_record(
        self, kind: str, payload: dict[str, JsonValue], claims: EvidenceClaims
    ) -> None: ...
    def covered(self, earlier: str, launched: str, kind: str) -> bool: ...
    def rollback_target(self, bundle: str) -> str: ...
    def mechanisms(self, bundle: str) -> frozenset[str]: ...
    def found_false(self, record_id: str) -> bool: ...
    def withdraw_launch(self, *, event: object, by: str, to_stage: int) -> object: ...
    def record_withdrawal(self, *, event: object, withdrawal: object) -> None: ...
    def record_gate(self, *, promotion: dict, verdict: "LaunchVerdict") -> None: ...
    def admit_launch(
        self, *, bundle: str, gate: "LaunchVerdict", owner: str, day: float
    ) -> object: ...


class GateFailure(LaunchRecord):
    row: Text
    kind: Text
    check: CheckName
    reason: Text


class RowVerdict(LaunchRecord):
    row: Text
    checks: dict[CheckName, bool]
    counted: tuple[str, ...]


class LaunchVerdict(LaunchRecord):
    verdict: Literal["launch", "hold"]
    launched_bundle: Text
    packet_fingerprint: Text
    requirement_fingerprint: Text
    day: Day
    rows: tuple[RowVerdict, ...] = Field(min_length=7, max_length=7)
    failures: tuple[GateFailure, ...]

    @model_validator(mode="after")
    def verdict_matches_checks(self):
        names = {"complete", "consistent", "signed_in_order", "reversible"}
        ids = {row.row for row in self.rows}
        if len(ids) != 7 or any(set(row.checks) != names for row in self.rows):
            raise ValueError(
                "verdict needs seven distinct rows and all four checks"
            )
        if any(f.row not in ids for f in self.failures):
            raise ValueError("failure names an unknown row")
        for row in self.rows:
            for name, passed in row.checks.items():
                if passed == any(
                    f.row == row.row and f.check == name for f in self.failures
                ):
                    raise ValueError(
                        "row checks differ from their failure reasons"
                    )
        if (self.verdict == "launch") != (not self.failures):
            raise ValueError("launch verdict differs from its failures")
        return self


def select(
    packet: Packet, need: RecordNeed, day: float
) -> ManifestEntry | None:
    candidates = [r for r in packet.records if r.claims.kind == need.kind]
    if need.period is not None:
        candidates = [r for r in candidates if r.claims.period == need.period]
    if not candidates:
        return None
    newest = max(r.claims.day for r in candidates)
    latest = [r for r in candidates if r.claims.day == newest]
    if len(latest) != 1:
        raise ValueError(
            "ambiguous latest record; no tie-breaking by filename"
        )
    if newest > day:
        raise ValueError("latest record was made after the gate day")
    return latest[0]


def complete(packet, need, record, registry) -> tuple[str, ...]:
    if record is None:
        return ("missing record",)
    try:
        raw = json.loads(within_run(packet.run_dir, record.path).read_text())
        if fingerprint(raw) != record.sha256:
            return ("file differs from manifest fingerprint",)
        actual = EvidenceFile.model_validate(raw)
        if actual.claims != record.claims:
            return ("indexed claims differ from file",)
        if registry is None:
            return ("producing record parser unavailable",)
        result = registry.validate_record(
            need.kind, actual.payload, actual.claims
        )
        if result is not None:
            raise ValueError(
                "producer parser must raise on failure, not return a verdict"
            )
        if registry.found_false(actual.claims.record_id):
            return ("registry marked counted record false",)
    except (ValueError, TypeError, KeyError, OSError) as error:
        return (f"record unreadable: {error}",)
    return (
        ()
        if record.claims.verdict == need.required_verdict
        else (
            f"requires {need.required_verdict}; record says {record.claims.verdict}",
        )
    )


def consistent(packet, req, need, record, registry) -> tuple[str, ...]:
    if record is None:
        return ()
    claim = record.claims
    reasons = []
    if need.bundle_rule == "strict" and claim.bundle != req.launched_bundle:
        reasons.append("strict record names an earlier or absent bundle")
    if need.bundle_rule == "covered" and claim.bundle != req.launched_bundle:
        if registry is None or claim.bundle is None:
            reasons.append("coverage proof unavailable")
        else:
            try:
                if not registry.covered(
                    claim.bundle, req.launched_bundle, need.kind
                ):
                    reasons.append("promotion diff requires this record again")
            except (ValueError, KeyError) as error:
                reasons.append(f"coverage refused: {error}")
    if need.kind == "rehearsal":
        try:
            if registry is None:
                raise ValueError("promotion registry unavailable")
            if claim.rollback_target != registry.rollback_target(
                req.launched_bundle
            ):
                reasons.append(
                    "rehearsal target differs from launched promotion target"
                )
        except (ValueError, KeyError) as error:
            reasons.append(str(error))
    # Policy sets and cohort identity are global, even for different rollout shares.
    policies = {
        r.claims.policy_versions
        for r in packet.records
        if r.claims.policy_versions
    }
    cohorts = {
        r.claims.cohort.fingerprint
        for r in packet.records
        if r.claims.cohort is not None
    }
    if len(policies) > 1:
        reasons.append("multiple policy-version sets")
    if len(cohorts) > 1:
        reasons.append("multiple cohort hash/canary rules")
    return tuple(reasons)


def signed_in_order(req, need, record, owners) -> tuple[str, ...]:
    reasons = []
    if record is not None:
        claim = record.claims
        if claim.signer_role != need.signer_role or not holds_role(
            owners["roles"], need.signer_role, claim.signed_by
        ):
            reasons.append("record signer does not hold required role")
        if not claim.signed_day < claim.run_started_day <= claim.day:
            reasons.append("governing signature must precede run")
        if claim.day > req.gate_day:
            reasons.append("record made after required gate day")
    layers = owners.get("layers", {})
    if set(layers) != {
        "interface",
        "context",
        "reasoning_and_orchestration",
        "model",
        "data",
        "evaluation_and_operations",
        "control_plane",
    }:
        reasons.append("layer map missing one or more owners")
    elif any(not owner for owner in layers.values()):
        reasons.append("layer-map row has no owner")
    return tuple(reasons)


def reversible(
    req, row, record, registry, logs, owners, day
) -> tuple[str, ...]:
    reasons = []
    for condition in row.reversals:
        if any(
            not owners["roles"].get(role) for role in condition.owner_roles
        ):
            reasons.append(
                f"{condition.condition_id}: reversal owner unavailable"
            )
        if registry is None or condition.mechanism not in registry.mechanisms(
            req.launched_bundle
        ):
            reasons.append(
                f"{condition.condition_id}: mechanism absent from bundle/registry"
            )
        since = req.signed_day if record is None else record.claims.signed_day
        for event in logs:
            if event.day <= since or event.day > day or event.planned_time_box:
                continue
            if event.condition_id == condition.condition_id:
                reasons.append(
                    f"{condition.condition_id} fired after signature: {event.record_id}"
                )
    return tuple(dict.fromkeys(reasons))


def check_row(packet, req, row, registry, logs, owners, day):
    failures, counted = [], []
    for need in row.records:
        try:
            record = select(packet, need, day)
        except ValueError as error:
            failures.append(
                GateFailure(
                    row=row.row_id,
                    kind=need.kind,
                    check="complete",
                    reason=str(error),
                )
            )
            continue
        reasons: dict[CheckName, tuple[str, ...]] = {
            "complete": complete(packet, need, record, registry),
            "consistent": consistent(packet, req, need, record, registry),
            "signed_in_order": signed_in_order(req, need, record, owners),
            "reversible": reversible(
                req, row, record, registry, logs, owners, day
            ),
        }
        failures.extend(
            GateFailure(
                row=row.row_id, kind=need.kind, check=name, reason=reason
            )
            for name, items in reasons.items()
            for reason in items
        )
        if record is not None and not any(reasons.values()):
            counted.append(record.claims.record_id)
    result = RowVerdict(
        row=row.row_id,
        counted=tuple(counted),
        checks={
            name: not any(f.check == name for f in failures)
            for name in (
                "complete",
                "consistent",
                "signed_in_order",
                "reversible",
            )
        },
    )
    return result, failures


def checked_context(packet, requirement, logs, owners, registry, day):
    from refund_agent.launch.capstone_blueprint_requirement import validate
    from refund_agent.launch.capstone_blueprint_scan import gate_events

    req = validate(requirement, owners["roles"])
    if (
        day != req.gate_day
        or packet.requirement_fingerprint != req.fingerprint
    ):
        raise ValueError("packet, requirement and gate clock differ")
    return req, gate_events(logs, registry, 0)


# region book:fnd.capstone-blueprint.check-01
def check(packet, requirement, registry, logs, owners, day):
    from refund_agent.launch import capstone_blueprint_gate as gate

    req, events = gate.checked_context(
        packet, requirement, logs, owners, registry, day
    )
    rows, failures = [], []
    for row in req.rows:
        result, reasons = gate.check_row(
            packet, req, row, registry, events, owners, day
        )
        rows.append(result)
        failures.extend(reasons)
    return gate.LaunchVerdict(
        verdict="hold" if failures else "launch",
        launched_bundle=req.launched_bundle,
        packet_fingerprint=packet.fingerprint,
        requirement_fingerprint=req.fingerprint,
        day=day,
        rows=tuple(rows),
        failures=tuple(failures),
    )


# endregion book:fnd.capstone-blueprint.check-01
