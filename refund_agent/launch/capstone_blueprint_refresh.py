"""A new bundle/target may earn new evidence; a failed verdict cannot be rerun."""

from collections.abc import Callable, Mapping
from pathlib import Path

from refund_agent.launch.capstone_blueprint_gate import (
    LaunchVerdict,
    RegistryPort,
    select,
)
from refund_agent.launch.capstone_blueprint_packet import (
    EvidenceFile,
    Packet,
    write_record,
)
from refund_agent.launch.capstone_blueprint_requirement import (
    LaunchRecord,
    LaunchRequirement,
)


class RefreshResult(LaunchRecord):
    refreshed: tuple[str, ...]
    retained: tuple[str, ...]
    unavailable: tuple[str, ...]


def refresh(
    verdict: LaunchVerdict,
    requirement: LaunchRequirement,
    run_dir: Path,
    *,
    packet: Packet,
    registry: RegistryPort,
    producers: Mapping[str, Callable[..., EvidenceFile]],
) -> RefreshResult:
    if (
        verdict.requirement_fingerprint != requirement.fingerprint
        or verdict.packet_fingerprint != packet.fingerprint
    ):
        raise ValueError(
            "refresh must use the judged packet and signed requirement"
        )
    grouped = {}
    for failure in verdict.failures:
        grouped.setdefault((failure.row, failure.kind), set()).add(
            failure.check
        )
    refreshed, retained, unavailable, done = [], [], [], set()
    for row in requirement.rows:
        for need in row.records:
            if (
                grouped.get((row.row_id, need.kind)) != {"consistent"}
                or not need.rerunnable
            ):
                continue
            if need.kind == "comparison" or need.kind in done:
                continue
            done.add(need.kind)
            old = select(packet, need, verdict.day)
            if old is None:
                raise ValueError(
                    "consistency refresh requires an existing result"
                )
            target = (
                registry.rollback_target(requirement.launched_bundle)
                if need.kind == "rehearsal"
                else None
            )
            if any(
                r.claims.kind == need.kind
                and r.claims.bundle == requirement.launched_bundle
                and r.claims.rollback_target == target
                for r in packet.records
            ):
                unavailable.append(
                    need.kind + ": this candidate/target already has a result"
                )
                continue
            if need.kind not in producers:
                unavailable.append(
                    need.kind + ": producing chapter function unavailable"
                )
                continue
            new = producers[need.kind](
                bundle=requirement.launched_bundle,
                rollback_target=target,
                requirement=requirement,
                previous=old.claims,
            )
            new = EvidenceFile.model_validate(new.model_dump(mode="json"))
            if (
                new.claims.kind != need.kind
                or new.claims.bundle != requirement.launched_bundle
                or new.claims.record_id == old.claims.record_id
                or new.claims.day < old.claims.day
                or need.kind == "rehearsal"
                and new.claims.rollback_target != target
            ):
                raise ValueError(
                    "refresh producer returned another kind, candidate, target or identity"
                )
            registry.validate_record(need.kind, new.payload, new.claims)
            write_record(run_dir, new)
            retained.append(old.claims.record_id)
            refreshed.append(new.claims.record_id)
    return RefreshResult(
        refreshed=tuple(refreshed),
        retained=tuple(retained),
        unavailable=tuple(unavailable),
    )
