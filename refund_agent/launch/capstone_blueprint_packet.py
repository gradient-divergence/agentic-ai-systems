"""Canonical evidence files and a manifest; no search outside the named run."""

import json
from pathlib import Path
from typing import Literal

from pydantic import JsonValue

from refund_agent.launch.capstone_blueprint_requirement import (
    Day,
    LaunchRecord,
    LaunchRequirement,
    Period,
    Role,
    Text,
)
from refund_agent.selection import fingerprint

INDEX = "capstone-blueprint-records.jsonl"


class CohortRule(LaunchRecord):
    canary_id: Text
    hash_rule: Literal[
        "SHA-256(canary_id:requester_id), first 8 bytes, modulo 10000"
    ]


class EvidenceClaims(LaunchRecord):
    record_id: Text
    kind: Text
    day: Day
    bundle: Text | None
    signed_by: Text
    signer_role: Role
    signed_day: Day
    run_started_day: Day
    verdict: Text
    period: Period | None = None
    policy_versions: tuple[Text, ...] = ()
    cohort: CohortRule | None = None
    rollback_target: Text | None = None
    label_dependent: bool = False
    last_payment_day: Day | None = None


class EvidenceFile(LaunchRecord):
    claims: EvidenceClaims
    payload: dict[str, JsonValue]


class ManifestEntry(LaunchRecord):
    path: Text
    sha256: Text
    claims: EvidenceClaims


class Packet(LaunchRecord):
    run_dir: Path
    requirement_fingerprint: Text
    records: tuple[ManifestEntry, ...]

    @property
    def fingerprint(self) -> str:
        # Location is not evidence identity. Relocating a run keeps its verdict.
        return fingerprint(
            {
                "requirement": self.requirement_fingerprint,
                "records": [r.model_dump(mode="json") for r in self.records],
            }
        )


class MissingRecord(LaunchRecord):
    kind: Text
    reason: Text
    integration_point: Text


def within_run(run_dir: Path, relative: str) -> Path:
    path = (run_dir / relative).resolve()
    if (
        not path.is_relative_to(run_dir.resolve())
        or Path(relative).is_absolute()
    ):
        raise ValueError("record path leaves the run directory")
    return path


def write_record(run_dir: Path, record: EvidenceFile) -> Path:
    record = EvidenceFile.model_validate(record.model_dump(mode="json"))
    # A digest in the filename avoids treating an external record id as a path.
    relative = (
        "records/capstone-blueprint-"
        + fingerprint(record.claims.record_id)
        + ".json"
    )
    path = within_run(run_dir, relative)
    text = (
        json.dumps(
            record.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text() != text:
            raise ValueError(
                "record identity already exists; append a new record"
            )
        return path
    path.write_text(text)
    with (run_dir / INDEX).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(relative) + "\n")
    return path


def assemble(
    run_dir: Path, registry, requirement: LaunchRequirement
) -> Packet:
    """Keep every result, including stale ones; the gate selects the latest."""
    run_dir = Path(run_dir).resolve()
    entries = []
    index = run_dir / INDEX
    paths = (
        [json.loads(line) for line in index.read_text().splitlines()]
        if index.exists()
        else []
    )
    if len(paths) != len(set(paths)):
        raise ValueError("duplicate path in record index")
    identities = set()
    for relative in paths:
        raw = json.loads(within_run(run_dir, relative).read_text())
        record = EvidenceFile.model_validate(raw)
        if record.claims.record_id in identities:
            raise ValueError("duplicate record identity")
        identities.add(record.claims.record_id)
        entries.append(
            ManifestEntry(
                path=relative, sha256=fingerprint(raw), claims=record.claims
            )
        )
    return Packet(
        run_dir=run_dir,
        requirement_fingerprint=requirement.fingerprint,
        records=tuple(entries),
    )


def missing(
    run_dir: Path, kind: str, reason: str, integration_point: str
) -> None:
    row = MissingRecord(
        kind=kind, reason=reason, integration_point=integration_point
    )
    target = run_dir / "missing" / ("capstone-blueprint-" + kind + ".json")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(row.model_dump_json(indent=2) + "\n")
