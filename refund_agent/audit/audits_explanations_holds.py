"""Append-only audit holds. A nondetection never releases a hold.

The declared owner controls this store. Hashes detect accidental alteration, not
forgery by someone who can rewrite the store and all its hashes.
"""

import fcntl
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Literal

from refund_agent.audit.audits_explanations_rates import AuditRecord
from refund_agent.contracts import Record, _utc_time
from refund_agent.release.rule import owner_ids
from refund_agent.selection import fingerprint


class HoldEvent(Record):
    sequence: int
    previous: str
    at: datetime
    segment: str
    reason: str
    action: Literal["open", "release", "audited"]
    audit: AuditRecord
    audit_fingerprint: str

    @property
    def fingerprint(self) -> str:
        return fingerprint(self.model_dump(mode="json"))


def _read(stream) -> tuple[HoldEvent, ...]:
    rows = tuple(HoldEvent.model_validate_json(line) for line in stream)
    previous = ""
    for i, row in enumerate(rows, 1):
        if (
            row.sequence != i
            or row.previous != previous
            or row.audit_fingerprint != row.audit.fingerprint
        ):
            raise ValueError("audit hold journal integrity differs")
        segment = next(
            (s for s in row.audit.segments if s.segment == row.segment), None
        )
        if row.at != row.audit.at:
            raise ValueError("hold transition has the wrong audit time")
        if row.action == "audited":
            if row.segment or row.reason:
                raise ValueError("audit receipt cannot open or release a hold")
        else:
            if segment is None:
                raise ValueError("hold transition lacks its segment")
            if (
                row.action == "open"
                and (
                    not segment.breach
                    or row.reason not in segment.held_reasons
                )
                or row.action == "release"
                and not segment.release_allowed
            ):
                raise ValueError(
                    "hold transition lacks its required audit evidence"
                )
        if i > 1 and (
            row.at < rows[i - 2].at
            or row.audit.period.end < rows[i - 2].audit.period.end
        ):
            raise ValueError("hold audit records are out of order")
        previous = row.fingerprint
    return rows


def _active(rows, at):
    active = set()
    for row in rows:
        if row.action == "audited":
            continue
        if row.at <= at:
            key = (row.segment, row.reason)
            if row.action == "open":
                active.add(key)
            else:
                active.discard(key)
    return active


class AuditHolds:
    def __init__(self, path: Path, owner: str = "morgan"):
        if owner not in owner_ids():
            raise ValueError(
                "audit hold owner must be listed by the release contract"
            )
        self.path, self.owner = Path(path).resolve(), owner

    def events(self) -> tuple[HoldEvent, ...]:
        if not self.path.exists():
            return ()
        with self.path.open(encoding="utf-8") as stream:
            fcntl.flock(stream.fileno(), fcntl.LOCK_SH)
            return _read(stream)

    def is_held(self, segment: str, reason: str, at: datetime) -> bool:
        return (segment, reason) in _active(self.events(), _utc_time(at))

    def apply(self, audit: AuditRecord) -> tuple[HoldEvent, ...]:
        audit = AuditRecord.model_validate(audit)
        if audit.rule.signed_by != self.owner:
            raise ValueError("audit signer is not this store's decision owner")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as writer:
            fcntl.flock(writer.fileno(), fcntl.LOCK_EX)
            with self.path.open(encoding="utf-8") as reader:
                rows = _read(reader)
            if any(r.audit_fingerprint == audit.fingerprint for r in rows):
                return ()
            if rows and (
                audit.at < rows[-1].at
                or audit.period.end < rows[-1].audit.period.end
            ):
                raise ValueError("hold audit records are out of order")
            active = _active(rows, audit.at)
            transitions = []
            for segment in audit.segments:
                if segment.breach:
                    transitions.extend(
                        (segment.segment, reason, "open")
                        for reason in segment.held_reasons
                        if (segment.segment, reason) not in active
                    )
                elif segment.release_allowed:
                    transitions.extend(
                        (g, reason, "release")
                        for g, reason in sorted(active)
                        if g == segment.segment
                    )
            # Record nondetections too, so restart cannot accept an older audit.
            transitions.append(("", "", "audited"))
            appended = []
            for g, reason, action in transitions:
                event = HoldEvent(
                    sequence=len(rows) + len(appended) + 1,
                    previous=appended[-1].fingerprint
                    if appended
                    else rows[-1].fingerprint
                    if rows
                    else "",
                    at=audit.at,
                    segment=g,
                    reason=reason,
                    action=action,
                    audit=audit,
                    audit_fingerprint=audit.fingerprint,
                )
                writer.write(
                    json.dumps(event.model_dump(mode="json"), sort_keys=True)
                    + "\n"
                )
                appended.append(event)
            writer.flush()
            os.fsync(writer.fileno())
            return tuple(e for e in appended if e.action != "audited")
