"""Append-only credential-call records; no seal, key or fetched payload."""

from datetime import datetime
from threading import RLock
from typing import Literal

from refund_agent import contracts as c


class CredentialEntry(c.Record):
    credential_id: str | None
    kind: str | None
    subject: str | None
    request_id: str
    trace_id: str
    run_id: str | None
    tool: str
    outcome: Literal["allowed", "refused"]
    reason: str
    at: datetime


class CredentialLog:
    def __init__(self) -> None:
        self._entries: tuple[CredentialEntry, ...] = ()
        self._lock = RLock()

    @property
    def entries(self) -> tuple[CredentialEntry, ...]:
        with self._lock:
            return c._detach(self._entries)

    def append(self, entry: CredentialEntry) -> None:
        with self._lock:
            self._entries += (c._validated(CredentialEntry, entry),)

    def trace(self, trace_id: str, run_id: str | None = None) -> tuple[CredentialEntry, ...]:
        return tuple(
            e
            for e in self.entries
            if e.trace_id == trace_id and (run_id is None or e.run_id == run_id)
        )
