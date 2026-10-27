"""HMAC issuance and temporal revocation; deployed keys stay outside the agent."""

from __future__ import annotations

import hmac
import json
from datetime import datetime, timedelta
from hashlib import sha256
from threading import RLock
from typing import Literal

from pydantic import ValidationError
from refund_agent import contracts as c
from refund_agent.security.security_privacy_accounts import AccountRecords
from refund_agent.security.security_privacy_scopes import (
    SCOPES,
    SCOPES_VERSION,
)

CREDENTIAL_LIFETIME = timedelta(minutes=30)  # Constructed, not a measured lifetime.


# fmt: off
# region book:fnd.security-privacy.credential-01
class Credential(c.Record):
    credential_id: c.Identity
    kind: Literal["request", "reviewer", "operator"]
    subject: c.Identity
    request_id: str | None = None
    order_ids: tuple[str, ...] = ()
    tools: tuple[str, ...]
    ceiling_cents: c.Cents | None = None
    role: str | None = None
    not_before: datetime
    expires_at: datetime
    issuer: c.Identity
    scopes_version: str = SCOPES_VERSION
    seal: str


def identity_problem(
    credential: Credential | None, at: datetime, verifier: Verifier
) -> str | None:
    if credential is None:  # (1)
        return "credential_missing"
    try:
        checked = c._validated(Credential, credential)
    except (ValidationError, TypeError, AttributeError):
        return "credential_seal"
    if (
        not verifier.verifies(checked)
        or checked.scopes_version != SCOPES_VERSION
    ):  # (2)
        return "credential_seal"
    if not checked.not_before <= c._utc_time(at) < checked.expires_at:  # (3)
        return "credential_expired"
    return None


def credential_problem(
    credential: Credential | None,
    call: c.ToolCall,
    state: c.State,
    at: datetime,
    verifier: Verifier,
    revocations: Revocations,
) -> str | None:
    problem = identity_problem(credential, at, verifier)
    if problem:
        return problem
    assert credential is not None
    if revocations.contains(credential, call.name, at):  # (4)
        return "credential_revoked"
    if (
        call.name not in credential.tools
        or call.name not in SCOPES[credential.kind]["tools"]
    ):  # (5)
        return f"credential_tool:{call.name}"
    if credential.kind != "request" or (
        credential.request_id != state.request.request_id
        or credential.subject != state.request.requester_id
    ):  # (6)
        return "credential_request"
    order_id = call.arguments.order_id or (
        state.order.order_id if state.order else None
    )
    if order_id is not None and order_id not in credential.order_ids:  # (7)
        return "order_not_found"
    if call.name in ("refund.propose", "refund.execute") and (
        credential.ceiling_cents is None
        or call.arguments.amount_cents is not None
        and call.arguments.amount_cents > credential.ceiling_cents
    ):  # (8)
        return "credential_amount"
    return None


# endregion book:fnd.security-privacy.credential-01
# fmt: on


def canonical(credential: Credential) -> bytes:
    return json.dumps(
        credential.model_dump(mode="json", exclude={"seal"}),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()


class Revocations:
    def __init__(self) -> None:
        self._credentials: dict[str, datetime] = {}
        self._scopes: list[tuple[datetime, str, bool]] = []
        self._lock = RLock()

    def revoke(self, credential_id: str, at: datetime) -> None:
        with self._lock:
            at = c._utc_time(at)
            self._credentials[credential_id] = min(self._credentials.get(credential_id, at), at)

    def scope(self, tool: str, at: datetime, revoked: bool) -> None:
        with self._lock:
            at = c._utc_time(at)
            previous = [t for t, name, _ in self._scopes if name == tool]
            if previous and at < max(previous):
                raise ValueError("scope revocations must be chronological")
            self._scopes.append((at, tool, revoked))

    def contains(self, credential: Credential, tool: str, at: datetime) -> bool:
        with self._lock:
            at = c._utc_time(at)
            revoked_at = self._credentials.get(credential.credential_id)
            if revoked_at is not None and revoked_at <= at:
                return True
            changes = [
                (time, revoked)
                for time, name, revoked in self._scopes
                if name == tool and time <= at
            ]
            issued = [revoked for time, revoked in changes if time <= credential.not_before]
            # Reinstatement restores the scope only for freshly issued grants.
            return bool(issued and issued[-1]) or any(
                revoked for time, revoked in changes if time >= credential.not_before
            )


class Verifier:
    def __init__(self, keys: dict[str, bytes]) -> None:
        if not keys or any(type(k) is not bytes or len(k) < 32 for k in keys.values()):
            raise ValueError("issuer verification requires keys of at least 32 bytes")
        self._keys = dict(keys)

    def verifies(self, credential: Credential) -> bool:
        if len(credential.seal) != 64 or any(
            character not in "0123456789abcdef" for character in credential.seal
        ):
            return False
        key = self._keys.get(credential.issuer)
        return key is not None and hmac.compare_digest(
            hmac.new(key, canonical(credential), sha256).hexdigest(),
            credential.seal,
        )


class Issuer:
    """Trusted intake; issue only after authenticating the supplied subject."""

    def __init__(self, key: bytes, *, name: str = "security-privacy-intake") -> None:
        if type(key) is not bytes or len(key) < 32:
            raise ValueError("issuer key must have at least 32 bytes")
        self._key, self.name = key, name
        self.revocations = Revocations()
        self.verifier = Verifier({name: key})
        self._issued: tuple[Credential, ...] = ()

    def _issue(self, kind: str, subject: str, at: datetime, **fields) -> Credential:
        at = c._utc_time(at)
        body = dict(
            kind=kind,
            subject=subject,
            not_before=at,
            expires_at=at + CREDENTIAL_LIFETIME,
            issuer=self.name,
            tools=SCOPES[kind]["tools"],
            **fields,
        )
        identity = sha256(json.dumps(body, sort_keys=True, default=str).encode()).hexdigest()[:24]
        record = Credential.model_validate(
            {
                "credential_id": f"security-privacy-{identity}",
                "seal": "",
                **body,
            }
        )
        issued = record.model_copy(
            update={"seal": hmac.new(self._key, canonical(record), sha256).hexdigest()}
        )
        self._issued += (issued,)
        return issued

    @property
    def issued_seals(self) -> tuple[bytes, ...]:
        """For the offline leakage oracle; never render these as telemetry."""
        return tuple(bytes.fromhex(record.seal) for record in self._issued)

    def issue_request(
        self, request: c.Request, accounts: AccountRecords, at: datetime
    ) -> Credential:
        request = c._validated(c.Request, request)
        return self._issue(
            "request",
            request.requester_id,
            at,
            request_id=request.request_id,
            order_ids=accounts.orders_of(request.requester_id),
            ceiling_cents=request.amount_cents,
        )

    def issue_reviewer(self, reviewer: c.Reviewer, at: datetime) -> Credential:
        reviewer = c._validated(c.Reviewer, reviewer)
        return self._issue("reviewer", reviewer.reviewer_id, at, role=reviewer.role)

    def issue_operator(self, subject: str, at: datetime) -> Credential:
        return self._issue("operator", subject, at)

    def revoke(self, credential_id: str, at: datetime) -> None:
        self.revocations.revoke(credential_id, at)

    def revoke_scope(self, tool: str, at: datetime) -> None:
        self.revocations.scope(tool, at, True)

    def reinstate_scope(self, tool: str, at: datetime) -> None:
        self.revocations.scope(tool, at, False)

    def control(
        self,
        credential: Credential | None,
        tool: str,
        target: str,
        at: datetime,
    ) -> c.ToolResult:
        """Authenticated operator intake; the issuer's direct methods are trusted."""
        at = c._utc_time(at)
        problem = identity_problem(credential, at, self.verifier)
        if problem:
            return c.ToolResult(ok=False, reason=problem)
        assert credential is not None
        if self.revocations.contains(credential, tool, at):
            return c.ToolResult(ok=False, reason="credential_revoked")
        if (
            credential.kind != "operator"
            or tool not in credential.tools
            or tool not in SCOPES["operator"]["tools"]
        ):
            return c.ToolResult(ok=False, reason=f"credential_tool:{tool}")
        if tool == "credential.revoke":
            self.revoke(target, at)
        elif tool == "scope.revoke":
            self.revoke_scope(target, at)
        else:
            self.reinstate_scope(target, at)
        return c.ToolResult(ok=True)
