"""Reviewer intake checks after the shared credential identity boundary."""

from datetime import datetime

from refund_agent import contracts as c
from refund_agent.security.security_privacy_credentials import (
    Credential,
    Revocations,
    Verifier,
    identity_problem,
)


def reviewer_problem(
    credential: Credential | None,
    approval: c.Approval,
    at: datetime,
    verifier: Verifier,
    revocations: Revocations,
) -> str | None:
    problem = identity_problem(credential, at, verifier)
    if problem:
        return problem
    assert credential is not None
    if revocations.contains(credential, "approval.intake", at):
        return "credential_revoked"
    if credential.kind != "reviewer":
        return "credential_reviewer"
    if "approval.intake" not in credential.tools:
        return "credential_tool:approval.intake"
    if credential.subject != approval.reviewer or credential.role != approval.role:
        return "credential_reviewer"
    return None
