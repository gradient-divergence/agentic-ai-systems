"""The starter fails expiry and onward delegation, for its designed reasons."""

import os
import runpy
from datetime import timedelta
from pathlib import Path

import pytest

from refund_agent.agent import Agent
from refund_agent.data.generate import generate
from refund_agent.security.security_privacy_accounts import AccountRecords, ScopeResolver
from refund_agent.security.security_privacy_credentials import Issuer
from refund_agent.security.security_privacy_experiment import FIXTURE_KEY
from refund_agent.security.security_privacy_log import CredentialLog
from refund_agent.security.security_privacy_tools import ScopedTools, SecurityRuntime

SOLUTION=Path(os.environ.get("SECURITY_PRIVACY_SOLUTION",Path(__file__).with_name("security_privacy_delegate_06.py")))
MODULE=runpy.run_path(str(SOLUTION))


def fixture():
    data = generate(7, orders=40, requests=40)
    request = next(r for r in data.requests if r.request_id == "r-7-0007")
    book = MODULE["Delegations"](AccountRecords.from_dataset(data))
    at, until = request.at, request.at + timedelta(minutes=5)
    grant = book.grant(request.requester_id, "customer-7-0012", request.order_reference, at, until)
    return data, request, book, grant


def run_as(data, original, book, subject, at):
    request = original.model_copy(update={"requester_id": subject})
    data = data.model_copy(update={"requests": [request]})
    issuer = Issuer(FIXTURE_KEY)
    accounts = book.accounts_at(at)
    security = SecurityRuntime(
        accounts,
        ScopeResolver.from_dataset(data),
        issuer.verifier,
        issuer.revocations,
        CredentialLog(),
    )
    tools = ScopedTools(
        data,
        security_privacy=security,
        security_credential=issuer.issue_request(request, accounts, at),
    )
    tools.security_clock = at
    return Agent(tools).run(request, at=at)


def test_within_period_pays_to_owners_order():
    data, request, book, grant = fixture()
    trace = run_as(data, request, book, grant.subject, grant.granted_at + timedelta(seconds=1))
    assert trace.receipt.amount_cents == 2499 and trace.receipt.order_id == grant.order_id


def test_after_period_is_indistinguishable_from_missing_order():
    data, request, book, grant = fixture()
    trace = run_as(data, request, book, grant.subject, grant.expires_at)
    assert trace.decision.reason == "order_not_found", "The starter never expires a grant"
    assert trace.receipt is None


def test_authorized_subject_cannot_pass_right_on():
    _, _, book, grant = fixture()
    with pytest.raises(ValueError, match="owner"):
        book.grant(
            grant.subject, "third-account", grant.order_id, grant.granted_at, grant.expires_at
        )


def test_grant_records_owner_and_time_and_preserves_owner():
    data, request, book, grant = fixture()
    assert grant.grantor == request.requester_id and grant.granted_at == request.at
    own = run_as(data, request, book, request.requester_id, grant.expires_at)
    assert own.decision.executed and own.receipt.amount_cents == 2499
