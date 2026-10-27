"""Chapter 35: authority is checked before a wrapped lookup can disclose."""

import json
from datetime import timedelta

import pytest

from refund_agent import contracts as c
from refund_agent.agent import Agent
from refund_agent.data.generate import generate
from refund_agent.security.security_privacy_accounts import AccountRecords, ScopeResolver
from refund_agent.security.security_privacy_attacks import approval_for, forged_reviewer, resend
from refund_agent.security.security_privacy_credentials import (
    credential_problem,
)
from refund_agent.security.security_privacy_experiment import (
    FIXTURE_KEY,
    attack_runs,
    fresh_tools,
    positive_runs,
    public_runs,
)
from refund_agent.security.security_privacy_hashing import export_records
from refund_agent.security.security_privacy_scan import scan


def test_resent_request_never_fetches_foreign_order():
    data = trusted = generate(7)
    honest = next(r for r in data.requests if r.request_id == "r-7-0007")
    request = honest.model_copy(
        update={
            "request_id": "security-privacy-resent",
            "requester_id": "customer-7-0012",
        }
    )
    snapshot = next(s for s in data.retrievals if s.request_id == honest.request_id)
    data = data.model_copy(
        update={
            "requests": [request],
            "retrievals": [snapshot.model_copy(update={"request_id": request.request_id})],
        }
    )
    tools, _, _ = fresh_tools(data, request, trusted_data=trusted)
    trace = Agent(tools).run(request)
    assert trace.decision.reason == "order_not_found"
    assert not tools.payments
    assert all(step.state.order is None for step in trace.steps)


def case():
    data = generate(7, orders=40, requests=40)
    request = next(r for r in data.requests if r.request_id == "r-7-0007")
    tools, issuer, security = fresh_tools(data, request)
    return data, request, tools, issuer, security


def test_accounts_do_not_learn_from_an_attack():
    data, request, _, _, _ = case()
    before = data.model_dump_json()
    accounts = AccountRecords.from_dataset(data)
    attacked = resend(data)
    assert accounts.orders_of("customer-7-0012") == ("o-0012",)
    assert not accounts.may_act(attacked.request.requester_id, request.order_reference)
    assert data.model_dump_json() == before
    assert not any(subject == "refund-agent" for subject, _ in accounts.grants)


@pytest.mark.parametrize(
    "allowed,expected",
    [
        ((), (None, "order_not_found")),
        (("o-0012",), (None, "order_not_found")),
        (("o-0000",), ("o-0000", None)),
        (("o-0000", "o-0001"), (None, "ambiguous_order")),
    ],
)
def test_reference_multiplicity_is_only_within_authorized_orders(allowed, expected):
    data, _, _, _, _ = case()
    assert ScopeResolver.from_dataset(data).resolve("ambiguous", allowed) == expected


@pytest.mark.parametrize("revoked", [None, "shipment.status", "policy.retrieve"])
def test_execution_authorizes_and_logs_every_reread(revoked):
    _, request, tools, issuer, security = case()
    state = c.State(request=request, trace_id="security-privacy-rereads")
    reads = ("order.lookup", "shipment.status", "policy.retrieve")
    for name in reads:
        assert tools.call(c.ToolCall(name=name), state).ok
    assert tools.call(
        c.ToolCall(
            name="refund.propose",
            arguments=c.ToolArguments(amount_cents=request.amount_cents, reason=request.reason),
        ),
        state,
    ).ok
    if revoked:
        issuer.revoke_scope(revoked, request.at)
    invocations = []
    for name, method in zip(reads, ("_order", "_shipment", "_policy"), strict=True):
        original = getattr(tools, method)

        def record(call, current, *, original=original, name=name):
            invocations.append(name)
            return original(call, current)

        setattr(tools, method, record)
    before = len(security.log.entries)
    result = tools.call(
        c.ToolCall(
            name="refund.execute",
            arguments=c.ToolArguments(amount_cents=2499, idempotency_key="security-privacy-reread"),
        ),
        state,
    )
    attempted = reads if revoked is None else reads[: reads.index(revoked) + 1]
    assert invocations == list(attempted if revoked is None else attempted[:-1])
    logged = [entry for entry in security.log.entries[before:] if entry.tool in reads]
    assert [entry.tool for entry in logged] == list(attempted)
    assert [entry.reason for entry in logged] == [
        "credential_revoked" if name == revoked else "" for name in attempted
    ]
    if revoked:
        assert not result.ok and result.reason == "credential_revoked"
        assert not tools.payments
        assert all(not audit.money_moved for audit in tools.audits.values())
    else:
        assert result.ok and sum(p.amount_cents for p in tools.payments) == 2499


@pytest.mark.parametrize(
    "kind,reason",
    [
        ("missing", "credential_missing"),
        ("seal", "credential_seal"),
        ("issuer", "credential_seal"),
        ("expired", "credential_expired"),
        ("future", "credential_expired"),
        ("revoked", "credential_revoked"),
        ("request", "credential_request"),
        ("subject", "credential_request"),
        ("tool", "credential_tool:refund.execute"),
        ("order", "order_not_found"),
        ("amount", "credential_amount"),
        ("operator", "credential_tool:refund.execute"),
    ],
)
def test_first_credential_failure(kind, reason):
    _, request, tools, issuer, security = case()
    credential, at = tools.security_credential, request.at
    call = c.ToolCall(name="refund.execute", arguments=c.ToolArguments(amount_cents=2499))
    state = c.State(request=request, trace_id="security-privacy-test")
    if kind == "missing":
        credential = None
    elif kind == "seal":
        credential = credential.model_copy(update={"ceiling_cents": 999999})
    elif kind == "issuer":
        credential = credential.model_copy(update={"issuer": "unknown"})
    elif kind == "expired":
        at = credential.expires_at
    elif kind == "future":
        at -= timedelta(seconds=1)
    elif kind == "revoked":
        issuer.revoke(credential.credential_id, at)
    elif kind in ("request", "subject"):
        state.request = request.model_copy(
            update={"request_id" if kind == "request" else "requester_id": "other"}
        )
    elif kind == "tool":
        credential = issuer._issue(
            "request",
            request.requester_id,
            at,
            request_id=request.request_id,
            ceiling_cents=2499,
            order_ids=(request.order_reference,),
        )
        # A valid seal over a narrower tool grant, issued by the trusted fixture issuer.
        import hmac
        from hashlib import sha256

        from refund_agent.security.security_privacy_credentials import canonical

        credential = credential.model_copy(update={"tools": ("order.lookup",)})
        credential = credential.model_copy(
            update={"seal": hmac.new(FIXTURE_KEY, canonical(credential), sha256).hexdigest()}
        )
    elif kind == "order":
        call = call.model_copy(
            update={"arguments": c.ToolArguments(amount_cents=2499, order_id="o-0012")}
        )
    elif kind == "amount":
        call = call.model_copy(update={"arguments": c.ToolArguments(amount_cents=4999)})
    elif kind == "operator":
        credential = issuer.issue_operator("morgan", at)
    assert (
        credential_problem(credential, call, state, at, security.verifier, security.revocations)
        == reason
    )


def test_expiry_and_reissuance():
    _, request, tools, issuer, security = case()
    state = c.State(request=request, trace_id="security-privacy-test")
    call = c.ToolCall(name="refund.execute", arguments=c.ToolArguments(amount_cents=2499))
    credential = tools.security_credential
    assert (
        credential_problem(
            credential,
            call,
            state,
            credential.expires_at - timedelta(seconds=1),
            security.verifier,
            security.revocations,
        )
        is None
    )
    assert (
        credential_problem(
            credential, call, state, credential.expires_at, security.verifier, security.revocations
        )
        == "credential_expired"
    )
    fresh = issuer.issue_request(request, security.accounts, credential.expires_at)
    assert fresh.credential_id != credential.credential_id
    assert (
        credential_problem(
            fresh, call, state, credential.expires_at, security.verifier, security.revocations
        )
        is None
    )
    issuer.revoke_scope("refund.execute", request.at)
    assert (
        credential_problem(
            credential, call, state, request.at, security.verifier, security.revocations
        )
        == "credential_revoked"
    )
    issuer.reinstate_scope("refund.execute", request.at + timedelta(seconds=1))
    assert (
        credential_problem(
            credential,
            call,
            state,
            request.at + timedelta(seconds=1),
            security.verifier,
            security.revocations,
        )
        == "credential_revoked"
    )
    restored_at = request.at + timedelta(seconds=2)
    fresh = issuer.issue_request(request, security.accounts, restored_at)
    assert (
        credential_problem(fresh, call, state, restored_at, security.verifier, security.revocations)
        is None
    )


def test_scope_reinstatement_never_revives_credentials_issued_while_revoked():
    _, request, tools, issuer, security = case()
    state = c.State(request=request, trace_id="security-privacy-test")
    call = c.ToolCall(name="refund.execute", arguments=c.ToolArguments(amount_cents=2499))
    issuer.revoke_scope(call.name, request.at)
    suspended = issuer.issue_request(request, security.accounts, request.at + timedelta(seconds=1))
    restored_at = request.at + timedelta(seconds=2)
    issuer.reinstate_scope(call.name, restored_at)
    for credential in (tools.security_credential, suspended):
        assert (
            credential_problem(
                credential, call, state, restored_at, security.verifier, security.revocations
            )
            == "credential_revoked"
        )
    fresh_at = restored_at + timedelta(seconds=1)
    fresh = issuer.issue_request(request, security.accounts, fresh_at)
    assert (
        credential_problem(fresh, call, state, fresh_at, security.verifier, security.revocations)
        is None
    )
    issuer.revoke(fresh.credential_id, fresh_at)
    issuer.reinstate_scope(call.name, fresh_at + timedelta(seconds=1))
    assert (
        credential_problem(fresh, call, state, fresh_at, security.verifier, security.revocations)
        == "credential_revoked"
    )


def test_only_operator_intake_changes_revocation_authority():
    _, request, tools, issuer, security = case()
    own = tools.security_credential
    result = issuer.control(own, "scope.revoke", "refund.execute", request.at)
    assert result.reason == "credential_tool:scope.revoke"
    assert not security.revocations.contains(own, "refund.execute", request.at)
    operator = issuer.issue_operator("morgan", request.at)
    assert issuer.control(operator, "scope.revoke", "refund.execute", request.at).ok
    assert security.revocations.contains(own, "refund.execute", request.at)
    assert issuer.control(operator, "scope.reinstate", "refund.execute", request.at).ok
    assert security.revocations.contains(own, "refund.execute", request.at)
    assert not issuer.control(operator, "refund.execute", "o-0007", request.at).ok


@pytest.mark.parametrize(
    "kind,reason",
    [
        ("missing", "credential_missing"),
        ("seal", "credential_seal"),
        ("non-ascii-seal", "credential_seal"),
        ("altered-role", "credential_seal"),
        ("issuer", "credential_seal"),
        ("request", "credential_reviewer"),
        ("operator", "credential_reviewer"),
        ("future", "credential_expired"),
        ("expired", "credential_expired"),
        ("revoked", "credential_revoked"),
        ("subject", "credential_reviewer"),
        ("role", "credential_reviewer"),
    ],
)
def test_approval_intake_rejects_and_stores_nothing(kind, reason):
    data, _, _, _, _ = case()
    attack = forged_reviewer(data)
    tools, issuer, security = fresh_tools(attack.data, attack.request, trusted_data=data)
    trace = Agent(tools).run(attack.request)
    approval = approval_for(tools, trace)
    reviewer = next(r for r in data.reviewers if r.reviewer_id == "alex")
    credential = issuer.issue_reviewer(reviewer, approval.approved_at)
    at = approval.approved_at
    if kind == "missing":
        credential = None
    elif kind == "seal":
        credential = credential.model_copy(update={"seal": "broken"})
    elif kind == "non-ascii-seal":
        credential = credential.model_copy(update={"seal": "\u00e9" * 64})
    elif kind == "altered-role":
        credential = credential.model_copy(update={"role": "named_approver"})
    elif kind == "issuer":
        credential = credential.model_copy(update={"issuer": "unknown"})
    elif kind == "request":
        credential = tools.security_credential
    elif kind == "operator":
        credential = issuer.issue_operator("morgan", at)
    elif kind == "future":
        at -= timedelta(seconds=1)
    elif kind == "expired":
        at = credential.expires_at
    elif kind == "revoked":
        issuer.revoke(credential.credential_id, at)
    elif kind in ("subject", "role"):
        reviewer = reviewer.model_copy(
            update={
                "reviewer_id" if kind == "subject" else "role": "casey"
                if kind == "subject"
                else "named_approver"
            }
        )
        credential = issuer.issue_reviewer(reviewer, at)
    result = tools.record_approval(approval, credential=credential, at=at)
    assert result.reason == reason
    assert not result.ok and not tools._read("approvals", tools._approvals) and not tools.payments
    assert security.log.entries[-1].reason == reason
    if kind == "missing":
        assert security.log.entries[-1].credential_id is None


@pytest.mark.parametrize("seal", ["\u00e9" * 64, "g" * 64, "a" * 63, ""])
def test_request_intake_returns_structured_refusal_for_malformed_seals(seal):
    _, request, tools, _, security = case()
    tools.security_credential = tools.security_credential.model_copy(update={"seal": seal})
    trace = Agent(tools).run(request)
    assert trace.decision.reason == "credential_seal"
    assert trace.decision.outcome == c.Outcome.REFER
    assert not tools.payments
    assert all(step.state.order is None for step in trace.steps)
    assert security.log.entries[-1].reason == "credential_seal"


def test_authorized_refund_returns_to_original_payment():
    _, request, tools, _, _ = case()
    trace = Agent(tools).run(request)
    assert trace.receipt.amount_cents == 2499
    assert trace.receipt.evidence.order.payment_id == trace.steps[0].state.order.payment_id
    assert trace.receipt.order_id == request.order_reference
    assert trace.receipt.evidence.order.order_id in tools.security_credential.order_ids


def test_same_request_identifier_cannot_disclose_another_senders_receipt():
    data, request, tools, issuer, security = case()
    paid = Agent(tools).run(request)
    assert paid.receipt is not None
    copied = request.model_copy(update={"requester_id": "customer-7-0012"})
    tools.security_credential = issuer.issue_request(copied, security.accounts, copied.at)
    refused = Agent(tools).run(copied)
    assert refused.receipt is None and refused.decision.reason == "order_not_found"
    assert all(step.state.order is None and step.state.receipt is None for step in refused.steps)
    assert len(tools.payments) == 1


def test_attack_effects_and_positive_public_controls():
    without, _, _ = attack_runs(scoped=False)
    with_scope, outputs, seals = attack_runs()
    assert sum(row.forbidden for row in without) == 2
    assert not any(row.forbidden for row in with_scope)
    assert scan(outputs, (FIXTURE_KEY, *seals)).matches == ()
    assert positive_runs() == positive_runs(scoped=False)
    assert sum(cents for _, cents in positive_runs()) == 62598
    public, bare = public_runs(), public_runs(scoped=False)
    assert public[1:] == bare[1:]
    assert public[0][0].request_id == "r-7-0000" and public[0][0].executed
    assert public[0][1] == 2499 and bare[0][0].reason == "ambiguous_order"
    assert sum(d.outcome == c.Outcome.DENY for d, _ in public_runs()) == 8


def test_revocation_refers_all_five_payments_and_preserves_denials():
    normal, revoked = public_runs(), public_runs(revoke_payment=True)
    for (decision, paid), (held, cents) in zip(normal, revoked, strict=True):
        assert cents == 0
        if paid:
            assert held.reason == "credential_revoked" and held.outcome == c.Outcome.REFER
        elif decision.outcome == c.Outcome.DENY:
            assert held == decision
    assert sum(paid > 0 for _, paid in normal) == 5


def test_export_breaks_all_identifier_joins_including_nested_reason():
    raw = {
        "requester_id": "customer-7-0012",
        "customer_id": "customer-7-0012",
        "request_id": "r-7-0012",
        "order_id": "o-0012",
        "trace_id": "trace-r-7-0012",
        "run_id": "run-r-7-0012-1",
        "reason": "refused r-7-0012",
        "decision": {"request_id": "r-7-0012", "outcome": "refer", "reason": "order_not_found"},
    }
    exported = export_records([raw], FIXTURE_KEY)
    text = json.dumps(exported)
    assert all(
        identifier not in text for identifier in ("customer-7-0012", "r-7-0012", "o-0012", "0012")
    )
    assert exported[0]["request_id"] == exported[0]["decision"]["request_id"]
    assert exported[0]["request_id"] != exported[0]["requester_id"]
    assert export_records([raw], b"x" * 32) != exported
    with pytest.raises(ValueError, match="unapproved"):
        export_records([{**raw, "text": "customer-7-0012"}], FIXTURE_KEY)
