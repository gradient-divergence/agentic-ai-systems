"""Offline, explicitly authenticated fixtures; no implicit production issuer."""

from dataclasses import dataclass
from datetime import timedelta
from hashlib import sha256

from refund_agent.agent import Agent
from refund_agent.batch_stream.generate import read_sources, snapshot_log
from refund_agent.batch_stream.generate import scan as carrier_scan
from refund_agent.batch_stream.log import EventLog
from refund_agent.batch_stream.tools import EventTimeTools
from refund_agent.batch_stream.view import first_complete
from refund_agent.data.generate import generate
from refund_agent.evaluation import POSITIVE_CASES, _positive_data
from refund_agent.security.security_privacy_accounts import AccountRecords, ScopeResolver
from refund_agent.security.security_privacy_attacks import (
    approval_for,
    forged_reviewer,
    hidden,
    paraphrase,
    policy_text,
    resend,
    secret_probe,
)
from refund_agent.security.security_privacy_credentials import CREDENTIAL_LIFETIME, Issuer
from refund_agent.security.security_privacy_log import CredentialLog
from refund_agent.security.security_privacy_model import ObedientModel
from refund_agent.security.security_privacy_oracle import effects
from refund_agent.security.security_privacy_scan import observations, scan
from refund_agent.security.security_privacy_tools import ScopedTools, SecurityRuntime
from refund_agent.tools import Tools

# Public labels illustrate calculation and protect nothing against this reader.
FIXTURE_KEY = sha256(b"security-privacy-public-fixture-issuer").digest()
HASH_KEY = sha256(b"security-privacy-public-fixture-export").digest()


def runtime(data, *, accounts=None, resolver=None, issuer=None):
    issuer = Issuer(FIXTURE_KEY) if issuer is None else issuer
    return issuer, SecurityRuntime(
        AccountRecords.from_dataset(data) if accounts is None else accounts,
        ScopeResolver.from_dataset(data) if resolver is None else resolver,
        issuer.verifier,
        issuer.revocations,
        CredentialLog(),
    )


def fresh_tools(data, request, *, tools_type=ScopedTools, trusted_data=None, event_time=False):
    trusted_data = data if trusted_data is None else trusted_data
    issuer, security = runtime(trusted_data)
    base = (
        type("SecurityPrivacyEventTools", (tools_type, EventTimeTools), {})
        if event_time
        else tools_type
    )
    kwargs = {"event_log": fixture_log(data)} if event_time else {}
    tools = base(data, security_privacy=security, **kwargs)
    issue_run(tools, issuer, request, request.at)
    return tools, issuer, security


def issue_run(tools, issuer, request, at):
    tools.security_credential = issuer.issue_request(request, tools.security_privacy.accounts, at)
    tools.security_clock = at


def fixture_log(data):
    log = snapshot_log(data)
    first = max(r.at for r in data.requests)
    ticks = [
        carrier_scan(
            f"security-privacy-feed-{n}",
            "shipped",
            first + timedelta(minutes=n),
            first + timedelta(minutes=n),
        )
        for n in range(5, 38 * 60 + 1, 5)
    ]
    return EventLog(log.entries + tuple(ticks))


def terminal_run(tools, request, *, issuer=None, model=None, observation_sink=None):
    at = tools.run_time(request)
    if issuer:
        issue_run(tools, issuer, request, at)
    trace = Agent(tools, model).run(request, at=at)
    if observation_sink is not None:
        observation_sink.extend(
            observations(trace, tools, tools.security_privacy.log if issuer else CredentialLog())
        )
    if isinstance(tools, EventTimeTools) and trace.decision.reason == "shipment_incomplete":
        at = first_complete(tools.event_log, request.at + tools.allowed_lateness, tools.margin)
        tools.data = read_sources(tools.data, at)
        if issuer:
            issue_run(tools, issuer, request, at)
        trace = Agent(tools, model).run(request, at=at)
        if observation_sink is not None:
            observation_sink.extend(
                observations(
                    trace, tools, tools.security_privacy.log if issuer else CredentialLog()
                )
            )
    return trace


@dataclass(frozen=True)
class AttackResult:
    name: str
    channel: str
    outcome: str
    reason: str
    paid_cents: int
    forbidden: bool
    scan_outputs: int
    scan_matches: int
    intake_reason: str = ""
    stored_approvals: int = 0


def attack_runs(*, scoped=True, tools_type=ScopedTools):
    trusted = generate(7, orders=40, requests=40)
    account_records = AccountRecords.from_dataset(trusted)
    results, all_observations, seals = [], [], []
    fixtures = [
        resend(trusted),
        hidden(trusted),
        paraphrase(trusted),
        policy_text(trusted),
        forged_reviewer(trusted),
        secret_probe(trusted),
    ]
    honest = next(r for r in trusted.requests if r.request_id == "r-7-0007")
    for name in ("replay", "expired", "revoked"):
        from refund_agent.security.security_privacy_attacks import Attack

        fixtures.append(Attack(name, trusted, honest, "credential"))
    for attack in fixtures:
        issuer, security = runtime(trusted)
        if scoped:
            tools = tools_type(attack.data, security_privacy=security)
        else:
            bare_type = Tools if tools_type is ScopedTools else tools_type
            tools = bare_type(attack.data)
        issued = issuer.issue_request(attack.request, account_records, attack.request.at)
        presented = issued
        if attack.name == "replay":
            presented = issuer.issue_request(
                attack.request.model_copy(update={"request_id": "security-privacy-other"}),
                account_records,
                attack.request.at,
            )
        elif attack.name == "expired":
            presented = issuer.issue_request(
                attack.request, account_records, attack.request.at - CREDENTIAL_LIFETIME
            )
        elif attack.name == "revoked":
            issuer.revoke(issued.credential_id, attack.request.at)
        if scoped:
            assert isinstance(tools, ScopedTools)
            tools.security_credential = presented
            tools.security_clock = attack.request.at
        seals.append(bytes.fromhex(presented.seal))
        trace = Agent(tools, ObedientModel()).run(attack.request)
        authenticated = ()
        intake_reason = ""
        if attack.name == "reviewer":
            approval = approval_for(tools, trace)
            admitted = tools.record_approval(approval)
            intake_reason = admitted.reason
            if admitted.ok:
                trace = Agent(tools, ObedientModel()).run(attack.request)
        visible = observations(trace, tools, security.log)
        scanned = scan(visible, (FIXTURE_KEY, bytes.fromhex(presented.seal)))
        honest_payments = (
            ((honest.order_reference, honest.amount_cents),)
            if attack.name not in ("resent", "hidden", "reviewer")
            else ()
        )
        graded = effects(
            trace,
            tools,
            account_records,
            honest_payments=honest_payments,
            authenticated_approvals=authenticated,
            scan_result=scanned,
        )
        results.append(
            AttackResult(
                attack.name,
                attack.channel,
                trace.decision.outcome.value,
                trace.decision.reason,
                sum(p.amount_cents for p in tools.payments),
                graded.forbidden,
                scanned.outputs,
                len(scanned.matches),
                intake_reason,
                len(tools._read("approvals", tools._approvals)),
            )
        )
        all_observations.extend(visible)
    return tuple(results), tuple(all_observations), tuple(seals)


def positive_runs(*, scoped=True, observation_sink=None, secret_sink=None):
    data = _positive_data()
    results = []
    for request in data.requests:
        if scoped:
            tools, issuer, _ = fresh_tools(data, request, event_time=True)
        else:
            tools, issuer = EventTimeTools(data, event_log=fixture_log(data)), None
        trace = terminal_run(tools, request, issuer=issuer, observation_sink=observation_sink)
        _, role = POSITIVE_CASES[request.request_id]
        if role:
            approval = approval_for(tools, trace)
            context = {}
            if issuer:
                reviewer = next(r for r in data.reviewers if r.reviewer_id == approval.reviewer)
                context = {
                    "credential": issuer.issue_reviewer(reviewer, approval.approved_at),
                    "at": approval.approved_at,
                }
            if not tools.record_approval(approval, **context).ok:
                raise ValueError("security_privacy_positive_approval_refused")
            trace = terminal_run(tools, request, issuer=issuer, observation_sink=observation_sink)
        if issuer and secret_sink is not None:
            secret_sink.extend(issuer.issued_seals)
        results.append((trace.decision, sum(p.amount_cents for p in tools.payments)))
    return tuple(results)


def public_runs(*, scoped=True, revoke_payment=False, observation_sink=None, secret_sink=None):
    trusted = generate(7, orders=40, requests=40)
    results = []
    for request in trusted.requests:
        if scoped:
            tools, issuer, _ = fresh_tools(trusted, request, event_time=True)
            if revoke_payment:
                issuer.revoke_scope("refund.execute", request.at)
        else:
            tools, issuer = EventTimeTools(trusted, event_log=fixture_log(trusted)), None
        trace = terminal_run(tools, request, issuer=issuer, observation_sink=observation_sink)
        if issuer and secret_sink is not None:
            secret_sink.extend(issuer.issued_seals)
        results.append((trace.decision, sum(p.amount_cents for p in tools.payments)))
    return tuple(results)
