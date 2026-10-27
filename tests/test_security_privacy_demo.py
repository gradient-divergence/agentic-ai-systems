"""Publication evidence is captured from the marked, runnable source regions."""

import json
import runpy
import sys
from pathlib import Path

import pytest

from demos.regions import capture_outputs, extract_snippets_from_text

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def local_outputs():
    return runpy.run_path(str(ROOT / "demos/security_privacy_demo.py"))[
        "create_outputs"
    ]()


def test_attack_publication_reports_entry_checks_approval_storage_and_secret_probe(local_outputs):
    output = local_outputs["attacks-01"]
    for name, channel, check in (
        ("resent", "order reference", "scope resolver"),
        ("hidden", "request text", "risk rule"),
        ("paraphrase", "request text", "credential amount"),
        ("policy", "policy text", "credential amount"),
        ("reviewer", "approval intake", "credential missing"),
        ("secret", "request text", "no secret in model context"),
        ("replay", "credential", "credential request"),
        ("expired", "credential", "credential validity times"),
        ("revoked", "credential", "credential revocation"),
        ("carrier", "carrier status", "ScanEvent and shipment contract"),
    ):
        assert f"{name} | {channel} | {check}" in output
    assert "Reviewer approvals stored: without 1; with 0" in output
    assert "Secret probe: eligible, paid 2499 cents; secret matches 0" in output
    assert "amount_mismatch, 0; with: credential_amount, 0" in output


def test_hash_publication_reports_search_size_and_recovery_position_without_digests(local_outputs):
    import re

    output = local_outputs["hash-01"]
    assert "Identifiers enumerated: 10000; plain recovery position: 13" in output
    assert "Plain SHA-256 recovered: customer-7-0012" in output
    assert "Keyed hash: not recovered after enumerating 10000 identifiers." in output
    assert len(output.splitlines()) == 6
    assert not re.search(r"\b[0-9a-f]{64}\b", output)


def test_exercise_key_names_intake_fault_and_revokes_all_its_day_credentials(local_outputs):
    output = local_outputs["keys-01"]
    assert "Exercise 5: security-privacy-day-05; corrupted trusted grant input." in output
    assert (
        "Fix issuance, revoke all intake credentials that day, list disclosed decisions." in output
    )


def test_security_privacy_outputs_reproduce_through_repository_capture():
    sources = [
        ROOT / "demos/security_privacy_demo.py",
        ROOT / "demos/security_privacy_integration_demo.py",
    ]
    snippets = {
        snippet.snippet_id: snippet
        for source in sources
        for snippet in extract_snippets_from_text(source.read_text(), source)
    }
    captured = capture_outputs(snippets, sys.executable, timeout=300)
    assert len(captured) == 10
    for name, value in captured.items():
        assert value == (ROOT / "book_outputs" / ("_" + name + ".txt")).read_text()
        assert "companion/" not in value
        assert max(map(len, value.splitlines())) <= 79


def test_credential_listing_and_plot_sidecar():
    source = ROOT / "refund_agent/security/security_privacy_credentials.py"
    region = next(
        s
        for s in extract_snippets_from_text(source.read_text(), source)
        if s.snippet_id.endswith("credential-01")
    )
    assert "class Credential" in region.content and "def credential_problem" in region.content
    reasons = (
        "credential_missing",
        "credential_seal",
        "credential_expired",
        "credential_revoked",
        "credential_tool:",
        "credential_request",
        "order_not_found",
        "credential_amount",
    )
    positions = [region.content.index(reason) for reason in reasons]
    assert positions == sorted(positions)
    assert tuple(region.callouts.values()) == tuple(range(1, 9))
    assert "check_credential" not in region.content
    assert max(map(len, region.content.splitlines())) <= 79
    plot = json.loads((ROOT / "demos/data/security_privacy_attempts.json").read_text())
    assert plot["independent"] and plot["p_pass"] == 0.05
    assert plot["attempts"] == list(range(1, 101))
    assert round(plot["chance"][13], 3) == 0.512


def test_exercise_day_has_one_fault_and_three_correct_distractors():
    support = runpy.run_path(
        str(ROOT / "exercises/security-privacy/security_privacy_day_05.py")
    )

    record = json.loads((ROOT / "exercises/security-privacy/day-05.json").read_text())
    assert record == support["fixture"]()
    assert len(record["requests"]) == 12
    assert support["diagnose"](record) == {
        "security-privacy-day-05": "corrupted trusted grant input"
    }
    assert record["requests"][1]["decision"] == record["requests"][2]["decision"]
    assert record["requests"][3]["reissued_credential"]["credential_id"].endswith("-fresh")
    assert record["scope_events"][0]["at"].endswith("14:00:00+00:00")


def test_exercise_day_approval_intake_uses_reviewers_own_credential():
    from refund_agent.agent import Agent
    from refund_agent.data.generate import generate
    from refund_agent.security.security_privacy_attacks import approval_for, forged_reviewer
    from refund_agent.security.security_privacy_experiment import fresh_tools

    record = json.loads((ROOT / "exercises/security-privacy/day-05.json").read_text())
    row = record["requests"][3]
    intake = next(entry for entry in row["credential_log"] if entry["tool"] == "approval.intake")
    declared = next(
        credential
        for key, credential in row.items()
        if key.endswith("credential") and credential["credential_id"] == intake["credential_id"]
    )
    trusted = generate(7, orders=40, requests=40)
    attack = forged_reviewer(trusted)
    tools, issuer, _ = fresh_tools(attack.data, attack.request, trusted_data=trusted)
    approval = approval_for(tools, Agent(tools).run(attack.request))
    credential = issuer._issue(
        declared.get("kind", "request"),
        declared["subject"],
        approval.approved_at,
        role=declared.get("role"),
    )
    result = tools.record_approval(approval, credential=credential)
    assert result.ok, result.reason
    assert tuple(tools._read("approvals", tools._approvals)) == (approval.approval_id,)


def test_exercise_day_revocation_records_match_actual_tool_refusals():
    from refund_agent import contracts as c
    from refund_agent.data.generate import generate
    from refund_agent.security.security_privacy_experiment import fresh_tools

    record = json.loads((ROOT / "exercises/security-privacy/day-05.json").read_text())
    data = generate(7, orders=40, requests=40)
    request = next(r for r in data.requests if r.request_id == "r-7-0007")
    for row in record["requests"]:
        for entry in row["credential_log"]:
            if entry["reason"] != "credential_revoked":
                continue
            tools, issuer, _ = fresh_tools(data, request)
            issuer.revoke_scope(record["scope_events"][0]["tool"], request.at)
            state = c.State(request=request, trace_id="security-privacy-exercise-5")
            result = tools.call(c.ToolCall(name=entry["tool"]), state)
            assert not result.ok and result.reason == entry["reason"]
            assert not tools.payments


@pytest.mark.parametrize("index", range(12))
def test_exercise_day_decisions_and_fault_records_reproduce_with_real_tools(index):
    from datetime import datetime, timedelta

    from refund_agent import contracts as c
    from refund_agent.agent import Agent
    from refund_agent.data.generate import Dataset
    from refund_agent.security.security_privacy_accounts import AccountRecords
    from refund_agent.security.security_privacy_attacks import approval_for
    from refund_agent.security.security_privacy_experiment import fresh_tools, issue_run, runtime
    from refund_agent.security.security_privacy_tools import ScopedTools

    record = json.loads((ROOT / "exercises/security-privacy/day-05.json").read_text())
    row = record["requests"][index]
    at = datetime.fromisoformat(row["credential"]["not_before"])
    own_order = record["accounts"][row["requester_id"]][0]
    request = c.Request(
        request_id=row["request_id"],
        requester_id=row["requester_id"],
        order_reference=own_order,
        at=at,
        amount_cents=row["decision"]["amount_cents"],
        reason="not_delivered" if index == 5 else "changed_mind",
    )
    orders = [
        c.Order(
            order_id=order_id,
            reference=order_id,
            purchased_at=at - timedelta(days=2),
            observed_at=at,
            items=[c.Item(name="desk lamp", quantity=1, unit_cents=20000)],
            paid_cents=20000,
            payment_id="payment-" + order_id,
        )
        for order_ids in record["accounts"].values()
        for order_id in order_ids
    ]
    policy = c.Policy(
        version="2026-10-01",
        effective_at=at.replace(hour=0, minute=0, second=0, microsecond=0),
        return_days=21,
        cap_cents=75000,
        text="Constructed refund policy.",
    )
    data = Dataset(
        orders=orders,
        requests=[request],
        shipments=[
            c.Shipment(
                order_id=order.order_id,
                observed_at=at,
                events=[c.ShipmentEvent(status="delivered", at=at - timedelta(days=1))],
            )
            for order in orders
        ],
        policies=[policy],
        retrievals=[
            c.PolicySnapshot(request_id=request.request_id, version=policy.version, observed_at=at)
        ],
        correct=[],
        reviewers=[c.Reviewer(reviewer_id="alex", name="Alex", role="reviewer")],
    )
    presented = request.model_copy(update={"order_reference": row["order_id"]})
    attacked = data.model_copy(update={"requests": [presented]})
    if index == 5:
        legitimate = AccountRecords(
            tuple((subject, order) for subject, ids in record["accounts"].items() for order in ids)
        )
        # One intake fault: both issuance and resolution trust this corrupted grant.
        corrupted = AccountRecords(
            tuple(
                sorted(
                    set(legitimate.grants)
                    | {(presented.requester_id, order.order_id) for order in data.orders}
                )
            )
        )
        issuer, security = runtime(data, accounts=corrupted)
        tools = ScopedTools(attacked, security_privacy=security)
        issue_run(tools, issuer, presented, at)
        tools.bind_run_record(
            {
                "name": "run",
                "trace_id": "trace-" + presented.request_id,
                "requester_id": presented.requester_id,
                "run_id": "run-" + presented.request_id + "-1",
            }
        )
    else:
        tools, issuer, security = fresh_tools(attacked, presented, trusted_data=data)
    if at >= datetime.fromisoformat(record["scope_events"][0]["at"]):
        issuer.revoke_scope(
            "refund.execute", datetime.fromisoformat(record["scope_events"][0]["at"])
        )
    trace = Agent(tools).run(presented)
    if index == 3:
        expired = next(
            entry for entry in row["credential_log"] if entry["reason"] == "credential_expired"
        )
        state = trace.steps[-1].state.model_copy(
            update={"execution_at": datetime.fromisoformat(expired["at"])}
        )
        refused = tools.call(c.ToolCall(name=expired["tool"]), state)
        assert not refused.ok and refused.reason == expired["reason"]
        intake = next(
            entry for entry in row["credential_log"] if entry["tool"] == "approval.intake"
        )
        approval = approval_for(tools, trace).model_copy(
            update={"approved_at": datetime.fromisoformat(intake["at"])}
        )
        reviewer = data.reviewers[0]
        assert tools.record_approval(
            approval, credential=issuer.issue_reviewer(reviewer, approval.approved_at)
        ).ok
        issued_at = datetime.fromisoformat(row["reissued_credential"]["not_before"])
        resumed_at = datetime.fromisoformat(row["audit"]["at"])
        tools.security_credential = issuer.issue_request(presented, security.accounts, issued_at)
        tools.security_clock = resumed_at
        tools.data = attacked.model_copy(
            update={
                "orders": [
                    order.model_copy(update={"observed_at": resumed_at}) for order in orders
                ],
                "shipments": [
                    s.model_copy(update={"observed_at": resumed_at}) for s in data.shipments
                ],
                "retrievals": [
                    snapshot.model_copy(update={"observed_at": resumed_at})
                    for snapshot in data.retrievals
                ],
            }
        )
        trace = Agent(tools).run(presented, at=resumed_at)
    actual = trace.decision.model_dump(mode="json", include=set(row["decision"]))
    assert actual == row["decision"]
    assert bool(tools.payments) == row["audit"]["money_moved"]
    if index == 5:
        assert trace.decision.outcome == c.Outcome.DENY
        assert trace.decision.reason == "already_delivered"
        assert trace.decision.policy_citation == "refund_cap_v4@2026-10-01#eligibility"
        fetched = trace.steps[-1].state.order
        assert fetched is not None and fetched.order_id == "order-07"
        assert "order-07" not in legitimate.orders_of(presented.requester_id)
        credential = tools.security_credential
        assert credential is not None
        assert (
            credential.model_dump(mode="json", include=set(row["credential"])) == row["credential"]
        )
        assert [entry.model_dump(mode="json") for entry in security.log.entries] == row[
            "credential_log"
        ]
        assert [audit.model_dump(mode="json") for audit in tools.audits.values()] == [row["audit"]]
        assert row["audit"]["decision"]["reason"] == "already_delivered"
        assert not row["audit"]["money_moved"]

        # Even a valid, over-broad credential cannot widen clean account authority.
        _, clean_security = runtime(data, accounts=legitimate, issuer=issuer)
        clean_tools = ScopedTools(attacked, security_privacy=clean_security)
        clean_tools.security_credential = credential
        clean_tools.security_clock = at
        refused = Agent(clean_tools).run(presented)
        assert refused.decision.reason == "order_not_found"
        assert all(step.state.order is None for step in refused.steps)
        assert not clean_tools.payments
