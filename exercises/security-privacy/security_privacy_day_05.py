"""Construct twelve records with one corrupted trusted grant input at intake.

Only customer-05's trusted account-to-orders grant is corrupted. Issuance and
the unchanged resolver both trust it, so a valid credential covers every order.
The other eleven records and the three distractors retain their correct behavior.
"""

from datetime import datetime, timedelta, timezone

from refund_agent import contracts as c
from refund_agent.agent import Agent
from refund_agent.data.generate import Dataset
from refund_agent.security.security_privacy_accounts import AccountRecords
from refund_agent.security.security_privacy_experiment import issue_run, runtime
from refund_agent.security.security_privacy_tools import ScopedTools


def _record_intake_fault(row, accounts):
    """Run the real tools with exactly one corrupted trusted grant input."""
    at = datetime.fromisoformat(row["credential"]["not_before"])
    request = c.Request(
        request_id=row["request_id"],
        requester_id=row["requester_id"],
        order_reference=row["order_id"],
        at=at,
        amount_cents=2499,
        reason="not_delivered",
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
        for order_ids in accounts.values()
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
    )
    grants = {(subject, order) for subject, ids in accounts.items() for order in ids}
    grants.update((request.requester_id, order.order_id) for order in orders)
    corrupted_trusted_grant_input = AccountRecords(tuple(sorted(grants)))
    issuer, security = runtime(data, accounts=corrupted_trusted_grant_input)
    tools = ScopedTools(data, security_privacy=security)
    issue_run(tools, issuer, request, at)
    tools.bind_run_record(
        {
            "name": "run",
            "trace_id": "trace-" + request.request_id,
            "requester_id": request.requester_id,
            "run_id": "run-" + request.request_id + "-1",
        }
    )
    trace = Agent(tools).run(request)
    credential = tools.security_credential
    assert credential is not None
    row["credential"] = credential.model_dump(mode="json", include=set(row["credential"]))
    row["credential_log"] = [entry.model_dump(mode="json") for entry in security.log.entries]
    (row["audit"],) = [audit.model_dump(mode="json") for audit in tools.audits.values()]
    row["decision"] = trace.decision.model_dump(mode="json", include=set(row["decision"]))


def fixture():
    start = datetime(2026, 10, 1, 9, tzinfo=timezone.utc)
    rows = []
    accounts = {f"customer-{i:02}": [f"order-{i:02}"] for i in range(12)}
    for i in range(12):
        at = start + timedelta(minutes=40 * i)
        request_id = f"security-privacy-day-{i:02}"
        subject, order = f"customer-{i:02}", f"order-{i:02}"
        reason, outcome, orders = "eligible", "approve", [order]
        if i == 5:
            order = "order-07"
        elif i in (1, 2):
            order, reason, outcome = (
                "missing-order" if i == 1 else "order-10",
                "order_not_found",
                "refer",
            )
        elif i >= 8:
            reason, outcome = "credential_revoked", "refer"
        credential = {
            "credential_id": request_id + "-credential",
            "kind": "request",
            "subject": subject,
            "request_id": request_id,
            "order_ids": orders,
            "not_before": at.isoformat(),
            "expires_at": (at + timedelta(minutes=30)).isoformat(),
        }
        credential_log = [
            {
                "trace_id": "trace-" + request_id,
                "run_id": "run-" + request_id + "-1",
                "tool": "refund.execute" if i >= 8 else "order.lookup",
                "reason": reason if outcome == "refer" else "",
                "at": at.isoformat(),
            }
        ]
        audit = {
            "trace_id": "trace-" + request_id,
            "at": at.isoformat(),
            "money_moved": outcome == "approve",
        }
        row = {
            "request_id": request_id,
            "requester_id": subject,
            "order_id": order,
            "credential": credential,
            "credential_log": credential_log,
            "audit": audit,
            "decision": {
                "outcome": outcome,
                "reason": reason,
                "amount_cents": 12000 if i == 3 else 2499,
                "tier": 2 if i == 3 else 1,
                "policy_citation": "refund_cap_v4@2026-10-01#eligibility"
                if reason != "order_not_found"
                else None,
            },
        }
        if i == 5:
            _record_intake_fault(row, accounts)
        if i == 3:
            resumed_at = (at + timedelta(minutes=32)).isoformat()
            fresh_id, reviewer_id = request_id + "-fresh", request_id + "-reviewer"
            audit["at"] = resumed_at
            row["reissued_credential"] = {
                **credential,
                "credential_id": fresh_id,
                "not_before": (at + timedelta(minutes=31)).isoformat(),
                "expires_at": (at + timedelta(minutes=61)).isoformat(),
            }
            row["reviewer_credential"] = {
                "credential_id": reviewer_id,
                "kind": "reviewer",
                "subject": "alex",
                "role": "reviewer",
                "not_before": (at + timedelta(minutes=31)).isoformat(),
                "expires_at": (at + timedelta(minutes=61)).isoformat(),
            }
            credential_log.append(
                {
                    "trace_id": "trace-" + request_id,
                    "run_id": "run-" + request_id + "-1",
                    "tool": "refund.execute",
                    "reason": "credential_expired",
                    "at": (at + timedelta(minutes=30)).isoformat(),
                }
            )
            credential_log.append(
                {
                    "trace_id": "trace-" + request_id,
                    "run_id": "run-" + request_id + "-2",
                    "tool": "approval.intake",
                    "reason": "",
                    "credential_id": reviewer_id,
                    "at": (at + timedelta(minutes=31)).isoformat(),
                }
            )
            credential_log.append(
                {
                    "trace_id": "trace-" + request_id,
                    "run_id": "run-" + request_id + "-2",
                    "tool": "refund.execute",
                    "reason": "",
                    "credential_id": fresh_id,
                    "at": resumed_at,
                }
            )
        rows.append(row)
    return {
        "constructed": True,
        "accounts": accounts,
        "requests": rows,
        "scope_events": [
            {
                "tool": "refund.execute",
                "at": start.replace(hour=14).isoformat(),
                "reason": "credential_revoked",
            }
        ],
    }


def diagnose(record):
    """Compare issued authority with clean accounts to find the intake fault."""
    wrong = {}
    for row in record["requests"]:
        granted = set(row["credential"]["order_ids"])
        legitimate = set(record["accounts"][row["requester_id"]])
        if granted - legitimate:
            wrong[row["request_id"]] = "corrupted trusted grant input"
    return wrong
