"""Constructed coverage and policy-change cases, scheduled by the core grader."""

from datetime import datetime, timedelta

from refund_agent import contracts as c
from refund_agent.data.generate import Dataset, _policies
from refund_agent.release.suite import Case
from refund_agent.runtimes.swap import SwapCase


def policy_fixture(policy, index, *, new=False):
    at = policy.effective_at + timedelta(hours=12, minutes=index)
    rid = f"mlops-mle-{'policy-change' if new else 'policy-coverage'}-{policy.version}-{index}"
    request = c.Request(
        request_id=rid,
        requester_id=rid + "-person",
        order_reference=rid,
        at=at,
        amount_cents=3499,
        reason="changed_mind",
    )
    delivered = at - timedelta(days=36 + index if new else 2)
    order = c.Order(
        order_id=rid,
        reference=rid,
        purchased_at=delivered - timedelta(days=2),
        observed_at=at - timedelta(seconds=60),
        paid_cents=100000,
        payment_id=rid + "-original",
        items=[c.Item(name="physical goods", quantity=1, unit_cents=100000)],
    )
    shipment = c.Shipment(
        order_id=rid,
        observed_at=at - timedelta(seconds=120),
        events=[c.ShipmentEvent(at=delivered, status="delivered")],
    )
    retrieval = c.PolicySnapshot(
        request_id=rid, version=policy.version, observed_at=at - timedelta(seconds=30)
    )
    data = Dataset(
        orders=[order],
        requests=[request],
        shipments=[shipment],
        policies=[policy],
        retrievals=[retrieval],
        correct=[],
    )
    return data, request


def policy_cases(settings):
    records = settings["register_snapshot"]["records"]
    new_versions = set(settings["versions"]) - {p.version for p in _policies()}
    policies = {p.version: p for p in _policies()}
    for version in new_versions:
        eligibility = next(
            r for r in records if r["version"] == version and r["section"] == "eligibility"
        )
        cap = next(r for r in records if r["version"] == version and r["section"] == "cap")
        policies[version] = c.Policy(
            version=version,
            effective_at=datetime.fromisoformat(eligibility["valid_from"]),
            return_days=eligibility["value"],
            cap_cents=cap["value"],
            named_approver=eligibility["named_approver"],
            text="constructed policy change",
        )
    cases = []
    for version in settings["versions"]:
        policy = policies[version]
        new = version in new_versions
        for index in range(8 if new else 1):
            data, request = policy_fixture(policy, index, new=new)
            paid = 0 if new else 3499
            cases.append(
                Case(
                    request.request_id,
                    "mlops-mle",
                    "policy-change" if new else "policy-coverage",
                    1,
                    SwapCase(request.request_id, "mlops-mle", data, request, paid),
                    handler="mlops-mle-policy",
                    expected={
                        "paid_cents": paid,
                        "outcome": "deny" if new else "approve",
                        "reason": "outside_return_window" if new else "eligible",
                    },
                )
            )
    return tuple(cases)
