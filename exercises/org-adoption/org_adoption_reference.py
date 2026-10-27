"""Reference solution; distinct from shipped admission and authority code."""

from refund_agent.rollout.org_adoption_desk import balance_case
from refund_agent.rollout.org_adoption_reconcile import reconcile

CLASSIFICATION = (
    "governed",
    "workaround",
    "governed",
    "governed",
    "workaround",
)
ROLES = {
    "sign": "decision_owner",
    "stop": "release_owner",
    "cause": "reconciler",
}


def count(fixture):
    return reconcile(**fixture)


def amounts(fixture, early):
    return balance_case(fixture, early=early)


def diagnosis(fixture):
    result = count(fixture)
    return {
        "denominator": "intake",
        "share": result.governed_percent,
        "decision": "hold",
    }


def reconcile_bank(intake, decisions, ledger, provider, bank, window):
    money = {}
    for row in (*provider, *bank):
        if row.payment_id in money:
            first = money[row.payment_id]
            facts = ("order_id", "amount_cents", "at", "request_id")
            if any(
                getattr(first, name) != getattr(row, name) for name in facts
            ):
                raise ValueError("bank/provider payment identity conflict")
        else:
            money[row.payment_id] = row
    return reconcile(intake, decisions, ledger, tuple(money.values()), window)
