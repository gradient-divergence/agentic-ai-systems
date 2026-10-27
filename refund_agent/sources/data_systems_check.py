"""Check meaning at assessment and the execution reread. Failure refers."""

import math

from refund_agent.sources.data_systems_contract import SPEC_LIMITS
from refund_agent.sources.data_systems_registry import Registry, default_registry


def contract_problem(
    source: str,
    record,
    served_version: str,
    age_seconds: float,
    *,
    registry: Registry | None = None,
) -> str | None:
    registry = default_registry() if registry is None else registry
    contract = registry.contract(source)
    if served_version != contract.version:
        return f"contract_version_unknown:{source}"
    if not contract.owner_role:
        return f"contract_owner_missing:{source}"
    if contract.budget.approver_role != contract.owner_role:
        return f"contract_budget_unapproved:{source}"
    if not math.isfinite(age_seconds) or age_seconds < 0:
        return f"contract_budget:{source}"
    # The base evidence check owns the spec's limit. Add only a tighter
    # approved budget here; the ledger has no separate cache-age check.
    tighter = contract.budget.limit_seconds < SPEC_LIMITS[source]
    if (tighter or source == "ledger") and age_seconds > contract.budget.limit_seconds:
        return f"contract_budget:{source}"
    if source == "shipment":
        for event in record.events:
            if event.status not in contract.vocabularies["status"]:
                return "contract_vocabulary:shipment.status"
    problem = _invariant(source, record)
    if problem and problem not in contract.invariants:
        raise ValueError("registry does not declare the enforced invariant")
    return f"contract_invariant:{source}.{problem}" if problem else None


def _invariant(source: str, record) -> str | None:
    if source == "order":
        total = sum(i.quantity * i.unit_cents for i in record.items)
        if record.paid_cents != total:
            return "paid_equals_items"
        if sum(p.amount_cents for p in record.prior_refunds) > record.paid_cents:
            return "prior_within_paid"
    elif source == "shipment":
        times = [e.at for e in record.events]
        shipped = [e.at for e in record.events if e.status == "shipped"]
        delivered = [e.at for e in record.events if e.status == "delivered"]
        if times != sorted(times):
            return "time_order"
        if shipped and delivered and min(delivered) < min(shipped):
            return "delivery_after_shipping"
        if any(t > record.observed_at for t in times):
            return "not_after_observation"
    elif source == "policy":
        if record.version != record.effective_at.date().isoformat():
            return "version_equals_effective_date"
    elif source == "ledger":
        if any(type(p.amount_cents) is not int or p.amount_cents <= 0 for p in record):
            return "positive_integer_cents"
        keys = [p.idempotency_key for p in record]
        if len(keys) != len(set(keys)):
            return "unique_idempotency_key"
    else:
        raise ValueError("unknown source")
    return None
