"""Indexes and receipt joins used by chapter 38's printed reconciliation."""

from collections import defaultdict

from refund_agent import contracts as c
from refund_agent.rollout.org_adoption_records import (
    DecisionEntry,
    Intake,
    Pending,
    ProviderPayment,
    WindowCount,
)


def window_count(
    window, received, entries, paid, around, governed, exceptions
):
    return WindowCount(
        window=window,
        received=len(received),
        governed=governed,
        around=around,
        missing=tuple(
            k
            for k, row in received.items()
            if k not in entries and row.decision_deadline <= window.closed_at
        ),
        pending=pending_requests(received, entries, window),
        referred=tuple(
            k
            for k in received
            if k in entries and not any(d.terminal for d in entries[k])
        ),
        late_decisions=tuple(
            k
            for k in received
            if any(
                d.audit.at > received[k].decision_deadline
                for d in entries.get(k, ())
                if d.terminal
            )
        ),
        payments_after_window=tuple(
            p.payment_id for p, _ in paid if p.at >= window.end
        ),
        exceptions=tuple(sorted(set(exceptions))),
    )


def pending_requests(received, entries, window):
    return tuple(
        Pending(
            request_id=key,
            decision_deadline=row.decision_deadline,
            overdue=row.decision_deadline <= window.closed_at,
        )
        for key, row in received.items()
        if (key in entries and not any(d.terminal for d in entries[key]))
        or (key not in entries and row.decision_deadline > window.closed_at)
    )


def unique(rows, field):
    result = {}
    for row in rows:
        key = field(row)
        if key in result and result[key] != row:
            raise ValueError("conflicting record identity: " + key)
        result[key] = row
    return result


def index_window(intake, decisions, ledger, provider, window):
    intake = [c._validated(Intake, row) for row in intake]
    decisions = [c._validated(DecisionEntry, row) for row in decisions]
    receipts = unique(
        [c._validated(c.Receipt, row) for row in ledger],
        lambda r: r.receipt_id,
    )
    payments = unique(
        [c._validated(ProviderPayment, row) for row in provider],
        lambda p: p.payment_id,
    )
    all_intake = unique(intake, lambda row: row.request.request_id)
    received = {
        key: row
        for key, row in all_intake.items()
        if window.start <= row.received_at < window.end
    }
    current_orders = {row.order_id for row in received.values()}
    entries = defaultdict(list)
    exceptions = []
    for row in unique(decisions, lambda d: d.audit.audit_id).values():
        if row.audit.at <= window.closed_at:
            if row.request_id not in all_intake:
                exceptions.append("decision_without_intake:" + row.request_id)
            elif row.path == "governed":
                if row.audit.at < all_intake[row.request_id].received_at:
                    raise ValueError("decision precedes intake")
                entries[row.request_id].append(row)
    for rows in entries.values():
        rows.sort(key=lambda d: (d.audit.at, d.audit.audit_id))
    by_order = defaultdict(list)
    for key, row in all_intake.items():
        by_order[row.order_id].append(key)
    attributed = []
    for payment in sorted(
        payments.values(), key=lambda p: (p.at, p.payment_id)
    ):
        if payment.at > window.closed_at:
            continue
        receipt = receipts.get(payment.payment_id)
        request_id = payment.request_id
        if request_id is None and receipt and receipt.request:
            request_id = receipt.request.request_id
        if request_id is None and len(by_order[payment.order_id]) == 1:
            request_id = by_order[payment.order_id][0]
        unmatched_order = (
            request_id not in all_intake
            or all_intake[request_id].order_id != payment.order_id
        )
        if (
            request_id in received
            or (unmatched_order and payment.order_id in current_orders)
            or window.start <= payment.at < window.end
        ):
            if request_id in all_intake:
                if all_intake[request_id].order_id != payment.order_id:
                    exceptions.append(
                        "payment_order_mismatch:" + payment.payment_id
                    )
                    request_id = None
            else:
                exceptions.append(
                    "payment_without_unique_intake:" + payment.payment_id
                )
                request_id = None
            attributed.append((payment, request_id))
    return received, entries, receipts, attributed, exceptions


def governed_payment(payment, request_id, receipts, entries):
    receipt = receipts.get(payment.payment_id)
    if (
        receipt is None
        or receipt.request is None
        or receipt.evidence is None
        or receipt.decision is None
        or not receipt.decision.executed
        or receipt.idempotency_key.startswith("prior-")
    ):
        return False
    if (
        receipt.request.request_id != request_id
        or receipt.order_id != payment.order_id
        or receipt.amount_cents != payment.amount_cents
        or receipt.at != payment.at
    ):
        return False
    return any(
        row.path == "governed"
        and row.audit.receipt_id == receipt.receipt_id
        and row.audit.money_moved
        and row.audit.paid_amount_cents == receipt.amount_cents
        and row.audit.decision == receipt.decision
        for row in entries.get(request_id, ())
    )
