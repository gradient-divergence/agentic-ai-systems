"""Constructed provider: console payments post to order history at settlement.

The provider is an independent record of money out. It never writes the agent's
ledger. A history import must not turn a console payment into governed work.
"""

from datetime import datetime, timedelta

from refund_agent import contracts as c
from refund_agent.rollout.org_adoption_records import ProviderPayment


class ConsoleRefund(c.Record):
    payment: ProviderPayment
    by: c.Identity
    settles_at: datetime


class Provider:
    def __init__(self):
        self._payments = {}
        self._console = {}

    @property
    def payments(self):
        return c._detach(tuple(self._payments.values()))

    def record_tools(self, tools):
        for receipt in tools.payments:
            assert receipt.request is not None
            payment = ProviderPayment(
                payment_id=receipt.receipt_id,
                order_id=receipt.order_id,
                amount_cents=receipt.amount_cents,
                at=receipt.at,
                request_id=receipt.request.request_id,
                channel="tool",
            )
            if (
                payment.payment_id in self._payments
                and self._payments[payment.payment_id] != payment
            ):
                raise ValueError("provider payment identity conflict")
            self._payments[payment.payment_id] = payment

    def refund_console(
        self, order_id, amount_cents, *, by, at, request_id, settles_at=None
    ):
        at = c._utc_time(at)
        if by == "refund-agent":
            raise ValueError("console simulation requires a person")
        if settles_at is None:
            settles_at = at.replace(hour=18, minute=0, second=0, microsecond=0)
            if settles_at < at:
                settles_at += timedelta(days=1)
        settles_at = c._utc_time(settles_at)
        if settles_at < at:
            raise ValueError("settlement precedes payment")
        payment = ProviderPayment(
            payment_id=f"org-adoption-console-{len(self._console) + 1}",
            order_id=order_id,
            amount_cents=amount_cents,
            at=at,
            request_id=request_id,
            channel="console",
        )
        row = ConsoleRefund(payment=payment, by=by, settles_at=settles_at)
        self._console[payment.payment_id] = row
        self._payments[payment.payment_id] = payment
        return c._detach(payment)

    def settle(self, tools, at):
        at = c._utc_time(at)
        data = tools.data
        orders = {order.order_id: order for order in data.orders}
        for row in self._console.values():
            if row.settles_at > at:
                continue
            payment = row.payment
            if payment.order_id not in orders:
                raise ValueError("provider settlement names an unknown order")
            order = orders[payment.order_id]
            prior = c.PriorRefund(
                payment_id=payment.payment_id,
                amount_cents=payment.amount_cents,
                at=payment.at,
            )
            existing = {p.payment_id: p for p in order.prior_refunds}
            if (
                payment.payment_id in existing
                and existing[payment.payment_id] != prior
            ):
                raise ValueError("settlement conflicts with order history")
            existing[payment.payment_id] = prior
            orders[payment.order_id] = order.model_copy(
                update={"prior_refunds": list(existing.values())}
            )
        # The public setter is the sole posting path. Lookup imports prior- receipts.
        tools.data = data.model_copy(update={"orders": list(orders.values())})
