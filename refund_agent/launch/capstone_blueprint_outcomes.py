"""Outcome labels follow actual human shadow payments and actual agent payments."""

from typing import Literal

from pydantic import Field, model_validator

from refund_agent.launch.capstone_blueprint_requirement import (
    Day,
    LaunchRecord,
    Text,
)


class PaymentOutcome(LaunchRecord):
    receipt_id: Text
    request_id: Text
    payment_day: Day
    amount_cents: int = Field(gt=0, strict=True)
    actor: Literal["human", "agent"]
    mode: Literal["shadow", "executed"]
    found_false_30d: bool | None
    generated_label: bool

    @model_validator(mode="after")
    def actual_payment(self):
        if self.actor == "agent" and self.mode == "shadow":
            raise ValueError(
                "a hypothetical shadow payment has no payment outcome"
            )
        return self


class OutcomeWindow(LaunchRecord):
    final: tuple[PaymentOutcome, ...]
    pending: tuple[str, ...]


def outcomes(
    payments: tuple[PaymentOutcome, ...], through: float
) -> OutcomeWindow:
    payments = tuple(
        PaymentOutcome.model_validate(p.model_dump(mode="json"))
        for p in payments
    )
    if len({p.receipt_id for p in payments}) != len(payments):
        raise ValueError(
            "an actual payment acquires one outcome, not one per feed copy"
        )
    if any(p.payment_day > through for p in payments):
        raise ValueError("future payment in outcome window")
    final, pending = [], []
    for payment in payments:
        if payment.found_false_30d is None or (
            not payment.generated_label and through < payment.payment_day + 30
        ):
            pending.append(payment.receipt_id)
        else:
            final.append(payment)
    return OutcomeWindow(final=tuple(final), pending=tuple(pending))
