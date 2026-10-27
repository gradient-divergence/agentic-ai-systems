"""Deterministic calculations over the constructed facts, without operating agents."""

import math
import random
from datetime import datetime, timedelta, timezone
from fractions import Fraction

from pydantic import Field, model_validator

from refund_agent.sources.data_systems_budget import budget_hours, stale_chance
from refund_agent.transfer.case_studies_domain_agnostic_check import expected_cost
from refund_agent.transfer.case_studies_domain_agnostic_facts import CostMatrix, Record, Text
from refund_agent.transfer.case_studies_domain_agnostic_packet import Rule


def alarm_numbers(rate: float, tolerance: float) -> dict[str, float]:
    return {
        "rate_per_hour": rate,
        "chance_20_minutes": stale_chance(rate, 1 / 3),
        "chance_30_minutes": stale_chance(rate, 1 / 2),
        "budget_seconds": 3600 * budget_hours(rate, tolerance),
    }


class Patient(Record):
    patient_id: Text
    urgency: str
    deadline_day: int = Field(gt=0, strict=True)
    discharged_hour: int = Field(ge=0, lt=24, strict=True)

    @model_validator(mode="after")
    def window(self):
        if self.urgency not in ("urgent", "routine"):
            raise ValueError("urgency must be urgent or routine")
        return self


class HospitalDay(Record):
    patients: tuple[Patient, ...]
    slots_per_day: int = Field(gt=0, strict=True)
    days: int = Field(gt=0, strict=True)
    batch_hour: int = Field(ge=0, lt=24, strict=True)
    basis: Text

    @model_validator(mode="after")
    def day(self):
        if len({p.patient_id for p in self.patients}) != len(self.patients):
            raise ValueError("patients must be unique")
        if any(
            p.deadline_day > self.days or p.discharged_hour > self.batch_hour for p in self.patients
        ):
            raise ValueError("the batch needs all discharges and their windows")
        return self


class Booking(Record):
    patient_id: str
    urgency: str
    day: int | None
    slot: int | None
    notification_hour: int


def hospital_day(day: HospitalDay, policy: str) -> tuple[Booking, ...]:
    if policy not in ("request", "batch"):
        raise ValueError("policy must be request or batch")
    pending = day.patients
    if policy == "batch":
        pending = tuple(sorted(pending, key=lambda p: p.deadline_day))
    free = [(d, slot) for d in range(1, day.days + 1) for slot in range(1, day.slots_per_day + 1)]
    booked = {}
    for patient in pending:
        available = next((pair for pair in free if pair[0] <= patient.deadline_day), None)
        if available is not None:
            free.remove(available)
        booked[patient.patient_id] = Booking(
            patient_id=patient.patient_id,
            urgency=patient.urgency,
            day=available[0] if available else None,
            slot=available[1] if available else None,
            notification_hour=patient.discharged_hour if policy == "request" else day.batch_hour,
        )
    return tuple(booked[p.patient_id] for p in day.patients)


def hospital_orders(
    day: HospitalDay, n: int = 10_000, seed: int = 7
) -> dict[str, int | float | Fraction]:
    if type(n) is not int or n <= 0:
        raise ValueError("a positive number of random orders is required")
    urgent = [p for p in day.patients if p.urgency == "urgent"]
    if not urgent or len({p.deadline_day for p in urgent}) != 1:
        raise ValueError("this calculation requires one urgent window")
    near = urgent[0].deadline_day * day.slots_per_day
    if len(urgent) != near or len(day.patients) < near:
        raise ValueError("the all-urgent probability requires urgent count equal to near capacity")
    if any(p.deadline_day < urgent[0].deadline_day for p in day.patients):
        raise ValueError("every discharge must be eligible for the near slots")
    rng, total, all_in_time = random.Random(seed), 0, 0
    for _ in range(n):
        order = list(day.patients)
        rng.shuffle(order)
        trial = HospitalDay.model_validate({**day.model_dump(), "patients": tuple(order)})
        count = sum(
            b.day is not None for b in hospital_day(trial, "request") if b.urgency == "urgent"
        )
        total += count
        all_in_time += count == len(urgent)
    return {
        "n": n,
        "seed": seed,
        "mean": total / n,
        "all_in_time_observed": all_in_time,
        "exact_mean": Fraction(len(urgent) * near, len(day.patients)),
        "exact_all_probability": Fraction(1, math.comb(len(day.patients), near)),
    }


def bank_costs(matrix: CostMatrix, policies: dict[str, Rule], risks: tuple[Fraction, ...]) -> dict:
    return {
        str(risk): {
            name: {
                "action": policy.action_at(risk),
                "cost": expected_cost(matrix, policy.action_at(risk), risk),
            }
            for name, policy in policies.items()
        }
        for risk in risks
    }


def holds_per_fraud(precision: Fraction) -> Fraction:
    if not 0 < precision <= 1:
        raise ValueError("precision must lie in (0, 1]")
    return (1 - precision) / precision


def order_balance() -> dict[str, int | str | None]:
    """B3's actual refund-agent execution, not a new domain's agent."""
    from refund_agent import contracts as c
    from refund_agent.agent import Agent
    from refund_agent.data.generate import Dataset
    from refund_agent.model import Reply, ScriptedModel
    from refund_agent.tools import Tools

    at = datetime(2026, 10, 1, 12, tzinfo=timezone.utc)
    requests = [
        c.Request(
            request_id=f"{__name__}-{n}",
            requester_id="owner",
            order_reference="order",
            at=at + timedelta(seconds=n),
            amount_cents=3000,
            reason="damaged",
        )
        for n in (1, 2)
    ]
    policy = c.Policy(
        version="2026-09-30",
        effective_at=at - timedelta(days=1),
        return_days=90,
        cap_cents=100000,
        text="Constructed refund policy.",
    )
    data = Dataset(
        requests=requests,
        orders=[
            c.Order(
                order_id="order",
                reference="order",
                payment_id="original",
                paid_cents=4000,
                purchased_at=at - timedelta(days=2),
                observed_at=at,
                items=[c.Item(name="book", quantity=1, unit_cents=4000)],
            )
        ],
        policies=[policy],
        correct=[],
        shipments=[
            c.Shipment(
                order_id="order",
                observed_at=at,
                events=[c.ShipmentEvent(status="delivered", at=at - timedelta(days=1))],
            )
        ],
        retrievals=[
            c.PolicySnapshot(request_id=r.request_id, version=policy.version, observed_at=at)
            for r in requests
        ],
    )
    tools = Tools(data)

    class InterleavedModel(ScriptedModel):
        first: c.Trace | None = None

        def reply(self, state: c.State) -> Reply:
            # Both assessments see 4000 cents. Another real run pays before
            # this proposal executes; the existing gate alone must refuse it.
            if state.phase == c.Phase.EXECUTE and self.first is None:
                self.first = Agent(tools).run(requests[0])
            return super().reply(state)

    schedule = InterleavedModel()
    second = Agent(tools, schedule).run(requests[1])
    first = schedule.first
    if first is None:
        raise ValueError("B3 fixture did not reach its execution interleaving")
    paid = sum(receipt.amount_cents for receipt in tools.payments)
    return {
        "first_payment_cents": first.receipt.amount_cents if first.receipt else None,
        "second_payment_cents": second.receipt.amount_cents if second.receipt else None,
        "second_outcome": second.decision.outcome.value,
        "second_reason": second.decision.reason,
        "second_assessed": next(
            step.reply.outcome.value
            for step in second.steps
            if step.phase == c.Phase.ASSESS and step.reply is not None
        ),
        "second_assessment_reason": next(
            step.result.reason for step in second.steps if step.phase == c.Phase.ASSESS
        ),
        "second_execution_attempts": sum(
            step.call is not None and step.call.name == "refund.execute" for step in second.steps
        ),
        "second_execution_refused": sum(
            step.call is not None
            and step.call.name == "refund.execute"
            and not step.result.ok
            and step.result.reason == "amount_over_policy"
            for step in second.steps
        ),
        "paid_cents": paid,
        "remaining_cents": 4000 - paid,
    }
