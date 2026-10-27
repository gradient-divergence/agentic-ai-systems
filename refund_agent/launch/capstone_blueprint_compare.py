"""Account for measured arms on the same requests, with a signed cost matrix."""

import json
from fractions import Fraction
from pathlib import Path
from typing import Literal

from pydantic import Field

from refund_agent.launch.capstone_blueprint_requirement import (
    Day,
    LaunchRecord,
    LaunchRequirement,
    Period,
    Text,
    validate,
)
from refund_agent.schedule.inputs import REVIEW_MINUTES

WorkClass = Literal[
    "tier2",
    "tier3",
    "referral",
    "recheck",
    "blind",
    "planted",
    "modification",
    "refusal_followup",
    "held_denial",
    "adjudication",
]
Outcome = Literal["approve", "deny"]
MANUAL_CLASS: dict[int, WorkClass] = {
    1: "referral",
    2: "tier2",
    3: "tier3",
}


class ScriptedModelDisagreement(ValueError):
    """An integrity failure that an operating driver must propagate."""


class ReviewWork(LaunchRecord):
    review_id: Text
    request_id: Text | None
    work_class: WorkClass
    minutes: float = Field(gt=0)


class LabelledRequest(LaunchRecord):
    request_id: Text
    tier: Literal[1, 2, 3]
    amount_cents: int = Field(gt=0, strict=True)
    requested_day: Day
    promised_hours: Literal[1, 48]
    person_outcome: Outcome
    person_paid: bool
    found_false_30d: bool
    label_day: Day


class ArmDecision(LaunchRecord):
    request_id: Text
    outcome: Outcome
    answer_day: Day
    paid_cents: int = Field(ge=0, strict=True)
    # Agent/rules values come from chapter 34's classify(), on the same trace.
    # A fast system referral is bad; a policy referral can be good before the
    # person's final decision. Final answer time alone cannot reconstruct this.
    timeliness_good: bool = Field(strict=True)


class MeasuredArm(LaunchRecord):
    name: Literal["agent", "rules"]
    bundle: Text
    # These are chapter 33's component identities, not a locally built bundle.
    components: dict[Text, Text]
    model_identity: Text
    decisions: tuple[ArmDecision, ...]
    reviews: tuple[ReviewWork, ...]


class ComparisonInputs(LaunchRecord):
    period: Period
    generated_labels: bool
    requests: tuple[LabelledRequest, ...] = Field(min_length=1)
    agent: MeasuredArm
    rules: MeasuredArm
    manual_answer_days: dict[Text, Day]
    manual_queue_record: Text


class ArmMeasures(LaunchRecord):
    arm: Literal["agent", "manual", "rules"]
    requests: int
    final_decisions: dict[str, Outcome]
    review_minutes: dict[str, float]
    reviews: int
    hours_per_week: float
    wrongful_denials: int
    wrong_payment_cents: int
    payments_later_false: int
    cost_cents: int
    tier1_answered_in_time: int
    tier1_requests: int
    tier1_share: float


class Comparison(LaunchRecord):
    period: Period
    launched_bundle: Text
    requirement_fingerprint: Text
    rule: Text
    signed_matrix_fingerprint: Text
    manual_reviews_reviewers_work: Literal[False] = False
    manual_assumption: Literal["no review of its reviewers' work"] = (
        "no review of its reviewers' work"
    )
    manual_queue_record: Text
    generated_labels: bool
    arms: tuple[ArmMeasures, ...]
    against_manual: Literal["better", "not better"]
    against_rules: Literal["better", "not better"]
    verdict: Literal["better", "not better"]
    last_payment_day: Day | None


def _decisions(rows, requests, day):
    by_id = {row.request_id: row for row in rows}
    if len(by_id) != len(rows) or set(by_id) != set(requests):
        raise ValueError(
            "each arm must decide the same distinct requests exactly once"
        )
    for rid, row in by_id.items():
        request = requests[rid]
        if row.answer_day < request.requested_day:
            raise ValueError("answer precedes request")
        if row.answer_day > day:
            raise ValueError("decision or payment is after evaluation clock")
        if row.paid_cents and (
            row.outcome != "approve" or row.paid_cents != request.amount_cents
        ):
            raise ValueError("payment differs from approved requested amount")
    return by_id


def _measure(name, rows, reviews, requests, req) -> ArmMeasures:
    choices = _decisions(rows, requests, req.gate_day)
    if len({r.review_id for r in reviews}) != len(reviews):
        raise ValueError("duplicate review item")
    minutes = {kind: 0.0 for kind in WorkClass.__args__}
    for review in reviews:
        if review.request_id is not None and review.request_id not in requests:
            raise ValueError("review names a request outside the comparison")
        minutes[review.work_class] += review.minutes
    wrongful = sum(
        row.outcome == "deny"
        and requests[rid].person_paid
        and not requests[rid].found_false_30d
        for rid, row in choices.items()
    )
    wrong_cents = sum(
        row.paid_cents
        for rid, row in choices.items()
        if requests[rid].person_outcome != "approve"
    )
    false = sum(
        bool(row.paid_cents) and requests[rid].found_false_30d
        for rid, row in choices.items()
    )
    tier1 = [r for r in requests.values() if r.tier == 1]
    timely = sum(choices[r.request_id].timeliness_good for r in tier1)
    matrix = req.cost_matrix
    return ArmMeasures(
        arm=name,
        requests=len(requests),
        final_decisions={k: r.outcome for k, r in choices.items()},
        review_minutes=minutes,
        reviews=len(reviews),
        hours_per_week=sum(minutes.values())
        / 60
        * 7
        / (req.comparison_period.end - req.comparison_period.start),
        wrongful_denials=wrongful,
        wrong_payment_cents=wrong_cents,
        payments_later_false=false,
        cost_cents=len(reviews) * matrix.review_cents
        + wrongful * matrix.wrongful_denial_cents
        + wrong_cents * matrix.wrong_payment_multiplier,
        tier1_answered_in_time=timely,
        tier1_requests=len(tier1),
        tier1_share=timely / len(tier1) if tier1 else 0,
    )


def better(agent: ArmMeasures, other: ArmMeasures) -> bool:
    # Integers/counts decide money and rates; the exact fraction avoids ratio drift.
    small_a = (
        agent.hours_per_week,
        agent.wrongful_denials,
        agent.wrong_payment_cents,
        agent.payments_later_false,
        agent.cost_cents,
    )
    small_b = (
        other.hours_per_week,
        other.wrongful_denials,
        other.wrong_payment_cents,
        other.payments_later_false,
        other.cost_cents,
    )
    timely_a = Fraction(
        agent.tier1_answered_in_time, agent.tier1_requests or 1
    )
    timely_b = Fraction(
        other.tier1_answered_in_time, other.tier1_requests or 1
    )
    return (
        all(a <= b for a, b in zip(small_a, small_b))
        and timely_a >= timely_b
        and (
            any(a < b for a, b in zip(small_a, small_b)) or timely_a > timely_b
        )
    )


def measure(
    inputs: ComparisonInputs, req: LaunchRequirement, roles
) -> Comparison:
    inputs = ComparisonInputs.model_validate(inputs.model_dump(mode="json"))
    validate(req, roles)
    if (
        inputs.period != req.comparison_period
        or inputs.agent.bundle != req.launched_bundle
    ):
        raise ValueError(
            "comparison period or agent bundle differs from signed requirement"
        )
    if inputs.agent.name != "agent" or inputs.rules.name != "rules":
        raise ValueError("comparison arm names differ")
    if set(inputs.agent.components) != {
        "model",
        "prompt",
        "tools",
        "policy",
        "index",
    }:
        raise ValueError("comparison needs all five component fingerprints")
    if set(inputs.rules.components) != set(inputs.agent.components) or any(
        inputs.rules.components[k] != inputs.agent.components[k]
        for k in inputs.agent.components
        if k != "model"
    ):
        raise ValueError("rules arm must keep every non-model component")
    if inputs.rules.model_identity != "RulesClient" or (
        inputs.rules.components["model"] == inputs.agent.components["model"]
    ):
        raise ValueError("rules arm must name its measured replacement model")
    requests = {r.request_id: r for r in inputs.requests}
    if len(requests) != len(inputs.requests) or any(
        not inputs.period.start <= r.requested_day < inputs.period.end
        for r in inputs.requests
    ):
        raise ValueError(
            "distinct requests must come from the signed comparison period"
        )
    if not inputs.generated_labels and any(
        r.label_day > req.gate_day for r in inputs.requests
    ):
        raise ValueError("comparison labels have not matured")
    if set(inputs.manual_answer_days) != set(requests):
        raise ValueError(
            "manual queue must serve every request; pending is not a fast answer"
        )
    manual = tuple(
        ArmDecision(
            request_id=rid,
            outcome=r.person_outcome,
            answer_day=inputs.manual_answer_days[rid],
            paid_cents=r.amount_cents if r.person_paid else 0,
            timeliness_good=(inputs.manual_answer_days[rid] - r.requested_day)
            * 24
            <= r.promised_hours + 1e-9,
        )
        for rid, r in requests.items()
    )
    manual_reviews = tuple(
        ReviewWork(
            review_id="capstone-blueprint-manual-" + rid,
            request_id=rid,
            work_class=MANUAL_CLASS[r.tier],
            minutes=REVIEW_MINUTES[r.tier],
        )
        for rid, r in requests.items()
    )
    a = _measure(
        "agent", inputs.agent.decisions, inputs.agent.reviews, requests, req
    )
    b = _measure("manual", manual, manual_reviews, requests, req)
    c = _measure(
        "rules", inputs.rules.decisions, inputs.rules.reviews, requests, req
    )
    if (
        inputs.agent.model_identity == "ScriptedModel"
        and a.final_decisions != c.final_decisions
    ):
        raise ScriptedModelDisagreement(
            "scripted model departs from rules; stop the code job"
        )
    result_b, result_c = better(a, b), better(a, c)
    paid = [
        row.answer_day
        for row in (*inputs.agent.decisions, *manual, *inputs.rules.decisions)
        if row.paid_cents
    ]
    return Comparison(
        period=inputs.period,
        launched_bundle=req.launched_bundle,
        requirement_fingerprint=req.fingerprint,
        rule=req.comparison_rule,
        signed_matrix_fingerprint=req.cost_matrix.fingerprint,
        manual_queue_record=inputs.manual_queue_record,
        generated_labels=inputs.generated_labels,
        arms=(a, b, c),
        against_manual="better" if result_b else "not better",
        against_rules="better" if result_c else "not better",
        verdict="better" if result_b and result_c else "not better",
        last_payment_day=max(paid) if paid else None,
    )


def compare(
    run_dir: Path, requirement: LaunchRequirement, *, roles
) -> Comparison:
    """Read actual recorded replay and queue results; absent arms raise, never tie."""
    path = Path(run_dir) / "capstone-blueprint-comparison-inputs.json"
    inputs = ComparisonInputs.model_validate(json.loads(path.read_text()))
    return measure(inputs, requirement, roles)
