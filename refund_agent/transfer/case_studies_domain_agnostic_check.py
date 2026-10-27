"""Four transfer checks. A missing launch-row implementation cannot pass."""

# The standalone printed region imports its helpers from this same module.
# ruff: noqa: E402

from fractions import Fraction
from importlib import import_module
from itertools import combinations
from typing import Literal

from pydantic import ValidationError

from refund_agent.sources.data_systems_budget import budget_hours, stale_chance
from refund_agent.transfer.case_studies_domain_agnostic_facts import (
    CostMatrix,
    DomainFacts,
    Record,
    SharedResource,
    Source,
)
from refund_agent.transfer.case_studies_domain_agnostic_packet import (
    DecisionPacket,
    Interval,
    Rule,
    map_requirement_roles,
)


class Failure(Record):
    check: str
    item: str
    reason: str
    numbers: dict[str, str]


class TransferVerdict(Record):
    domain: str
    form: str
    verdict: Literal["pass", "hold"]
    failures: tuple[Failure, ...]


def expected_cost(matrix: CostMatrix, action: str, risk: Fraction) -> Fraction:
    if not 0 <= risk <= 1:
        raise ValueError("risk must lie in [0, 1]")
    sound, bad = matrix.entries[action]
    return (1 - risk) * sound + risk * bad


def cheapest(matrix: CostMatrix, actions: tuple[str, ...], referral: str, risk: Fraction) -> str:
    values = {action: expected_cost(matrix, action, risk) for action in actions}
    minimum = min(values.values())
    tied = [action for action in actions if values[action] == minimum]
    return referral if referral in tied else tied[0]


def breakpoints(matrix: CostMatrix, actions: tuple[str, ...]) -> set[Fraction]:
    points = {Fraction(0), Fraction(1)}
    for left, right in combinations(actions, 2):
        a, b = matrix.entries[left], matrix.entries[right]
        slope = (a[1] - a[0]) - (b[1] - b[0])
        if slope:
            risk = (b[0] - a[0]) / slope
            if 0 <= risk <= 1:
                points.add(risk)
    return points


def atoms(points: set[Fraction]) -> list[Interval]:
    ordered = sorted(points)
    result = []
    for index, point in enumerate(ordered):
        result.append(Interval(lower=point, upper=point, action="probe", right_closed=True))
        if index + 1 < len(ordered):
            result.append(
                Interval(
                    lower=point,
                    upper=ordered[index + 1],
                    action="probe",
                    left_closed=False,
                    right_closed=False,
                )
            )
    return result


def _join(pieces: list[tuple[Interval, str, str]]) -> list[tuple[Interval, str, str]]:
    result: list[tuple[Interval, str, str]] = []
    for interval, actual, wanted in pieces:
        if result:
            prior, old_actual, old_wanted = result[-1]
            if (
                (actual, wanted) == (old_actual, old_wanted)
                and prior.upper == interval.lower
                and (prior.right_closed or interval.left_closed)
            ):
                result[-1] = (
                    Interval(
                        lower=prior.lower,
                        upper=interval.upper,
                        action=actual,
                        left_closed=prior.left_closed,
                        right_closed=interval.right_closed,
                    ),
                    actual,
                    wanted,
                )
                continue
        result.append((interval, actual, wanted))
    return result


def optimal_rule(
    matrix: CostMatrix, actions: tuple[str, ...], referral: str, alone: tuple[str, ...]
) -> Rule:
    if not actions or len(set(actions)) != len(actions) or referral not in actions:
        raise ValueError("distinct allowed actions must include the referral")
    pieces = []
    for interval in atoms(breakpoints(matrix, actions)):
        action = cheapest(matrix, actions, referral, (interval.lower + interval.upper) / 2)
        pieces.append((interval, action, action))
    joined = _join(pieces)
    return Rule(
        intervals=tuple(
            Interval.model_validate({**i.model_dump(), "action": action}) for i, action, _ in joined
        ),
        actions_alone=alone,
        referral=referral,
        status="constructed",
    )


def rule(packet: DecisionPacket, facts: DomainFacts) -> list[Failure]:
    matrix, configured = facts.cost_matrix, packet.rule
    if matrix is None:
        return (
            []
            if configured is None
            else [
                Failure(
                    check="rule",
                    item="rule",
                    reason="this domain declares no scored decision",
                    numbers={},
                )
            ]
        )
    if configured is None:
        return [
            Failure(check="rule", item="rule", reason="scored decision lacks a rule", numbers={})
        ]
    if configured.referral not in facts.actions_on_score or not {
        i.action for i in configured.intervals
    } <= set(facts.actions_on_score):
        return [
            Failure(
                check="rule",
                item="boundary",
                reason="rule exceeds the score's authority",
                numbers={},
            )
        ]
    referral = facts.role_map["refer"]
    failures = []
    if configured.referral != referral:
        failures.append(
            Failure(
                check="rule",
                item="referral",
                reason="referral differs from domain authority",
                numbers={"declared": configured.referral, "required": referral},
            )
        )
    points = breakpoints(matrix, facts.actions_on_score) | {
        x for interval in configured.intervals for x in (interval.lower, interval.upper)
    }
    wrong = []
    for interval in atoms(points):
        risk = (interval.lower + interval.upper) / 2
        actual = configured.action_at(risk)
        wanted = cheapest(matrix, facts.actions_on_score, referral, risk)
        if actual != wanted:
            wrong.append((interval, actual, wanted))
    for interval, actual, wanted in _join(wrong):
        risk = (interval.lower + interval.upper) / 2
        failures.append(
            Failure(
                check="rule",
                item="risk",
                reason="minimum cost; referral wins ties",
                numbers={
                    "lower": str(interval.lower),
                    "upper": str(interval.upper),
                    "left_closed": str(interval.left_closed),
                    "right_closed": str(interval.right_closed),
                    "copied": actual,
                    "cheapest": wanted,
                    "midpoint": str(risk),
                    "copied_cost": str(expected_cost(matrix, actual, risk)),
                    "cheapest_cost": str(expected_cost(matrix, wanted, risk)),
                },
            )
        )
    return failures


def structure(packet: DecisionPacket, facts: DomainFacts) -> list[Failure]:
    # Chapter 39 owns row semantics. No local validator or shipped test double.
    launch = import_module("refund_agent.launch")
    failures = []
    if len(packet.requirement) != 7:
        failures.append(
            Failure(
                check="structure",
                item="rows",
                reason="seven launch questions required",
                numbers={"rows": str(len(packet.requirement))},
            )
        )
    for index, raw in enumerate(packet.requirement):
        try:
            if packet.domain != "refund":
                raw = map_requirement_roles((raw,), {title: role for role, title in facts.roles.items()})[0]
            row = launch.LaunchRow.model_validate(raw)
        except (ValidationError, KeyError, TypeError) as error:
            failures.append(
                Failure(check="structure", item=str(index + 1), reason=str(error), numbers={})
            )
            continue
        reasons = launch.validate_row(row, facts.roles)
        if not isinstance(reasons, (tuple, list)) or any(not isinstance(s, str) for s in reasons):
            raise TypeError("chapter 39 validate_row must return a sequence of reasons")
        failures.extend(
            Failure(check="structure", item=str(index + 1), reason=reason, numbers={})
            for reason in reasons
        )
    return failures


def _clock_source(packet: DecisionPacket, facts: DomainFacts, source: Source) -> Failure | None:
    if source.version_checked:
        if source.name in packet.version_checks:
            return None
        return Failure(check="clock", item=source.name, reason="version check missing", numbers={})
    if source.name not in packet.age_limits:
        return Failure(check="clock", item=source.name, reason="age limit missing", numbers={})
    if source.rate_per_hour is None:
        raise ValueError("the source has neither a rate nor a version check")
    seconds = packet.age_limits[source.name]
    chance = stale_chance(source.rate_per_hour, seconds / 3600)
    if chance <= facts.tolerance:
        return None
    return Failure(
        check="clock",
        item=source.name,
        reason="out-of-date chance exceeds tolerance",
        numbers={
            "chance": f"{chance:.12g}",
            "tolerance": str(facts.tolerance),
            "age_seconds": str(seconds),
            "budget_seconds": f"{budget_hours(source.rate_per_hour, facts.tolerance) * 3600:.12g}",
        },
    )


def _resource_failure(packet: DecisionPacket, resource: SharedResource) -> Failure | None:
    grouped = packet.decision_unit in (f"batch:{resource.name}", f"reserve:{resource.name}")
    if packet.recheck_at_execution and (resource.requirement == "limit" or grouped):
        return None
    return Failure(
        check="coupling",
        item=resource.name,
        reason="resource constraint not enforced",
        numbers={
            "requires": resource.requirement,
            "capacity": str(resource.capacity),
            "unit": resource.unit,
            "decision_unit": packet.decision_unit,
            "recheck_at_execution": str(packet.recheck_at_execution),
        },
    )


# fmt: off
# isort: off
# region book:fnd.case-studies-domain-agnostic.check-01
import refund_agent.transfer.case_studies_domain_agnostic_check as transfer

def clock(packet, facts):
    return [failure for source in facts.sources
            if (failure := transfer._clock_source(packet, facts, source))
            is not None]

def coupling(packet, facts):
    actions = {item.action for item in packet.classes}
    return [failure for resource in facts.shared_resources
            if actions.intersection(resource.actions)
            and (failure := transfer._resource_failure(packet, resource))
            is not None]

def check(packet, facts) -> transfer.TransferVerdict:
    if packet.domain != facts.domain:
        raise ValueError("packet and facts name different domains")
    failures = tuple(f for test in (transfer.structure, clock, coupling,
                                   transfer.rule)
                     for f in test(packet, facts))
    return transfer.TransferVerdict(domain=packet.domain, form=packet.form,
        verdict="hold" if failures else "pass", failures=failures)
# endregion book:fnd.case-studies-domain-agnostic.check-01
# isort: on
# fmt: on
