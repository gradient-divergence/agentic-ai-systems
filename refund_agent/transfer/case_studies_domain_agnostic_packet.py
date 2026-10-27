"""Packet answers and exact risk intervals; launch rows remain chapter 39's."""

from fractions import Fraction
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, JsonValue, model_validator

from refund_agent.transfer.case_studies_domain_agnostic_facts import (
    DomainFacts,
    Record,
    Text,
    require_inventory,
)


class DecisionClass(Record):
    name: Text
    action: Text
    level: int = Field(ge=1, le=4, strict=True)
    above_role: Text
    condition: Text


class Interval(Record):
    lower: Fraction = Field(ge=0, le=1)
    upper: Fraction = Field(ge=0, le=1)
    action: Text
    left_closed: bool = True
    right_closed: bool = False

    @model_validator(mode="after")
    def bounds(self):
        if (
            self.lower > self.upper
            or self.lower == self.upper
            and not (self.left_closed and self.right_closed)
        ):
            raise ValueError("an interval must contain at least one risk")
        return self

    def contains(self, risk: Fraction) -> bool:
        return (self.lower < risk or self.left_closed and risk == self.lower) and (
            risk < self.upper or self.right_closed and risk == self.upper
        )


class Rule(Record):
    intervals: tuple[Interval, ...] = Field(min_length=1)
    actions_alone: tuple[Text, ...]
    referral: Text
    status: Literal["designed-held", "constructed"]

    @model_validator(mode="after")
    def partition(self):
        if tuple(sorted(self.intervals, key=lambda i: (i.lower, i.upper))) != self.intervals:
            raise ValueError("risk intervals must be ordered")
        ends = sorted(
            {Fraction(0), Fraction(1)}
            | {x for interval in self.intervals for x in (interval.lower, interval.upper)}
        )
        probes = ends + [(a + b) / 2 for a, b in zip(ends, ends[1:])]
        if any(sum(i.contains(p) for i in self.intervals) != 1 for p in probes):
            raise ValueError("risk intervals must cover [0, 1] once, including every boundary")
        if not set(self.actions_alone) <= {i.action for i in self.intervals}:
            raise ValueError("an autonomous action must occur in the rule")
        return self

    def action_at(self, risk: Fraction) -> str:
        if not 0 <= risk <= 1:
            raise ValueError("risk must lie in [0, 1]")
        return next(i.action for i in self.intervals if i.contains(risk))


class Cap(Record):
    action: Text
    value: Fraction = Field(gt=0)
    unit: Text
    dimension: Literal["money", "sessions", "bookings"]
    inclusive: bool
    basis: Text


class Cell(Record):
    question: Text
    answer: Text
    kind: Literal["same", "refilled", "redesigned"]

    @model_validator(mode="after")
    def short(self):
        if len(self.answer.split()) > 12:
            raise ValueError("table answers may contain at most 12 words")
        return self


class DecisionPacket(Record):
    domain: Text
    form: Literal["refund", "copy", "redesign"]
    boundary: dict[Text, Text]
    classes: tuple[DecisionClass, ...] = Field(min_length=1)
    age_limits: dict[Text, float]
    version_checks: tuple[Text, ...]
    rule: Rule | None
    decision_unit: Text
    recheck_at_execution: bool
    caps: tuple[Cap, ...]
    # Raw transport rows are parsed by the real LaunchRow, never a local substitute.
    requirement: tuple[dict[str, JsonValue], ...]
    cells: tuple[Cell, ...] = Field(min_length=7, max_length=7)
    label_windows: dict[Text, float]

    @model_validator(mode="after")
    def answers(self):
        if set(self.boundary) != {
            "permitted",
            "forbidden",
            "approval",
            "reversal",
            "ceiling",
            "logging",
        }:
            raise ValueError("the boundary needs its six fields")
        if any(seconds < 0 for seconds in self.age_limits.values()):
            raise ValueError("read ages must be nonnegative")
        if set(self.age_limits) & set(self.version_checks):
            raise ValueError("a source cannot be aged and version checked")
        if self.decision_unit != "request" and not any(
            self.decision_unit.startswith(prefix) and len(self.decision_unit) > len(prefix)
            for prefix in ("batch:", "reserve:")
        ):
            raise ValueError("decision unit must be request, batch:resource or reserve:resource")
        if len({c.question for c in self.cells}) != 7:
            raise ValueError("the seven questions must be distinct")
        return self


def strip(requirement: BaseModel) -> tuple[dict[str, JsonValue], ...]:
    """Keep only chapter 39's rows; bundle, gate day and signatures stay out."""
    raw = requirement.model_dump(mode="json")
    return tuple(raw["rows"])


def map_requirement_roles(rows, roles) -> tuple[dict[str, JsonValue], ...]:
    """Map transport roles; chapter 39 still validates canonical authority roles."""
    return tuple({
        **row,
        "records": [{**need, "signer_role": roles[need["signer_role"]]}
                    for need in row["records"]],
        "reversals": [{**condition, "owner_roles": [roles[role] for role in condition["owner_roles"]]}
                      for condition in row["reversals"]],
    } for row in rows)


def copy_refund(facts: DomainFacts, refund: DecisionPacket) -> DecisionPacket:
    """Map one supplied refund requirement, without retuning its numeric answers."""
    if refund.domain != "refund" or refund.form != "refund":
        raise ValueError("copy_refund requires the refund packet")
    mapped = facts.role_map
    rule = None
    if facts.actions_on_score:
        if refund.rule is None:
            raise ValueError("the supplied refund packet lacks its designed rule")
        rule = Rule(
            intervals=tuple(
                Interval.model_validate(
                    {
                        **i.model_dump(),
                        "action": mapped[i.action],
                    }
                )
                for i in refund.rule.intervals
            ),
            actions_alone=tuple(mapped[a] for a in refund.rule.actions_alone),
            referral=mapped[refund.rule.referral],
            status="designed-held",
        )
    raw = refund.model_dump(mode="json")
    raw.update(
        domain=facts.domain,
        form="copy",
        classes=[
            DecisionClass(
                name="copied execution",
                action=mapped["refund.execute"],
                level=4,
                above_role="reviewer",
                condition="copied refund checks pass",
            ).model_dump(mode="json")
        ],
        age_limits={
            source.name: refund.age_limits[source.refund_role]
            for source in facts.sources
            if not source.version_checked
        },
        version_checks=[source.name for source in facts.sources if source.version_checked],
        rule=rule.model_dump(mode="json") if rule else None,
        caps=[{**cap.model_dump(mode="json"), "action": mapped[cap.action]} for cap in refund.caps],
        label_windows=facts.label_windows,
        requirement=map_requirement_roles(refund.requirement, facts.roles),
    )
    return DecisionPacket.model_validate(raw)


def load_packet(path: str | Path) -> DecisionPacket:
    return DecisionPacket.model_validate_json(Path(path).read_text())


def dump_packet(packet: DecisionPacket, path: str | Path) -> None:
    require_inventory(packet.__dict__, DecisionPacket.model_fields, "packet field")
    validated = DecisionPacket.model_validate(packet.model_dump())
    Path(path).write_text(validated.model_dump_json(indent=2) + "\n")
