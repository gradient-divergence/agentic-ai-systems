"""Typed, constructed facts. Rates describe decision-relevant changes."""

from collections import Counter
from collections.abc import Hashable, Iterable
from fractions import Fraction
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Text = Annotated[str, Field(min_length=1, strict=True)]
Dimension = Literal["money", "sessions", "bookings"]


def require_inventory(actual: Iterable[Hashable], required: Iterable[Hashable], name: str) -> None:
    counts = Counter(iter(actual))
    expected = set(required)
    missing = sorted(expected - counts.keys(), key=repr)
    extra = sorted(counts.keys() - expected, key=repr)
    duplicate = sorted((item for item, count in counts.items() if count > 1), key=repr)
    if missing or extra or duplicate:
        raise ValueError(
            f"{name} inventory: missing {missing}; extra {extra}; duplicate {duplicate}"
        )


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class Source(Record):
    name: Text
    refund_role: Text
    observed_at: Text
    rate_per_hour: float | None = Field(default=None, ge=0)
    basis: Text
    version_checked: bool = False

    @model_validator(mode="after")
    def currency(self):
        if self.version_checked == (self.rate_per_hour is not None):
            raise ValueError("a source needs a rate or a version check, exclusively")
        return self


class Action(Record):
    name: Text
    role: Text
    effect: Literal["read", "write"]
    reversible: bool
    undo_seconds: float | None = Field(default=None, ge=0)
    harm_unit: Text
    harm_dimension: Dimension
    largest_loss: Fraction | None = Field(default=None, ge=0)
    basis: Text

    @model_validator(mode="after")
    def reversal(self):
        if self.reversible != (self.undo_seconds is not None):
            raise ValueError("reversible actions need an undo time; irreversible ones have none")
        return self


class SharedResource(Record):
    name: Text
    capacity: Fraction = Field(gt=0)
    unit: Text
    actions: tuple[Text, ...] = Field(min_length=1)
    scope: Literal["order", "domain"]
    requirement: Literal["limit", "allocation"]
    basis: Text


class CostMatrix(Record):
    outcomes: tuple[Text, Text]
    entries: dict[Text, tuple[Fraction, Fraction]]
    unit: Text
    basis: Text

    @model_validator(mode="after")
    def costs(self):
        if len(set(self.outcomes)) != 2 or len(self.entries) < 2:
            raise ValueError("two distinct outcomes and at least two actions required")
        if any(cost < 0 for pair in self.entries.values() for cost in pair):
            raise ValueError("costs must be nonnegative")
        return self


class Clock(Record):
    name: Text
    kind: Literal["read", "review", "undo", "outcome", "window"]
    seconds: float | None = Field(default=None, gt=0)
    basis: Text

    @model_validator(mode="after")
    def never(self):
        if self.seconds is None and self.kind != "undo":
            raise ValueError("only an irreversible action has no finite clock")
        return self


class DomainFacts(Record):
    domain: Text
    constructed: Literal[True]
    sources: tuple[Source, ...] = Field(min_length=1)
    actions: tuple[Action, ...] = Field(min_length=1)
    actions_on_score: tuple[Text, ...] = ()
    cost_matrix: CostMatrix | None = None
    base_rate: Fraction | None = Field(default=None, gt=0, lt=1)
    tolerance: float = Field(gt=0, lt=1)
    shared_resources: tuple[SharedResource, ...] = ()
    roles: dict[Text, Text]
    role_map: dict[Text, Text]
    label_windows: dict[Text, float]
    clocks: tuple[Clock, ...]
    planted_checks: tuple[Literal["clock", "coupling", "rule"], ...] = ()
    notes: tuple[Text, ...] = ()

    @model_validator(mode="after")
    def references(self):
        if set(self.roles) != {"decision_owner", "release_owner", "reviewer", "named_approver"}:
            raise ValueError("all four authority roles are required")
        names = [action.name for action in self.actions]
        if len(set(names)) != len(names) or len({s.name for s in self.sources}) != len(
            self.sources
        ):
            raise ValueError("action and source names must be unique")
        if not set(self.actions_on_score) <= set(names):
            raise ValueError("scored actions must be in the domain's boundary")
        if bool(self.actions_on_score) != (self.cost_matrix is not None):
            raise ValueError("scored actions require their cost matrix")
        if self.cost_matrix and not set(self.actions_on_score) <= self.cost_matrix.entries.keys():
            raise ValueError("every permitted scored action needs costs")
        if any(not set(r.actions) <= set(names) for r in self.shared_resources):
            raise ValueError("a resource names an unknown consuming action")
        if len({r.name for r in self.shared_resources}) != len(self.shared_resources):
            raise ValueError("resource names must be unique")
        if any(seconds <= 0 for seconds in self.label_windows.values()):
            raise ValueError("label windows must be positive")
        return self


def load_facts(path: str | Path) -> DomainFacts:
    return DomainFacts.model_validate_json(Path(path).read_text())
