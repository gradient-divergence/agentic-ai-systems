"""Meaning and approved freshness, beyond the existing strict record types."""

from datetime import datetime
from typing import Literal

from pydantic import Field, model_validator

from refund_agent.contracts import Record

SPEC_LIMITS = {"order": 1800, "shipment": 1800, "policy": 2592000, "ledger": 0}


class FieldMeaning(Record):
    field: str
    unit: str
    meaning: str


class Budget(Record):
    limit_seconds: float = Field(ge=0, allow_inf_nan=False)
    basis: str = Field(min_length=1)
    approver_role: str | None
    at: datetime
    count: int | None = Field(default=None, strict=True, ge=0)
    exposure_hours: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    estimate: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    upper: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    chance_estimate: float | None = Field(default=None, ge=0, le=1)
    chance_upper: float | None = Field(default=None, ge=0, le=1)


class NoticeOfChange(Record):
    days: int = Field(strict=True, ge=0)
    basis: str


class SourceContract(Record):
    source: Literal["order", "shipment", "policy", "ledger"]
    version: str
    owner_role: str | None
    fields: tuple[FieldMeaning, ...]
    vocabularies: dict[str, tuple[str, ...]]
    invariants: tuple[str, ...]
    budget: Budget
    notice: NoticeOfChange
    failure_response: Literal["refer"] = "refer"

    @model_validator(mode="after")
    def within_specification(self):
        if self.budget.limit_seconds > SPEC_LIMITS[self.source]:
            raise ValueError("budget exceeds the specification's source limit")
        if self.version.split("@")[0] != self.source:
            raise ValueError("contract version must name its source")
        return self
