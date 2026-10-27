"""Contracts for the physical-goods refund example."""

from __future__ import annotations

import json
from copy import deepcopy
from datetime import datetime, timezone
from enum import Enum, IntEnum
from pathlib import Path
from types import MappingProxyType as _ReadOnly
from typing import Annotated, Literal, TypeVar
from typing import Callable as _Callable
from typing import ClassVar as _ClassVar
from typing import cast as _cast

from pydantic import (
    BaseModel,
    Field,
    ModelWrapValidatorHandler,
    ValidationInfo,
    field_validator,
    model_validator,
)

# All amounts use one currency, USD, represented as integer cents.
Cents = Annotated[int, Field(strict=True, gt=0)]
Nonnegative = Annotated[int, Field(strict=True, ge=0)]
Identity = Annotated[str, Field(strict=True, min_length=1, pattern=r"^\S+$")]
AuditEvent = Literal["decision", "approval", "intent", "confirmation"]
_T = TypeVar("_T")
_Model = TypeVar("_Model", bound=BaseModel)


def _detach(value: _T) -> _T:
    return deepcopy(value)


def _validated(cls: type[_Model], value: _Model) -> _Model:
    validate = getattr(cls, "_entry", cls.__pydantic_validator__.validate_python)
    return _cast(_Model, validate(value))


def _utc_time(value: object) -> datetime:
    if type(value) is not datetime or value.utcoffset() is None:
        raise ValueError("timestamps must be exact timezone-aware datetimes")
    return value.astimezone(timezone.utc).replace()


class Record(BaseModel):
    model_config = {"extra": "forbid", "frozen": True, "revalidate_instances": "always"}
    _entry: _ClassVar[_Callable[[object], BaseModel]]

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: object) -> None:
        if cls.__pydantic_complete__:
            cls._entry = cls.__pydantic_validator__.validate_python

    @model_validator(mode="wrap")
    @classmethod
    def exact_record(cls, value: object, handler: ModelWrapValidatorHandler[Record]) -> Record:
        if isinstance(value, cls) and type(value) is not cls:
            raise ValueError("records must have their exact declared type")
        return handler(value)

    @field_validator("*", mode="before")
    def enum_types(cls, value: object, info: ValidationInfo) -> object:
        if (
            isinstance(value, Enum)
            and type(value) is not cls.model_fields[info.field_name or ""].annotation
        ):
            raise ValueError("enum type must match its declared field")
        return value

    @field_validator("*", mode="before")
    def exact_values(cls, value: object) -> object:
        if isinstance(value, BaseModel):
            return value
        if type(value) not in (
            str,
            int,
            float,
            bool,
            datetime,
            list,
            tuple,
            dict,
            type(None),
            Outcome,
            Tier,
            Phase,
        ):
            raise ValueError("values must have exact built-in or declared enum types")
        if isinstance(value, (list, tuple)):
            for item in value:
                cls.exact_values(item)
        elif type(value) is dict:
            for key, item in value.items():
                cls.exact_values(key)
                cls.exact_values(item)
        return value

    @field_validator(
        "at",
        "observed_at",
        "purchased_at",
        "effective_at",
        "approved_at",
        "created_at",
        "execution_at",
        mode="before",
        check_fields=False,
    )
    def timestamp_input(cls, value: object, info: ValidationInfo) -> object:
        if value is None or (info.mode == "json" and type(value) is str):
            return value
        return _utc_time(value)

    @field_validator("*")
    def utc_dates(cls, value: object) -> object:
        if isinstance(value, datetime):
            return _utc_time(value)
        return value


class Outcome(str, Enum):
    APPROVE = "approve"
    DENY = "deny"
    REFER = "refer"


class Tier(IntEnum):
    AGENT = 1
    REVIEWER = 2
    NAMED_APPROVER = 3


# region book:fnd.agentic-loop.phases-01
class Phase(str, Enum):
    ORDER = "order"
    SHIPMENT = "shipment"
    POLICY = "policy"
    ASSESS = "assess"
    PROPOSE = "propose"
    APPROVAL = "approval"
    EXECUTE = "execute"
    AUDIT = "audit"
    DONE = "done"


PHASE_TO_TOOL = _ReadOnly(
    {
        Phase.ORDER: "order.lookup",
        Phase.SHIPMENT: "shipment.status",
        Phase.POLICY: "policy.retrieve",
        Phase.PROPOSE: "refund.propose",
        Phase.APPROVAL: "approval.request",
        Phase.EXECUTE: "refund.execute",
    }
)
# endregion book:fnd.agentic-loop.phases-01


class Request(Record):
    request_id: str
    requester_id: Identity
    preparer_id: Identity = "refund-agent"
    order_reference: str
    at: datetime
    amount_cents: Cents
    reason: str
    text: str = ""
    failure: str | None = None  # Scenario label, never an authorization signal.


class Item(Record):
    name: str
    quantity: Annotated[int, Field(strict=True, gt=0)]
    unit_cents: Cents


class PriorRefund(Record):
    payment_id: Identity
    amount_cents: Cents
    at: datetime


class Order(Record):
    order_id: str
    reference: str
    purchased_at: datetime
    observed_at: datetime
    items: list[Item]
    paid_cents: Cents
    payment_id: str
    prior_refunds: list[PriorRefund] = Field(default_factory=list)


class ShipmentEvent(Record):
    at: datetime
    status: str


class Shipment(Record):
    order_id: str
    observed_at: datetime
    events: list[ShipmentEvent]


class Policy(Record):
    policy_id: Literal["refund_cap_v4"] = "refund_cap_v4"
    version: str
    effective_at: datetime
    return_days: Nonnegative
    cap_cents: Cents
    text: str
    named_approver: Identity = "morgan"

    @model_validator(mode="after")
    def dated_version(self) -> Policy:
        if self.version != self.effective_at.date().isoformat():
            raise ValueError("policy version must be its effective date")
        return self


class PolicySnapshot(Record):
    request_id: str
    version: str
    observed_at: datetime


class Decision(Record):
    request_id: str
    outcome: Outcome
    amount_cents: Cents
    tier: Tier
    reason: str
    policy_citation: str | None = None
    executed: bool = False

    @model_validator(mode="after")
    def consistent_outcome(self) -> Decision:
        if self.outcome == Outcome.DENY and not self.policy_citation:
            raise ValueError("denials require a policy citation")
        if self.executed and self.outcome != Outcome.APPROVE:
            raise ValueError("only approved decisions can execute")
        return self


class Approval(Record):
    approval_id: str
    request_id: str
    order_id: str
    proposal_id: str
    amount_cents: Cents
    reason: str
    policy_version: str
    reviewer: Identity
    role: str = "reviewer"
    approved_at: datetime


class Reviewer(Record):
    reviewer_id: Identity
    name: str
    role: Literal["reviewer", "named_approver"]


class State(Record):
    model_config = {"extra": "forbid", "frozen": False}
    request: Request
    trace_id: str
    phase: Phase = Phase.ORDER
    order: Order | None = None
    refunded_cents: Nonnegative = 0  # Display snapshot derived from the payment ledger.
    shipment_events: list[ShipmentEvent] = Field(default_factory=list)
    policy: Policy | None = None
    policy_version: str | None = None
    freshness: dict[str, float] = Field(default_factory=dict)
    risk_score: float | None = None
    decision: Decision | None = None
    approvals: list[Approval] = Field(default_factory=list)
    proposal_id: str | None = None
    execution_at: datetime | None = None
    failure_detail: str | None = None
    receipt: Receipt | None = None  # Display snapshot; tools derive it from the ledger.


class Evidence(Record):
    order: Order
    shipment_events: list[ShipmentEvent]
    policy: Policy
    freshness: dict[str, float]
    risk_score: float
    trace_id: str


class Receipt(Record):
    receipt_id: str
    request: Request | None = None  # Imported prior payments have no request or evidence.
    order_id: str
    amount_cents: Cents
    idempotency_key: str
    proposal_id: str | None = None
    approvals: tuple[Approval, ...] = ()
    at: datetime
    evidence: Evidence | None = None
    decision: Decision | None = None


class Proposal(Record):
    proposal_id: str
    request_id: str
    request: Request
    amount_cents: Cents
    reason: str
    created_at: datetime
    evidence: Evidence


class ReviewItem(Record):
    review_id: str
    proposal: Proposal | None = Field(default=None, exclude_if=lambda value: value is None)
    request: Request | None = Field(default=None, exclude_if=lambda value: value is None)
    named_approver: Identity | None = Field(default=None, exclude_if=lambda value: value is None)
    required_role: str
    policy_reason: str
    open: bool = True

    @model_validator(mode="after")
    def review_packet(self) -> ReviewItem:
        if self.proposal is not None:
            if self.request is not None or self.named_approver is not None:
                raise ValueError("review must contain exactly one packet")
        elif (
            self.request is None
            or self.request.amount_cents <= 50000
            or self.named_approver is None
            or self.required_role != "named_approver"
        ):
            raise ValueError("request-only review requires a tier 3 request and named approver")
        return self


class AuditRecord(Record):
    audit_id: str
    trace_id: str
    at: datetime
    decision: Decision
    freshness: dict[str, float]
    risk_score: float | None
    approval_ids: list[str]
    event: AuditEvent = "decision"
    proposal_id: str | None = None
    receipt_id: str | None = None
    money_moved: bool = False
    paid_amount_cents: Nonnegative = 0
    failure_detail: str | None = None


class ToolArguments(Record):
    order_id: str | None = None
    amount_cents: Cents | None = None
    reason: str = ""
    idempotency_key: str | None = None
    approval_id: str | None = None


class ToolCall(Record):
    name: str
    arguments: ToolArguments = Field(default_factory=ToolArguments)


class ToolResult(Record):
    ok: Annotated[bool, Field(strict=True)]
    reason: str = ""
    age_seconds: Annotated[float, Field(allow_inf_nan=False)] | None = None
    order: Order | None = None
    shipment: Shipment | None = None
    policy: Policy | None = None
    record_id: str | None = None


class TraceStep(Record):
    index: int
    phase: Phase
    call: ToolCall | None
    reply: Decision | None = None
    result: ToolResult
    state: State
    rejected_reply: str | None = None


class Trace(Record):
    trace_id: str
    steps: list[TraceStep]
    decision: Decision
    receipt: Receipt | None = None
    failure_detail: str | None = None

    def write(self, path: Path) -> None:
        write_json(path, self)

    @classmethod
    def read(cls, path: Path) -> Trace:
        return cls.model_validate_json(path.read_text(encoding="utf-8"))


State.model_rebuild()
State._entry = State.__pydantic_validator__.validate_python


def write_json(path: Path, record: BaseModel) -> None:
    record = _validated(type(record), record)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(record.model_dump(mode="json"), sort_keys=True, indent=2, allow_nan=False)
    path.write_bytes((text + "\n").encode("utf-8"))
