"""One reproducible solve, with gaps computed from its value and certified bound."""

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from hashlib import sha256
from pathlib import Path

import scipy

from .certificate import gaps
from .inputs import INPUTS_VERSION, TOL


@dataclass(frozen=True)
class Assignment:
    request_id: str
    person: str
    start: datetime
    end: datetime

    def to_dict(self):
        return dict(
            request_id=self.request_id,
            person=self.person,
            start=self.start.isoformat(),
            end=self.end.isoformat(),
        )

    @classmethod
    def from_dict(cls, row):
        return cls(
            row["request_id"],
            row["person"],
            datetime.fromisoformat(row["start"]),
            datetime.fromisoformat(row["end"]),
        )


def queue_inputs(queue, minutes):
    rows = [
        {
            **asdict(d),
            "allowed": list(d.allowed),
            "at": d.at.isoformat(),
            "plan_at": d.plan_at.isoformat(),
        }
        for d in queue
    ]
    return dict(queue=rows, minutes=minutes, inputs_version=INPUTS_VERSION)


def queue_fingerprint(queue, minutes):
    text = json.dumps(queue_inputs(queue, minutes), sort_keys=True, separators=(",", ":"))
    return sha256(text.encode()).hexdigest()


@dataclass
class SolveRecord:
    queue_fingerprint: str
    budget: int
    status: str
    plan: list[Assignment]
    value: float
    bound: float
    multipliers: list[float]
    certificate: dict
    plan_check: list[str]
    inputs_version: str = INPUTS_VERSION
    solver_name: str = "scipy.optimize.milp/linprog (bundled HiGHS)"
    solver_version: str = scipy.__version__
    tolerance: float = TOL
    secondary_status: str = "not requested"
    inputs: dict = field(default_factory=dict)
    absolute_gap: float = field(init=False)
    relative_gap: float | str = field(init=False)
    relative_gap_reason: str | None = field(init=False)

    def __post_init__(self):
        self.absolute_gap, self.relative_gap, self.relative_gap_reason = gaps(
            self.value, self.bound
        )

    def to_dict(self, printed=False):
        result = asdict(self)
        result["plan"] = [a.to_dict() for a in self.plan]
        if printed:
            result.pop("solver_name")
            result.pop("solver_version")
        return result

    def write_json(self, path):
        Path(path).write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n"
        )

    @classmethod
    def from_dict(cls, row):
        result = dict(row)
        expected = [
            result.pop(key) for key in ("absolute_gap", "relative_gap", "relative_gap_reason")
        ]
        result["plan"] = [Assignment.from_dict(a) for a in row["plan"]]
        record = cls(**result)
        if expected != [record.absolute_gap, record.relative_gap, record.relative_gap_reason]:
            raise ValueError("stored gaps do not recompute")
        return record
