"""The weekly controller changes samples and requests staff; no tier mutation."""

from dataclasses import dataclass, field
from datetime import datetime
from math import isfinite


@dataclass
class Controller:
    recheck_share: float = .25
    blind_share: float = .10
    changes: list[dict] = field(default_factory=list)

    def __post_init__(self):
        if self.recheck_share not in (.25, .10) or self.blind_share not in (.10, .05):
            raise ValueError("controller samples must be the signed settings or floors")

    def weekly(self, utilization: float, at: datetime) -> tuple[dict, ...]:
        if not isfinite(utilization) or utilization < 0 or at.utcoffset() is None:
            raise ValueError("controller needs a finite load and an aware clock")
        if self.changes and at <= self.changes[-1]["at"]:
            raise ValueError("weekly observations must advance")
        if utilization <= .8:
            return ()
        added = []
        for name, floor in (("recheck_share", .10), ("blind_share", .05)):
            before = self.recheck_share if name == "recheck_share" else self.blind_share
            if before > floor:
                row = {"at": at, "kind": name, "before": before, "after": floor}
                added.append(row)
                if name == "recheck_share":
                    self.recheck_share = floor
                else:
                    self.blind_share = floor
        added.append({"at": at, "kind": "staffing_request", "utilization": utilization,
                      "owner_role": "decision_owner"})
        self.changes.extend(added)
        return tuple(added)
