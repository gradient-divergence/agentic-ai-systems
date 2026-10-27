"""A signed, immutable plan; the hash fixes membership before arrival."""

import json
from datetime import datetime
from hashlib import sha256
from pathlib import Path

from pydantic import model_validator

from refund_agent import contracts as c
from refund_agent.selection import fingerprint

OWNERS = Path(__file__).resolve().parents[1] / "release/owners.json"


def has_role(owner: str | None, role: str) -> bool:
    return any(
        row["id"] == owner and role in row["roles"]
        for row in json.loads(OWNERS.read_text())["owners"]
    )


def in_cohort(canary_id: str, requester_id: str, share: float = 0.05) -> bool:
    if not canary_id or not requester_id or not 0 < share < 1:
        raise ValueError("cohort requires identifiers and a share in (0, 1)")
    value = int.from_bytes(sha256(f"{canary_id}:{requester_id}".encode()).digest()[:8], "big")
    return value % 10000 < share * 10000


class CanaryPlan(c.Record):
    canary_id: c.Identity
    candidate_fingerprint: c.Identity
    suite_fingerprint: c.Identity
    rollback_target_fingerprint: c.Identity
    requirement_ref: c.Identity
    rehearsal_ref: c.Identity
    error_budget_ref: str = ""
    share: float = 0.05
    max_cents_per_payment: int = 4999
    worst_day_cents: int = 75000
    recheck_hours: int = 4
    time_box_days: int = 28
    cohort_rule: str = "SHA-256(canary_id:requester_id), first 8 bytes, modulo 10000"
    comparator: str = "tier 1 outside the cohort, same days, blind person decisions"
    stop_triggers: tuple[str, ...] = (
        "recheck", "correction", "untrusted_record", "owner", "recheck_overdue"
    )
    clean_count_rule: str = "settled; pending counts neither clean nor failed"
    offer_rule: str = "fixed"
    signed_by: str | None = None
    signed_at: datetime | None = None

    @model_validator(mode="after")
    def settings_and_signature(self):
        if (self.share, self.max_cents_per_payment, self.worst_day_cents,
                self.recheck_hours, self.time_box_days) != (0.05, 4999, 75000, 4, 28):
            raise ValueError("stage 5 requires the decided canary settings")
        if (self.signed_by is None) != (self.signed_at is None):
            raise ValueError("plan signature requires owner and time")
        if self.signed_by is not None and not has_role(self.signed_by, "decision_owner"):
            raise ValueError("plan signer must hold decision_owner")
        return self

    @property
    def daily_cap_count(self) -> int:
        return self.worst_day_cents // self.max_cents_per_payment

    @property
    def daily_cap_cents(self) -> int:
        return self.daily_cap_count * self.max_cents_per_payment

    @property
    def fingerprint(self) -> str:
        return fingerprint(self.model_dump(mode="json"))


def sign_plan(plan: CanaryPlan, owner: str, at: datetime) -> CanaryPlan:
    return c._validated(CanaryPlan, plan.model_copy(update={"signed_by": owner, "signed_at": at}))
