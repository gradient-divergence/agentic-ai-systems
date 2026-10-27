"""The commission fixed before delegation, and the resulting offline selection record."""

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Literal


def fingerprint(value) -> str:
    """SHA-256 of a record's canonical JSON, excluding its own fingerprint."""
    text = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(text.encode()).hexdigest()


@dataclass(frozen=True)
class Commission:
    label_policy: str
    inputs: tuple[str, ...]
    windows: dict
    space: dict
    baseline: str = "boosted trees with map"
    measure: str = "raw validation log loss"
    budget: int = 20
    seed: int = 7
    threshold: float = 0.080
    review_cost: float = 4.0
    alpha: float = 0.05
    challengers: tuple[str, ...] = ("logistic regression", "search winner", "agent delivery")
    acceptance: str = "all checks pass; paired cost difference < 0 and its 95% upper bound < 0; tie keeps baseline"
    delegate_changes: str = (
        "search: declared family/settings; agent: pipeline and allowed derived inputs"
    )
    sha256: str = field(init=False)

    def __post_init__(self):
        if self.budget < 1 or not 0 < self.alpha < 1 or not self.challengers:
            raise ValueError("positive budget, probability alpha and named challengers required")
        object.__setattr__(self, "sha256", "")
        object.__setattr__(self, "sha256", fingerprint(self.payload()))

    def payload(self):
        return {k: v for k, v in asdict(self).items() if k != "sha256"}

    def intact(self):
        return self.sha256 == fingerprint(self.payload())


def decide(
    eligible: bool, cost_difference: float, interval: tuple[float, float]
) -> Literal["accepted", "sent back", "not adopted"]:
    """Apply the predeclared rule to candidate minus baseline, in dollars/request."""
    if not eligible:
        return "sent back"
    if cost_difference < 0 and interval[1] < 0:
        return "accepted"
    return "not adopted"


@dataclass(frozen=True)
class SelectionRecord:
    commission: dict
    receipt: str
    deliveries: tuple[dict, ...]
    checks: dict
    frozen: dict
    log: tuple[dict, ...]
    evidence: tuple[dict, ...]
    decisions: dict
    selected: str
    spent: str
    grants: str = "none; drafts only, a person executes"
    dependencies: tuple[str, ...] = (
        "calibration record",
        "tier threshold",
        "threshold experiment plan",
    )
    signer: str = "model owner; illustrative costs await the decision owner's signature"
    sha256: str = field(init=False)

    def __post_init__(self):
        object.__setattr__(self, "sha256", "")
        object.__setattr__(self, "sha256", fingerprint(self.payload()))

    def payload(self):
        return {k: v for k, v in asdict(self).items() if k != "sha256"}

    def intact(self):
        return self.sha256 == fingerprint(self.payload())
