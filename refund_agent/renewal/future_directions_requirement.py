"""A frozen pre-evidence requirement; missing incumbent admission fails closed."""

from datetime import datetime

from pydantic import Field

from refund_agent.canary.cloud_native_deployment_plan import has_role
from refund_agent.contracts import Record
from refund_agent.data.history import END_AT
from refund_agent.selection import fingerprint


class BundleIdentity(Record):
    """Read chapter 33's bundle hash and its separate chapter 29 cross-reference."""

    bundle_fingerprint: str = Field(min_length=1)
    candidate_fingerprint: str = Field(min_length=1)


class ReplacementRequirement(Record):
    incumbent_bundle_fingerprint: str | None = Field(default=None, min_length=1)
    incumbent_candidate_fingerprint: str = Field(min_length=1)
    candidate_bundle_fingerprint: str | None = Field(default=None, min_length=1)
    candidate_fingerprint: str | None = Field(default=None, min_length=1)
    policy_version: str = Field(min_length=1)
    selection_fingerprint: str = Field(min_length=1)
    sealed_receipt: str = Field(min_length=1)
    score_fingerprint: str = Field(min_length=1)
    generator: str = "history builders forward; disjoint indices >= 36000"
    seed: int = 7
    paid_per_window: int = 2000
    maturity_days: int = 30
    threshold: float = 0.080
    review_cents: int = 400
    cost_matrix: str = "wrong payment: amount; review: $4; genuine denial: $20"
    cost_signer: str = "morgan"
    calibration: str = "mean prediction inside inclusive 95% Wilson interval"
    cost_rule: str = "candidate minus incumbent < 0; 95% upper end < 0; tie keeps incumbent"
    wrongful_denials: str = "no more than incumbent on the same requests (D6)"
    capacity: str = "Human Oversight and Risk Management check_design; added blocking tier 1 referrals at five minutes"
    release_required: bool = True
    failure_criterion: str = "any failed check keeps incumbent"
    attempts: int = 2
    rebuild_on: str = "calibration alone"
    reversal_owner: str = "morgan"
    reversal_target: str | None = Field(default=None, min_length=1)
    reversal_mechanism: str = (
        "persisted watch and MLOps and ML Engineering rollback; budget callback withdraws tier 1"
    )
    reversal_signals: tuple[str, ...] = ("watch.calibration.confirmed", "budget_spent")
    experiment: str = "signed decline"
    experiment_assumption: str = "a review is always right; no review loss beyond $4"
    experiment_signer: str = "morgan"
    signed_by: str = "morgan"
    signer_role: str = "decision_owner"
    signed_day: float = 0.0
    first_request_day: float = Field(gt=0)

    @property
    def fingerprint(self):
        return fingerprint(self.model_dump(mode="json"))

    @property
    def cost_fingerprint(self):
        return fingerprint(
            {
                "matrix": self.cost_matrix,
                "signer": self.cost_signer,
                "review_cents": self.review_cents,
                "signed_day": self.signed_day,
            }
        )

    def as_table(self):
        rows = [
            (
                "Incumbent",
                "bundle "
                + (self.incumbent_bundle_fingerprint or "unavailable")[:16]
                + "; candidate "
                + self.incumbent_candidate_fingerprint[:16]
                + "; stage-6 admission must bind both",
            ),
            (
                "Candidate",
                "selected trees, map and threshold; bundle "
                + (
                    self.candidate_bundle_fingerprint[:16]
                    if self.candidate_bundle_fingerprint
                    else "unavailable"
                )
                + "; candidate "
                + (self.candidate_fingerprint or "unavailable")[:16]
                + "; text screen kept",
            ),
            (
                "Cases",
                f"first {self.paid_per_window:,} incumbent-paid tier 1 requests after opening; disjoint forward draws",
            ),
            ("Calibration", self.calibration),
            ("Cost", self.cost_rule + "; signed " + self.cost_fingerprint[:16]),
            ("Counter-measures", self.wrongful_denials + "; " + self.capacity),
            (
                "Cases checked",
                "exactly 2,000 unique selected paid requests; each outcome final after 30 days",
            ),
            ("Release", "signed passing record for this candidate and suite"),
            ("Failure criterion", self.failure_criterion),
            (
                "Rebuild",
                "once, after calibration alone; fit that window; judge the next; retain first failure",
            ),
            (
                "Reversal",
                f"failed month -> pending next 2,000 requests; matured paid confirmation clears or reverses; owner {self.reversal_owner}",
            ),
            (
                "Target and mechanism",
                "bundle "
                + (self.reversal_target or "unavailable")[:16]
                + "; candidate "
                + self.incumbent_candidate_fingerprint[:16]
                + "; "
                + self.reversal_mechanism,
            ),
            (
                "Budget",
                "spent -> immediate tier 1 shadow/person withdrawal and rollback; restore never renews authority",
            ),
            ("Not run", self.experiment + "; " + self.experiment_assumption),
            ("Policy", self.policy_version),
            ("Signer", f"{self.signed_by}, {self.signer_role}, day {self.signed_day:g}"),
        ]
        return "\n".join(
            ["| Requirement | Signed setting |", "|---|---|"]
            + [f"| {key} | {value} |" for key, value in rows]
        )


def validate(req, admitted_targets):
    req = ReplacementRequirement.model_validate(req.model_dump())
    if any(isinstance(v, str) and not v.strip() for v in req.model_dump().values()):
        raise ValueError("empty requirement field")
    if req.candidate_fingerprint is None:
        raise ValueError("candidate identity unavailable; requirement is not admissible")
    if req.candidate_bundle_fingerprint is None:
        raise ValueError("candidate bundle identity unavailable; requirement is not admissible")
    if req.signer_role != "decision_owner" or not has_role(req.signed_by, "decision_owner"):
        raise ValueError("requirement needs decision_owner signature")
    if (
        req.cost_signer != req.signed_by
        or req.experiment_signer != req.signed_by
        or (not has_role(req.reversal_owner, "decision_owner"))
    ):
        raise ValueError("matrix, decline and reversal need the decision owner")
    from demos.data_science_strategy_demo import EvidenceRequirement, check

    dated = EvidenceRequirement(
        step="risk replacement",
        claim="calibrated risk at lower paired cost",
        comparator=req.incumbent_candidate_fingerprint,
        cases=req.generator,
        measures=[req.calibration, req.cost_rule],
        threshold=f"{req.threshold:.3f}",
        signer=req.signed_by,
        signed_day=round(req.signed_day * 86400 * 10**6),
        reversal=req.reversal_mechanism,
    )
    # The date rule compares integers; retain sub-day ordering in microseconds.
    if check(dated, round(req.first_request_day * 86400 * 10**6)) or req.signed_day < 0:
        raise ValueError("signature must precede window 1's first request")
    if req.attempts != 2 or req.rebuild_on != "calibration alone":
        raise ValueError("one rebuild allowed only after calibration alone")
    if (req.paid_per_window, req.maturity_days, req.threshold, req.review_cents) != (
        2000,
        30,
        0.080,
        400,
    ):
        raise ValueError("signed cases, maturity, threshold and costs differ")
    if (
        not req.release_required
        or req.experiment != "signed decline"
        or (req.reversal_signals != ("watch.calibration.confirmed", "budget_spent"))
    ):
        raise ValueError("release, signed decline and both observable reversal signals required")
    if req.reversal_target != "person" and (
        req.incumbent_bundle_fingerprint is None
        or req.reversal_target != req.incumbent_bundle_fingerprint
        or not any(
            r.get("bundle_fingerprint") == req.incumbent_bundle_fingerprint
            and r.get("candidate_fingerprint") == req.incumbent_candidate_fingerprint
            and r.get("stage") == 6
            and r.get("status") == "admitted"
            for r in admitted_targets
        )
    ):
        raise ValueError("incumbent stage-6 admission record missing")
    return req


def at_day(day: float) -> datetime:
    from datetime import timedelta

    return END_AT + timedelta(days=day)
