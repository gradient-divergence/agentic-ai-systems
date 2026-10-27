"""The seven questions and the requirement signed before their comparison."""

from typing import Annotated, Literal

from pydantic import BaseModel, Field, StringConstraints

from refund_agent.selection import fingerprint

Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
Day = Annotated[float, Field(ge=0, allow_inf_nan=False, strict=True)]
Role = Literal["decision_owner", "release_owner", "reviewer", "named_approver"]
BundleRule = Literal["strict", "covered", "any"]
NeedDefinition = tuple[str, str, BundleRule, Role, bool]

# Final chapter titles, keyed by manuscript section rather than draft filename.
PRODUCER_CHAPTERS = {
    "sec-what-is-agentic-ai": ("What Is Agentic AI?", ("boundary",)),
    "sec-action-guardrails": ("Action, Guardrails, and Policies", ("loss_bound",)),
    "sec-security-privacy": (
        "Security and Privacy", ("scope", "injection_suite", "revocation_drill"),
    ),
    "sec-human-oversight-risk": (
        "Human Oversight and Risk Management", ("routing", "review_design"),
    ),
    "sec-cloud-native-deployment": (
        "Deployment and Rollback", ("exposure", "shadow_window", "stop", "rehearsal"),
    ),
    "sec-data-systems": ("Data Systems for Agentic AI", ("data_contracts",)),
    "sec-batch-stream": ("Batch and Streaming Pipelines", ("authority_table",)),
    "sec-audits-explanations": (
        "Outcome Audits and Decision Explanations", ("segment_audit", "explanation"),
    ),
    "sec-data-science-strategy": (
        "Decision Strategy and Evidence Requirements", ("evidence_requirements",),
    ),
    "sec-agent-evaluation": ("Agent Evaluation and Release Evidence", ("release",)),
    "sec-mlops-mle": ("MLOps and ML Engineering", ("promotion",)),
    "sec-observability-reliability": (
        "Observability and Reliability", ("error_budget", "incident", "incident_actions"),
    ),
    "sec-org-adoption": ("Organizational Adoption", ("rollout_plan", "rollout_step")),
    "sec-capstone-blueprint": (
        "Capstone: Design, Build, Deploy, Operate", ("comparison", "tier3"),
    ),
    "sec-case-studies-domain-agnostic": ("Domain-Agnostic Transfer Cases", ()),
    "sec-future-directions": ("Future Directions and System Renewal", ()),
}


class LaunchRecord(BaseModel):
    model_config = {"extra": "forbid", "frozen": True, "allow_inf_nan": False}

    @property
    def fingerprint(self) -> str:
        return fingerprint(self.model_dump(mode="json"))


class Period(LaunchRecord):
    start: Day
    end: Day


class RecordNeed(LaunchRecord):
    kind: Text
    required_verdict: Text
    bundle_rule: BundleRule
    signer_role: Role
    rerunnable: bool = False
    period: Period | None = None
    producer: Text = PRODUCER_CHAPTERS["sec-capstone-blueprint"][0]


class ReversalCondition(LaunchRecord):
    condition_id: Text
    text: Text
    withdraws: Literal["tier1", "tier2", "all", "launch"]
    returns_to: Text
    owner_roles: tuple[Role, ...] = Field(min_length=1)
    mechanism: Text


class LaunchRow(LaunchRecord):
    row_id: Text
    question: Text
    source: Text
    records: tuple[RecordNeed, ...] = Field(min_length=1)
    reversals: tuple[ReversalCondition, ...] = Field(min_length=1)


class CostMatrix(LaunchRecord):
    review_cents: int = Field(default=400, ge=0, strict=True)
    wrongful_denial_cents: int = Field(default=2000, ge=0, strict=True)
    wrong_payment_multiplier: int = Field(default=1, ge=0, strict=True)
    signed_by: Text
    signed_day: Day


class LaunchRequirement(LaunchRecord):
    rows: tuple[LaunchRow, ...] = Field(min_length=7, max_length=7)
    launched_bundle: Text
    comparison_period: Period
    comparison_rule: Literal[
        "better than each alternative on at least one business measure; worse on none"
    ] = "better than each alternative on at least one business measure; worse on none"
    cost_matrix: CostMatrix
    gate_day: Day
    signed_by: Text
    signed_day: Day

    def as_table(self) -> str:
        lines = ["row | question | records (bundle rule; signer)"]
        for row in self.rows:
            kinds = "; ".join(
                f"{r.kind} ({r.producer}; {r.bundle_rule}; {r.signer_role})"
                for r in row.records
            )
            lines.append(f"{row.row_id} | {row.question} | {kinds}")
        return "\n".join(lines)


def validate_row(row: LaunchRow, roles: dict[str, object]) -> tuple[str, ...]:
    """Domain-neutral row validation; chapter 40 supplies its own role holders."""
    reasons = []
    if len({r.kind for r in row.records}) != len(row.records):
        reasons.append("duplicate record kind in row")
    for need in row.records:
        if not roles.get(need.signer_role):
            reasons.append(f"{need.kind}: signer role has no holder")
        if need.kind == "comparison" and need.rerunnable:
            reasons.append("comparison is never rerunnable")
        if need.period is not None and need.period.end <= need.period.start:
            reasons.append(f"{need.kind}: empty or reversed period")
    for reversal in row.reversals:
        if any(not roles.get(role) for role in reversal.owner_roles):
            reasons.append(
                f"{reversal.condition_id}: reversal owner has no holder"
            )
    return tuple(reasons)


def validate(
    req: LaunchRequirement, roles: dict[str, object]
) -> LaunchRequirement:
    req = LaunchRequirement.model_validate(req.model_dump(mode="json"))
    reasons = [
        reason for row in req.rows for reason in validate_row(row, roles)
    ]
    if not holds_role(roles, "decision_owner", req.signed_by):
        reasons.append("requirement needs decision_owner")
    if req.signed_day >= req.gate_day:
        reasons.append("requirement signature must precede gate")
    if req.comparison_period.start <= req.signed_day:
        reasons.append("comparison must open after the signature")
    if req.comparison_period.end <= req.comparison_period.start:
        reasons.append("empty comparison period")
    if req.comparison_period.end > req.gate_day:
        reasons.append("comparison closes after gate")
    if len({row.row_id for row in req.rows}) != 7:
        reasons.append("seven distinct questions required")
    comparison = [n for n in req.rows[0].records if n.kind == "comparison"]
    if len(comparison) != 1 or comparison[0].period != req.comparison_period:
        reasons.append(
            "first row must name the exact signed comparison period"
        )
    elif (
        comparison[0].required_verdict,
        comparison[0].bundle_rule,
        comparison[0].rerunnable,
        comparison[0].signer_role,
    ) != ("better", "strict", False, "decision_owner"):
        reasons.append(
            "comparison requires better, strict, decision_owner and no rerun"
        )
    conditions = {}
    for row in req.rows:
        for condition in row.reversals:
            if (
                conditions.setdefault(condition.condition_id, condition)
                != condition
            ):
                reasons.append("conflicting reversal definitions")
    if (req.cost_matrix.signed_by, req.cost_matrix.signed_day) != (
        req.signed_by,
        req.signed_day,
    ):
        reasons.append("cost matrix must carry the requirement's signature")
    if reasons:
        raise ValueError("; ".join(reasons))
    return req


def holds_role(roles, role, owner) -> bool:
    holders = roles.get(role, ())
    return owner == holders if isinstance(holders, str) else owner in holders


def reversals() -> tuple[ReversalCondition, ...]:
    return (
        ReversalCondition(
            condition_id="forbidden_payment",
            text="tier 1 policy or age-limit breach",
            withdraws="tier1",
            returns_to="tier 1 shadow; a person decides",
            owner_roles=("decision_owner",),
            mechanism="observe.monitor:forbidden_payment",
        ),
        ReversalCondition(
            condition_id="budget_spent",
            text="tier 1 error budget spent",
            withdraws="tier1",
            returns_to="tier 1 shadow; new candidate needs a new canary",
            owner_roles=("decision_owner",),
            mechanism="observe.monitor:budget_spent",
        ),
        ReversalCondition(
            condition_id="tier2_unbound_approval",
            text="tier 2 outcome without bound review",
            withdraws="tier2",
            returns_to="a person decides; the agent drafts",
            owner_roles=("decision_owner",),
            mechanism="launch.scan:tier2_unbound_approval",
        ),
        ReversalCondition(
            condition_id="tier3_without_named_approver",
            text="tier 3 outcome without named approver",
            withdraws="all",
            returns_to=(
                "a person decides up to $500; named approver decides every "
                "request over $500, denials included"
            ),
            owner_roles=("release_owner",),
            mechanism="launch.scan:tier3_without_named_approver",
        ),
        ReversalCondition(
            condition_id="launch_gate_held",
            text="promotion or false record reopens gate and holds",
            withdraws="launch",
            returns_to="stage 6 under its own conditions",
            owner_roles=("decision_owner", "release_owner"),
            mechanism="promotion.registry:launch_gate",
        ),
    )


def refund_rows(period: Period) -> tuple[LaunchRow, ...]:
    conditions = reversals()
    definitions: tuple[tuple[str, str, tuple[NeedDefinition, ...]], ...] = (
        (
            "What can this agent do that a workflow cannot?",
            "intro:1759",
            (("comparison", "better", "strict", "decision_owner", False),),
        ),
        (
            "What is the smallest safe action scope?",
            "intro:1760",
            (
                ("boundary", "ok", "any", "decision_owner", False),
                ("scope", "pass", "strict", "decision_owner", False),
                ("routing", "pass", "strict", "decision_owner", False),
                ("tier3", "ok", "strict", "decision_owner", False),
            ),
        ),
        (
            "Which failures become public or expensive?",
            "intro:1761",
            (
                ("loss_bound", "ok", "any", "decision_owner", False),
                ("exposure", "pass", "covered", "release_owner", True),
                ("data_contracts", "pass", "covered", "release_owner", True),
                ("authority_table", "pass", "covered", "release_owner", True),
                ("injection_suite", "pass", "covered", "release_owner", True),
                ("revocation_drill", "pass", "covered", "release_owner", True),
                (
                    "segment_audit",
                    "no_breach",
                    "strict",
                    "decision_owner",
                    False,
                ),
                ("explanation", "pass", "strict", "decision_owner", False),
            ),
        ),
        (
            "What evidence would show that the agent is improving?",
            "data-science-strategy",
            (
                (
                    "evidence_requirements",
                    "signed",
                    "any",
                    "decision_owner",
                    False,
                ),
                ("release", "pass", "strict", "release_owner", False),
                ("shadow_window", "pass", "covered", "decision_owner", True),
                ("promotion", "promoted", "strict", "release_owner", False),
                ("error_budget", "unspent", "covered", "decision_owner", True),
            ),
        ),
        (
            "How do we turn incidents into evaluation cases?",
            "intro:1763",
            (
                ("incident", "resolved", "strict", "release_owner", False),
                ("incident_actions", "pass", "strict", "release_owner", False),
            ),
        ),
        (
            "What can a person override, and how fast?",
            "org-adoption",
            (
                ("review_design", "pass", "any", "decision_owner", False),
                ("rollout_plan", "signed", "strict", "decision_owner", False),
                ("rollout_step", "widen", "strict", "decision_owner", False),
                ("stop", "clean", "covered", "release_owner", False),
            ),
        ),
        (
            "Which rollback has actually been rehearsed?",
            "cloud-native-deployment",
            (("rehearsal", "pass", "strict", "release_owner", True),),
        ),
    )
    producers = {
        kind: title
        for title, kinds in PRODUCER_CHAPTERS.values()
        for kind in kinds
    }
    rows = tuple(
        LaunchRow(
            row_id=str(index),
            question=question,
            source=source,
            records=tuple(
                RecordNeed.model_validate(
                    dict(
                        kind=kind,
                        required_verdict=verdict,
                        bundle_rule=rule,
                        signer_role=role,
                        rerunnable=again,
                        period=period if kind == "comparison" else None,
                        producer=producers[kind],
                    )
                )
                for kind, verdict, rule, role, again in needs
            ),
            reversals=conditions,
        )
        for index, (question, source, needs) in enumerate(definitions, 1)
    )
    return rows


def refund_requirement(
    bundle: str,
    *,
    signed_day: float = 1,
    gate_day: float = 64,
    comparison_period: Period | None = None,
    owner: str = "morgan",
) -> LaunchRequirement:
    """Constructed settings, not a claim that an operating run completed."""
    period = comparison_period or Period(start=50, end=64)
    return LaunchRequirement(
        rows=refund_rows(period),
        launched_bundle=bundle,
        comparison_period=period,
        cost_matrix=CostMatrix(signed_by=owner, signed_day=signed_day),
        gate_day=gate_day,
        signed_by=owner,
        signed_day=signed_day,
    )
