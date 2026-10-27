"""Evidence named before judging; later components contribute their own rows."""

from dataclasses import dataclass

from refund_agent.release import extensions


@dataclass(frozen=True)
class Requirement:
    component: str
    items: tuple[str, ...]
    evidence: tuple[str, ...]
    description: str


REQUIREMENTS = (
    Requirement(
        "model", ("scripted_model",), ("shadow_comparison",), "release and shadow comparison"
    ),
    Requirement(
        "model",
        ("risk_score", "risk_threshold"),
        ("selection", "calibration", "signed_cost_matrix", "threshold_experiment", "rebuild"),
        "selection, current calibration, signed costs, experiment run or signed decline, rebuild",
    ),
    Requirement("prompt", ("*",), ("regression",), "passing regression suite"),
    Requirement(
        "tools", ("agent", "agent_code", "core"), (), "release suite including safety cases"
    ),
    Requirement("tools", ("tool_contracts",), ("tool_contract_tests",), "tool contract tests"),
    Requirement(
        "tools",
        ("data_contracts",),
        ("data_contract_tests", "source_notice"),
        "source tests and notice",
    ),
    Requirement(
        "tools", ("event_time",), ("replay_tests", "late_scan_tests"), "replay and late-scan tests"
    ),
    Requirement(
        "tools",
        ("runtime", "adapter"),
        ("runtime_swap",),
        "runtime swap including test_runtime_swap.py",
    ),
    Requirement("tools", ("verifier",), ("verifier_arms",), "matched-case verifier arms"),
    Requirement(
        "tools", ("environment",), ("environment_release",), "suite rerun in the new environment"
    ),
    Requirement(
        "tools", ("components",), ("component_tests",), "tests for changed selected components"
    ),
    Requirement(
        "policy",
        ("versions", "register_snapshot"),
        ("policy_change_cases", "support", "policy_notice"),
        "new version's policy-change cases, support and policy-owner notice",
    ),
    Requirement("policy", ("offer_rule",), ("offer_evaluation",), "offer evaluation record"),
    Requirement(
        "index",
        ("*",),
        ("frozen_questions", "dated_recall"),
        "no frozen question loses its clause; dated recall",
    ),
)
ALWAYS = ("release", "policy_coverage", "release_owner")


def required(changes, *, rows=REQUIREMENTS, candidate=None, to_stage="shadow"):
    rows = tuple(rows)
    if candidate is not None:
        for additions in extensions.contributions(candidate, "promotion_requirements").values():
            rows += tuple(Requirement(**row) for row in additions)
    result = list(ALWAYS)
    for change in changes:
        matches = [
            row
            for row in rows
            if row.component == change["component"]
            and ("*" in row.items or change["item"] in row.items)
        ]
        if not matches:
            raise ValueError("changed bundle item has no requirement row")
        for row in matches:
            result.extend(row.evidence)
    if to_stage == "canary":
        result.extend(("shadow_window", "canary_plan", "rehearsal"))
    return tuple(dict.fromkeys(result))
