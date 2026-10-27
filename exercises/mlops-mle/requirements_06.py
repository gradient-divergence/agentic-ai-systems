"""Exercise 6: complete evidence matching and the missing runtime requirement."""

from refund_agent.promotion.mlops_mle_requirements import ALWAYS as ALWAYS
from refund_agent.promotion.mlops_mle_requirements import Requirement

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


def required(changes):
    """Return unique evidence names: ALWAYS plus matching component/item rows.

    A row's '*' matches every item of that component. Preserve first occurrence.
    Refuse a changed item with no matching row.
    """
    raise NotImplementedError("match every change, including wildcard rows")
