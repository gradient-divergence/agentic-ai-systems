"""Reference solutions; numerical answers come from the given costs and clocks."""

import json
import math
from fractions import Fraction as F

from refund_agent.sources.data_systems_budget import budget_hours, stale_chance
from refund_agent.transfer.case_studies_domain_agnostic_check import optimal_rule
from refund_agent.transfer.case_studies_domain_agnostic_data import EXERCISES
from refund_agent.transfer.case_studies_domain_agnostic_facts import CostMatrix


def responses():
    matrix = CostMatrix(
        outcomes=("legitimate", "fraudulent"),
        entries={"clear": (F(0), F(600)), "refer": (F(5), F(95)), "hold": (F(20), F(5))},
        unit="dollars",
        basis="exercise 4 constructed costs",
    )
    band = optimal_rule(matrix, ("clear", "refer", "hold"), "refer", ("clear", "hold"))
    long_hold = CostMatrix.model_validate(
        {**matrix.model_dump(), "entries": {**matrix.entries, "hold": (F(365), F(5))}}
    )
    late = optimal_rule(long_hold, ("clear", "refer", "hold"), "refer", ("clear", "hold"))
    insurer = json.loads((EXERCISES / "insurer_05.json").read_text())
    return {
        "01": {"packet": ["a", "c", "e", "h"], "refund": ["b", "d", "f", "g"]},
        "02": {
            "triage": ("clock", "number"),
            "hospital": ("coupling", "unit of decision"),
            "bank": ("rule", "autonomous action"),
        },
        "03": {
            "budget_seconds": budget_hours(math.log(2) / 1.5, 0.02) * 3600,
            "chance_30_minutes": stale_chance(math.log(2) / 1.5, 0.5),
        },
        "04": {
            "clear_edge": str(band.intervals[0].upper),
            "hold_edge": str(band.intervals[-1].lower),
            "decisions": [band.action_at(r) for r in (F(3, 100), F(1, 4))],
            "day_hold_edge": str(late.intervals[-1].lower),
        },
        "05": {
            "window_days": insurer["facts"]["all_findings_mature_by_days"],
            "decision": "hold",
            "hidden_fraud_share": insurer["facts"]["fraud_findings_after_30_days"],
        },
    }
