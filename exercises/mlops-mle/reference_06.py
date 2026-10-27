"""Reference solution for the constructed mlops-mle requirements exercise."""

from refund_agent.promotion.mlops_mle_requirements import REQUIREMENTS as BOOK_REQUIREMENTS
from refund_agent.promotion.mlops_mle_requirements import Requirement

REQUIREMENTS = (
    *(row for row in BOOK_REQUIREMENTS if row.items != ("runtime", "adapter")),
    Requirement(
        "tools",
        ("runtime", "adapter"),
        ("runtime_swap",),
        "runtime swap including test_runtime_swap.py",
    ),
)


def required(changes):
    from refund_agent.promotion.mlops_mle_requirements import ALWAYS

    needs = list(ALWAYS)
    for change in changes:
        matches = [
            row
            for row in REQUIREMENTS
            if row.component == change["component"]
            and ("*" in row.items or change["item"] in row.items)
        ]
        if not matches:
            raise ValueError("changed bundle item has no requirement row")
        for row in matches:
            needs.extend(row.evidence)
    return tuple(dict.fromkeys(needs))
