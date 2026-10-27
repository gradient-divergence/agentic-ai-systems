"""Reference solution; the checker also accepts other structurally valid cases."""

from refund_agent.learning_case import threshold_case


def submission():
    case = threshold_case()
    return case, case.render()
