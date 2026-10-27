"""Reference solution for the fifth check; earlier checks still apply."""

from refund_agent.transfer.case_studies_domain_agnostic_check import Failure


def check_units(packet, facts):
    actions = {action.name: action for action in facts.actions}
    failures = []
    for cap in packet.caps:
        action = actions[cap.action]
        if cap.dimension != action.harm_dimension:
            failures.append(
                Failure(
                    check="units",
                    item=cap.action,
                    reason="cap and harm have different dimensions",
                    numbers={
                        "cap_unit": cap.unit,
                        "cap_dimension": cap.dimension,
                        "harm_unit": action.harm_unit,
                        "harm_dimension": action.harm_dimension,
                    },
                )
            )
    return failures
