"""Read chapter 34's plan/window-bound exhaustion before withdrawing stage 6."""

from refund_agent.launch.capstone_blueprint_decision import reverse
from refund_agent.launch.capstone_blueprint_scan import ReversalEvent
from refund_agent.rollout.org_adoption_authority import BudgetExhaustion


def budget_spent(
    status: BudgetExhaustion, *, authority, registry, requirement, day: float
):
    status = BudgetExhaustion.model_validate_json(status.model_dump_json())
    withdrawal = authority.on_budget_spent(status)
    event = ReversalEvent(
        record_id=withdrawal.record_id,
        condition_id="budget_spent",
        at=status.at,
        day=day,
        reason="chapter 34 monitor's bound budget is spent",
    )
    return reverse(
        event,
        requirement=requirement,
        authority=authority,
        registry=registry,
        by=authority.plan.signed_by,
    )
