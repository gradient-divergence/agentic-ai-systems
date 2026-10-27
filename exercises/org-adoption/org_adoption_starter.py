"""Exercises 38.1-5: full, deliberately wrong responses to diagnose and replace."""

from refund_agent.rollout.org_adoption_desk import balance_case
from refund_agent.rollout.org_adoption_reconcile import reconcile

# Exercise faults: disagreeing with an agent is mistaken for going around it;
# the agent is assigned authority that belongs to named people.
CLASSIFICATION = (
    "workaround",
    "workaround",
    "workaround",
    "governed",
    "workaround",
)
ROLES = {
    "sign": "refund-agent",
    "stop": "refund-agent",
    "cause": "refund-agent",
}


def count(fixture):
    result = reconcile(**fixture)
    # Exercise fault: removes received requests with no decision from denominator.
    return result.model_copy(update={"received": len(fixture["decisions"])})


def amounts(fixture, early):
    # Exercise fault: assumes the console refund is always already posted.
    return balance_case(fixture, early=True)


def diagnosis(fixture):
    return {
        "denominator": "decision records",
        "share": 100.0,
        "decision": "widen",
    }
