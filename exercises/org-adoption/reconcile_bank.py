"""Exercise 38.6 starter: deliberately misses bank-statement refunds."""

from refund_agent.rollout.org_adoption_reconcile import reconcile


def reconcile_bank(intake, decisions, ledger, provider, bank, window):
    # Exercise fault: only the provider's money-out record is reconciled.
    return reconcile(intake, decisions, ledger, provider, window)
