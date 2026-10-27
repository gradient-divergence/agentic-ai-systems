"""Exercise 38.6: bank-only money, duplicate feeds and conflicting identities."""

import os

import org_adoption_reference
import pytest
import reconcile_bank

from demos.org_adoption_fixtures import accounting_fixture
from refund_agent.rollout.org_adoption_records import ProviderPayment

solution = (
    org_adoption_reference
    if os.environ.get("ORG_ADOPTION_REFERENCE") == "1"
    else reconcile_bank
)


def test_bank_only_refund_holds_the_window_with_order_and_amount():
    fixture = accounting_fixture(20, console=True)
    request = fixture["intake"][2]
    bank = ProviderPayment(
        payment_id="org-adoption-bank-1",
        order_id=request.order_id,
        amount_cents=2500,
        at=request.received_at,
        request_id=request.request.request_id,
        channel="bank",
    )
    result = solution.reconcile_bank(**fixture, bank=(bank,))
    assert len(result.around) == 2, (
        "bank transfer was not reconciled against the ledger"
    )
    assert (result.around[-1].order_id, result.around[-1].amount_cents) == (
        request.order_id,
        2500,
    ) or (result.around[0].order_id, result.around[0].amount_cents) == (
        request.order_id,
        2500,
    )
    assert result.governed == 17 and result.governed_percent < 99


def test_refund_found_in_both_feeds_counts_once():
    fixture = accounting_fixture(20, console=True)
    duplicate = fixture["provider"][0].model_copy(update={"channel": "bank"})
    result = solution.reconcile_bank(**fixture, bank=(duplicate,))
    assert len(result.around) == 1


def test_conflicting_identity_is_not_silently_deduplicated():
    fixture = accounting_fixture(20, console=True)
    conflicting = fixture["provider"][0].model_copy(update={"amount_cents": 1})
    with pytest.raises(ValueError, match="identity conflict"):
        solution.reconcile_bank(**fixture, bank=(conflicting,))
