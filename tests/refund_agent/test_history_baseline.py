from datetime import timedelta

import numpy as np
import pytest

pytest.importorskip("sklearn")

from refund_agent.data.history import generate_history
from refund_agent.model.history_baseline import (
    TEST_AT,
    cost_summary,
    feature_matrix,
    reliability_table,
    time_split,
)


def test_time_split_only_uses_mature_paid_labels() -> None:
    rows = generate_history(rows=1000)
    train, gap, test = time_split(rows)
    assert train and gap and test
    assert all(
        row.paid
        and row.at < TEST_AT
        and row.label_final_at is not None
        and row.label_final_at <= TEST_AT
        for row in train
    )
    assert all(
        row.paid and row.label_final_at is not None and row.at < TEST_AT < row.label_final_at
        for row in gap
    )
    assert all(row.paid and row.at >= TEST_AT for row in test)
    assert len(train) + len(gap) + len(test) == sum(row.paid for row in rows)
    row = train[0].model_copy(
        update={"at": TEST_AT - timedelta(days=30), "label_final_at": TEST_AT}
    )
    assert time_split([row])[0] == [row]


def test_feature_allowlist_ignores_outcome_and_later_metadata() -> None:
    rows = generate_history(rows=100)
    changed = [
        row.model_copy(
            update={
                "found_false_30d": 1,
                "paid": False,
                "policy_decision": "deny",
                "label_final_at": TEST_AT,
                "request_id": "label-1",
                "at": TEST_AT,
            }
        )
        for row in rows
    ]
    # Missing delivery scans stay NaN in both matrices.
    np.testing.assert_array_equal(feature_matrix(rows), feature_matrix(changed))


def test_costs_include_review_and_wrong_payment_but_never_score_denial() -> None:
    costs = cost_summary([1, 0, 1], [0.01, 0.08, 0.9], [1000, 4999, 2000])
    assert costs["paid_count"] == 1 and costs["review_count"] == 2
    assert costs["denied_count"] == costs["wrong_denial_count"] == 0
    assert costs["expected_total_usd"] == pytest.approx(0.01 * 10 + 4 + 4)
    assert costs["realized_total_usd"] == pytest.approx(10 + 4 + 4)
    assert costs["wrong_denial_cost_usd"] == 20
    assert costs["deny_edge"] == 0.8


def test_ten_bin_reliability_preserves_counts_and_ties() -> None:
    table = reliability_table([0, 1, 0, 1], [0.1, 0.1, 0.2, 0.8])
    assert len(table) == 10
    assert sum(bin_["count"] for bin_ in table) == 4
    assert sum(bin_["positives"] for bin_ in table) == 2
    tied = next(bin_ for bin_ in table if bin_["mean_probability"] == 0.1)
    assert tied["count"] == 2 and tied["observed_rate"] == 0.5
    assert all(bin_["observed_rate"] is None for bin_ in table if bin_["count"] == 0)
