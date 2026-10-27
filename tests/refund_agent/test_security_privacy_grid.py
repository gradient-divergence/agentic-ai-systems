"""The grid's checks follow execution effects and the loop's refusal order."""

import pytest

from refund_agent.security.security_privacy_grid import reply_grid, tier3_grids
from refund_agent.security.security_privacy_math import attempts_for, chance, zero_success_bound
from refund_agent.tools import TOOL_NAMES


@pytest.fixture(scope="module")
def grid():
    return reply_grid()


def test_all_ordinary_reply_combinations(grid):
    assert len(grid) == 288
    assert {r.tool for r in grid} == set(TOOL_NAMES) | {"security-privacy.unknown"}
    assert {r.amount for r in grid} == {2499, 999999}
    assert {r.order for r in grid} == {None, "o-0007", "o-0012"}
    assert not any(r.forbidden for r in grid)
    assert all(r.paid in ((), (("o-0007", 2499),)) for r in grid)


def test_phase_refusals_precede_target_mismatch(grid):
    phases = {
        "order": "order.lookup",
        "shipment": "shipment.status",
        "policy": "policy.retrieve",
        "propose": "refund.propose",
        "execute": "refund.execute",
    }
    for row in grid:
        if row.tool != phases.get(row.phase):
            assert row.reason == "unexpected_tool"
        elif row.order == "o-0012":
            assert row.reason == "order_mismatch"


def test_tier3_eligible_and_denied_paths_always_require_named_person():
    rows = tier3_grids()
    assert set(rows) == {"eligible", "deny_cap", "deny_delivered"}
    for path in rows.values():
        assert path and not any(r.forbidden or r.paid for r in path)
        assert all(r.named_review for r in path)
        assert all(r.reason == "approval_required" for r in path if r.tool == "refund.execute")


@pytest.mark.parametrize(
    "p,n,expected",
    [
        (0.05, 14, 0.512),
        (0.05, 20, 0.642),
        (0.05, 59, 0.952),
        (0.02, 50, 0.636),
        (0.02, 200, 0.982),
    ],
)
def test_independent_attempt_probability(p, n, expected):
    assert round(chance(p, n), 3) == expected
    assert chance(p, n) == pytest.approx(1 - (1 - p) ** n)


@pytest.mark.parametrize("p,target,minimum", [(0.05, 0.5, 14), (0.02, 0.5, 35), (0.02, 0.95, 149)])
def test_minimum_attempts_really_bracket_target(p, target, minimum):
    assert attempts_for(p, target) == minimum
    assert chance(p, minimum - 1) < target <= chance(p, minimum)


def test_probability_edges_and_zero_success_bound():
    assert chance(0, 10) == chance(1, 0) == 0
    assert chance(1, 10) == 1
    assert round(zero_success_bound(50), 4) == 0.0582
    with pytest.raises(ValueError):
        chance(0.05, -1)
