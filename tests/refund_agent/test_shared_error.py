"""Joint error identities, enumeration and the limits of pairwise dependence."""

import itertools
import math

import pytest


@pytest.mark.parametrize(
    "cov,through,caught", [(0, 0.04, 0.16), (0.08, 0.12, 0.08), (0.16, 0.2, 0)]
)
def test_pair_identity_at_fixed_rates(cov, through, caught):
    from refund_agent.verifier.shared_error import catches, pass_through

    assert pass_through(0.2, 0.2, cov) == pytest.approx(through)
    assert catches(0.2, 0.2, cov) == pytest.approx(caught)


def test_observed_covariance_divides_by_case_count():
    from refund_agent.verifier.shared_error import error_table

    table = error_table(
        [0] * 790 + [0] * 60 + [1] * 90 + [1] * 60, [0] * 790 + [1] * 60 + [0] * 90 + [1] * 60
    )
    assert table["counts"] == {"00": 790, "01": 60, "10": 90, "11": 60}
    assert table["rho_p"] == 0.15 and table["rho_v"] == 0.12
    assert table["covariance"] == pytest.approx(0.042)
    assert table["pass_through"] == 0.06 and table["catches"] == 0.09


@pytest.mark.parametrize("n_a,rho", [(1, 0), (3, 0.2), (5, 0.15), (7, 1)])
def test_independent_majority_matches_full_enumeration(n_a, rho):
    from refund_agent.verifier.shared_error import majority_error

    exact = sum(
        rho ** sum(bits) * (1 - rho) ** (n_a - sum(bits))
        for bits in itertools.product((0, 1), repeat=n_a)
        if sum(bits) > n_a // 2
    )
    assert majority_error(n_a, rho) == pytest.approx(exact)


def test_shared_cause_matches_enumeration_of_the_mixture():
    from refund_agent.verifier.shared_error import joint_law_summary, shared_cause_majority

    law = {
        bits: 0.9 * (1 / 9) ** sum(bits) * (8 / 9) ** (3 - sum(bits))
        for bits in itertools.product((0, 1), repeat=3)
    }
    law[(1, 1, 1)] += 0.1
    summary = joint_law_summary(law)
    assert summary["rates"] == pytest.approx([0.2] * 3)
    assert summary["majority"] == pytest.approx(53 / 405)
    assert shared_cause_majority(0.1, 1 / 9) == pytest.approx(53 / 405)
    assert summary["covariances"] == pytest.approx([16 / 225] * 3)


def test_equal_rates_and_zero_pairwise_covariances_do_not_fix_majority():
    from refund_agent.verifier.shared_error import JOINT_A, JOINT_B, joint_law_summary

    for law, want in ((JOINT_A, 0.12), (JOINT_B, 0.04)):
        summary = joint_law_summary(law)
        assert summary["rates"] == pytest.approx([0.2] * 3)
        assert summary["covariances"] == pytest.approx([0] * 3, abs=1e-15)
        assert summary["majority"] == pytest.approx(want)


def test_seeded_simulation_repeats_and_matches_within_three_standard_errors():
    from refund_agent.verifier.shared_error import simulation

    rows = simulation(seed=7)
    assert rows == simulation(seed=7)
    # Normal approximation: 0.27% two-sided per estimate, at most about 1.08% for four.
    for row in rows:
        assert abs(row["observed"] - row["exact"]) <= 3 * row["standard_error"]


@pytest.mark.parametrize("n_a,rho", [(2, 0.2), (0, 0.2), (True, 0.2), (3, -1), (3, math.nan)])
def test_majority_rejects_unstated_tie_rules_and_invalid_probabilities(n_a, rho):
    from refund_agent.verifier.shared_error import majority_error

    with pytest.raises(ValueError):
        majority_error(n_a, rho)


@pytest.mark.parametrize("rho_p,rho_v,cov", [(0.2, 0.2, 0.3), (0.2, 0.2, -0.1), (0.2, 2, 0)])
def test_pair_rejects_impossible_joint_probabilities(rho_p, rho_v, cov):
    from refund_agent.verifier.shared_error import pass_through

    with pytest.raises(ValueError):
        pass_through(rho_p, rho_v, cov)


@pytest.mark.parametrize(
    "rho_p,rho_v,cov,want",
    [
        (0, 0.2, -5e-13, 0),
        (0.2, 0.2, 0.16 + 5e-13, 0.2),
        (0.8, 0.7, -0.06 - 5e-13, 0.5),
        (0.8, 0.7, 0.14 + 5e-13, 0.7),
    ],
)
def test_pair_clamps_accepted_roundoff_to_feasible_bounds(rho_p, rho_v, cov, want):
    from refund_agent.verifier.shared_error import pass_through

    assert pass_through(rho_p, rho_v, cov) == want
