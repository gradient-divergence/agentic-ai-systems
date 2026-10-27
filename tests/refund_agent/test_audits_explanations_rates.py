"""Independent arithmetic and signed audit boundaries for chapter 36."""

from datetime import datetime, timedelta, timezone
from fractions import Fraction
from math import comb

import pytest

from refund_agent.audit.audits_explanations_rates import (
    AuditObservation,
    AuditRule,
    Period,
    audit_period,
    ceiling_test,
    exact_upper_count,
    fpr_from_identity,
    hypergeom_tail,
    minimum_release_sample,
    power,
    wilson_interval,
    wrongful_denial_rate,
)

AT = datetime(2026, 5, 1, tzinfo=timezone.utc)
PERIOD = Period(start=AT, end=AT + timedelta(days=28))


def rule(**changes):
    return AuditRule.model_validate(
        {"signed_at": AT - timedelta(days=1), **changes}
    )


def observations(n=239, owed=395, k=22, *, held=False):
    return tuple(
        AuditObservation(
            request_id=f"audits-explanations-r-{i}",
            order_id=f"audits-explanations-o-{i}",
            segment="C",
            at=AT + timedelta(days=1),
            outcome="refer"
            if held and i < n
            else "deny"
            if i < n
            else "approve",
            reason="already_delivered" if i < n else "eligible",
            held_proposal=held and i < n,
            owed=i < k if i < n else True,
            found_false_if_paid=None if i < n else False,
            label_mature_at=AT + timedelta(days=32),
        )
        for i in range(n + owed)
    )


def test_hypergeometric_matches_independent_counting_and_normalizes():
    for N in range(1, 12):
        for K in range(N + 1):
            for m in range(N + 1):
                probabilities = [
                    Fraction(comb(K, j) * comb(N - K, m - j), comb(N, m))
                    if j <= K and 0 <= m - j <= N - K
                    else Fraction(0)
                    for j in range(m + 1)
                ]
                assert sum(probabilities) == 1
                for k in range(m + 2):
                    assert hypergeom_tail(k, N, K, m) == sum(probabilities[k:])


def test_tail_monotone_in_population_wrongful_count():
    tails = [hypergeom_tail(3, 25, K, 9) for K in range(26)]
    assert tails == sorted(tails)


def test_upper_bound_has_exact_at_least_95_percent_coverage():
    for N in range(1, 13):
        for K in range(N + 1):
            for m in range(N + 1):
                misses = sum(
                    (
                        hypergeom_tail(k, N, K, m)
                        - hypergeom_tail(k + 1, N, K, m)
                        for k in range(m + 1)
                        if exact_upper_count(k, N, m) < K
                    ),
                    Fraction(0),
                )
                assert misses <= Fraction(1, 20)


def test_brief_ceiling_boundaries():
    K, chance = ceiling_test(3, 239, 395, 80, Fraction(1, 100))
    assert K == 3
    assert chance == Fraction(comb(80, 3), comb(239, 3))
    assert float(chance) == pytest.approx(0.0366, abs=0.00005)
    assert ceiling_test(4, 239, 395, 80, Fraction(1, 100)) == (3, 0)
    K, chance = ceiling_test(4, 239, 396, 80, Fraction(1, 100))
    assert K == 4
    assert float(chance) == pytest.approx(comb(80, 4) / comb(239, 4))
    assert ceiling_test(1, 5, 98, 5, Fraction(1, 100)) == (0, 0)


def test_population_incapable_of_exceeding_ceiling_b4():
    for m in range(6):
        for k in range(m + 1):
            assert ceiling_test(k, 5, 990, m, Fraction(1, 100)) == (5, 1)
    assert power(rule(), 5, 5, 990, 5) == 0


def test_zero_entitlement_is_unassessable_b4():
    rows = tuple(
        r.model_copy(update={"owed": False}) for r in observations(5, 0, 0)
    )
    row = audit_period(
        rows, PERIOD, rule(), 7, at=AT + timedelta(days=33)
    ).segments[2]
    assert row.status == "unassessable"
    assert row.estimate is None and row.release_allowed is False
    assert row.upper_owed_denial_share is None
    bad = audit_period(
        observations(5, 0, 1), PERIOD, rule(), 7, at=AT + timedelta(days=33)
    ).segments[2]
    assert bad.breach and bad.estimate == 1


def test_hold_reads_proposals_and_requires_positive_safety_evidence():
    at = AT + timedelta(days=33)
    record = audit_period(observations(held=True), PERIOD, rule(), 7, at=at)
    assert record.segments[2].denials == 239
    assert record.segments[2].held_proposals == 239
    clean = audit_period(
        observations(k=0, held=True), PERIOD, rule(), 7, at=at
    )
    C = clean.segments[2]
    assert not C.breach and not C.release_allowed
    assert (
        C.upper_owed_denial_share is not None
        and C.upper_owed_denial_share > 0.01
    )
    needed = minimum_release_sample(239, 395, Fraction(1, 100))
    assert needed is not None
    enough = audit_period(
        observations(k=0, held=True),
        PERIOD,
        rule(read_per_segment=needed),
        7,
        at=at,
    ).segments[2]
    assert enough.upper_owed_denial_share is not None
    assert enough.release_allowed and enough.upper_owed_denial_share < 0.01
    assert exact_upper_count(0, 239, needed - 1) >= 4
    assert exact_upper_count(0, 239, needed) <= 3


def test_all_read_estimate_equals_oracle_and_small_period_reads_all():
    C = audit_period(
        observations(9, 27, 3), PERIOD, rule(), 7, at=AT + timedelta(days=33)
    ).segments[2]
    assert C.read == 9 and C.wrongful == 3
    assert C.estimate == pytest.approx(3 / 30)
    assert C.chance == 0 and C.breach
    assert C.sample_ids == tuple(sorted(C.sample_ids))


def test_sampling_is_independent_of_order_truth_and_period_metadata():
    rows = observations()
    at = AT + timedelta(days=33)
    a = audit_period(rows, PERIOD, rule(), 7, at=at).segments[2]
    changed = tuple(
        r.model_copy(update={"owed": not r.owed}) if r.outcome == "deny" else r
        for r in reversed(rows)
    )
    b = audit_period(changed, PERIOD, rule(), 7, at=at).segments[2]
    assert a.sample_ids == b.sample_ids
    assert len(set(a.sample_ids)) == 80


def test_no_denials_pending_labels_and_late_signature():
    at = AT + timedelta(days=33)
    C = audit_period(
        observations(0, 20, 0), PERIOD, rule(), 7, at=at
    ).segments[2]
    assert C.status == "no_denials" and C.estimate == 0 and C.read == 0
    C = audit_period(
        observations(), PERIOD, rule(), 7, at=PERIOD.end
    ).segments[2]
    assert C.status == "pending_labels" and not C.release_allowed
    with pytest.raises(ValueError, match="signed before"):
        audit_period((), PERIOD, rule(signed_at=AT), 7, at=at)


def test_refuses_duplicate_ids_unknown_segments_and_invalid_counts():
    rows = observations(1, 3, 0)
    with pytest.raises(ValueError, match="duplicate"):
        audit_period(
            rows + rows[:1], PERIOD, rule(), 7, at=AT + timedelta(days=33)
        )
    with pytest.raises(ValueError, match="unnamed"):
        audit_period(
            (rows[0].model_copy(update={"segment": "D"}),),
            PERIOD,
            rule(),
            7,
            at=AT + timedelta(days=33),
        )
    for args in ((1, 5, -1, 3), (-1, 5, 2, 3), (0, 5, 1, 6)):
        with pytest.raises(ValueError):
            hypergeom_tail(*args)


def test_identity_and_ppv_one_boundary():
    assert fpr_from_identity(0.1, 0.5, 0.8) == pytest.approx(80 / 900)
    assert fpr_from_identity(0.3, 0.5, 0.8) == pytest.approx(240 / 700)
    with pytest.raises(ValueError):
        fpr_from_identity(0.1, 1, 0.8)


def test_wilson_zero_and_conversion():
    z = 1.96
    lo, hi = wilson_interval(0, 80)
    assert lo == 0 and hi == pytest.approx(z * z / (80 + z * z))
    assert wrongful_denial_rate(0.4, 3 / 80, 12 / 360) == pytest.approx(
        9 / (348 + 9)
    )
    assert wrongful_denial_rate(0.4, 0, 1) is None


@pytest.mark.parametrize(
    "denied_share, wrongful_share, found_false_share, expected",
    (
        (0, 0.25, 0.2, 0), (1, 0.25, 0.2, 1),
        (0, 0.25, 1, None), (1, 0, 0.2, None),
    ),
)
def test_conversion_accepts_denied_share_endpoints(
    denied_share, wrongful_share, found_false_share, expected
):
    assert wrongful_denial_rate(
        denied_share, wrongful_share, found_false_share
    ) == expected


@pytest.mark.parametrize("denied_share", (-0.01, 1.01))
def test_conversion_rejects_denied_shares_outside_unit_interval(denied_share):
    with pytest.raises(ValueError, match="denied share"):
        wrongful_denial_rate(denied_share, 0.25, 0.2)


def test_exact_power_expected_and_impossible_detection():
    assert float(power(rule(), 239, 22, 395, 80)) == pytest.approx(
        0.97, abs=0.01
    )
    assert power(rule(), 239, 3, 395, 80) == 0


def test_wilson_matches_chapter_four_grid_without_running_its_demo():
    from demos.ml_data_evaluation_demo import wilson_interval as chapter_four

    for n in (1, 5, 20, 80, 239):
        for k in range(n + 1):
            assert wilson_interval(k, n) == pytest.approx(
                chapter_four(k, n), abs=1e-12
            )
