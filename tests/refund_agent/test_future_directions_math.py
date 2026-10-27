import pytest

from refund_agent.renewal.future_directions_math import (
    calibrated,
    calibration_power,
    false_reversals,
    wilson_interval,
)


def test_power_and_cutoff_precede_evidence():
    power = calibration_power()
    assert power["first_high_rejection"] == 117
    assert power["power"] == pytest.approx(0.891, abs=0.001)
    assert power["correct_score_failure"] == pytest.approx(0.055, abs=0.001)
    assert false_reversals(0.1, months=2)["confirmation"] == pytest.approx(0.01)
    assert false_reversals(power["correct_score_failure"])["confirmation"] < 0.04


def test_small_counts_and_endpoints():
    from scipy.stats import binomtest

    assert calibrated(0.05, 0, 40)
    assert wilson_interval(0, 40)[1] == pytest.approx(0.088, abs=0.001)
    assert calibrated(wilson_interval(128, 2000)[0], 128, 2000)
    exact = binomtest(0, 40).proportion_ci(method="exact")
    assert exact.low <= 0.05 <= exact.high
    assert exact.high == pytest.approx(0.088, abs=0.001)
    with pytest.raises(ValueError):
        wilson_interval(1, 0)


def test_opposite_band_errors_cancel():
    assert calibrated(0.07, 140, 2000)
    assert not calibrated(0.12, 80, 1000)
    assert not calibrated(0.02, 60, 1000)
