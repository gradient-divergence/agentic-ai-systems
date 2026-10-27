"""Calibration in the large; binomial calculations state their assumptions."""

from math import isfinite

from scipy.stats import binom


def wilson_interval(wrong: int, paid: int, z: float = 1.96) -> tuple[float, float]:
    if type(paid) is not int or type(wrong) is not int or not 0 <= wrong <= paid or paid < 1:
        raise ValueError("positive paid count and bounded integer wrong count required")
    p = wrong / paid
    denominator = 1 + z * z / paid
    center = (p + z * z / (2 * paid)) / denominator
    half = z * (p * (1 - p) / paid + z * z / (4 * paid * paid)) ** 0.5 / denominator
    return max(0.0, center - half), min(1.0, center + half)


def calibrated(prediction: float, wrong: int, paid: int) -> bool:
    if not isfinite(prediction) or not 0 <= prediction <= 1:
        raise ValueError("prediction must be a finite probability")
    low, high = wilson_interval(wrong, paid)
    return low <= prediction <= high


def calibration_power(paid=2000, prediction=0.049, actual=0.065) -> dict:
    rejected = [k for k in range(paid + 1) if not calibrated(prediction, k, paid)]
    high_cut = next(k for k in rejected if k / paid > prediction)
    return {
        "paid": paid,
        "prediction": prediction,
        "actual": actual,
        "first_high_rejection": high_cut,
        "wilson_at_cut": wilson_interval(high_cut, paid),
        "power": float(binom.pmf(rejected, paid, actual).sum()),
        "correct_score_failure": float(binom.pmf(rejected, paid, prediction).sum()),
    }


def false_reversals(failure: float, months: int = 12) -> dict:
    """Independent, homogeneous, all-paid monthly batches of 2,000 requests.

    A confirmation consumes the next batch; it is not another check on the
    failed month's rows. Real paid subsets and heterogeneous risks differ.
    """
    if not 0 <= failure <= 1 or type(months) is not int or months < 1:
        raise ValueError("probability and positive months required")
    clear, pending, reversed_probability = 1.0, 0.0, 0.0
    for _ in range(months):
        reversed_probability += pending * failure
        clear, pending = (clear + pending) * (1 - failure), clear * failure
    return {
        "months": months,
        "single": 1 - (1 - failure) ** months,
        "confirmation": reversed_probability,
        "pending": pending,
    }
