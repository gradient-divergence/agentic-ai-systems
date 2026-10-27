"""Fixed independent-attempt calculations, never estimates of a live attacker."""

from math import ceil, expm1, isfinite, log1p


def chance(p: float, n: int) -> float:
    if not isfinite(p) or not 0 <= p <= 1 or type(n) is not int or n < 0:
        raise ValueError("chance needs a probability and nonnegative attempt count")
    if n == 0 or p == 0:
        return 0.0
    return 1.0 if p == 1 else -expm1(n * log1p(-p))


def attempts_for(p: float, target: float) -> int:
    if not 0 < p < 1 or not 0 < target < 1:
        raise ValueError("attempt threshold needs probabilities strictly between zero and one")
    return ceil(log1p(-target) / log1p(-p))


def zero_success_bound(n: int) -> float:
    if type(n) is not int or n <= 0:
        raise ValueError("positive independent trial count required")
    return 1 - 0.05 ** (1 / n)
