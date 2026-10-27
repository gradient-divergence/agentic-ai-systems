"""Agreement with an adjudicated reference, never an assertion of truth."""


def agreement(fail_fail, fail_pass, good_fail, good_pass):
    counts = (fail_fail, fail_pass, good_fail, good_pass)
    if any(type(c) is not int or c < 0 for c in counts) or sum(counts) == 0:
        raise ValueError("nonnegative integer cells and a nonempty table required")
    total = sum(counts)
    bad = fail_fail + fail_pass
    good = good_fail + good_pass
    judge_fail = fail_fail + good_fail
    judge_pass = fail_pass + good_pass
    observed = (fail_fail + good_pass) / total
    chance = (bad * judge_fail + good * judge_pass) / total**2
    return dict(
        total=total,
        cells=list(counts),
        agreement=observed,
        chance=chance,
        kappa=(observed - chance) / (1 - chance) if chance < 1 else None,
        miss_rate=fail_pass / bad if bad else None,
        false_hold_rate=good_fail / good if good else None,
    )
