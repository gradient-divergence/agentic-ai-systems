"""Listings and output for the chapter on decision strategy and evidence requirements.

`bound-01` computes how far a run of cases without a failure bounds a failure rate.
`bound-02` reads the same clean cases with a flat prior beside that bound.
`record-01` defines an evidence requirement and the check that refuses one with an
empty field, without an outcome measure and a counter-measure, or signed on or
after the day the run that judges it began; further required measures are allowed.
Its records are in `data_science_strategy_records.py`; `record-02` prints the
signed drafts-to-shadow requirement's seven fields and its signing and run days.
`scorecard-01` prints the rule-only baseline's line from the default run,
`python -m refund_agent.run --seed 7`, and the public scenarios by tier.
`keys-01` prints the keys to the chapter's numeric exercises.

Importing this module defines `zero_failure_bound`, `cases_needed`,
`EvidenceRequirement` and `check`, and runs nothing: the printing, and the refund
agent's run (which writes `refund-output` under the working directory), happen only
when the file runs as a script. Nothing here changes the refund agent.
"""

import contextlib
import io
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from refund_agent.data.generate import generate  # noqa: E402, I001
from refund_agent.run import main  # noqa: E402


def baseline_lines() -> str:
    """The default run's baseline line, on two lines, and the scenarios by tier."""
    with contextlib.redirect_stdout(io.StringIO()) as run_output:
        main(["--seed", "7", "--out", "refund-output"])
    line = next(line for line in run_output.getvalue().splitlines() if line.startswith("baseline:"))
    data = generate(7)
    labels = {decision.request_id: decision.tier for decision in data.correct}
    tiers = Counter(int(labels[request_id]) for request_id in data.public_scenarios)
    by_tier = " ".join(f"tier {tier}={tiers[tier]}" for tier in sorted(tiers))
    return line.replace("; ", ";\n", 1) + f"\npublic scenarios by tier: {by_tier}"


# region book:fnd.data-science-strategy.bound-01 output
import math


def zero_failure_bound(n: int, chance: float = 0.05) -> float:
    """Largest rate at which n clean cases have this chance."""
    return 1 - chance ** (1 / n)


def cases_needed(rate: float, chance: float = 0.05) -> int:
    """Smallest n whose clean run bounds the rate at `rate`."""
    return math.ceil(math.log(chance) / math.log(1 - rate))


if __name__ == "__main__":
    print(f"{'cases':>6}{'bound':>9}{'3/n':>8}")
    for n in (10, 40, 100, 300, 1000):
        print(f"{n:>6}{zero_failure_bound(n):>9.4f}{3 / n:>8.4f}")
    print(f"40 clean cases at a rate of 0.05: {0.95**40:.4f}")
    print(f"cases for a bound of 0.01: {cases_needed(0.01)}")
    print("40 clean cases (constructed comparison):")
    label = "  independent: effective cases 40"
    print(f"{label}; bound {zero_failure_bound(40):.4f}")
    label = "  one shared failure: effective cases 1"
    print(f"{label}; bound {zero_failure_bound(1):.2f}")
# endregion book:fnd.data-science-strategy.bound-01


# region book:fnd.data-science-strategy.record-01 output
from dataclasses import dataclass, fields

from demos.data_science_strategy_records import CASES


@dataclass(frozen=True)
class EvidenceRequirement:
    step: str  # the class, and the grant it asks for
    claim: str
    comparator: str
    cases: str
    # Outcome measure, counter-measure, further required measures.
    measures: list[str]
    threshold: str
    signer: str
    signed_day: int  # counted from a day 0 the team picks
    reversal: str


def check(req: EvidenceRequirement, start_day: int) -> list[str]:
    """Reasons to refuse `req` for the run that will judge it."""
    problems = []
    for f in fields(req):
        if f.name != "signed_day" and not getattr(req, f.name):
            problems.append(f"no {f.name}")
    if len(req.measures) < 2 or not all(req.measures):
        problems.append("no measure pair")
    if req.signed_day >= start_day:
        problems.append("signed after its run began")
    return problems


if __name__ == "__main__":
    print(f"{'step':<25}{'signed':>6}{'run':>5}  verdict")
    for record, start_day in CASES:
        req = EvidenceRequirement(**record)
        verdict = "; ".join(check(req, start_day)) or "accepted"
        row = f"{req.step:<25}{req.signed_day:>6}{start_day:>5}"
        print(f"{row}  {verdict}")
# endregion book:fnd.data-science-strategy.record-01


# region book:fnd.data-science-strategy.record-02 output
from demos.data_science_strategy_records import DRAFTS_TO_SHADOW
from demos.output_wrap import print_wrapped

if __name__ == "__main__":
    record = DRAFTS_TO_SHADOW
    print_wrapped(f"| Field | {record['step'].capitalize()} |")
    print_wrapped("|---|---|")
    for name in ("claim", "comparator", "cases", "measures", "threshold", "signer", "reversal"):
        value = record[name]
        cell = "; ".join(value) if isinstance(value, list) else value
        print_wrapped(f"| {name.title()} | {cell} |")
    print_wrapped()
    print_wrapped(
        f": The requirement for {record['step']}, signed on day {record['signed_day']} "
        f"for a run from day {CASES[0][1]} "
        "{#tbl-data-science-strategy-drafts-to-shadow}"
    )
# endregion book:fnd.data-science-strategy.record-02


# region book:fnd.data-science-strategy.scorecard-01 output
if __name__ == "__main__":
    print(baseline_lines())
# endregion book:fnd.data-science-strategy.scorecard-01


# region book:fnd.data-science-strategy.keys-01 output
# Keys to exercises 3 and 6, and the retry figure the chapter quotes; not included.
if __name__ == "__main__":
    print(f"bound at 60 cases: {zero_failure_bound(60):.4f}")
    print(f"bound at 200 cases: {zero_failure_bound(200):.4f}")
    print(f"cases needed for a bound of 0.02: {cases_needed(0.02)}")
    print(f"cases needed for a bound of 0.005: {cases_needed(0.005)}")
    print(f"bound at {cases_needed(0.005)} cases: {zero_failure_bound(cases_needed(0.005)):.6f}")
    # A search over two versions: each independent run has true failure rate 0.01.
    print(
        f"two versions, one prespecified 299-case run each at 0.01, one clean: {1 - (1 - 0.99**299) ** 2:.4f}"
    )
# endregion book:fnd.data-science-strategy.keys-01


# region book:fnd.data-science-strategy.bound-02 output
from demos.data_science_strategy_demo import cases_needed, zero_failure_bound


def flat_prior_bound(n: int, chance: float = 0.05) -> float:
    """Upper rate with this posterior tail after n clean cases, flat prior."""
    return zero_failure_bound(n + 1, chance)


if __name__ == "__main__":
    print("One-sided 95% bounds; flat prior on the failure rate.")
    print(f"{'cases':>6}{'frequentist':>13}{'flat prior':>12}")
    for n in (10, 40, 100, 300, 1000):
        frequentist = zero_failure_bound(n)
        print(f"{n:>6}{frequentist:>13.4f}{flat_prior_bound(n):>12.4f}")
    print(
        f"40 clean cases: frequentist={zero_failure_bound(40):.1%}; "
        f"flat prior={flat_prior_bound(40):.1%}"
    )
    print(
        f"cases for a bound of 0.01: frequentist={cases_needed(0.01)}; "
        f"flat prior={cases_needed(0.01) - 1}"
    )
# endregion book:fnd.data-science-strategy.bound-02
