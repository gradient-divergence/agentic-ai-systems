# Chapter 3: Decision Strategy and Evidence Requirements

[Companion README](../../README.md)

Chapter 3 asks what evidence a class of refunds needs before it gets its next grant of
authority. The demo computes how far a run of cases without a failure bounds a failure
rate, defines an evidence requirement and the check that refuses an incomplete or late
one, and prints the rule-only baseline's line from the refund agent's default run.
Nothing here changes the refund agent.

## Files

| What | Where |
|---|---|
| Demo | `demos/data_science_strategy_demo.py` |
| Records | `demos/data_science_strategy_records.py`: the requirements the listing checks, as plain records |
| Code | `refund_agent/run.py` (the default run), `refund_agent/baseline.py` |
| Exercises | `exercises/data-science-strategy/` |
| Stored outputs | `book_outputs/_fnd.data-science-strategy.*.txt` (6 files) |
| Tests | `tests/test_data_science_strategy.py` |

## Run it

From the companion folder, after the install in the README:

```sh
python -m demos.data_science_strategy_demo
python -m pytest tests/test_data_science_strategy.py
```

Run as a script, the demo also makes the refund agent's default run, which writes
`refund-output/` under the working directory (ignored by git). Importing the module
defines `zero_failure_bound`, `cases_needed`, `EvidenceRequirement` and `check`, and
runs nothing.

## What the output shows

- `bound-01` is the one-sided 95% upper confidence bound on a failure rate after `n`
  independent clean cases, beside the rule of three, `3/n`. Forty clean cases bound
  the rate at 0.0722 (`3/n` gives 0.0750); 300 give 0.0099. At a true rate of 0.05,
  forty clean cases still happen with probability 0.1285. A claim of "below 1%" needs
  299 clean cases. If the forty cases share one failure, they count as one effective
  case, and the bound is 0.95.
- `bound-02` reads the same clean runs with a flat prior on the failure rate. At forty
  cases the flat-prior bound is 7.0% against the frequentist 7.2%, and it reaches
  "below 1%" one case sooner, at 298. The two numbers mean different things: the
  bound's 95% is a frequency over runs, while the flat prior's 5% is a chance about
  this class's rate, bought with an assumption made before any case is seen.
- `record-01` runs `check` on three requirements for a run that starts on day 10. Tier
  1's drafts-to-shadow requirement signed on day 8 is accepted; the same one signed on
  day 12 is refused (signed after its run began); tier 2's draft is refused for having
  no threshold.
- `record-02` prints the signed drafts-to-shadow requirement's seven fields: claim,
  comparator, cases, measures, threshold, signer and reversal. Its cases are the 15
  tier 1 requests among the forty public scenarios; they show coverage of chosen cases
  and bound no failure rate.
- `scorecard-01` is the baseline's line from `python -m refund_agent.run --seed 7`:
  40/40 agreement with generated labels (approve 4, deny 22, refer 14, executed 0),
  and the public scenarios by tier: 15 at tier 1, 10 at tier 2, 15 at tier 3.
- `keys-01` holds the keys to exercises 3 and 6 and a retry figure the chapter quotes:
  the bounds at 60 and 200 cases (0.0487 and 0.0149), the cases needed for 2% (149)
  and for 0.5% (598, whose bound is 0.004997), and 0.0966, the chance that at least one
  of two versions, each run once for 299 prespecified cases at a true rate of 0.01,
  comes out clean.

The bound assumes independent cases; a check accepts a requirement only if every
field is filled, it names an outcome measure and a counter-measure, and it was signed
before its run began. Further required measures are allowed.

## Exercises

Exercise 6 writes tier 1's shadow-to-canary requirement as a JSON file for the claim
"below 0.5%" and checks it for a run that starts on day 10. The reference solution record is
`exercises/data-science-strategy/requirement-06.json`. To check it, or your own file:

```sh
python -m pytest -p no:cacheprovider exercises/data-science-strategy/test_acceptance.py
STRATEGY_REQUIREMENT=path/to/your_requirement.json \
  python -m pytest -p no:cacheprovider exercises/data-science-strategy/test_acceptance.py
```

The reference solution record's fixed run has 598 cases and bound 0.004997 (0.00499705869425
before rounding), a signature before day 10, a referral threshold that says "no
higher," and no promise to undo paid refunds. Its third measure counts wrongful
denials: requests the agent denied that the person paid, with the payment not found
false within 30 days. Its threshold says "no more than the baseline's" on the same
requests, and the baseline is the only comparator. These tests check the exercise's
arithmetic and the record's shape, not canary authorization, and are not part of the
default test suite.

## What it leaves out

- The scenarios are generated, and the baseline's labels repeat its own fictional
  rules, so 40/40 agreement shows coverage of chosen cases, not a failure rate.
- The requirements are records and a check; nothing here grants authority or moves
  money.
