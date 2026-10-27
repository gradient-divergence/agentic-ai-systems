# Human Oversight and Risk Management

Run from the repository root with the companion Python, `PYTHONPATH` including
the repository and its companion directory, `OMP_NUM_THREADS=1`, and
`PYTHONDONTWRITEBYTECODE=1`. Each pytest call names one file and uses
`-p no:cacheprovider`.

Exercises 1–5 have an executable worksheet in
`human_oversight_risk_answers_01_05.py`. Its five answers contain intentional
mistakes: confusing accumulated work with payments, ignoring the weekend,
reporting service time as total waiting time, sizing the insurer for seven
staffed days, and counting escalations as catches. The corresponding test file
fails five tests. Replace the answers and run it again.

Exercise 5's `log-05.csv` is a constructed week with 99 review items, including
20 planted cases. Its last row reports "Planted cases caught: 15 of 20 (75%).
Review healthy." Diagnose that report, recompute it by load band, and write the
reply to the decision owner. The fixture includes opening and closing clocks,
the queue length, reviewer, planted flag, action and an unresolved outcome.

Exercise 6's `timeout_06.py` intentionally writes an approval at the deadline.
`test_timeout_06.py` rejects that action. The corrected handler must keep the
item open, escalate along its path, and send the requester the new deadline.
Chapter 37's lifecycle tests additionally verify payment after a real review,
fresh execution reads, stale evidence, eligibility expiry, identity rejection,
and a new policy/proposal/review during the approver's absence. The test
credentials are constructed partner fixtures, confined to tests.

```sh
python -m pytest -p no:cacheprovider exercises/human-oversight-risk/test_human_oversight_risk_answers_01_05.py
python -m pytest -p no:cacheprovider exercises/human-oversight-risk/test_timeout_06.py
```

Those commands assume the working directory is the companion directory. The
repository-root commands prefix each test path with the companion directory.

Reference solutions live under `reference/`. To check them, use
`HUMAN_OVERSIGHT_RISK_REFERENCE=1` with the same commands. The five-answer
reference passes five tests; the deadline reference passes its acceptance
test. It imports the actual deadline handler rather than duplicating it.

The files under `reference/` are reference solutions. No credential issuer or denial
explanation generator is shipped in this exercise directory.
