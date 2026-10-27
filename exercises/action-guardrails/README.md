# Action Guardrails, exercise 6

Reference solution: `reference.patch`. It adds `_TIER1_DAILY_CAP` and the cap
check to the gate before the approval branch. Authorization reads every ledger
receipt and sums tier 1 cents on the execution's UTC date, including imported
receipts without a request. The existing tool boundary serializes authorization
and payment in the offline example.

Run from the repository root with the companion environment's Python:

```sh
python companion/exercises/run_reference.py action-guardrails
```

The runner applies the patch to a temporary copy, executes `test_acceptance.py`
and writes `run-06.json` with the command, environment, source hashes, timestamp
and pytest result. It does not patch the book's runnable agent. As in the latest
chapters, the acceptance file lives outside the default `companion/tests` suite.

The operator setting defaults to `None`, so an unconfigured copy retains the
original behavior. Tests set it to 5,000 cents. Two 2,499-cent payments pass and
the third is refused with `tier1_daily_cap`, audited and unpaid, with or without
a validated approval. The next UTC day's first payment passes. Tests also cover
same-day imported receipts, previous-day receipts, execution date rather than
request date, approved payments below the cap and the exact-cap boundary.

This receipt proves the acceptance tests named in its pytest result. It is not a
claim that the entire companion regression suite or a book render ran.
