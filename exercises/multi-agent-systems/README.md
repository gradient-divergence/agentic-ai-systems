# Multi-Agent Systems, exercises 5 and 6

Exercise 5's `review_05.json` contains 120 compliance decisions and a claimed
error rate based on verifier agreement. Diagnose it using the canonical
clauses and adjudicated outcomes.

For exercise 6, complete `ProofCheckingVerifier.verify` in `starter_06.py`.
`proof_of_delivery.json` supplies a second source: object when its delivery
timestamp differs from the carrier's by more than one day. Preserve every
existing objection and do not grant additional payment authority.

From the companion root:

```sh
EXERCISE_VERIFIER_FILE="$PWD/exercises/multi-agent-systems/starter_06.py" \
  OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  python -m pytest exercises/multi-agent-systems/test_exercise_06.py -p no:cacheprovider
```

The untouched starter fails the exercise check. The tests also reject losing
existing catches or using purchase age instead of delivery evidence, and check
both sides of the one-day boundary. Select `instructor_reference_06.py` instead
to check the public reference solution; its existing filename is retained.
