Chapter 36, exercises 5 and 6
===========================

Exercise 5 diagnoses an appeals-based audit. `audit_starter.py` deliberately
substitutes appeals for blind readings. The reference solution `reference.py` uses
the signed finite-population ceiling test. The supplied seed-11
`audit_report_05.json` contains 2,000 observations from a second constructed
month. The tests' small constructed report is labelled separately.

Exercise 6's `explain_starter.py` supplies a worked amount-only explanation.
Complete its local `collect_failures` and `joint_counterfactual` mechanisms;
no complete explanation, clause collector or counterfactual generator is imported.
Retain all six replay checks. Changing only the amount leaves an expired execution
window. Counterfactual eligibility never supplies a reviewer's or named approver's
decision.

From the companion directory, with its environment already available:

```sh
PYTHONPATH=. OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  python -m pytest exercises/audits-explanations/test_acceptance.py -p no:cacheprovider
AUDITS_EXPLANATIONS_SOLUTION=explain_starter PYTHONPATH=. \
  OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  python -m pytest exercises/audits-explanations/test_acceptance.py -p no:cacheprovider
```

Set `AUDITS_EXPLANATIONS_SOLUTION=reference` to check the public reference.
The reference passes. The untouched starter fails completeness and its
counterfactual on `two_clause_06.json`. The remaining tests prove those failures
and show why the exercise-5 starter holds B while blind readings hold C.
Acceptance also exercises independent delivery, expired-window, amount-cap and
exhausted-balance records with all six replay checks. It runs the selected
solution on every canonical seed-7 denial when that artifact exists. Until
then the canonical population test explicitly skips with its integration reason.
