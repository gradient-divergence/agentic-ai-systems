# Organizational Adoption exercises

All records are constructed accounting fixtures. The full stage 6 window awaits
the predecessor integration described in the companion README. These exercises
do not issue a release, credential, promotion or rollout grant.

Run one test file per call with the companion environment and repository/companion
on `PYTHONPATH`, `OMP_NUM_THREADS=1`, `PYTHONDONTWRITEBYTECODE=1`, and pytest's
`-p no:cacheprovider` flag.

Exercises 1-5 start in `org_adoption_starter.py`. Their five tests in
`test_org_adoption_acceptance.py` fail respectively on path classification,
named roles, the intake denominator, delayed settlement and the planted
denominator error. Acceptance imports only the student file. To validate the reference solution,
copy `org_adoption_reference.py` to `org_adoption_starter.py` in an isolated
exercise copy and run the same test file.

Exercise 6 starts in `reconcile_bank.py`; run `test_exercise_06.py`. The starter
misses bank-only money and fails to detect conflicting identities across feeds.
The reference reads both money-out records, counts duplicate payment identities
once, and refuses contradictory facts. Its rerun reports the bank refund's order
and amount, 17 governed of 20 requests and two payments around the path: hold.

Fixtures printed in the chapter live at `demos/data/org_adoption_window_03.json`,
`org_adoption_record_05.json` and `org_adoption_balance_04.json`. Regenerate them
with `python -m demos.org_adoption_demo --write-data`. Exercise 4 has its own
9,000/6,000/4,500-cent fixture; the text's separate case remains 4,000 cents.

Payment amounts and public worked keys are in
`book_outputs/_fnd.org-adoption.keys-01.txt`.
