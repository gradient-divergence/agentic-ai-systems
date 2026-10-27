# Capstone exercises

The JSON files are constructed diagnostic fixtures with given verdicts and
signatures. They are not records of a deployment or substitutes for the
predecessor chapters' operating producers.

`capstone_blueprint_starters.py` has one designed error per exercise. The public
acceptance checks fail on the untouched starter functions. The maturity starter ignores the
last payment day; deployed mode must hold label-dependent records until that day
plus 30, naming the maturity day. Companion mode changes nothing.

Run `python -m pytest exercises/capstone-blueprint/test_exercise_06.py
-p no:cacheprovider` from the companion directory. Check
`capstone_blueprint_reference.py` with `CAPSTONE_BLUEPRINT_USE_REFERENCE=1`.
Additional reference checks are in `test_capstone_blueprint_instructor.py`,
enabled by the same environment variable. The default demo omits exercise 5's key.

Exercise 3 has exactly two stale records: the strict rehearsal and the injection
suite required again by the scope change. Exercise 5 has exactly one planted
error in `decision_05.json`'s claimed launch. Diagnose it and decide whether the
claim can stand. Its other signatures, covered window and given comparison are correct.
