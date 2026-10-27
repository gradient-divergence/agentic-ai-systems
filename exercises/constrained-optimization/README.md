# Exercise 8: respond-by plans

Implement `plan_morning(queue, minutes, budget)` in `starter.py`, using its return
schema. The checker reconstructs all training mornings, deadlines and tier limits;
it checks the usual-plan fallback, the counting proof and the minimum time repair.
It also preserves a temporary Chapter 4 golden manifest and prior access journal.

From the repository root, with the companion environment:

```sh
CONSTRAINED_OPTIMIZATION_SOLUTION="$PWD/companion/exercises/constrained-optimization/starter.py" \
  python -m pytest -p no:cacheprovider companion/exercises/constrained-optimization/test_acceptance.py
```

The starter fails at its unfinished function. Select `reference_solution.py` to
check the reference solution, or set the variable to the learner's Python file.
