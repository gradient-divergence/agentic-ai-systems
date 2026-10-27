# Exercise 8: a submitted learning case

Implement `submission()` in `starter.py`. Return a `LearningCase` (or its JSON
object) and the rendered table. Sources use `companion/path.py:line` or a stored
output's `companion/book_outputs/path.txt:line`; additional line numbers may follow
after semicolons. The checker resolves every locator and validates the JSON round
trip, verdict and unchanged execution authority. A person must still check that
each cited passage supports the answer.

From the repository root, with the companion environment:

```sh
RL_FOUNDATIONS_SOLUTION="$PWD/companion/exercises/rl-foundations/starter.py" \
  python -m pytest -p no:cacheprovider companion/exercises/rl-foundations/test_acceptance.py
```

The starter intentionally fails. Select `reference_solution.py` for the reference
solution, or set the variable to a learner's file. Run the unchanged
`companion/tests/refund_agent/test_learning_case.py` separately.
