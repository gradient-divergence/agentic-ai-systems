# Supervised Learning and Calibrated Risk: Build

Fit the balanced boosted-tree classifier on training data and the Platt map on validation. Read August only after both fits, apply the fixed 0.080 gate, and justify the cost decision and a measurement that would reverse it.

Complete `build(directory, seed=7)` in `starter.py`. The result's evidence fields
are named in `test_acceptance.py`, which checks the chapter's printed acceptance
conditions. Keep the supplied data and earlier journals intact. A person also
reads the judgment or defense where the exercise asks for one.

From the companion root:

```sh
SUPERVISED_LEARNING_SOLUTION="$PWD/exercises/supervised-learning/starter.py" \
  python -m pytest exercises/supervised-learning/test_acceptance.py -p no:cacheprovider
```

Set `SUPERVISED_LEARNING_SOLUTION` to your submission's Python path. The public reference solution is
`reference_solution.py`; use that path to check it. Run it directly to print its
report. These constructed reruns grant no authority to pay or promote a change.
