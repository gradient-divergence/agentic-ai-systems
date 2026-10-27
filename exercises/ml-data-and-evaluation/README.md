# ML Data and Evaluation: Build

Freeze the complete golden set and its September slice with one shared read journal. Report baseline errors and classifier recall and precision, with Wilson intervals, separately by tier.

Complete `build(directory, seed=7)` in `starter.py`. The result's evidence fields
are named in `test_acceptance.py`, which checks the chapter's printed acceptance
conditions. Keep the supplied data and earlier journals intact. A person also
reads the judgment or defense where the exercise asks for one.

From the companion root:

```sh
ML_DATA_EVALUATION_SOLUTION="$PWD/exercises/ml-data-and-evaluation/starter.py" \
  python -m pytest exercises/ml-data-and-evaluation/test_acceptance.py -p no:cacheprovider
```

Set `ML_DATA_EVALUATION_SOLUTION` to your submission's Python path. The public reference solution is
`reference_solution.py`; use that path to check it. Run it directly to print its
report. These constructed reruns grant no authority to pay or promote a change.
