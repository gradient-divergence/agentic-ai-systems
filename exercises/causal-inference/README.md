# Causal Inference and Experiments: Build

Build the capped assignment plan on the complete validation population, then report its balance, expected cost, worst case and sample size. Do not run payments, create reviews or consult a frozen set.

Complete `build(directory, seed=7)` in `starter.py`. The result's evidence fields
are named in `test_acceptance.py`, which checks the chapter's printed acceptance
conditions. Keep the supplied data and earlier journals intact. A person also
reads the judgment or defense where the exercise asks for one.

From the companion root:

```sh
CAUSAL_INFERENCE_SOLUTION="$PWD/exercises/causal-inference/starter.py" \
  python -m pytest exercises/causal-inference/test_acceptance.py -p no:cacheprovider
```

Set `CAUSAL_INFERENCE_SOLUTION` to your submission's Python path. The public reference solution is
`reference_solution.py`; use that path to check it. Run it directly to print its
report. These constructed reruns grant no authority to pay or promote a change.
