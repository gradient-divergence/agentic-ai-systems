# Exercise 6: judge Candidate C

For a check without opening the evaluation package, run
`python -m pytest exercises/practical-rl/test_unsealed_acceptance.py -p no:cacheprovider`
from the companion root, with `PRACTICAL_RL_SOLUTION` selecting the submission.
This checks Candidate C's specification, preserved Chapter 4 golden evidence,
and independently recomputed support and estimates on the shipped constructed
exercise-5 data. The complete Candidate C record and evaluation-package diagnostics
require a separately authorized evaluation run. The public task is to implement
the fixed policy and pass `test_unsealed_acceptance.py`; its report can state these
checks and their limits without producing the complete evaluation record.
The exercise-5 fixture has unsupported not-delivered dated notices; its estimate
checks use only the supported contexts. That fixture check does not establish
Candidate C's full frozen-window verdict.

`candidate_c.json` chooses the referenced notice for not-delivered requests and
the standard notice elsewhere. It is fixed before reading the frozen window.
Complete `candidate_c(frozen_at)` in `starter.py` to return that fixed `Policy`.
The provided evaluator writes the rule before consulting outcomes once, delivers
the record JSON and its rendered output, and follows the rule's decision.
The candidate has support; the exercise does not promise that it will pass.

From the companion root, with the existing companion environment:

```sh
export PYTHONPATH="$PWD" OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
python -m pytest exercises/practical-rl/test_unsealed_acceptance.py -p no:cacheprovider
```

The unfinished starter fails with `NotImplementedError` for Candidate C's policy.
Set `PRACTICAL_RL_SOLUTION` to a submission's Python path to check another file.
The reference solution is checked with:

```sh
PRACTICAL_RL_SOLUTION="$PWD/exercises/practical-rl/reference_solution.py" \
  python -m pytest exercises/practical-rl/test_unsealed_acceptance.py -p no:cacheprovider
```

A complete evaluation run additionally produces
`candidate-c-output/candidate-c-record.json` and
`candidate-c-output/candidate-c-output.txt`. Those records require the separately
available evaluation package and its full acceptance check. They are outside the
public check above. Its preserved journals must not be truncated for a rerun.
