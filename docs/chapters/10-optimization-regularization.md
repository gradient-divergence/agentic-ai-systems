# Chapter 10: Training Optimization and Regularization

[Companion README](../../README.md)

The chapter controls training on problems small enough to solve exactly. The
demo runs one-weight and two-weight quadratics, momentum and Adam, a rotation
check, independent gradient noise, warmup and cosine decay, weight decay and
early stopping, and a constructed heating-energy run. It uses NumPy only, with
seed 7.

## Files

| What | Where |
|---|---|
| Demo | `demos/training_control_demo.py` |
| Exercises | `exercises/optimization-regularization/` |
| Stored outputs | `book_outputs/_fnd.optimization-regularization.*.txt` (10 files) |
| Tests | `tests/test_training_control.py` |

## Run it

From the companion folder, after the install in the README:

```sh
OMP_NUM_THREADS=1 python -m demos.training_control_demo
OMP_NUM_THREADS=1 python -m pytest tests/test_training_control.py
```

Importing the demo does no work and writes nothing. `--write-data` rewrites
`demos/data/training_control_figures.json`, the numbers behind the chapter's
step-size, path, schedule and stopping plots.

## What the output shows

Loss is half the mean squared prediction error throughout, and a penalty is
lambda/2 times the squared norm of the weights. A step count means the first
step below the threshold; an oscillating method may cross it and leave it again.

The four printed listings, `descent-01`, `valley-01`, `adam-01` and `decay-01`,
run together in the chapter's order, without the output-only code between them.
The ten stored outputs, in the chapter's order:

| Output | What it shows |
|---|---|
| `overshoot-01` | Chapter 9's two-layer ReLU network after one step along its checked gradient: at step size 0.05 the loss falls from 0.5 to 0.0028125; at 0.10 the hidden input reaches zero and the loss rises to 0.605 |
| `descent-01` | One weight with curvature 4: step size 0.25 lands on the minimum in one step, 0.50 oscillates forever, 0.60 diverges |
| `valley-01` | A two-weight valley with curvatures 4 and 0.04 (condition number 100): 653 steps at step size 0.25, 332 at 0.49, never at 0.50, divergence at 0.51; after standardizing with training statistics, one step at 1.0 |
| `adam-01` | First steps and step counts for momentum (130 at step size 1, 105 at 5) and Adam (135 at 0.1, 112 at 0.3), with and without bias correction |
| `rotate-01` | The valley turned 45 degrees: descent and momentum take the same number of steps, Adam does not (291 instead of 135 at 0.1) |
| `noise-01` | Seeded gradient noise against the stationary formula, and the warmup and cosine schedule |
| `decay-01` | A penalty and decoupled weight decay: equal for plain descent, different once an adaptive denominator divides the gradient |
| `stop-01` | 30 inputs, 40 training and 4,000 validation examples: penalty selection (lambda 1, validation loss 0.684049, against 6.008191 with no penalty) and early stopping (step 3, against 6.008133 at step 2,000) |
| `report-01` | A constructed heating-energy run whose stable loss hides two nearly unlearned directions; the decision is to refuse the raw run, and a rerun with training-fitted standardization follows |
| `keys-01` | The exercise keys |

`stop-01` and exercise 6 fit means, scales, intercepts and weights on training
data only. They compare completed fits on validation, spending that window on
the selection, so a low validation loss there is not an independent estimate.
There is no test or golden window in this chapter.

## Exercises

Exercise 5's `exercises/optimization-regularization/run-05.json` holds the
initial state and sixty exact-gradient updates of an agent's run on a
constructed hospital bed-occupancy problem. The updates are correct; the one
planted error is the agent's conclusion. `generate.py` in the same folder
rebuilds the file.

Exercise 6's `starter.py` raises `NotImplementedError` until you complete
`decay_step`. Its acceptance tests are not part of the default suite; run them
explicitly:

```sh
python -m pytest exercises/optimization-regularization/test_acceptance.py
TRAINING_CONTROL_SOLUTION=path/to/your_solution.py python -m pytest exercises/optimization-regularization/test_acceptance.py
TRAINING_CONTROL_SOLUTION=demos/training_control_demo.py python -m pytest exercises/optimization-regularization/test_acceptance.py
```

The first command fails on purpose while the starter is unfinished. The second
checks your file. The third runs the checks against the reference solution,
which is the demo itself. In exercise 6, decay in Adam's updates is decoupled
from its gradient average; it need not improve the validation loss at the three
supplied coefficients.

## What it leaves out

The exercise compares three completed training runs on validation. It does not
refit on validation, and it does not certify a winning setting. All data are
generated; no refund records are read.
