# Chapter 9: Neural Networks and Backpropagation

[Companion README](../../README.md)

The chapter's network is small enough to check by hand: four counted words feed
two ReLU units and three class scores. The demo runs its forward pass, its
backward pass, a finite-difference gradient check and a short training run, in
NumPy, on generated text only.

## Files

| What | Where |
|---|---|
| Demo | `demos/backprop_demo.py` |
| Generated data | `demos/backprop_data.py`: three constructed tables of fifteen patterns each, for refund requests, incident tickets and clinic messages |
| Exercises | `exercises/neural-networks-backprop/` |
| Stored outputs | `book_outputs/_fnd.neural-networks-backprop.*.txt` (11 files) |
| Tests | `tests/test_backprop.py` |

## Run it

From the companion folder, after the install in the README:

```sh
OMP_NUM_THREADS=1 python -m demos.backprop_demo
OMP_NUM_THREADS=1 python -m pytest tests/test_backprop.py
```

The demo needs only NumPy. Importing it prints nothing and runs no experiment.
`--write-data` rewrites `demos/data/backprop_figures.json`, the numbers behind
the chapter's plots.

## What the output shows

The four printed listings, `forward-01`, `backward-01`, `check-01` and
`train-01`, run in the chapter's order with no setup between them. Seven more
outputs cover the scalar network that chapter 10 picks up (`scalar-01`), a ReLU
corner (`kink-01`), the precision of finite differences (`epsilon-01`), training
step sizes (`curves-01`), a delegated backward pass (`delegated-01`),
initialization (`depth-01`) and the exercise keys (`keys-01`).

- **Forward pass.** The network has 19 parameters. On the chapter's request its
  loss is 2.407605964, against 1.098612289 for a uniform guess.
- **Gradient check.** All 19 parameters are nudged up and down, 38 forward
  passes plus one base pass, in float64 with epsilon 0.00001. No nudge switches
  a unit, and the largest relative gaps are 5.110e-12 for `W1` and `b1` and
  2.238e-11 for `W2` and `b2`.
- **Training.** The default request draw has 300 examples, seed 7, and class
  probabilities 0.55, 0.25 and 0.20 for damaged, changed_mind and
  not_delivered. With the ReLU gate, 300 updates at step size 0.2 take the
  training loss from 2.296565265 to 0.103123609, with 287 of 300 correct, but
  the loss rises once, at step 61. With the gate removed, the loss falls at
  every update and ends at 0.193380547 with 280 of 300 correct: a missing gate
  can lower the loss at every update and still fail the gradient check. Four
  counted words cannot separate every generated label.
- **Corners.** A central difference across a ReLU's corner is not an estimate
  of a derivative, whether or not the sum started exactly at zero.
  `central_difference` returns its estimates together with the entries whose
  nudge, up or down, switched a unit on or off, and `check_arrays` sets those
  entries aside before it reports a gap. At the hand-set weights, 229 of the 300
  requests have a unit exactly at zero.
- **Delegated code.** `delegated-01` checks a backward pass handed over by
  someone else, on a constructed IT service desk (300 tickets, three queues).
  Its `W1` and `b1` gradients are wrong, so the decision is to reject it, and
  the output prints the corrected line.
- **Depth.** Ten ReLU layers of width 200, 200 generated inputs and five seeds
  (7 to 11), followed by a three-score softmax head. The head's small weight
  scale is fixed across the three initialization variances. At epsilon 0.00001
  the check screens out nudges that cross ReLU gates before checking 25 sampled
  weights in each of layers 1, 5 and 10; it prints both the excluded counts and
  the unscreened errors.

A correct gradient can still produce dead units, or a rising loss with a long
step. A passing check says that the coded gradient matches the coded loss,
locally.

## Exercises

Exercise 7 hands you a delivery to audit for a constructed clinic's message
router (three desks: appointments, prescriptions and billing):
`check_report_07.py` and the report it wrote, `check-report-07.json`. One error
is planted in the delivery, and finding it is the exercise, so neither file says
where it is. To print the report again:

```sh
python exercises/neural-networks-backprop/check_report_07.py
```

(`--write-report` rewrites the JSON beside the script.) The keys to the
chapter's exercises, exercise 7 included, are in `keys-01`; read it after you
have tried them.

## What it leaves out

Every loss reported here is a training loss on generated data, not a held-out
estimate. The demo uses none of the refund agent's code or records.
