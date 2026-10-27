# Chapter 12: Sequence Processing, Transformers and Foundation Models

[Companion README](../../README.md)

This chapter's code works through sequence models by hand: a one-unit recurrence,
gradients through time, an LSTM step, attention and a transformer block, then the
arithmetic of a small decoder, contrastive loss, in-context reading and serving.
Every number comes from given inputs or a seeded random model; nothing is a
trained language model.

## Files

| What | Where |
|---|---|
| Demo | `demos/deep_learning_demo.py` |
| Exercises | `exercises/transformers-foundation-models/` |
| Stored outputs | `book_outputs/_fnd.transformers-foundation-models.*.txt` (18 files) |

## Run it

From the companion folder, after the install in the README:

```sh
python -m demos.deep_learning_demo
```

The script runs top to bottom. Its first line, `{'l2_penalty': 0.0035}`, is
chapter 10's regularization example, which shares the file. Then come chapter
12's outputs, in the chapter's order. The last listing defines a small PyTorch
block (with an optional Lightning harness) and prints nothing. It needs the `ml`
extra (`pip install -e '.[ml]'`); without PyTorch, the script prints every stored
output and then stops with `ModuleNotFoundError: No module named 'torch'`.

## What the output shows

Reading sequences:

- `recurrence-01`: a one-unit recurrence tells the orders "paid, returned" and
  "returned, paid" apart (final states 0.707 and 0.843), although both codes sum
  to 1.5. It is still a fixed-width summary.
- `bptt-01`: the gradient through ten steps for weights 0.5, 1.0 and 1.2, linear
  and with tanh, checked against central differences. The linear factor is a
  ceiling; with weight 1 the tanh gradient already falls below 1 (0.908).
- `lstm-01`: one LSTM step with given weights, and the weight ratio of an LSTM,
  a GRU and a plain recurrence (4:3:1).
- `rollout-01`: a supplied parcel multiplier (1.05, not fitted) fed its own
  predictions. The one-step error stays at 5.00 while the free-running error
  grows to 40.71 by step 7. This illustrates the gap between teacher forcing and
  free running; it does not diagnose a trained model or show that attention
  closes the gap. Five tokens need 4 recurrent links but one causal-attention
  link; 2,048 tokens need 2,047 and one.

Attention and the block:

- `core-01` defines scaled dot-product attention and prints nothing.
- `example-01`: unmasked and causally masked weights for three positions.
- `scale-01`: dot-product variance grows with the key width (about 4, 65 and
  510 for widths 4, 64 and 512), and dividing by its square root brings it back
  near 1. Seed 0.
- `positions-01`: without position codes, attention is permutation-equivariant;
  with them, it is not.
- `block-01`: the shapes through one transformer block (seed 0).
- `loss-01`: next-token losses for "The sky is blue", mean 1.213 nats.

Foundation models and serving:

- `decoder-01`: a random decoder (seed 7) with a seven-token vocabulary, one
  greedy step, and one training step that moves only the output weights `W_U`
  (loss 2.441 to 2.283). No language model is trained.
- `contrastive-01`: a symmetric contrastive loss on constructed unit vectors,
  with a fixed temperature of 0.1: 0.127 for matched image-caption pairs, 3.461
  once captions 2 and 3 are swapped.
- `icl-01`: a constructed one-head reading mechanism. Changing the
  demonstrations changes the prediction while no weight changes; it is not
  evidence about a trained model.
- `rollout-02`: the random decoder generating from its own output; nothing in
  it is trained.
- `sota-01`: a 96 MiB key-value cache per request, and 2,098,176 attention
  scores with the cache against 2,865,409,024 without it.
- `serving-01`: an illustrative 1 GiB budget for a 24-layer configuration. At 2,
  1 and 0.5 bytes per number, 9, 19 and 41 full-window requests fit beside the
  weights. It also gives the multiplications per weight byte for prefill and
  batched decoding, and the chance that at least one of k independent calls is
  slow (1% each), checked by a 10,000-run simulation (seed 19). Counts are
  multiplications and weight traffic only; latencies are constructed, not
  measured.
- `keys-07` and `keys-12`: answers to the chapter's exercises.

The arithmetic uses given inputs; no training, held-out or frozen refund data is
read.

## Exercises

`exercises/transformers-foundation-models/` holds the generation exercise:
complete `generate` in `starter.py` so that it returns the generated vectors and
the cumulative count of attention scores, one row per step, with and without a
cache. The cached run keeps earlier keys and values and computes only the new
query's attention row. `test_acceptance.py` checks the file named by
`TRANSFORMERS_SOLUTION`; by default it runs the unfinished starter, which fails
on purpose. `reference.py` is the reference solution.

## What it leaves out

No model here is trained beyond the one head-only step, and none is a language
model. `demos/data/transformers_sequence_paths.json`, the data behind the
chapter's dependency figure, records possible reads, not learned attention
weights. Serving numbers count multiplications and bytes; they are not measured
on hardware.
