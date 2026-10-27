# Chapter 14: Prompting, Fine-Tuning and Retrieval Adaptation

[Companion README](../../README.md)

This chapter's code compares three ways to adapt the refund agent's drafting:
a structured prompt carrying the whole policy, the same prompt with dated clauses
from chapter 11's index, and an adapter that carries the policy terms itself. It
also works through low-rank adapters, quantization and cost on small examples.
No language model is trained, and no live model or network access is needed.

## Files

| What | Where |
|---|---|
| Demo | `demos/adaptation_demo.py` |
| Code | `refund_agent/adaptation/` (prompt, arms, drafter, experiment, suite, `lora.py`) |
| Stored outputs | `book_outputs/_fnd.prompting-finetuning-rag.*.txt` (13 files) |
| Tests | `tests/refund_agent/test_adaptation.py` |

## Run it

From the companion folder, after the install in the README:

```sh
OMP_NUM_THREADS=1 python -m demos.adaptation_demo
OMP_NUM_THREADS=1 python -m pytest tests/refund_agent/test_adaptation.py
```

`--write-data` rewrites the figure inputs, the selection records and exercise
5's deliberately flawed report under `demos/data/`.

## What the code does

`refund_agent/adaptation/` runs three drafting arms outside `Agent`: the whole
policy returned by `policy.retrieve`, dated clauses from chapter 11's index, or
no policy text with the terms carried by an adapter. All three use
`ScriptedDrafter` through `ModelClient`, the same evidence and the same checks.
The adapter looks up, whole, the terms of the latest policy version cited by
training drafts. `lora.py` separately trains a frozen-layer toy by gradient
descent and checks the low-rank arithmetic. The agent's default
`ScriptedModel`, loop, tool contracts and generated data are unchanged.

## What the output shows

- `prompt-01`: the drafting prompt, section by section, with word counts. Words
  and punctuation stand in for tokens; these are not a deployed tokenizer's
  counts. The first three sections share 300 words, inside chapter 12's
  illustrative 2,048-word window. A malformed reply is refused.
- `experiment-01` and `results-01`: the comparison on 121 golden requests.
  `results-01` is the short table the chapter prints.

| Arm | Agreement (95% interval) | Only challenger / only prompt | Forbidden |
|---|---|---|---|
| Prompt | 121/121 [0.969, 1.000] | -- | 0/121 |
| Retrieval | 121/121 [0.969, 1.000] | 0 / 0 | 0/121 |
| Adapter | 83/121 [0.599, 0.762] | 0 / 38 | 0/121 |

  The prompt is selected; neither challenger is adopted. The adapter agrees on
  61/61 August requests but 22/60 September ones: it learned July's terms (no
  development draft is labeled September), so it makes 33 wrong denials citing
  July and misses 5 requests newly inside the return window.
- `paired-01`: a paired comparison on 200 constructed hospital notes, whose
  interval excludes zero where an independent one does not, then the exact test
  on discordant cases. Adoption needs a positive interval at the split's level
  (0.025) and an exact two-sided chance no larger than it: five discordant wins
  to none in 121 cannot pass; seven can.
- `suite-01`: the spent golden cases become a regression suite with 5 injected
  failures, 21 policy-change cases and 484 frozen retrieval questions. No
  passing case may start failing and no forbidden draft may appear; the prompt
  passes and the adapter fails 38 cases.
- `lora-01`, `lowrank-01`, `objective-01`, `quantize-01`: low-rank adapter
  sizes and scaling; a seeded 16-by-16 frozen layer where ranks 2, 4 and 8 come
  within 10% of a full update; the draft-only training loss; and 8-bit and
  4-bit quantization of the trained toy.
- `resources-01` and `cost-01`: context and draft words per arm, and
  constructed prices. With a $0.001 word price and a $4.00 redo, the adapter
  costs $1.67 per accepted draft against the prompt's $0.47.
- `keys-01` and `keys-02`: answers to the exercises.

The data is seed 7: 398 training requests, 81 July settings requests and 121
golden requests. The golden labels had already been read once, on 10 October
2026, before the experiment was commissioned; the log records the experiment as
the second access, so its result was already known. A later adaptation decision
needs untouched cases.

## Exercises

The exercises have no folder; their keys are in `keys-01`. Exercise 5 audits
`demos/data/adaptation_report_05.json`, a deliberately flawed report. Exercise
6's `constructed_requests` helper adds twenty November regression cases without
changing the golden manifest or adding them to training. When forbidden drafts
occur, the output gives a Wilson interval; the zero-failure bound applies only
to a clean run.

## What it leaves out

No language model is trained or called; the adapter is a lookup of terms, and
word counts stand in for tokens. No latency or price is measured; the costs are
constructed. The regression suite has no release authority here; the agent
evaluation chapter adds it.
