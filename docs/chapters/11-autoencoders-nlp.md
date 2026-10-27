# Chapter 11: Autoencoders and NLP Representations

[Companion README](../../README.md)

This chapter's code turns refund-policy clauses into vectors and builds a dated
policy index for the refund agent. It compares word counts with the codes of a
linear autoencoder, and shows that a clause's dates, not its vector, decide which
policy version answers a question.

## Files

| What | Where |
|---|---|
| Demo | `demos/autoencoders_nlp_demo.py` |
| Code | `refund_agent/retrieval/` (clauses, encoders, index, questions, evaluation) |
| Exercises | `exercises/autoencoders-nlp/` |
| Stored outputs | `book_outputs/_fnd.autoencoders-nlp.*.txt` (7 files) |
| Tests | `tests/refund_agent/test_retrieval.py` |

## Run it

From the companion folder, after the install in the README:

```sh
python -m demos.autoencoders_nlp_demo
python -m pytest tests/refund_agent/test_retrieval.py
```

The demo needs only the core dependencies and writes nothing.
`--write-data` rewrites the figure inputs and exercise 5's report under
`demos/data/`.

## What the code does

`refund_agent/retrieval/` supplies count vectors (optionally weighted by rarity),
a centered linear autoencoder fitted by SVD or by untied gradient descent, and an
exact policy index. Every clause keeps its citation, effective interval, text
hash, source-read time, encoder identity and vector. Retrieval filters by date
before it ranks, refuses a policy read older than 30 days (`stale_policy`), and
breaks similarity ties by citation. A question made only of unknown words
returns no clause. This index is meant to sit behind `policy.retrieve` later; the
agent's tool contract is unchanged.

## What the output shows

The demo prints seven outputs, in the chapter's order:

- `counts-01`: the policy has ten clauses and a vocabulary of 26 words. The
  worked question has the same cosine, 0.617, with the eligibility clause of
  all five policy versions, so similarity alone cannot pick a version.
- `autoencoder-01`: a four-point example of the linear autoencoder, then the
  clause codes. Clause reconstruction error is 0.720 at width 1 and 0.560 at
  width 2; the closed form and untied gradient descent agree (0.559541149 and
  0.559541155).
- `index-01`: one indexed clause with its interval, read time and manifest; a
  read 30 days and one second old is refused as `stale_policy`.
- `recall-01` and `recall-02`: recall on the frozen requests. `recall-02` is
  the table the chapter prints. With dates, counts and both linear codes reach
  242/242 at rank 1 on policy wording. On customer wording, counts reach
  121/242 and the linear codes 0/242. Lower reconstruction error does not earn
  the codes a place in this index.
- `versions-01`: the worked request `r-7-0117` (damaged, 33 days after
  delivery, 65,000 cents). Similarity alone picks the January clause, the
  newest-version rule the September clause, and the dated index the July clause
  in force. Over the 242 frozen questions at rank 1: similarity 0/242, newest
  120/242, dated 242/242.
- `keys-01`: answers to exercises 3, 4 and 6.

The data is seed 7 with chapter 4's split: 398 training requests, 81 July
validation requests and 121 frozen requests. Each request yields an eligibility
question and a cap question, each in two preset wordings; the relevance label
supplies the policy version, never words in the question. All settings are fixed
on July before one logged read of the frozen set. The version comparisons use all
five policy versions after the fact, with a simulated fresh source read one hour
before each question, so they do not measure a live ingestion system. Wilson
intervals treat questions as independent trials, although two questions share a
request.

## Exercises

`exercises/autoencoders-nlp/` holds exercise 6: extend the policy history with a
constructed 1 October 2026 version, write questions across that boundary, and
build a dated index. Complete the three functions in `starter.py`, then run:

```sh
AUTOENCODERS_NLP_SOLUTION=path/to/your_solution.py \
  python -m pytest exercises/autoencoders-nlp/test_acceptance.py
```

Without the variable, the test runs the unfinished starter, which fails on
purpose; `reference.py` is the reference solution. The key runs 20 boundary
questions and checks historical recall again, with counts and with rarity
weighting. Exercise 5's report, `demos/data/autoencoders_nlp_index_report_05.json`,
is deliberately wrong: it leaves both September clauses' `effective_to` open
after an October version has been added.

## What it leaves out

`PairEncoder` (NumPy cosine InfoNCE, trained only on pre-July question and
passage pairs) is implemented and its gradient is tested, but it has no row in
the comparison. The index is not yet wired into the agent's `policy.retrieve`,
and the version comparisons simulate source reads rather than measure a live
ingestion system.
