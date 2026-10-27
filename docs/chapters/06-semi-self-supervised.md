# Chapter 6: Semi- and Self-Supervised Learning and Unlabeled Structure

[Companion README](../../README.md)

Half the refund history has no outcome label: an unpaid request never gets one.
This chapter's demo asks what the unlabeled requests can add: clusters,
self-training with pseudo-labels, a contrastive loss and a label budget. It reuses chapter 5's history, time windows
and boosted trees, and chapter 4's golden-set machinery, and it tests every
claim against the frozen months of August and September.

## Files

| What | Where |
|---|---|
| Demo | `demos/semi_self_supervised_demo.py` |
| Golden sets and their access journals | `refund_agent/split.py` (`GoldenSet`) |
| Exercises | `exercises/semi-self-supervised/` (exercise 6); `demos/data/semi_self_supervised_selftrain_05.json` (exercise 5) |
| Stored outputs | `book_outputs/_fnd.semi-self-supervised.*.txt` (15 files) |
| Tests | `tests/test_semi_self_supervised.py` |

## Run it

From the companion folder, after the install in the README:

```sh
BOOK_GOLDEN_DIRECTORY="$(mktemp -d)" python -m demos.semi_self_supervised_demo
python -m pytest tests/test_semi_self_supervised.py
```

Run as a script, the demo sets its numerical libraries to one thread. Importing
it fits nothing, freezes nothing, prints nothing and writes nothing.

The demo keeps each golden set's access journal on disk, in `golden-sets/` under
the working directory, or in the folder `BOOK_GOLDEN_DIRECTORY` names. In a new,
empty folder, the printed access log matches the book's two-entry record. Run it
again against the same folder and it appends a third entry, so its log differs
from the book's on purpose: a consultation, once made, stays on the record. The
`mktemp -d` above gives every run a new folder.

## What the output shows

The three printed listings, `kmeans-01`, `selftrain-01` and `contrast-01`, run
together in that order without the other code, and their setup imports are
explicit.

**The frozen months.** Before any fit, the demo freezes August and September.
After all fits, it resumes each set's durable access journal and consults each
once. On first use it replays chapter 5's actual first consultation from its
stored output and verifies the set's fingerprint; the second consultation is
appended beside the manifest, in `golden-sets/<full fingerprint>.access.jsonl`,
and each line carries the fingerprint (`holdout-01`). `GoldenSet.persist(directory)`
also works without a replay, across processes, and on parent and slice
consultations. It locks and appends the journal, rejects corruption, and never
replaces an existing log. Chapters 4 and 5 keep their original in-process
records, and their stored outputs are unchanged.

**The pools** (`pools-01`). Of 36,000 requests in the history, 14,754 are
labeled by 1 August (13,766 training, 988 validation). The open paid pool before
1 August has 985 requests, and 9,899 tier 1 requests were refused before
1 August. Both pools expose only decision-time inputs.

**Self-training** (`selftrain-01`). Confidence 0.9 and three diagnostic rounds;
each accepted pseudo-label is kept, and no case is added twice. Every accepted
pseudo-label is 0: 869 then 7 on the open pool, and 9,277, 615 and 7 on the
refused tier 1 pool, against a labeled base rate of 0.0362. The demo prints the
validation audit and the class shares, including those all-zero labels. The
fixed rounds show the damage; they do not implement the validation or
class-share stopping guard that exercise 6 asks you to add. Every choice comes
before the frozen consultations, and the agent's score and execution rule stay
unchanged. The refused pool still has no observed outcomes of its own.

**Label budgets** (`budget-01`). Twenty label draws. Three fictional
service-desk queues are drawn as clusters (a pool of 3,000 tickets), and a
random disjoint holdout of 1,000 is fixed before fitting. Three-nearest-neighbor
classification compares random labels, self-training and cluster
representatives: with 6 labels, random labels average 0.6636 accuracy and
labels on the tickets nearest the k-means centers 0.9634. The refund experiment
hides training labels and compares boosted trees with and without self-training
on validation, by log loss and by absolute distance from the planted risk.

The bands are percentiles across label draws, not uncertainty from new holdout
samples. Cluster representatives use the same number of labels as random
acquisition but select different cases; the supervised and self-trained methods
share exactly the same labeled cases.

`figures-01` holds the data behind the chapter's scatter, label-budget and
contrastive-margin figures. `keys-01` prints the exercise keys, including the
honest frozen-holdout figures from the same fitted models.

## Exercises

Exercise 5's fixture, `demos/data/semi_self_supervised_selftrain_05.json`, is a
constructed self-training report on the three service-desk queues, with a single
planted evaluation defect; its key is in `keys-01`. `--write-fixture` rebuilds
it.

Exercise 6 asks you to add the stopping guard to `self_train` in
`exercises/semi-self-supervised/starter.py`. Keep its signature, and return the
fitted model and per-round dictionaries with `accepted` (the candidate count),
`share_one`, the original labeled `base_rate`, `guard` (`stop` or `continue`) and
`total` (labels actually added). Print the candidate count, share, base rate and
guard decision for each round. Stop before adding labels or refitting if the
share leaves the inclusive band from 0.5 to 1.5 times the original labeled base
rate.

The acceptance tests are outside the default suite. Point
`SEMI_SELF_SUPERVISED_SOLUTION` at your file, or at `reference.py` for the
reference solution; the untouched starter fails on purpose:

```sh
SEMI_SELF_SUPERVISED_SOLUTION=path/to/your_solution.py python -m pytest exercises/semi-self-supervised/test_acceptance.py
SEMI_SELF_SUPERVISED_SOLUTION=exercises/semi-self-supervised/reference.py python -m pytest exercises/semi-self-supervised/test_acceptance.py
```

They cover both band edges, rejection above and below the band before refitting,
the original labeled base rate in later rounds, the 9,277-candidate seed-7
stopped round, repeatability and zero frozen-window consultations.

## Tests

`tests/test_semi_self_supervised.py` covers arithmetic and finite differences,
the standalone listings, time-safe pools, feature-only reads, all fits before
the consultations, journals across processes, fixture provenance, seeded
reproduction, and byte-identical regeneration of the stored outputs of
chapters 4, 5 and 6.

## What it leaves out

No encoder is trained, and no result changes the refund agent or its proposed
score. The demo's fixed self-training rounds have no stopping guard; that is
exercise 6.
