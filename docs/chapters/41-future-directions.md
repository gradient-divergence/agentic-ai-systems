# Chapter 41: Future Directions and System Renewal

[Companion README](../../README.md)

The chapter's code asks whether a replacement score may take over from the
incumbent rule, and how its adoption would be watched. The demo signs a
replacement requirement before any evidence, compares the selected trees, map
and threshold from chapter 8 with the incumbent's fixed text rule on fresh
requests, and decides `keep`, `rebuild` or `replace`. No comparison pays a
refund. Without the records of the earlier chapters' runs, the decision is
`keep`.

## Files

| What | Where |
|---|---|
| Demo | `demos/future_directions_demo.py` |
| Code | `refund_agent/renewal/` (requirement, windows, evidence, decision, adoption, watch); `refund_agent/future_directions_risk.py` (the runtime component) |
| Exercises | `exercises/future-directions/` |
| Stored outputs | `book_outputs/_fnd.future-directions.*.txt` (9 files) |
| Tests | `tests/refund_agent/test_future_directions_{math,renewal,risk,watch}.py` |

## Run it

From the companion folder, after the install in the README:

```sh
OMP_NUM_THREADS=1 python -m demos.future_directions_demo --run-dir runs/future-directions
python -m demos.future_directions_demo --read-run --run-dir runs/future-directions
python -m refund_agent.renewal decide --run-dir runs/future-directions
OMP_NUM_THREADS=1 python -m pytest tests/refund_agent/test_future_directions_renewal.py
```

The first command writes the run's records into the run folder, with missing
records under `missing/`. `--read-run` reprints a finished run without rerunning
it, and checks that the stored decisions match a recomputation. The `decide`
command prints the decision as JSON and exits 1 for `keep` or `rebuild`. A run
folder is immutable: after a change to the code or the requirement, use a fresh
folder. Set `OMP_NUM_THREADS=1` for the renewal tests too; with more threads the
model fit starts a worker pool, which the release guard refuses.

## What the output shows

`requirement-01` is the signed requirement, as a table: the incumbent and the
candidate, the cases (the first 2,000 incumbent-paid tier 1 requests after
opening, drawn forward and disjoint from earlier data), calibration (mean
prediction inside the inclusive 95% Wilson interval), cost (candidate minus
incumbent below zero at the 95% upper end, a tie keeps the incumbent, a signed
$4 review cost), counter-measures, release, rebuild (once, after a calibration
failure alone), reversal and budget rules, signed by morgan on day 0.

`decision-01` judges window 1: 2,000 paid requests with 136 wrong payments, an
observed rate of 0.068000 with Wilson 95% [0.057774, 0.079882], against a mean
prediction of 0.047724, so calibration fails. Cost passes: the candidate saves
$0.462240 a request, 95% [$0.234899, $0.701511] by a paired bootstrap, with 261
referrals and 1,305 added blocking review minutes. Wrongful denials are 0 against
0. Capacity, incumbent admission and release fail because their records are
missing. Decision 1 is `keep` on day 91.391222, when the last outcome closes.
`decision-02` follows: window 2 is not generated, so no second candidate is
judged and no `replace` verdict is issued. `adopt-01` holds adoption and lists
the seven missing records, and `watch-01` claims no post-adoption month.

`power-01` and `confirmation-01` size the watch. With 2,000 paid requests, a
prediction of 0.049 and an actual rate of 0.065, the check rejects from 117
wrong payments, with power 0.890968989; a correct score fails 0.055023377 of the
time. Over twelve independent monthly batches, a single failed month reverses a
correct score with yearly probability 0.492947676; requiring a matured
confirmation on the next month's separate batch lowers it to 0.031339779.

`threshold-01` shows why calibration in the large is not enough: two bands of
1,000 paid requests, predicted .020 and .120 but observed .060 and .080, pass
the aggregate check at .070 while the threshold band fails. Window 1's band
around the operating threshold, [.060, .100], has 54 paid requests, 6 wrong, a
mean prediction of 0.074800 and Wilson 95% [0.051930, 0.221949].

The watch, once a replacement is adopted, persists its state in an append
journal. A failed month makes the next 2,000 requests a confirmation cohort and
the watch pending; the confirmation reads only matured paid outcomes, then
clears or reverses. A spent budget withdraws tier 1 at once, before the
rollback; restoring the old bundle never restores authority.

## Exercises

Exercises 1 and 2 are written readings. `keys-01` gives exercise 3's numbers and
exercise 4's route arithmetic. Exercise 5 diagnoses `decision_05.json`, a
constructed team's decision with one wrong reversal signal. Exercise 6 repairs
`watch_06.py`:

```sh
python -m pytest exercises/future-directions/test_exercise_06.py
FUTURE_DIRECTIONS_EXERCISE_REFERENCE=1 python -m pytest exercises/future-directions/test_exercise_06.py
```

The untouched starter fails five checks; the second command checks the
reference. The reference fixtures are not an adoption, a grant of authority or a
deployment.

## What it leaves out

The partner records that a replacement needs: the capacity design and its
demand (chapter 37), the signed release (chapter 29), the stage 6 registry and
bundle identities (chapter 33), the held launch record (chapter 39), the
withdrawal client (chapter 38) and the budget callback (chapter 34). With them,
`RenewalInputs` would run window 2, adoption and the first watched month; here
those outputs say they are absent. The annual figures assume independent,
homogeneous, all-paid monthly batches; real paid subsets after adoption differ.
