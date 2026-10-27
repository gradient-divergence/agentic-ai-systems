# Chapter 31: Batch and Streaming Pipelines

[Companion README](../../README.md)

The refund agent reads a carrier's shipment scans, and a scan reaches the agent's
store some time after it was made. This chapter's code keeps every scan with both
of its times, event time (when the scan was made) and arrival time (when the store
saw it), and decides what the agent may treat as known at each moment. A fast read
pays at once on whatever has arrived; an event-time read holds a claim until a
watermark says the evidence is complete. Everything runs offline on a constructed
stream of 14 days, so every number below is reproducible.

## Files

| What | Where |
|---|---|
| Demo | `demos/batch_stream_demo.py` |
| Code | `refund_agent/batch_stream/` (log, views, watermark, windows, replay, tools) |
| Frozen inputs and decisions | `demos/data/batch-stream-records.json` |
| Exercises | `exercises/batch-stream/` (exercises 1 to 6) |
| Stored outputs | `book_outputs/_fnd.batch-stream.*.txt` (7 files) |
| Tests | `tests/refund_agent/test_batch_stream.py`, `tests/test_batch_stream_demo.py`, `tests/test_batch_stream_review_fixes.py` |

## Run it

From the companion folder, after the install in the README:

```sh
python -m demos.batch_stream_demo
python -m pytest tests/refund_agent/test_batch_stream.py tests/test_batch_stream_demo.py tests/test_batch_stream_review_fixes.py
```

The demo prints the seven stored outputs in this order: the opening case
(`opening-01`), the first day's counts (`counts-01`), the 700-claim comparison as
printed in the chapter (`claims-01`) and in full with the generator's settings
(`claims-full-01`), the replay (`replay-01`), the data behind the chapter's figure
(`figure-01`) and the exercise arithmetic (`keys-01`). Importing the demo prints
and writes nothing. `--write-data` rewrites the frozen records and the two exercise
inputs, `scans-03.json` and `payment-05.json`.

## The code

`refund_agent/batch_stream/` keeps an append-only carrier log of scan events and
builds event-time views over it. Its watermark is the latest scan time seen, less
a nine-hour margin. Allowed lateness is 27 hours: a day's count closes when the
watermark passes the day's end plus 27 hours, and later scans go to correction.

Two tool sets read the same log. `LogShipmentTools` is the fast read. With
`EventTimeTools`, a claim that is eligible but lacks a delivery scan is held as
`shipment_incomplete` until the watermark passes the request time plus the
allowed lateness, and is checked again at execution. A claim whose delivery scan
has already arrived can be denied at once; only absent evidence waits. Every later
run rereads the order and the policy at the time it is given.

The chapter adds a versioned component to chapter 29's release suite. Add it and
its must-pass cases with `batch_stream.suite.extend(suite)`, or pass
`batch_stream.component.selector()` to `release.suite.build_suite(components=...)`.
`release.candidate.build_candidate(suite)` selects exactly those components; pass
the scan feed through `release.candidate.build_tools` with `event_log=log`. A
selected component refuses to run without a log. The release record hashes the
component's code, version and settings beside the fixed core, and registering the
component without selecting it leaves chapter 29's release identity unchanged. The
release rule, its journal and shadow admission stay chapter 29's.

The event log, the watermark and the checkpoints live in memory. The component
registers one local run journal, owned by `morgan` (an owner listed in
`refund_agent/release/owners.json`), in append mode: reads and appends are
allowed, and replacing, truncating or deleting the journal is refused, so earlier
entries stay in place.

## What the output shows

**The generator.** The stream has 400 scans a day for 14 days, 5,600 in all, and
sends one scan in fifty twice. The generator's 95%, 4% and 1% are selection probabilities for each scan's delay: 95% for an arrival within minutes (a
mean of 120 seconds), 4% for a depot sync 1 to 9 hours later, and 1% for a
contractor's daily file 12 to 36 hours later. They are the chances of each draw,
not shares the stream must hit. The stored 14 days realize 5,307, 224 and 69 of
the 5,600 original scans; copies are not counted.

**The opening case.** A parcel's shipping scan was made on January 18 at 18:05.
Its delivery scan was made on January 20 at 11:40 but arrived at 20:30, 31,800
seconds later. A refund request for 3,499 cents arrives at 18:05 on January 20,
and the shipment record it reads was observed at 18:03, 120 seconds earlier. The
fast read pays it
(`approve, eligible; executed True`). The event-time read holds it (`refer,
shipment_incomplete`), and once the evidence is final, at 06:08 on January 22,
it denies it (`deny, already_delivered`). The delivery scan was not late relative
to the watermark. One correction is recorded against the fast payment, and the
paid amount stays 3,499 cents: corrections never change the ledger.

**The first day's counts.** `counts-01` counts the stream's first day, January
20, at three moments. At midnight the fast count reads 398 against the day's 400:
7 copies have arrived and 9 scans are still out, and 398 = 400 + 7 - 9. When the
watermark passes midnight, at 09:02:52 on January 21, the fast count reads 406.
Once the count is final, at 12:03:04 on January 22, 36 hours and 3 minutes after
the day ends, it reads 408 against a final count of 400. Counted by the
day they arrived, the same log gives 403. The pane column says whether the count
is provisional, on time or final. Over the 14 days, 69 of the 5,600 scans were
late, a share of 0.012321 with a Wilson 95% interval of [0.009748, 0.015563].

**The 700 claims.** 552 claims whose delivery scan has already arrived are denied
at once. The other 148 have no delivery scan yet, and three rules decide them:
pay on what has arrived (fast), wait for the watermark, or wait until the
evidence is final.

| Rule | Paid | Paid, then contradicted | Share contradicted | Honest claimants' mean wait |
|---|---|---|---|---|
| fast | 148 | 8 | 0.054054 | 0.00 hours |
| watermark | 144 | 4 | 0.027778 | 9.06 hours |
| final | 140 | 0 | 0.000000 | 36.04 hours |

The 140 honest claimants are paid under every rule; waiting buys fewer
contradicted payments at the cost of a longer wait. Over all 700 requests the
evidence becomes final after a mean of 36.06 hours, 36.24 at the longest, and the
longest original scan delay is 35.75 hours. Corrections are counted twice, once `executed` and once `shadow`,
because the shadow count replays the same recorded payments and moves no money.

**The replay.** Replay takes an arrival-time cutoff and records the processing
version (`batch-stream-1`). It rebuilds the views, counts each scan identity once
and never calls a tool. The original replay reads the log through 18:05 on
January 20 and reproduces the hold; the corrected replay reads it through 06:08 two days after the request, January 22 for the January 20 request, and reproduces
the denial. Both match the recorded decisions, and the ledger is unchanged. Each
correction carries its mode (`executed` or `shadow`), the payment and request
identifiers, the contradicting scan's times and both statuses. Chapter 32 reads
the executed corrections of its cohort; chapter 34 reads both modes.

## Exercises

`exercises/batch-stream/README.md` describes the inputs: `scans-03.json`, ten
records (nine distinct scans) for exercise 3, whose fixture's watermark is
`W(t) = t - 2 hours` with six hours of allowed lateness, and `payment-05.json`,
exercise 5's planted arrival-clock error. For exercise 6, edit `starter_06.py`:

```sh
python -m pytest exercises/batch-stream/test_exercise_06.py
BATCH_STREAM_SUBMISSION="$PWD/exercises/batch-stream/reference_06.py" python -m pytest exercises/batch-stream/test_exercise_06.py
```

The untouched starter fails the three checks (copies, late scans kept in their
event day, the final cutoff with replay), and the reference solution,
`reference_06.py`, passes them. Exercises 1 to 5 have `starter_checks.py`, with one
planted fault each, and `reference_checks.py`; run `test_checks.py`, setting
`BATCH_STREAM_CHECK_SUBMISSION` to a submission's absolute path. All the
exercise arithmetic is in `book_outputs/_fnd.batch-stream.keys-01.txt`; read it
after working the exercises.

## What it leaves out

The stream is constructed, not measured, and the carrier, depot and contractor
are simulated. The log, the watermark and the checkpoints are kept in memory. No
partner validator and no canary are implemented here.
