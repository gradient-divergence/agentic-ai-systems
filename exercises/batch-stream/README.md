# Chapter 31: batch and streaming

`scans-03.json` is exercise 3's ten records (nine distinct scans). Its UTC event
day starts at `2026-01-20T00:00:00+00:00`. A separate background feed supplies
`W(t) = t - 2 hours`; the allowed lateness is six hours. These are constructed
inputs. Use event time to place scans and arrival time to decide what was known.

`payment-05.json` is exercise 5's planted arrival-clock error. Diagnose it from
the decision and the raw log, including the copied shipping scan. A correction
records the discrepancy and leaves the recorded payment standing.
The background parcel's scan supplies the recorded observation time and the
watermark, 9 hours behind its scan time. This watermark is before the request;
a damaged claim's return window uses the delivery's event time and does not
require the final-evidence wait for a missing delivery scan. The key still
compares 745 event-time hours with 697 arrival-time hours against a 720-hour
limit.

For exercise 6, edit `starter_06.py` and run from the companion directory:

```sh
python -m pytest exercises/batch-stream/test_exercise_06.py
```

The untouched starter deliberately fails the three checks: copies, late scans
assigned to their event day, and the final cutoff with replay. The reference
solution is `reference_06.py`. To check a submission at another path:

```sh
BATCH_STREAM_SUBMISSION=/absolute/path/submission.py \
python -m pytest exercises/batch-stream/test_exercise_06.py
```

Exercises 1 to 5 also have `starter_checks.py`, a separate fault for each:
confusing delay with age, using fast evidence for irreversible acts, counting
copies as distinct, ignoring the base rate, and measuring from arrival.
`reference_checks.py` is the reference solution. Run `test_checks.py`, setting
`BATCH_STREAM_CHECK_SUBMISSION` to a submission's absolute path if needed.
The untouched checks starter fails all five, and the reference passes all five.

Exercise 1 uses the opening's two timestamps and observation age. Exercise 2
uses the fixed authority table in `refund_agent/batch_stream/authority.py`.
Exercise 4 uses the base rate and delayed-event chances given in the chapter.
All arithmetic is reproduced in `_fnd.batch-stream.keys-01.txt` under
`book_outputs/`. Reference solutions should be read after working the exercises.
