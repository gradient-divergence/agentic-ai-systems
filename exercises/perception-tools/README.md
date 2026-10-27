# Perception and Tools, exercise 10

Reference solution: `reference.patch`. It adds a carrier-observed `CarrierEvent`,
`carrier.track`, the carrier read phase and a 1,800-second freshness limit to an
isolated copy of `refund_agent`. Reads return the latest event and its observation
time. Execution reads it again and recomputes its age from the source.

Run from the repository root with the companion environment's Python:

```sh
python companion/exercises/run_reference.py perception-tools
```

The runner applies the patch to a temporary copy, executes `test_acceptance.py`
and writes `run-10.json` with the command, environment, source hashes, timestamp
and pytest result. It does not patch the book's runnable agent. As in the latest
chapters, the acceptance file lives outside the default `companion/tests` suite.

Checks cover the 1,800/1,801-second boundary, future and missing timestamps,
timezone validation, absent records, undeclared arguments, a forged model age,
execution rereads, unread evidence and the instructor split. The refusal reasons
are `stale_carrier_event` and `future_carrier_event`. The `carrier_event` freshness
key avoids confusing the new refusal with the existing `stale_carrier` scenario
label, which denotes a stale shipment source.

This exercise adds an eighth tool. The book's exact-seven-tool namespace test and
contract lint still apply to the unchanged canonical agent, not this exercise
copy. Its older fixtures lack the new evidence field and are not an acceptance
suite for this extension. No historical nine-test or full-suite count is reused.
