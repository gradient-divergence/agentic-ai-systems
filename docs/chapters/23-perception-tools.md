# Chapter 23: Perception and Tools

[Companion README](../../README.md)

What the refund agent reads, and what its tools promise. The demo runs the agent's
default command (`python -m refund_agent.run --seed 7`) into `refund-output/` under
the working directory, then prints what the chapter quotes from that run: the age
of each read at the moment it is used, the tool contracts, calls made directly to
the tools, the payment tool's one-key-one-payment rule, and the arithmetic of the
chapter's vision and screen-reading sections. Nothing in it changes the agent.

## Files

| What | Where |
|---|---|
| Demo | `demos/perception_tools_demo.py`, with `demos/perception_tools_enrichment.py` (offline arithmetic for the vision sections) |
| Code | `refund_agent/rules.py` (freshness limits), `refund_agent/contracts.py` (tool arguments and results), `refund_agent/tools.py` |
| Exercises | `exercises/perception-tools/` |
| Stored outputs | `book_outputs/_fnd.perception-tools.*.txt` (16 files) |
| Tests | `tests/test_perception_tools_demo.py` |

## Run it

From the companion folder, after the install in the README:

```sh
python -m demos.perception_tools_demo
python -m pytest tests/test_perception_tools_demo.py
```

## What the output shows

- The age of a read: request `r-7-0002` (2,499 cents, damaged) has a shipment record
  1,801 seconds old at use, one second past the 1,800-second limit, so it is referred
  with `stale_shipment` and no money moves. The limits are 1,800 seconds for the
  order and the shipment and 2,592,000 seconds (30 days) for the policy text. For
  the paid request `r-7-0007`, the evidence expires 1,680 seconds (28 minutes) after
  the request, by the shipment read.
- The freshness check's boundaries: 1,800 seconds is fresh, 1,801 is stale, a
  negative age is `future_shipment`, a missing age and a non-number each have their
  own reason.
- Versions: a cached policy version is refused with `non_current_policy`, and a
  policy text retrieved 2,592,001 seconds earlier with `stale_policy`.
- Contracts: the strict Pydantic records refuse an amount given as a string, an
  undeclared field and an unknown tool (`policy.check`), and nothing is paid; a lax
  schema would have accepted the same two inputs.
- The payment tool: the same key with the same arguments returns the first receipt;
  the same key with a changed amount is an `idempotency_conflict`; a new key on a
  paid request is `request_already_executed`; a blank key is refused. Two calls at
  once pay once.
- A table of the injected failures, each refused at its step, audited once, with no
  money moved.
- Convolution and vision arithmetic: cross-correlation against convolution on a
  small example, output sizes and parameter counts, receptive fields, how a
  false-positive rate changes `Pr(defect | flag)`, and Wilson intervals for a
  slice's sensitivity: 0.90 from 50 defects gives about [0.786, 0.957], and from 500
  about [0.871, 0.923], at a given defect rate of 0.02 (about 2,500 and 25,000
  parts).
- Reading a screen: a constructed tracking page whose displayed update makes it
  1,801 seconds old, and a click located by intersection over union: an offset of 10
  pixels gives 0.39 and is refused at 0.5, an offset of 4 gives 0.68 and is accepted.
- The worked answers for the chapter's numbered exercises, and the JSON Schema the
  shared tool arguments generate.

## Exercises

Exercise 10 adds an eighth tool, `carrier.track`, with a carrier read phase and a
1,800-second freshness limit. `exercises/perception-tools/reference.patch` is a
worked reference. This command applies it to a temporary copy of `refund_agent`,
runs `test_acceptance.py`, and rewrites the receipt `run-10.json` with the command,
environment, source hashes, time and pytest result:

```sh
python exercises/run_reference.py perception-tools
```

The book's agent is not patched. The checks cover the 1,800 and 1,801-second
boundary, future and missing timestamps, time-zone validation, absent records,
undeclared arguments, a forged model age, rereads at execution and unread evidence.
The new refusal reasons are `stale_carrier_event` and `future_carrier_event`. The
acceptance file sits outside the default `tests/` suite.

## What it leaves out

A contract checks form, authority, age and version, not truth: a fresh, well-formed
record that is wrong passes every check. The vision and screen sections are
arithmetic on constructed numbers; no image model runs.
