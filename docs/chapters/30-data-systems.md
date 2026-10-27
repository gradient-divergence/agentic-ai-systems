# Chapter 30: Data Systems for Agentic AI

[Companion README](../../README.md)

This chapter's code gives each of the refund agent's four sources (order,
shipment, policy and ledger) a data contract, an owner, a freshness budget with a
recorded basis, and a lineage record for every decision: stage 5 of the agent. The
demo renames the carrier's delivery code and shows that the renamed records pass
the type, the gate and every freshness check while the agent pays the wrong
requests. With the contracts in place, the same requests are referred instead.

## Files

| What | Where |
|---|---|
| Demo | `demos/data_systems_demo.py`, with its runs and records in `demos/data_systems_records.py` |
| Code | `refund_agent/sources/data_systems_*.py`: contracts, budgets, the version registry, checks, constructed source releases, lineage, and the release component |
| Exercises | `exercises/data-systems/` |
| Stored outputs | `book_outputs/_fnd.data-systems.*.txt` (11 files) |
| Tests | `tests/refund_agent/test_data_systems.py`, `tests/refund_agent/test_data_systems_extensions.py` |

## Run it

From the companion folder, after the install in the README:

```sh
python -m demos.data_systems_demo
python -m pytest tests/refund_agent/test_data_systems.py
```

The demo also runs the whole release suite on the contracts candidate, so it takes
a few minutes. To run that release yourself with the chapter's component selected:

```sh
python - <<'PY' > data-systems-components.json
import json
from refund_agent.sources.data_systems_release import SELECTOR
print(json.dumps([SELECTOR]))
PY
python -m refund_agent.release run --components data-systems-components.json \
  --trials 5 --run-dir runs/data-systems --at 2026-10-12T12:00:00+00:00
```

In your own code, `build_candidate(build_suite(components=[SELECTOR]))` selects the
contracts. Left unselected, chapter 29's suite and candidate are unchanged.
`refund_agent.sources` exports `ContractedTools`, `contract_problem` and
`default_registry`; the registry's version is `data-systems.registry@1`.

## What the output shows

- **The contracts** (`contract-01`). Each source is at version 1 with seven days'
  notice of a change; the policy's notice is its register's. The shipment
  contract has a closed vocabulary (`shipped`, `delivered`), an owner (the carrier
  integration team), a unit and meaning for each field, and three invariants:
  `time_order`, `delivery_after_shipping` and `not_after_observation`. A renamed
  status is refused as `contract_vocabulary:shipment.status`.
- **A freshness budget** (`budget-01`). From 48 scans in 2,400 transit
  parcel-hours, the change rate is 0.02 an hour, with a 95% upper bound of
  0.025439. A read 30 minutes old is out of date with chance 0.995% at the
  estimate and 1.264% at the bound, so at a 1% tolerance the budget is 30.15
  minutes at the estimate and 23.70 minutes at the bound. In the 1,200 daytime
  parcel-hours the rate doubles. The order source saw no change in 6,000
  order-hours (upper bound 0.000499 an hour); the policy's budget is the
  specification's judgment, with no measured rate; the ledger is read at
  execution, with no cached reuse.
- **The forty public scenarios, four ways** (`run-table-01`, `run-01`): as
  generated, under the renamed code without contracts and with them, and with
  contracts under a harmless added code, `out_for_delivery`. As generated, the
  contracts change no decision and the agent pays four tier 1 requests, 14,996
  cents. Under the rename without contracts it pays 14,996 cents again, but 9,997
  of them wrongly: `r-7-0014`, `r-7-0029` and `r-7-0035`, paid 7, 1 and 2 days
  after their parcels were delivered. The fourth payment, `r-7-0008`'s 4,999
  cents, is right in every column: its record has no delivery scan to rename. Four
  other tier 1 requests on delivered parcels are referred as `delivery_unknown`.
  Of the six not-delivered claims at tiers 2 and 3 on delivered parcels, three now
  draft an approval and three still draft an `amount_over_policy` denial. With
  contracts, 26 requests are referred and only `r-7-0008` is paid. The harmless
  code changes no decision without contracts; with them it refers the same 26
  requests, which is what a closed vocabulary costs until the contract names the
  new code.
- **Releases of a source** (`releases-01`). Amounts moved to dollars and announced
  as `order@2` are refused as an unknown version. Served silently as `order@1`, no
  check fires, because every amount moves together and the invariants still hold.
  Chapter 2's single-field release, denied as `amount_over_policy` without
  contracts, is referred as `contract_invariant:order.paid_equals_items` with them.
- **The generator's own flaw** (`generator-01`). Of the 2,000 generated shipment
  records, 23 hold a scan dated 120 seconds after the record was observed. Five of
  the 600 requests read one, none of them a public scenario. The contract refers
  them as `contract_invariant:shipment.not_after_observation`; the generator is
  left unchanged.
- **A lineage record** (`lineage-01`) for `r-7-0007`: the candidate's fingerprint,
  the registry version, each source's contract version, age, fingerprint, observed
  time and capture time, the balance, the step budget and the recorded model
  proposal. All forty decisions rebuild equal, under the generated and the renamed
  code.
- **The release** (`release-01`, `release-record-01`). The contracts candidate
  passes the release rule on 330 cases in 5 trials, with the five must-pay cases
  paid (62,598 cents) in every trial. The record is unsigned, so it grants nothing.
- `inputs-access-01` notes that planning read the inputs only, never the labels.
  `keys-01` holds the exercises' numbers.

A contract failure refers, on the denial path and on the payment path, after the
earlier evidence checks; a failed check never denies. The audit's denial append
checks the contracts too, so a denial on broken evidence cannot be stored by
calling `audit.append` directly. Lineage keeps the simulator's capture time apart
from the source's claimed observation time, and a later record substituted into a
payment still fails. A rebuild uses the recorded model outputs, the request, the
decision time, the source snapshots and any approval evidence, and makes no fresh
model call; a changed replay code hash refuses the rebuild.

## Exercises

`exercises/data-systems/README.md` has the details. For exercises 1 to 5, edit
`data_systems_starter.py` and run
`python -m pytest exercises/data-systems/test_data_systems_acceptance.py`; each
starter answer fails its own check until you correct it. Exercise 5 diagnoses the
planted error in `lineage-05.json`. For exercise 6, write the card feed's contract
in `contract_06.py` against `feed-06.json` and run
`python -m pytest exercises/data-systems/test_exercise_06.py`. The starter fails
two tests: its vocabulary is open and its version unpinned. The new version is
announced seven days before 1 October 2026, and the budget uses 30 refunds in 600
hours, a one-sided 95% bound and a 1% tolerance. The reference solutions are
`data_systems_reference.py` and `data_systems_reference_06.py`; run the same tests
with `DATA_SYSTEMS_SOLUTION=data_systems_reference.py` or
`DATA_SYSTEMS_CONTRACT=data_systems_reference_06.py` to check them.

## What it leaves out

- Every setting and record is constructed, including the four notice periods and
  the order source's quiet log.
- The lineage store is in memory and writes no file.
- The shipment contract promises that its scan history is complete, and no check
  of a single record can test that promise. Chapter 31 takes it up.
- A silent change of units that moves every amount together passes every check.
- The release record is not signed, so the agent stays in shadow.
