# Chapter 39: Capstone: Design, Build, Deploy, Operate

[Companion README](../../README.md)

The capstone's code turns the launch decision into records. It holds the
seven-question launch requirement, the evidence packet, a gate with four checks,
a refresh bound to the launched bundle, three-arm business accounting, the named
approver's evidence, a launch scan, signatures and the withdrawal conditions.
The gate returns `hold` or `launch`; it never grants payment authority. The
operating run that would feed it is not assembled in the companion, so the demo
prints what is missing instead of inventing it.

## Files

| What | Where |
|---|---|
| Demo | `demos/capstone_blueprint_demo.py` |
| Code | `refund_agent/launch/` (schema exported as `LaunchRow`, `LaunchRequirement`, `validate_row`, `validate`) |
| Exercises | `exercises/capstone-blueprint/` |
| Stored outputs | `book_outputs/_fnd.capstone-blueprint.*.txt` (8 files) |
| Tests | `tests/refund_agent/test_capstone_blueprint_*.py` (six files) |

## Run it

From the companion folder, after the install in the README:

```sh
CAPSTONE_BLUEPRINT_RUN=runs/capstone-blueprint python -m demos.capstone_blueprint_demo
python -m pytest tests/refund_agent/test_capstone_blueprint_gate.py
```

The run folder receives the run's configuration, its summary and one file per
missing record under `missing/`. Run the six test files one at a time; in one
process, a resource guard set by one file makes the others fail.
`--write-data` rewrites the figure data and the exercise fixtures.

The gate also has a command line, `python -m refund_agent.launch gate --run RUN
--requirement REQUIREMENT`, with `refresh`, `admit` and `sign` beside `gate`. It
needs a saved requirement and, through `--context refund_agent.MODULE:FUNCTION`,
the registry and producer adapters of a full deployment; without them the gate
holds.

## What the output shows

`packet-01` lists the requirement's seven rows, one per launch question, and the
records each needs, with the chapter that produces it, its bundle rule (`strict`,
`covered` or `any`) and its signer. The operating packet's status is
`integration_missing`: no launched stage 6 bundle or signature is made up.

`gate-01` lists, row by row, the required records that are missing (row 3 alone
needs eight), and records no launch admission or decision-owner signature.
`gate-02` says the refresh did not run: no rehearsal rerun, no earlier result
replaced, and a comparison held on its verdict is never refreshed.

`refused-01` runs the full gate on a labeled, constructed packet whose only
rehearsal names an earlier bundle. Rows 1 to 6 pass all four checks (complete,
consistent, signed in order, reversible); row 7 fails `consistent` with "strict
record names an earlier or absent bundle", so the verdict is `hold`.

`compare-01` states the accounting the comparison needs without values: tier 2,
tier 3, tier 1 referrals, rechecks, blind second reviews and planted cases, at
the constructed review times of 5, 15 and 30 minutes for tiers 1, 2 and 3. The
manual arm assumes no review of its reviewers' work. The rule is that the agent
must be better than each alternative (manual and rules) on at least one business
measure and worse on none. `tier3-01` restates that every request over $500,
denials included, needs its named approver. `decision-01` prints the five
reversal conditions, what each withdraws, where the decision returns and the
mechanism that fires it; payments already made stand under every reversal.

`keys-01` gives the exercise keys: the four checks; budget exhaustion withdraws
tier 1 authority and a person decides; exercise 3 holds with two records not
counted (the rehearsal and the injection suite); exercise 4 gives 61.666667
manual and 48.066667 agent reviewer-hours a week and holds; exercise 6's
deployed gate on day 64 must hold until day 84, 30 days after the last payment on
day 54.

How the measured comparison works: `capstone_blueprint_compare.measure` reads the
agent run and the `RulesClient` run on the same requests, with every non-model
component unchanged, and a scripted-model and rules decision mismatch stops the
run. Human payments made during shadow acquire outcomes; a hypothetical shadow
payment cannot. Deployed labels mature 30 days after each payment.

## Exercises

`exercises/capstone-blueprint/` has one deliberately failing starter per
exercise in `capstone_blueprint_starters.py`, the reference, two diagnostic
fixtures (`packet_03.json`, `decision_05.json`) and the acceptance checks:

```sh
python -m pytest exercises/capstone-blueprint/test_exercise_06.py
CAPSTONE_BLUEPRINT_USE_REFERENCE=1 python -m pytest exercises/capstone-blueprint/test_exercise_06.py
```

The first command fails on the untouched starters; the second checks the
reference. Exercise 3 has exactly two stale records; exercise 5 has exactly one
planted error in `decision_05.json`'s claimed launch, and the default demo omits
its key. The maturity exercise never changes companion mode.

## What it leaves out

The assembled operating run: the mixed request stream, the stage 6 bundle, the
promotion registry, budget, scope, audit, queue and rollout evidence. Without
them `gate-01`, `gate-02`, the comparison, the tier 3 report and the held
decision print as unavailable, and the figure data has an empty record array.
The constructed-record tests verify the earlier-bundle hold and the target-bound
refresh.
