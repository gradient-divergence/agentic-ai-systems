# Chapter 32: Deployment and Rollback

[Companion README](../../README.md)

This chapter's code lets the refund agent pay a small cohort's tier 1 refunds
under a signed canary plan, bounds what that canary can pay before it stops, and
rehearses the rollback with requests in flight. The demo builds the shadow window,
the plan and its start checks, the exposure arithmetic, and two rollback
rehearsals: one that passes and one that fails because the rollback target cannot
read what the newer version wrote.

## Files

| What | Where |
|---|---|
| Demo | `demos/cloud_native_deployment_demo.py` |
| Code | `refund_agent/canary/cloud_native_deployment_*.py`: the plan, window, binding, stop rules, store, exposure, rehearsal, restart and start; `python -m refund_agent.canary` is its command line |
| Exercises | `exercises/cloud-native-deployment/` |
| Stored outputs | `book_outputs/_fnd.cloud-native-deployment.*.txt` (7 files) |
| Tests | `tests/refund_agent/test_cloud_native_deployment*.py` (five files) |

## Run it

From the companion folder, after the install in the README:

```sh
python -m demos.cloud_native_deployment_demo
python -m pytest tests/refund_agent/test_cloud_native_deployment.py
```

The canary has a local command line with no remote effects. A start reads the
plan, the release record, the window and the rehearsal, and checks them:

```sh
python -m refund_agent.canary start --plan PLAN --release-record RELEASE \
  --window WINDOW --rehearsal REHEARSAL --owner morgan --at UTC_TIME
```

`python -m refund_agent.canary rehearse --candidate as-built --cadence 4h --out FILE`
runs a rehearsal (`--candidate planted-field` for the failing one, `--cadence daily`
for the daily review).

The canary keeps its grants, starts, stops, rechecks, receipts and proposals in an
append-only store and journal that its component declares. An operating run can
choose those paths before it builds its suite, and must select `canary` in both
candidates, since the paths are part of each candidate's identity:

```python
from refund_agent.canary.cloud_native_deployment_component import selector
from refund_agent.release.extensions import RuntimeStore

canary = selector(
    store=RuntimeStore("/absolute/operating-run/store.jsonl", "morgan", "append"),
    journal=RuntimeStore("/absolute/operating-run/journal.jsonl", "morgan", "append"),
)
```

`runtime_paths(candidate)` returns the paths; a host rebuilding recorded selections
calls `restore_selection(candidate_record)` before `build_suite`.

## What the output shows

- **The shadow window** (`window-01`). The first 299 eligible request identifiers
  are frozen in arrival order, and a held request keeps its place until it has a
  terminal decision. On those 299 the agent and the baseline each refer 87, with no
  forbidden payment and no wrongful denial, so the three measures pass. Generated
  labels stand in for the person's decisions and 30-day outcomes; the record names
  when deployed labels would mature (9 November 2026). The arrival times are a
  constructed model: live request times joined to the delay and copy patterns of
  chapter 31's generated carrier stream.
- **The canary plan** (`plan-01`). The cohort is a fixed 5% of requesters by hash
  (131 of 3,262, a realized 4.0%). The agent may pay a cohort's tier 1 request
  through the unchanged gate, at most 4,999 cents a payment, with daily caps of 15
  payments and 74,985 cents derived from a named $750 worst day. A person
  rechecks every payment within 4 hours; the plan's time box is 28 days. Over $500,
  the named approver decides every request, denials included. Payments outside
  that authority go to the shadow recorder and the person's path.
- **The start** (`start-01`). The release, window, plan and rehearsal checks pass;
  the error budget is not named, so the start is refused and the binding stays in
  shadow. Chapters 33 and 35 add their own start checks, which run on every start.
- **Exposure** (`exposure-01`), from a constructed model at a steady rate. At
  25/9 tier 1 requests an hour, a 4-hour recheck expects 5/9 further cohort
  requests before a stop, so the expected loss, counting the first wrong payment,
  is at most 7,777 cents (7,776.22 rounded up). A daily review would give 10/3 and
  21,663 cents, but an overdue recheck pauses the canary at 4 hours, so that figure
  is a counterfactual. A stop that waited for settled evidence, about 36 hours,
  gives 5 and 29,994 cents. Evidence took 36.06 hours on average in chapter 31's
  stream, so 299 clean requests take 91.2 days at a 5% share, 46.4 at 10% and 19.4
  at 25%. In its 28 days the canary settles about 88; if none fails, the one-sided
  95% upper bound on the failure rate is 3.35%, and a fault touching 1 request in
  1,000 shows among 88 with probability 0.084.
- **The rollback rehearsals** (`rehearsal-01`, `rehearsal-02`). A test hook plants
  one forbidden 4,999-cent payment past the gate, counted apart, and a simulated
  crash cuts off two requests in flight. Under the 4-hour recheck the canary stops
  at 04:00 on day 3. Under a daily review the overdue rule pauses it three times,
  the third at 04:00:01 on day 3, and the review finds the payment only at the
  next midnight. Neither arm pays a further request before its stop.
  Both pass all nine checks: intents closed unpaid, the next cohort to a person,
  receipts kept and confirmed at restart, idempotent reconciliation, readable
  records, components restored, no request paid twice, and the restarted target
  deciding the forty public scenarios as its record says. The policy in force is
  not part of the candidate record, so the rollback does not restore it. In the
  second rehearsal the newer version writes `canary_id` into each receipt and
  proposal; the rollback target's closed record contracts refuse the field, six
  checks fail and the verdict is fail.
- `keys-01` holds the exercises' numbers.

Restart reconciliation never pays. A person's desk uses the agent's idempotency
key, `execute-{request_id}`. A paused or ended grant keeps its original expiry
across restarts and version changes, and re-signing cannot renew spent evidence. A
defect, a spent budget or the 28-day expiry ends the canary for good. A store that
fails its schema, key, chain or read-back check stops the canary as
`untrusted_record`; if the store cannot take that stop, payment authority closes
without rewriting the damaged file, and the stop is written once the store is
trusted again.

## Exercises

Run `python -m pytest exercises/cloud-native-deployment/test_cloud_native_deployment_acceptance.py`.
Starters 1 to 5 supply proposed answers that fail their designed checks; correct
them. Exercise 5's `rehearsal-05.json` holds a day's payment records and a claimed
rehearsal result: find the violated property and write the reply to the release
owner. For exercise 6, keep `cloud_native_deployment_canary_06.py` unchanged,
expand the reader in `cloud_native_deployment_target_06.py`, and rehearse again with
`python exercises/cloud-native-deployment/cloud_native_deployment_rehearse_06.py`.
Acceptance checks a real fresh-process restart, no request paid twice and unchanged
decisions on the forty public scenarios. `cloud_native_deployment_reference.py` is
the reference solution; acceptance imports only the student files.

## What it leaves out

- The companion has no payment provider. Idempotency across providers, uncertain
  provider outcomes and a crash of the store itself are outside the model.
- A rehearsal pays under a simulated grant of its own and never gives authority to
  start the canary.
- The window's labels are generated, not deployed outcomes.
- The canary never starts in the demo: no error budget is named (chapter 34 names
  one).
