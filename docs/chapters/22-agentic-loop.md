# Chapter 22: The Agentic Loop

[Companion README](../../README.md)

The refund agent runs each request through a bounded loop: it reads the order, the
shipment and the policy, drafts a decision, proposes a refund, executes it through
the guarded payment tool, and appends an audit record. The demo runs the agent's
default command and prints what the chapter quotes from the files that run writes:
the scorecard, one paid request's trace and audit records, two runs that stop
without paying, and the arithmetic of errors that compound over a run's steps.

## Files

| What | Where |
|---|---|
| Demo | `demos/agentic_loop_demo.py` |
| Code | `refund_agent/run.py` (the command), `refund_agent/agent/` (the loop), `refund_agent/tools.py` (the seven tools and the payment boundary) |
| Stored outputs | `book_outputs/_fnd.agentic-loop.*.txt` (7 files) |
| Tests | `tests/test_agentic_loop_demo.py`, `tests/refund_agent/test_stage1.py` |

## Run it

From the companion folder, after the install in the README:

```sh
python -m refund_agent.run --seed 7
python -m demos.agentic_loop_demo
python -m pytest tests/test_agentic_loop_demo.py tests/refund_agent/test_stage1.py
```

`python -m refund_agent.run` writes `scorecard.json`, `audits.json` and one trace
per request under `traces/` in `refund-output/` (choose another folder with
`--out`). The demo runs the same command into `refund-output/` under the working
directory, then reads those files. Neither needs a network or a wall clock: the same
arguments give identical scorecards, audits and traces.

## The refund agent at this stage

`refund_agent` is an offline refund desk for fictional physical goods. It has
generated orders and policies, scripted and replayed models, seven tools, a bounded
loop, simulated payments, audits and JSON traces. All amounts are in one currency,
USD, in integer cents. The agent can execute tier 1 refunds by itself and tier 2
refunds after one person approves. At tier 3 it prepares a packet for the named
approver, and a person can execute it through the guarded tool.

The generator (seed 7) produces 2,000 orders, 600 labeled requests, five dated
policies, a reviewer list and 40 public scenarios. The default run scores exactly
those 40 public scenarios; `--requests N` scores the first N requests instead.

`refund.execute` reads the order, shipment and policy again at execution time and
checks the ledger balance, the current policy version, the return window, the
sources' ages and a fixed text risk rule. A retrieved policy may be at most 30 days
old, and order and shipment observations at most 30 minutes old; a cached older
policy version or a stale observation causes a referral. A retried payment with the
same idempotency key returns the first receipt and pays once. The policy versions,
return windows, caps and risk threshold are fictional, and the risk score is not
calibrated.

## What the output shows

- `scorecard-01`: the baseline agrees with the generated labels on 40 of 40; the
  agent on 26 of 40 (approve 4, deny 8, refer 28, executed 4). The `safety` line
  refuses and audits 8 of 8 independently specified cases with 0 unauthorized
  cents. `gate_removed` reruns them with the gate inside `refund.execute` removed:
  7 of 8 are still refused, and the eighth, a request for 2,499 cents, pays
  999,999 cents. `positive_payments` checks five hand-written payments against the
  ledger (100, 2,499 and 4,999 cents without a person; 5,000 and 50,000 cents after
  one reviewer approves), 62,598 cents in all. A loop that never paid would fail it.
- `trace-01` and `audits-01`: request `r-7-0007`, 2,499 cents at tier 1, through its
  seven steps, and its two audit records: the intent (no money moved) and the
  confirmation (2,499 cents paid, receipt `refund-r-7-0007`).
- `endings-01`: `r-7-0002` stops at its assessment with `stale_shipment` (its
  shipment record is 1,801 seconds old), and `r-7-0009`, 5,000 cents at tier 2,
  waits with `approval_required`. Neither pays.
- `compound-01` and `outcomes-01`: with each step right with a given probability of
  0.98, seven steps are all right 0.868 of the time and eight steps 0.851. In a
  constructed comparison over 1,000 runs, a checker that catches 0.90 of wrong runs
  and refers 0.02 of right ones cuts wrong finishes from about 132 to about 13, at
  the cost of about 136 referrals.
- `disagreement-01`: where the agent and the baseline differ. All 14 differences
  are tier 3 requests labeled deny, which the agent refers to the named approver.

The 26 of 40 measures agreement with labels that repeat the fictional rules, not
correctness. With a scripted model, the scores say the code runs as written and
nothing about a real model.

## What it leaves out

There is no live model adapter, real payment integration or deployment.
`ModelClient` is the adapter boundary; an adapter must enforce its own deadline and
raise `TimeoutError`. The ledger lives in memory, and a process crash between
payment and confirmation is outside this stage. A deployed system would add a
durable ledger, idempotency at the payment provider, and reconciliation between the
ledger and the audit log.
