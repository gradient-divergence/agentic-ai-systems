# Chapter 24: Reasoning and Validated Planning

[Companion README](../../README.md)

Plans for the refund agent, checked before their first write. The demo models the
seven tools as operators (preconditions, facts added, facts removed) and runs
`validate_plan` on plans it is given, among them the call sequences of the agent's
runs on the default data (seed 7). The loop itself does not call the validator.
Nothing in the demo changes the agent, and nothing is written to disk.

## Files

| What | Where |
|---|---|
| Demo | `demos/reasoning_planning_demo.py`, with `demos/reasoning_planning_search.py` (unit-cost search and checked proposals) |
| Data | `demos/data/reasoning_planning_report.csv` (the eight plans of the report) |
| Stored outputs | `book_outputs/_fnd.reasoning-planning.*.txt` (11 files) |
| Tests | `tests/test_reasoning_planning_demo.py` |

## Run it

From the companion folder, after the install in the README:

```sh
python -m demos.reasoning_planning_demo
python -m pytest tests/test_reasoning_planning_demo.py
```

## What the output shows

- The operator table of the seven tools, and the plan of `r-7-0007` (2,499 cents,
  tier 1) checked step by step: six preconditions and one goal (`audited`, `paid`)
  hold, so the plan is valid. It uses 6 steps of a budget of 12, pays once, and pays
  2,499 cents against a cap of 80,000. There are 117,649 sequences of six tool names
  from seven tools.
- A second run of `r-7-0009` (5,000 cents, tier 2): paying at once fails for lack of
  the three reads, and the plan that reads again in six steps is valid.
- Seven faulty plans and the verdict that names each fault: a policy check called
  as a tool, a payment before the proposal, an approval only in the text, two
  payments, a payment above what the order has left, no audit, and thirteen steps
  against a budget of twelve.
- The forty public runs, each run's calls checked as a plan: all valid, none
  refused.
- A boundary case: a plan for 11,000 cents is valid, yet `refund.execute` refuses it
  with `amount_mismatch` because the approval covers a proposal of 12,000; nothing
  is paid.
- Search: breadth-first search and A* with the relaxed-plan estimate `h+` find the
  same six calls for the refund and for a constructed network repair; A* expands
  fewer states (8 against 10, and 6 against 7), and `h+` never overestimates on any
  reachable state. The network search is also printed expansion by expansion.
- Checked proposals under a budget: with a valid-proposal rate of 0.125 (one valid
  plan in a pool of 8), the exact expected proposals used and the referral
  probability for budgets of 3, 6, 12 and 24, beside a 10,000-request simulation
  (seed 7). At a budget of 12, about 0.20 of requests are referred.

## What it leaves out

The operator model is the chapter's, not the running agent's: the loop keeps its
pinned order. The step budget counts calls, while the loop counts steps including
its assessment, so a plan within twelve calls need not fit in the loop's twelve
steps. Validity and bounds say nothing about the return window; the eligibility
rules and the payment gate check that.
