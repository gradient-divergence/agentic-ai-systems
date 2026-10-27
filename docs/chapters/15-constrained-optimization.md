# Chapter 15: Constrained Optimization for Agentic Decisions

[Companion README](../../README.md)

This chapter's code plans the work of the people who act on the refund agent's
drafts: who reviews which draft each morning, within their minutes, and with a
bound that proves no better plan exists. It uses SciPy's open-source HiGHS for
linear and integer programs, and OR-Tools CP-SAT for one scheduling example.

## Files

| What | Where |
|---|---|
| Demo | `demos/constrained_optimization_demo.py` |
| Code | `refund_agent/schedule/` (inputs, queue, model, bound, certificate, check, record) |
| Exercises | `exercises/constrained-optimization/` |
| Stored outputs | `book_outputs/_fnd.constrained-optimization.*.txt` (12 files) |
| Tests | `tests/refund_agent/test_schedule.py` |

## Run it

From the companion folder, after the install in the README:

```sh
python -m demos.constrained_optimization_demo
python -m pytest tests/refund_agent/test_schedule.py
```

The demo needs SciPy (core) and OR-Tools (the `opt` extra:
`pip install -e '.[opt]'`) for the incident-task example. `--write-data` writes
the figure inputs and `review_plan_05.json` under `demos/data/`, and the solve
records beside the stored outputs.

## What the output shows

- `certificate-01`: a small laboratory program, its optimal plan (value 7.5)
  and the multipliers that certify it, checked arithmetically.
- `plan-01`: Monday 23 March 2026, 13 drafts waiting. The plain rule answers 9;
  maximizing the count, then the hours already waited, answers 10, and the
  certificate proves 10 is the most possible (gap 0). With 90 minutes for
  `morgan`, 11.
- `relax-01`, `branches-01` and `search-01`: an integer program's relaxation
  bound, branch and bound, and a search tree over task orders.
- `cp-01`: three incident tasks with due slots, solved by CP-SAT; with a
  stricter due slot it is infeasible, and a counting proof shows why.
- `window-01`: all 114 training mornings. Count-then-wait answers 397 of 398
  requests, matching the rounded bound every morning; the plain rule falls
  short of it on four mornings.
- `absence-01`, `robust-01` and `uncertain-01`: a constructed absence, a slow
  tier 3 case, and review times drawn at random (seed 15, 10,000 draws).
- `relief-01`: relief depots and shelters through the same linear-program
  interface and certificate.
- `keys-01`: answers to exercises 1 to 8.

The inputs are constructed (`review-inputs-v1`): reviews take 5, 15 and 30
minutes by tier, `alex` has 120 minutes and `morgan` 60 from 09:00 UTC, Monday
to Friday with no holidays, and tier 3 drafts go to `morgan`, the named
approver. Plans maximize the count, then the total hours already waited, and
order each person's work oldest first. The bound certifies the count; it does
not certify the waiting-time criterion. Node budgets are deterministic and the
arithmetic tolerance is 1e-7. Relative gaps divide by the absolute incumbent: zero
for 0/0, and undefined, with a reason, for a zero incumbent under a positive
bound. A solver error records "unknown" and proposes no plan; an error in the
second solve keeps the checked count plan and reports that waiting time was not
optimized.

Only the 398 training requests before 1 July are used. Only timestamps are read
to exclude later requests; no validation or golden content or outcome label is
read. Carry-over keeps every unanswered draft. The planner calls no agent tool,
records no approval, and leaves the loop, policy and tools unchanged.

## Exercises

Exercise 8, in `exercises/constrained-optimization/`, adds the respond-by rule:
a draft is due by 17:00 UTC on the next working day after it arrives, and an
already-late draft is due now. Implement `plan_morning(queue, minutes, budget)`
in `starter.py`, then run:

```sh
CONSTRAINED_OPTIMIZATION_SOLUTION=path/to/your_solution.py \
  python -m pytest exercises/constrained-optimization/test_acceptance.py
```

The checker rebuilds every training morning. It checks the fallback to the usual
plan, the counting proof on each infeasible morning and the smallest extra time
that repairs it, and that the golden set is neither read nor changed. The
starter fails at its unfinished function; `reference_solution.py` is the
reference solution. Exercise 5 audits `demos/data/review_plan_05.json`, a
planted plan that gives one tier 3 draft to `alex` and claims 11 answers.

## What it leaves out

The review times, staff minutes and calendar are constructed. The plan proposes
work for people; it approves nothing and changes no part of the agent.

Solver interfaces: [linear programs](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linprog.html),
[integer programs](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.milp.html),
[constraint solver](https://developers.google.com/optimization/cp/cp_solver).
