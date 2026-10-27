# Chapter 25: Action, Guardrails, and Policies

[Companion README](../../README.md)

The policy gate inside `refund.execute`, the only one of the agent's seven tools
that cannot be undone. The gate's conditions are read again at the moment of
payment, and the first that fails names the reason recorded. The demo builds the
requests the chapter walks through, calls the public tools and the agent, and
prints what they return; every printed line comes from the refund agent package.

## Files

| What | Where |
|---|---|
| Demo | `demos/refund_gate_demo.py`; a smaller example of a guarded tool call with the `agents` package, `demos/action_guardrails_demo.py` |
| Code | `refund_agent/rules.py` (`gate`, `approval_problem`), `refund_agent/tools.py` (the commit sequence), `refund_agent/evaluation.py` (the eight refusal cases) |
| Data | `demos/data/refund_gate_exercise_log.csv` (the audit log of exercise 5) |
| Exercises | `exercises/action-guardrails/` |
| Stored outputs | `book_outputs/_fnd.action-guardrails.*.txt` (9 files) |
| Tests | `tests/test_refund_gate_demo.py` |

## Run it

From the companion folder, after the install in the README:

```sh
python -m demos.refund_gate_demo
python -m pytest tests/test_refund_gate_demo.py
python -m demos.action_guardrails_demo
```

The last command registers one tool behind an allowlist and a cost cap and prints
the allowed call's result.

## What the output shows

- Limits: 4,999 cents with a shipment record 1,800 seconds old passes the gate;
  5,000 cents needs an approval (`approval_required`); 1,801 seconds is
  `stale_shipment`.
- A tier 1 payment of 2,499 cents, with its intent and confirmation audit records.
- A direct call to `refund.execute` for 999,999 cents on a 2,499-cent request is
  refused with `amount_mismatch` and not paid.
- Execution-time checks: a 12,000-cent request is approved but its shipment read has
  gone stale by then, so it is referred; after a fresh read it is paid.
- What an approval approves: without an approval a tier 2 request waits; approved by
  the listed reviewer `alex`, it is paid, with approval, intent and confirmation
  records.
- Tier 3: a 50,001-cent request opens a review for the named approver; after the
  approval the agent still refers it, and the named approver's direct call pays
  50,001 cents once; the same key again pays nothing more. A tier 3 denial becomes a
  referral with no payment.
- Rates over the forty public scenarios: 4 attempts at `refund.execute`, none
  refused by the gate, 4 payments, none against the generated labels. That says
  little, because the agent's own assessment refers every bad request first.
- The eight refusal cases, as built and with the gate removed: 8 of 8 refused as
  built; without the gate, 7 of 8, and the eighth pays 999,999 cents. The five
  positive payments are found in the ledger.
- The audit log of exercise 5, with three planted faults.

## Exercises

Exercise 6 adds a daily ceiling on tier 1 payments. `exercises/action-guardrails/reference.patch`
is a worked reference; this command applies it to a temporary copy of `refund_agent`,
runs `test_acceptance.py`, and rewrites the receipt `run-06.json`:

```sh
python exercises/run_reference.py action-guardrails
```

The setting defaults to `None`, so an unconfigured copy behaves as before. The tests
set it to 5,000 cents: two 2,499-cent payments pass, and the third that UTC day is
refused with `tier1_daily_cap`, audited and unpaid, with or without an approval; the
next day's first payment passes. The book's agent is not patched.

## What it leaves out

A paid refund cannot be recalled; after a wrong payment, only a compensating action
remains. The gate trusts the age and version of what it reads. The tier limits
($50 for tier 1) are given by the specification, not derived.
