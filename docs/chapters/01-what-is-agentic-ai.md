# Chapter 1: What Is Agentic AI?

[Companion README](../../README.md)

Chapter 1 meets the refund agent at stage 0. There the rule-only baseline reads the
order, the shipment and the policy, and drafts a decision that a person executes. The
demo runs that baseline on two requests: r-7-0007, the constructed scenario the
chapter follows, and r-7-0000, one of the forty public scenarios. It prints the three
reads, the draft and the audit records of each.

## Files

| What | Where |
|---|---|
| Demo | `demos/what_is_agentic_ai_demo.py` |
| Listing | `demos/what_is_agentic_ai_baseline.py`: the `baseline` function set to the page's width; a test keeps it identical to the package's |
| Code | `refund_agent/baseline.py` (`baseline`), `refund_agent/rules.py` (`assess`, `tier_for`), `refund_agent/tools.py`, `refund_agent/contracts.py` |
| Exercises | `exercises/what-is-agentic-ai/` |
| Stored outputs | `book_outputs/_fnd.what-is-agentic-ai.*.txt` (1 file, `run-01`) |
| Tests | `tests/test_what_is_agentic_ai_demo.py` |

## Run it

From the companion folder, after the install in the README:

```sh
python -m demos.what_is_agentic_ai_demo
python -m pytest tests/test_what_is_agentic_ai_demo.py
```

The chapter prints the first command as `python demos/what_is_agentic_ai_demo.py`,
which runs the same file. The demo needs no optional package and no network access.

## What the output shows

The data are the default seed's (7), generated offline: 2,000 orders, 600 labeled
requests, five dated policies, a reviewer list and 40 public scenarios. Amounts are
integer cents in one currency, USD. The demo sets order o-0007 and request r-7-0007 to
the values the chapter gives (one item paid 4,200 cents, no earlier refund, 3,800
cents asked), using chapter 2's construction.

The first seven lines follow r-7-0007 from the request through its three reads to the
draft:

- `order.lookup`: o-0007 paid 4,200 cents, of which 0 has been refunded.
- `shipment.status`: delivered 12 days before the request.
- `policy.retrieve`: `refund_cap_v4`, version 2026-03-01, with a 21-day window.
- The draft approves at tier 1 with the reason `eligible` and cites
  `refund_cap_v4@2026-03-01#eligibility`. It leaves 0 audit records.

The draft approves because every condition holds: the delivery falls inside the
policy's window for a change of mind, the evidence is fresh and the policy current, the
amount is within the policy's cap and the order's unrefunded balance, and the risk
score is low. That score is a placeholder, a fixed rule that looks for the substring
"ignore" in the request's text, whatever its case.

The next lines follow r-7-0000, whose order identifier matches two orders. The lookup
fails with `ambiguous_order`, the baseline stops reading, and the draft refers the
request at tier 1 with that reason and no citation. That referral leaves the one audit
record of the two runs; the read tool writes it itself when the lookup fails.

The last line counts three possible drafts (approve, deny, refer), 0 calls to a write
tool and 0 executions. The baseline creates no proposal or review item and pays
nothing: at stage 0 a person executes every refund.

## Exercises

Exercise 6 runs the baseline on the forty public scenarios with seed 7, then lowers
the tier 1 threshold in `tier_for` from 5,000 to 2,500 cents and runs it again. The
reference solution is `exercises/what-is-agentic-ai/reference.patch`. To run it:

```sh
python exercises/run_reference.py what-is-agentic-ai
```

The runner first checks the unchanged baseline on the public scenarios (4 approvals,
22 denials, 14 referrals, no execution). It then applies the patch to a temporary copy
of `refund_agent` (it needs `git` on your path), runs the acceptance tests there, and
writes its receipt to `exercises/what-is-agentic-ai/run-06.json`, replacing the stored
one. The changed baseline must draft 2 approvals, 22 denials and 16 referrals without
calling a write tool.

To check your own change, run `exercises/what-is-agentic-ai/test_acceptance.py` from
your patched copy of the companion, with `PYTHONPATH` set to that copy and pytest's
`pythonpath` option empty (`-o pythonpath=`), as the runner does. These acceptance
tests are not part of the default test suite.

## What it leaves out

- The baseline doesn't yet meet the decision boundary's logging rule: a draft that
  settles a request, like r-7-0007's, leaves no audit record. Chapter 22 adds it.
- The risk score is the placeholder text rule, not a model; policy versions, windows
  and caps are fictional.
- Nothing here pays. The later stages' writes, approvals and simulated ledger are in
  the same package, but this chapter's demo calls only the three read tools.
