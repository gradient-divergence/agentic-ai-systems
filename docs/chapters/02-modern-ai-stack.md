# Chapter 2: Architecture, Components and Contracts

[Companion README](../../README.md)

Chapter 2 tests one layer boundary of the refund agent: the order record that the
order system sends. The demo prints the record's contract from
`refund_agent/contracts.py`, checks four releases of the order system against it, and
shows what the rule-only baseline drafts from each. Nothing here changes the refund
agent: every check is the package's own `Order` record and `baseline` function.

## Files

| What | Where |
|---|---|
| Demo | `demos/modern_ai_stack_demo.py` |
| Code | `refund_agent/contracts.py` (`Record`, `Item`, `Order`), `refund_agent/baseline.py` |
| Exercises | `exercises/modern-ai-stack/` |
| Stored outputs | `book_outputs/_fnd.modern-ai-stack.*.txt` (6 files) |
| Tests | `tests/test_modern_ai_stack_demo.py` |

## Run it

From the companion folder, after the install in the README:

```sh
python -m demos.modern_ai_stack_demo
python -m pytest tests/test_modern_ai_stack_demo.py
```

The chapter prints the first command as `python demos/modern_ai_stack_demo.py`, which
runs the same file.

## What the output shows

The data are the default seed's (7). Order o-0007 and request r-7-0007 take the values
the chapter gives: the order paid 4,200 cents for one item with no earlier refund, and
the request asks for 3,800 cents back after a change of mind. The demo builds that
record, `sent`, from the generated order.

- `contract-01` is an excerpt of `refund_agent/contracts.py`: the `Cents` type, the
  `Record` base class and its validator, `Item` and `Order`.
- `compat-01` checks four versions of `sent` against `Order`. Three are rejected: the
  renamed field (`paid_cents` missing, `amount_paid_cents` not permitted), the added
  field (`currency` not permitted) and the naive time (timestamps must be
  timezone-aware). The release in dollars is accepted as 42 cents, because 42 is a
  valid number of cents.
- `drafts-01` shows what the baseline drafts for r-7-0007 from each record. As sent,
  the balance is 4,200 cents and the draft approves at tier 1. The rename stops the
  data load before any of the 600 requests is drafted, naming `orders.7.paid_cents`.
  In dollars, the balance is 42 cents and the draft denies at tier 1 with
  `amount_over_policy`, citing the cap.
- `scorecard-01` is the baseline's scorecard on the forty public scenarios before and
  after the dollars release: 40/40 agreement (4 approve, 22 deny, 14 refer, 0 paid)
  before, and 16/40 (0 approve, 35 deny, 5 refer, 0 paid) after.
- `invariant-01` checks a rule on the whole record: `paid_cents` equals the items'
  total, the sum of quantity times `unit_cents`. It holds on all 2,000 generated
  orders, fails for the record in dollars (42 against 4,200), and holds again when
  every amount moves to dollars (42 and 42), so that record is accepted.
- `exercise-03` is the key to exercise 3: the text "4200" and a pending payment of 0
  are rejected; a local time with its offset and an amount in hundredths of a cent
  (420,000) are accepted and drafted as approvals. At 5,000 cents, the order as sent
  is denied at tier 2 for `amount_over_policy`, and the hundredths record is referred
  with `approval_required`.

No model takes part: at stage 0 the model layer is still empty, and a person reads
every draft.

## Exercises

Exercise 6 adds a check on the whole record to `Order`, a `model_validator` like
`Decision`'s, that rejects an order whose `paid_cents` differs from its items' total.
The reference solution is `exercises/modern-ai-stack/reference.patch`. To run it:

```sh
python exercises/run_reference.py modern-ai-stack
```

The runner applies the patch to a temporary copy of `refund_agent` (it needs `git` on
your path), runs the acceptance tests and writes its receipt to
`exercises/modern-ai-stack/run-06.json`, replacing the stored one. The runnable agent's
`Order` stays unchanged. Acceptance covers all generated orders, the rejection of the
dollar release and of the three other incompatible releases, orders of several items
and quantities, all forty public baseline decisions, and no call to a write tool.

To check your own change, run `exercises/modern-ai-stack/test_acceptance.py` against
your patched package, as [chapter 1's guide](01-what-is-agentic-ai.md#exercises)
describes. These acceptance tests are not part of the default test suite.

## What it leaves out

- The contract checks types and fields, not meaning: a release that changes the unit
  of a valid field passes it, which is why the release in dollars gets through until
  the whole-record rule is added.
- The releases are constructed; there is no real order system, and no refund is
  executed.
