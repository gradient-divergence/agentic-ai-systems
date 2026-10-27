# Chapter 27: Multi-Agent Systems and Independent Verification

[Companion README](../../README.md)

A delegated check on the refund agent, measured before adoption. `VerifiedTools`
calls the unchanged payment gate first and then a read-only tier 1 verifier; an
objection, a timeout or a mismatched verdict refers the request to a person.
Agreement grants no authority. Cooperative inheritance lets another tools class
redirect the payment write while keeping the same verification path. The demo
compares three arms on the same constructed requests, offline, with seed 7.

## Files

| What | Where |
|---|---|
| Demo | `demos/multi_agent_systems_demo.py` |
| Code | `refund_agent/verifier/` (`independent.py`, `agreeing.py`, `experiment.py`, `shared_error.py`) |
| Exercises | `exercises/multi-agent-systems/` |
| Stored outputs | `book_outputs/_fnd.multi-agent-systems.*.txt` (13 files) |
| Tests | `tests/refund_agent/test_verifier.py`, `tests/refund_agent/test_shared_error.py`, `tests/test_multi_agent_systems_demo.py` |

## Run it

From the companion folder, after the install in the README:

```sh
python -m demos.multi_agent_systems_demo
python -m refund_agent.verifier.independent
python -m demos.agentic_systems_demo
python -m pytest tests/refund_agent/test_verifier.py tests/refund_agent/test_shared_error.py tests/test_multi_agent_systems_demo.py
```

The demo prints eleven of the stored outputs. The second command prints
`verifier-01` (three verdicts: a contradiction, a plain case, a phrase trip), and the
third prints `sota-01`, a weighted-vote counterexample, among other lines. One line
of `diagnostic-01`, a local wall-clock time per request, differs from machine to
machine; its counts do not. `--write-data` writes the figure data and the two
constructed exercise fixtures.

## What the output shows

- Shared errors in closed form: two checkers that each fail 0.2 of the time pass
  0.04 of cases together when independent, 0.12 at correlation 0.5, and 0.20 when
  identical. A seeded simulation of 200,000 cases matches each exact value; the
  fully shared point is 0.201435 against 0.2, 1.60 standard errors away.
- One verified request, `verifier-r-7-0003`: the gate passes, the verifier reads the
  order, shipment and policy with their ages, objects because the customer's words
  contradict the carrier record, and the request is referred with no money moved.
- The matched comparison: all three arms run on the same 400 tier 1 requests of
  2,499 cents (24 contradictions, 16 false carrier dates, 8 phrase trips, 352
  plain). Truth is kept apart from the runtime data and read once, after the arms
  finish, through an append-only access journal. The single agent and the agreeing
  verifier, which repeats the eligibility rules, are each right on 360 and pay 40
  wrongly; the independent verifier is right on 376, pays 16 wrongly and refers 32.
- The adoption rule, fixed before the run: zero additional payments, a positive
  97.5% paired interval and an exact discordant chance of at most 2.5%. A tie keeps
  the single agent. The agreeing verifier is not adopted. The independent one is
  accepted: a lead of 0.04, interval [0.0086, 0.0714], exact chance 0.0070. Adoption
  means eligibility for chapter 29's suite and shadow run, not permission to deploy.
- Costs: at illustrative costs (a wrong payment loses 2,499 cents, a review costs
  400), the independent verifier saves $1.18 a request before its own call, with a
  97.5% interval of $0.61 to $1.74, so a call must cost less than about $0.61.
- Hand-offs: a verifier that answers after its 1-second deadline is a
  `verifier_timeout` referral, and a shipment date moved after the verdict voids it.
- A diagnostic putting the same check inside the single agent, the cost
  break-even lines, a referral game between two desks and a second-price auction,
  and the worked answers for the exercises.

The independent verifier compares the customer's words with the carrier record by a
phrase rule, a placeholder for a model reader. No model service is called, and
response latency is simulated.

## Exercises

`exercises/multi-agent-systems/review_05.json` is a constructed bank review whose
report confuses agreement with correctness. Exercise 6 supplies a proof-of-delivery
source (`proof_of_delivery.json`) and a `ProofCheckingVerifier` starter
(`starter_06.py`): implement the subclass and run `test_exercise_06.py`. The
untouched starter fails on purpose. The core verifier is unchanged.

## What it leaves out

In the companion, payments go into a simulated ledger; in the specified deployment
a person carries out every refund. Autonomy is unchanged. A live verifier adapter
must enforce its supplied deadline and raise `TimeoutError` when it expires.
