# Agent Evaluation and Release Evidence exercises

Paths here are relative to the companion root. All fixtures are constructed.
Exercises 1 and 2 are conceptual, 3 uses `refund_agent/release/trials.py`, and 4 uses
`refund_agent/release/judges.py`. `record-05.json` supplies exercise 5; `practice.md` holds
additional practice and its keys.

Exercise 6 is a **fixed-scenario wiring exercise**. It starts with the
`approval-reuse` tier 2 cell in `coverage-06.json` empty. `candidate_06.py` pays
after an approval's proposal changes. Add a JSON record with a new case ID,
part `approval`, failure class `approval-reuse`, tier 2, handler `approval-reuse`,
and `must_pass: true`. Set `expected` exactly to
`{"stores": ["ledger", "audits", "reviews", "decision"]}`. This wires the fixed
scenario and its fixed no-payment grader into the suite. Additional submitted
assertions, including contradictory payment expectations, are rejected.
The starter deliberately has no scenario. No golden or instructor case is needed.

Run the acceptance tests from the companion root with your file:

```sh
AGENT_EVALUATION_SCENARIO=exercises/agent-evaluation/my-scenario.json python -m pytest exercises/agent-evaluation/test_acceptance.py
```

The tests run the release rule for the candidate and the agent as built and
require exit codes 1 and 0, your scenario in the failures, and a changed suite
fingerprint. An reference solution is supplied separately in
`reference_solution.py`; the reader test never reads it.

The current as-built release passes, and the reference wiring passes acceptance.
The stale-approval candidate passes with the missing cell and fails when it is
wired in. A named approver reviews every unpaid tier 3 request on both runtimes.

The exercise command in `exercises/agent-evaluation/run_06.py` starts from the
same suite as acceptance: the default suite without
`approval-reused-after-proposal-change`. Before wiring your scenario, the
candidate passes (exit 0):

```sh
python exercises/agent-evaluation/run_06.py --candidate stale-approval --trials 1 --run-dir exercise-06-before
```

Generate command records with your scenario, using a fresh run directory for each
run, without reading an instructor case:

```sh
python exercises/agent-evaluation/run_06.py --candidate stale-approval --trials 1 --scenario exercises/agent-evaluation/my-scenario.json --run-dir exercise-06-candidate
python exercises/agent-evaluation/run_06.py --candidate as-built --trials 1 --scenario exercises/agent-evaluation/my-scenario.json --run-dir exercise-06-as-built
```

The candidate exits 1 with your case in its failures; as-built exits 0.
Software and suite identities are distinct; a changed suite requires explicit failure dispositions.
