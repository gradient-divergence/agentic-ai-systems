# Chapter 28: Agent Frameworks and Orchestration

[Companion README](../../README.md)

The chapter runs the refund agent on two offline runtimes, its own loop and a
second, library-shaped runtime, and compares the records the two leave on the same
cases. An adapter carries the agent's rules onto the second runtime's hooks, and the
swap record shows what each of the adapter's settings changes. A second demo is a
small harness for an agent that edits code: disposable files, a checkpoint, an
injected crash, and held tests that only the owner runs.

## Files

| What | Where |
|---|---|
| Demos | `demos/agent_frameworks_demo.py` (the runtime swap), `demos/harness_demo.py` (the coding harness) |
| Code | `refund_agent/runtimes/`: `runner.py` (the second runtime), `adapter.py` (the refund adapter), `faults.py` (injected faults), `swap.py` (the cases and the comparison) |
| Exercises | `exercises/agent-frameworks/` |
| Stored outputs | `book_outputs/_fnd.agent-frameworks.*.txt` (7 files), and `_fnd.agent-frameworks.swap-record.json`, the complete swap record |
| Tests | `tests/refund_agent/test_runtime_swap.py` |

## Run it

From the companion folder, after the install in the README:

```sh
python -m demos.agent_frameworks_demo
python -m demos.harness_demo
python -m pytest tests/refund_agent/test_runtime_swap.py
```

The first demo prints `swap-01`, `swap-02`, `resume-01`, `keys-01` and `replay-01`;
the second prints `harness-01`. The harness works in a temporary directory, starts
its workers with the same Python, and leaves nothing behind. `--write-data` rewrites
the exercise fixtures in `demos/data/` and the swap record in `book_outputs/`.

## What the output shows

**The second runtime and the adapter.** `refund_agent/runtimes/runner.py` is a
generic, offline runtime shaped like an agent library. Its defaults are the
companion's own and represent no vendor's retry policy. `adapter.py` reimplements
the own loop's step semantics at the runner's hooks, without editing the agent
(`refund_agent/agent/__init__.py`) or its tools (`refund_agent/tools.py`). It mounts
the tools instance the caller supplies, including any subclass's authorization check
after the gate. The caller's tools keep the contracts, the proposal-bound approvals,
the ledger and the audit; the session and the spans of each attempt belong to the
runtime. The listing at the top of the demo, `mount`, shows the five settings: end
at audit, one tool per phase, refer on failure, no retry, and the agent's budget of
12 steps with one kept for the audit record.

**The swap (`swap-01`).** Seed 7 gives 60 chosen cases: 40 public, eight
independent refusals, five positive payment checks, five injected faults or limits,
and two cross-runtime resumes. Each side starts with fresh tools from one supplied
factory. Public cases keep their reference-matching orders, all policies and the
reviewer list, and their own shipment and retrieval records; no held-out or frozen
set is read.

| Configuration | Cases that differ | Money differs | Cents against policy |
|---|---:|---:|---:|
| runtime defaults | 60/60 | 12 | 0 |
| audit ending | 10/60 | 3 | 0 |
| ordered calls | 8/60 | 3 | 0 |
| referral ending | 4/60 | 3 | 0 |
| no retries | 1/60 | 1 | 0 |
| full adapter | 0/60 | 0 | 0 |

No row pays against the policy, yet money moves differently in every row but the
last. The comparison covers ten fields: decision, outcome, reason, tier, amount,
executed, tool calls, ledger receipts, audits and failure detail. The agent's
autonomy stays fixed under every configuration: a request above $500, a denial
included, still goes to the named approver.

**A retry that pays (`swap-02`).** When `refund.execute` times out before acting,
with the audit ending and the refusal handling carried but the retry left on, the
own loop refers the request (`tool_error`) and pays nothing, while the second
runtime calls `refund.execute` twice and pays 2,499 cents.

**Resumes (`resume-01`).** The two tier 2 resume cases start on one runtime and
finish on the other, in both directions. Each is approved and pays once (5,000 and
50,000 cents) with no difference. The approval records stay in the supplied tools;
no hosted session is exported.

**One request end to end (`replay-01`).** Request `r-7-0085` (4,999 cents,
`changed_mind`) shows the separate memory experiment beside the loop (an April copy
of the policy is refused at use as beyond notice; a seven-day-old copy is accepted),
the 30 words of required facts, and the loop as it stands paying once on each
runtime, with the verifier agreeing and none of the ten fields differing.

Passing all 60 cases makes the full adapter eligible for chapter 29's shadow run. It
grants no release or shadow authority. The scripted model, the runtime, the adapter,
the tools composition and the suite have fingerprints that chapter 29's candidate
records.

**The harness (`harness-01`).** A scripted worker may propose file replacements,
not shell commands, on a two-function repository with four visible tests and a
budget of three patches. `apply_patch` lets a patch change only `toy.py`, and
refuses before writing if the patch names another file or `toy.py` is a symbolic
link. The checkpoint is an immutable, content-addressed file commit; Git is never
called. After cycle 3 the worker exits with status 23. A fresh worker restores the
checkpoint and reruns the tests before any edit. Its patch to the test file is
refused, changes nothing and still spends an attempt. The source patch then passes
all four visible tests, and the owner's six held tests, which the worker never
reads, accept the change: fail-to-pass 4/4, keep-passing 2/2, three patch attempts
of three.

## Exercises

Exercises 3 and 5 use `demos/data/agent_frameworks_swap_03.json` and
`agent_frameworks_swap_05.json`; exercise 7 uses `exercises/agent-frameworks/harness_07.json`.
The public keys to exercises 3, 4, 5 and 7 are in
`book_outputs/_fnd.agent-frameworks.keys-01.txt`.

Exercise 6 mounts the agent on `exercises/agent-frameworks/runner_c.py`, a third
runtime. Its defaults differ from the second runtime's: a reply naming no tool keeps
the run going, no call is retried, an error raises instead of returning to the
model, the turn limit is six, and several named reads run concurrently. Write
`mount(tools, model=None, settings=None)` in a file of your own, mounting the exact
tools you are given, and run:

```sh
AGENT_FRAMEWORKS_SOLUTION=path/to/your_file.py \
  python -m pytest exercises/agent-frameworks/test_exercise_06.py
```

The variable is required. The tests ask for the complete comparison, both
cross-runtime resumes, one receipt per request, and the rejection of a reply that
names two reads before either runs, keeping `model_error:order:TypeError`.
`starter.py` runs but leaves call retries on, so its complete-suite test fails when a
timeout changes a payment, and the other three pass. `reference.py` is a reference
solution.

## What it leaves out

Chosen cases establish coverage, not equivalence or a rate. The scripted model is
held fixed, so live-model variation is not measured. The swap tests no vendor SDK,
concurrent refund, process crash or hosted-session export, and authorizes no shadow
run or deployment. The harness shows an authority boundary, not an operating-system
sandbox. In the companion, payments go into a simulated ledger; deployed, a person
still carries out every refund through Part V.
