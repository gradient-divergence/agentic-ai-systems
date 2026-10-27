# Chapter 26: Memory and Retrieval

[Companion README](../../README.md)

A policy memory beside the refund loop. `refund_agent/memory/` keeps an append-only
policy register with valid intervals, announcement times and recording intervals.
Texts live separately, keyed by their hashes. A correction appends a record, and
retention expires a text without changing any record. `project(snapshot, known_at)`
supplies one complete `Policy` per version to chapter 11's unchanged `PolicyIndex`,
and `PolicyMemory` refuses a copy older than the register's seven-day notice
promise. The agent, `policy.retrieve`, the check at use and the payment gate are
unchanged.

## Files

| What | Where |
|---|---|
| Demo | `demos/memory_rag_demo.py` |
| Code | `refund_agent/memory/` (`register.py`, `memory.py`, `assemble.py`, `funnel.py`, `support.py`) |
| Exercises | `exercises/memory-rag/` |
| Stored outputs | `book_outputs/_fnd.memory-rag.*.txt` (10 files) |
| Tests | `tests/refund_agent/test_memory.py`, `tests/test_memory_rag_demo.py` |

## Run it

From the companion folder, after the install in the README:

```sh
OMP_NUM_THREADS=1 python -m demos.memory_rag_demo
OMP_NUM_THREADS=1 python -m pytest tests/refund_agent/test_memory.py tests/test_memory_rag_demo.py exercises/memory-rag/test_exercise_06.py
```

The run is seeded, offline and uses training data only. `--write-data` writes
`demos/data/memory_rag_figures.json`, `demos/data/memory_rag_support.json` (every
training draft and the clauses it needs) and the exercise fixture
`exercises/memory-rag/hospital_register_05.json`.

## What the output shows

The refund measurements use only the 398 training requests, before 1 July 2026; no
validation or frozen rows are read.

- The retrieval funnel on a seeded telecom library: 24 services with four
  procedures each, 202 dated passages, and 400 questions written three ways
  (identifier, paraphrase, plain). The word budget is 180, the first depth 12 and
  the main cutoff 5; words stand in for tokens. Reranking helps identifier and plain
  wording but lowers paraphrase survival from 49.25% to 45%. The sweep shows the
  product plateau from cutoff 4 with dates. The library is constructed, not a
  dense-against-lexical benchmark, and these are results, not tuned parameters.
- Budget: a request whose raw evidence (2,040 words) and draft (400) exceed a
  2,048-word window. Assembly keeps the required facts and cuts 13 carrier records;
  the decision is preserved on 398 of 398 training requests, and keeping only the
  last five scans would have referred with `delivery_unknown`.
- As-of reads: the same request read as known on 20 April and on 12 May gets the
  21-day and the 30-day version. An April copy at 12 May is refused as
  `beyond_notice`; exactly 7 days is accepted, 7 days and 1 second is not.
- Monthly copies read on the 20th: 94 of 398 training requests would read an ended
  version, and 9 policy conclusions would change (tiers 1, 2 and 3: 6, 2 and 1). The
  unchanged check at use refers all 94 with `non_current_policy`; copies no older
  than 7 days change nothing.
- Lifecycle: a constructed correction, recorded on 1 October, shortens May's window
  from 30 to 25 days. The audit of `r-7-0085` then reads approve as known then and
  deny as known now, and the correction breaks the notice promise. Retention expires
  seven texts while the records stay identical.
- Support: of 398 training drafts, 201 are `supported` and 197 referrals are
  `not_checked`. Five planted unsupported drafts are each refused with a named
  reason, and the request is referred.

The promise of seven days' notice and the monthly reads on the 20th are constructed
settings, with announcements recorded at the same instant. A request above $500,
including a denial, still goes to the named approver.

## The support check for chapter 29

`support.support(draft, register, request_facts)` takes the existing `Decision`
and `assemble.required_facts(state)`, with `at` fixed to the decision time and
`known_at` to the draft time. Its report names every required clause, record
identifier and text hash, and returns `supported`, `refused` with a reason, or
`not_checked` for a referral. An approval needs the eligibility and cap clauses plus
the unrefunded balance. Support is not truth or execution authority: a constructed
on-time but wrong register still yields a supported denial.

## Exercises

Exercise 6 audits a late policy: complete `audit(register)` in a copy of
`exercises/memory-rag/starter_06.py`, returning the training drafts whose decisions
change between the request's knowledge time and 1 October 2026. Run the acceptance
tests against your file:

```sh
MEMORY_RAG_SOLUTION=/absolute/path/to/solution.py OMP_NUM_THREADS=1 python -m pytest exercises/memory-rag/test_exercise_06.py
```

The harness runs your file in a fresh process with a training-only view of the
demo: the 398 training requests, their orders, shipments, snapshots and labels.
Data generators and other row-exposing modules are refused. It is a teaching guard,
not a security sandbox. Without `MEMORY_RAG_SOLUTION`, the tests check that the
untouched starter is rejected and the register setup is correct.

## What it leaves out

The companion still pays into a simulated ledger; the specified deployment leaves
every refund to a person.
