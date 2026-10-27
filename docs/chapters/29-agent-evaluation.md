# Chapter 29: Agent Evaluation and Release Evidence

[Companion README](../../README.md)

The chapter gives the refund agent an evaluation that can say no. A fixed release
suite grades effects (the ledger, the audit log, the review queue and the decision),
a release rule stops a candidate on any failing must-pass case, a named owner signs
the record, and a shadow switch admits only a signed, journaled pass. The demo runs
the agent as built and three planted candidates through the suite, prints the
trial statistics and a grader's agreement with adjudicated cases, and starts one
generated shadow day.

## Files

| What | Where |
|---|---|
| Demo | `demos/agent_evaluation_release_demo.py` |
| Code | `refund_agent/release/` (suite, candidate, rule, journal, graders, judges, trials, extensions), `refund_agent/shadow/` |
| Exercises | `exercises/agent-evaluation/` |
| Stored outputs | `book_outputs/_fnd.agent-evaluation.*.txt` (7 files) |
| Tests | `tests/refund_agent/test_release.py`, `tests/refund_agent/test_shadow.py`; the `test_release_*.py` files beside them test the extension mechanism |

## Run it

From the companion folder, after the install in the README:

```sh
python -m demos.agent_evaluation_release_demo
python -m refund_agent.release.trials
python -m pytest tests/refund_agent/test_release.py tests/refund_agent/test_shadow.py
```

The demo prints six outputs (`suite-01`, `release-01`, `judge-01`, `shadow-01`,
`keys-01`, `cli-01`); `python -m refund_agent.release.trials` prints the seventh,
`trials-01`. The demo runs four candidates through 330 cases five times each, so it
takes a few minutes. `--write-data` rewrites `demos/data/agent_evaluation_*.json`
and `exercises/agent-evaluation/coverage-06.json`.

The release rule is also a command. It exits 0 on a pass and 1 otherwise:

```sh
python -m refund_agent.release --run-dir runs/agent-evaluation
python -m refund_agent.release sign --run-dir runs/agent-evaluation --owner morgan
python -m refund_agent.shadow --release-record runs/agent-evaluation/release.json \
  --output runs/agent-evaluation/shadow.json
```

Options: `--candidate` (`as-built` by default), `--runtime own|second`,
`--verifier off|independent|timeout|void`, `--trials` (5), `--requirement PATH`,
`--components PATH`, `--scenario PATH`, `--dispositions PATH`. Every verdict,
aborted trials included, goes into the run directory's `journal.jsonl`.

## What the output shows

**The suite (`suite-01`).** Seed 7, five trials, 330 cases in 15 parts: 40 public,
8 refusals, 5 positive payments, 5 faults, 2 cross-runtime resumes, 252 listed
cases, and the state-continuation, contract, approval, execution-time,
memory-support, regression, runtime-swap, offer-record and verifier cases. All are
must-pass. The comparison arms are the candidate agent bundle and the rules-only
baseline on the same requests. Chapter 14's spent artifact enters as a regression
guard with an access record, not as a quality measure, and no golden case is
consulted again. The generated scenario sets and the memory funnel are reported
apart, with no release authority of their own.

**The release (`release-01`).** The agent as built passes. The planted candidates
fail: `resume-saved` on 3 cases (the state-continuation cases, where a saved
source pays 5,000 cents that unchanged tools would not pay), `deny-all` on 208 and
`gate-removed` on 5 (among them `refusal-2-amount_mismatch`, which pays 999,999
cents without the gate). The referral floor is signed in advance in
`refund_agent/release/requirement.json`: on seed 7's 15 tier 1 public requests,
no forbidden payment and no more than 7 referrals, the baseline's count. The
deny-all control passes that floor, since a denial is neither a forbidden payment
nor a referral; the rest of the suite stops it. Every unexecuted tier 3 outcome
opens a review for the named approver, who decides every request above $500.

**Trials (`trials-01`).** `pass@k` (at least one of `k` trials passes) and `pass^k`
(all `k` pass) are estimated without bias from `n` trials with `c` passes, beside
their plug-in values. On constructed suites, pooling scenarios into one pass chance
overstates `pass@k` and understates `pass^k`. The agent passes its 40 public
cases on each of five trials, but those 200 deterministic runs are 40 scenarios,
not 200 independent requests, and bound no live rate.

**The grader (`judge-01`).** On a constructed bank of 200 adjudicated letters, 20
failing, a judge agrees on 94.5% with Cohen's kappa 0.6875, and misses 6 of the 20
failing letters (95% Wilson interval 0.1455 to 0.5190). A judge that passes every
letter agrees on 90% with kappa 0. A soft grader may hold a case for a person; its
pass alone cannot pass a must-pass case.

**Shadow (`shadow-01`).** The signed pass admits the first generated shadow day of
120 requests. The agent would pay 16 and send 60 for review, yet the real ledger
and review queue stay unchanged and no money moves. On the 35 tier 1 requests it
makes no forbidden payment and refers 7, as the baseline does. Wrongful denials,
counted against the baseline on the same 299 tier 1 requests, can't be counted
yet: a denial is wrongful when the person paid the request and no one found the
payment false within 30 days, and those labels are still open. So the day grants no
canary sign-off. Generated labels stand in for a blind person's decisions.

**The command line (`cli-01`).** The as-built record passes, is signed and starts
shadow; the `resume-saved` record fails, its signing is refused and shadow stays
unstarted.

## How the release record works

- A candidate's record names `name`, `agent`, `model`, `runtime`, `adapter`,
  `tools_composition`, `offer_policy` and `suite`. Its fingerprint names the
  software. A change to the cases, the trials or the signed requirement changes
  the evaluation's identity but clears none of the software's failures.
- Unchanged software can't rerun a hard failure into a pass, even on a changed
  suite. Changed software runs the whole suite and carries every earlier failure
  until a disposition covers it: `fixed` (the case passes every new trial) or
  `retired` (a listed signer, a reason and a time). Must-pass failures can't be
  waived; signed overrides settle only soft holds.
- Changing the ceiling under the old sign-off is refused; `--requirement` supplies
  a newly signed one. Signatures are recorded owner-and-time sign-offs from
  `refund_agent/release/owners.json`, not cryptographic ones.
- The shadow entry point checks the exact verdict, the sign-off and the software's
  failure history before it starts. In shadow, payment and review writes are
  recorded, never sent.
- At the freshness limit a payment must be audited; past it, payment is forbidden.

Later chapters add their checks as components registered with
`refund_agent/release/extensions.py`. A component declares its modules, every file
it reads and every store it writes (`RuntimeStore(path, owner, mode)`, with modes
`read`, `append` or `read-write`), and hooks such as `compose`, `wrap_model`,
`cases` and `run_case`. Undeclared reads, writes beyond a store's mode, process
spawning, threads and native loading are refused during a component's import,
selection and hooks. **This is an integrity check against honest mistakes, not a
sandbox against hostile code**; untrusted code needs operating-system isolation,
the subject of chapter 35.

## Exercises

`exercises/agent-evaluation/README.md` describes the six exercises; `practice.md`
holds extra practice with keys, and `keys-01` prints the keys to exercises 3 to 6.
Exercise 3 uses `refund_agent/release/trials.py`, exercise 4
`refund_agent/release/judges.py`, and exercise 5 `record-05.json`. Exercise 6 wires
the suite's fixed approval-reuse scenario into the empty tier 2 cell of
`coverage-06.json` with a JSON record of your own; `candidate_06.py` pays on a
reused approval. Run the acceptance tests with your file:

```sh
AGENT_EVALUATION_SCENARIO=exercises/agent-evaluation/my-scenario.json \
  python -m pytest exercises/agent-evaluation/test_acceptance.py
```

They require the stale-approval candidate to exit 1 with your case among its
failures, the agent as built to exit 0, and a changed suite fingerprint. The
README gives the `run_06.py` commands that produce the deliverable's records.

## What it leaves out

Every fixture is constructed, and the sign-off is a constructed owner's. Chosen
cases and repeated deterministic trials are coverage, not a live rate. The shadow
day uses generated requests and labels. In the companion, payments go into a
simulated ledger; deployed, a person still carries out every refund through
Part V.
