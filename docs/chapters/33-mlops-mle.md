# Chapter 33: MLOps and ML Engineering

[Companion README](../../README.md)

This chapter's code treats a release of the refund agent as one version bundle of
five components (model, prompt, tools, policy and index) with a single
fingerprint. It states what evidence each kind of change must show before
promotion, keeps signed promotions and rollbacks in an append-only registry, and
refers any request decided under a policy version the running bundle was not
judged under. The demo runs a constructed fortnight in shadow in which a new policy
version takes effect, and measures what promoting the bundle late or early costs.

## Files

| What | Where |
|---|---|
| Demo | `demos/mlops_mle_demo.py`, with its runs in `demos/mlops_mle_demo_helpers.py` |
| Code | `refund_agent/promotion/mlops_mle_*.py`: bundle, differences, evidence requirements, policy coverage, lineage stamp, promotion record, registry and component; `python -m refund_agent.promotion` is its command line |
| Exercises | `exercises/mlops-mle/` |
| Stored outputs | `book_outputs/_fnd.mlops-mle.*.txt` (7 files) |
| Tests | `tests/refund_agent/test_mlops_mle_bundle_contract.py`, `tests/refund_agent/test_mlops_mle_promotion.py` |

## Run it

From the companion folder, after the install in the README:

```sh
python -m demos.mlops_mle_demo
python -m pytest tests/refund_agent/test_mlops_mle_promotion.py tests/refund_agent/test_mlops_mle_bundle_contract.py
python -m refund_agent.promotion show --bundle exercises/mlops-mle/incumbent-05.json
```

`show` prints a bundle's record and fingerprint. `check` decides a promotion:

```sh
python -m refund_agent.promotion check --candidate B1.json --incumbent B0.json \
  --to shadow --evidence EVIDENCE_DIR
```

Put the paired `release.json` and the other JSON records the requirements name in
the evidence directory, which is the component's declared journal directory. The
command prints the promotion record and exits 1 when the promotion is held;
`sign --record RECORD --owner OWNER` adds the owner's sign-off. The demo's
`--write-data` flag rewrites the data files in `demos/data/` and the exercise
records.

To use the bundle in your own run, select `mlops-mle.bundle@1` with
`mlops_mle_component.selector(snapshot, versions)` beside the data-contract,
event-time and canary components of chapters 30 to 32.

## What the output shows

- **The bundle** (`bundle-01`, also a listing you can run). Each component has a
  fingerprint prefix, the chapters that built it, and whether the loop reads it:
  the model (scripted, with a text risk score at threshold 0.2), the prompt, the
  tools, the policy (versions, register and offer rule) and the retrieval index.
  The fingerprint covers 130 files, and the bundle declares the five policy
  versions it was judged under, from 2026-01-20 to 2026-09-01. It covers chapter
  29's fixed core and the selected components' code, including their grading;
  changing only an evaluation manifest or unselected code leaves it unchanged. No
  environment variable or secret enters it.
- **What a change must show** (`requirements-01`). Every promotion needs the
  release, passing cases for the declared versions, and the release owner. Each
  changed item adds its own evidence: a passing regression suite for the prompt, a
  runtime swap record for the runtime or adapter, a rerun of the suite for a new
  environment, policy-change cases and the owner's notice for a new policy
  version, and for the index, that no frozen question loses its clause.
- **The fortnight** (`week-01`). Four requests a day with generated labels, all in
  shadow. A new policy version is announced on day 1 and takes effect on day 8.
  Promoted late, on day 11, the old bundle B0 refers all 12 requests decided on
  days 8 to 10 as `policy_not_in_bundle`; promoted early, on day 5, nothing is
  referred. The matched controls run the full shadow pipeline without this
  chapter; 8 of the late arm's 12 controls match the generated labels, and the
  four exceptions are `stale_order` once and named review of two
  `amount_over_policy` denials and one `outside_return_window` denial. Every
  lineage record is stamped with its bundle.
- **Promotion records** (`records-01`). B0 from offline to shadow, B1 (changed
  index and policy) to shadow, B0 to canary, and a bundle with a changed model
  (which also needs calibration, a signed cost matrix and a threshold experiment)
  are all held, with the release requirement unmet. The rebuilt receipt matches.
  The canary start runs six checks and is refused: the release record is
  unavailable, no error budget is named, and the bundle is not admitted to the
  canary.
- **Rollback** (`rollback-01`). A mixed state, B1 with B0's register snapshot, has
  a fingerprint no record names, so the registry admits it to neither stage. Put
  B0 back on day 12 and it refers every request from then on, 12 in all, because
  it was never judged under the new version. Restoring software does not renew
  spent, ended or expired payment authority.
- **Fingerprint collisions** (`collision-01`). With exact fractions, the chance
  that two of a million bundles share a 16-hex-digit prefix is bounded by about
  2.71e-8; for 10,000 bundles and 7 digits the bound is 0.186 (0.170 by the
  birthday approximation). `keys-01` holds the exercises' numbers.

`Registry(path, namespace)` appends bundles and transitions to one JSON-lines file.
`admitted(bundle_fp, stage, at=...)` reads the latest effective signed transition;
`rollback(record, bundle, owner=..., at=...)` restores the reversed record's exact
target; `demote(stop, bundle, canary_id=...)` records every ended canary stop,
including a spent budget and the time box, while a pause leaves admission alone.
`PolicyCoverage` checks the policy in force at assessment and again at execution,
and stamps every lineage record before it is stored.

## Exercises

Exercise 5's `record-05.json` is a deliberately faulty promotion record: compare
`candidate-05.json` with `incumbent-05.json`, recompute their fingerprints, and
name the evidence its incomplete diff left out. For exercise 6, run

```sh
python exercises/mlops-mle/mlops_mle_check_06.py
python exercises/mlops-mle/mlops_mle_check_06.py --candidate exercises/mlops-mle/b1_06.json
python -m pytest exercises/mlops-mle/test_acceptance.py
```

The untouched `requirements_06.py` lacks the runtime and adapter row, so it wrongly
prints `promoted` for `candidate_06.json` and the first acceptance test fails. Add
that one row: the command must then exit 1 with exactly `runtime_swap` unmet, and
B1 must still exit 0. `reference_06.py` is the reference solution. Exercises 3
and 4 (collision arithmetic, and the three days before a June sign-off) have their
numbers in `keys-01` and `collision-01`.

## What it leaves out

- The fortnight, its policy and its requests are constructed, the labels are
  generated, and every run is in shadow.
- A bundle cannot see a component nobody listed. Its fingerprint says which
  bundle runs, not whether it is good.
- A rollback cannot take back a policy change.
- No canary starts: the release records are held and no error budget is named.
