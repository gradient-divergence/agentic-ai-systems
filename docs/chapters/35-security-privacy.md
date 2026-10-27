# Chapter 35: Security and Privacy

[Companion README](../../README.md)

Chapter 35 puts scoped credentials around the refund agent's tools and attacks
them. `demos/security_privacy_demo.py` runs constructed attacks against an
effect oracle that reads the actual stores and returned decisions, then runs
positive controls, revocation, keyed hashing for exports, a leakage scan and the
reply grids. Everything is constructed and offline: no live model, no real
customer data, no operating-system sandbox and no live canary.

## Files

| What | Where |
|---|---|
| Demos | `demos/security_privacy_demo.py`; `demos/security_privacy_integration_demo.py` (the integration preflight) |
| Code | `refund_agent/security/` (every module starts with `security_privacy_`) |
| Exercises | `exercises/security-privacy/` |
| Stored outputs | `book_outputs/_fnd.security-privacy.*.txt` (10 files) |
| Plot data | `demos/data/security_privacy_attempts.json` |
| Tests | `tests/refund_agent/test_security_privacy.py`, `test_security_privacy_grid.py`, `test_security_privacy_integration.py` |

The package holds the accounts and the trusted metadata resolver; credentials,
the credential check, scopes and the append-only credential log; the outermost
tools wrapper; and the scripted model, attack copies, effect oracle, reply grid,
experiment, leakage scan, hashing, arithmetic, release checks and the component
registration. The printed credential listing is
`fnd.security-privacy.credential-01`.

## Run it

From the companion folder, after the install in the README:

```sh
python -m demos.security_privacy_demo
python -m demos.security_privacy_integration_demo
python -m pytest tests/refund_agent/test_security_privacy.py tests/refund_agent/test_security_privacy_grid.py
```

The first demo prints nine stored outputs (`attack`, `filter`, `grid`,
`revoke`, `hash`, `scan`, `attacks`, `positive` and `keys`); the second prints
`integration-01`.

## What the output shows

- **One attack, two runs** (`attack-01`). A constructed resent request names
  another account's order. Without scopes the agent approves and pays 2,499
  cents; with scopes it refers with `order_not_found` and pays nothing. No
  fetched foreign-order field reaches state, traces or replies.
- **Every constructed attack** (`attacks-01`). Each row names the attack, where
  it enters and the check that stops it with scopes: a resent order reference
  (the scope resolver), hidden instructions in the request text (the risk rule),
  paraphrased and policy text (the credential's amount), an approval sent to
  the reviewer intake without a credential, a secret probe, replay, expiry,
  revocation and carrier status text (the `ScanEvent` and shipment contract).
  Forbidden attack runs: 2 without scopes, 0 with them. Both fault controls (the
  scope removed, and a scope that skips the order) fail the local suite, as they
  should. An altered reviewer role fails the credential seal and stores no
  approval. The secret probe still pays the honest 2,499 cents and finds no
  secret.
- **Reply grids** (`grid-01`). The ordinary grid tries 288 replies with fresh
  tools for each, with no forbidden effect. All 14 payments equal the honest
  refund, 2,499 cents on `o-0007`. The three tier 3 paths (eligible, over the
  cap, already delivered) try 864 replies: tier 3 pays nothing, and every
  terminal path opens named-approver review.
- **Positive controls** (`positive-01`). Event-time runs reissue credentials at
  every resumed run. All five must-pay cases pay, 62,598 cents in total. Of the
  forty public decisions, thirty-nine are unchanged, including eight denials.
  `r-7-0000` has one authorized match to the shared reference and now pays 2,499
  cents instead of being referred as globally ambiguous. `positive-100` and
  `r-7-0008` get fresh credentials on timed event-time reruns.
- **Revocation** (`revoke-01`). Revoking the payment scope while reads continue
  turns the public forty's five payments into `credential_revoked` referrals
  and pays 0 cents; no grant is widened. Revoking shipment or policy access stops
  the read and the payment. Reinstating a scope needs a fresh credential:
  grants and credentials issued while it was revoked stay refused for it, and
  other scopes keep working.
- **Keyed hashing** (`hash-01`). Enumerating 10,000 identifiers recovers a plain
  SHA-256 identifier at position 13; the keyed hash is not recovered after all
  10,000. Exports key five join fields, which breaks shared suffixes, but anyone
  holding the key can relink them: the data are pseudonymized, not anonymous.
- **Leakage scan** (`scan-01`). Every recorded trace, audit, decision,
  credential log and prompt of the demo, 19,609 objects, holds no raw,
  hexadecimal or base64 copy of a secret.
- **Attempts arithmetic** (`filter-01`). A constructed text rule and a fixed 5%
  chance per independent attempt, not a measured attack rate: at least one
  attempt gets through with probability 0.512 after 14 attempts and 0.952 after
  59 (even odds after 13.51, so 14 is the fewest). Fifty clean independent
  trials bound the per-attempt chance at 0.0582 (95% upper bound). `keys-01`
  checks the exercise answers.

The public issuer key and the separate export key are fixture labels that
illustrate HMAC; they protect nothing against a reader who can derive them, and
no key or seal is ever printed. A deployment keeps unpredictable keys in the
issuer and the tool service.

| Recipient | Purpose and boundary | Status |
|---|---|---|
| Requester | Decision on orders within the authenticated account grant | Companion tool mediation |
| Reviewer | Open reviews matching the credential's subject and role | Companion approval intake |
| Operator | Credential and scope revocation; no agent tool permission | Companion operator intake |
| Telemetry reader | Authorized operations or breach review; least needed records | Deployment role and purpose control |
| Export recipient | Keyed identifiers; no raw prompt or customer payload | Companion closed export schema |

## How it joins the release

The component is `security-privacy.scoped` (version `security-privacy.scoped@1`,
with `scopes@1`) and declares its own code through chapter 29's fixed hooks.
Selecting it requires an explicit `SecurityRuntime`, built from authenticated
intake and account records frozen before any attack; it never infers an
authenticated sender or a grant from attack data. The resolver refuses before
the wrapped lookup, and the same scope limits chapter 30's run-entry snapshots
and stored continuations. The gate's three internal rereads are each authorized
and logged before dispatch. Chapter 34's run span binds to the credential log
through `ScopedTools.bind_run_record`.

The release suite's injection slot adds a must-pass full-effects case. Its
handler runs every constructed attack, both reply grids, the leakage scan, the
positive controls, the public comparison and both fault controls, and a failed
check aborts the release trial.

## Exercises

`exercises/security-privacy/README.md` has the details. Exercises 1 to 5 each
start from a deliberately wrong answer, `security_privacy_0N.py`; exercise 5
diagnoses `day-05.json`, a constructed day with a corrupted account grant.
Exercise 6 starts at `security_privacy_delegate_06.py`, whose starter fails
exactly two tests: expiry and onward delegation. Run each test file separately:

```sh
python -m pytest exercises/security-privacy/test_security_privacy_acceptance.py
python -m pytest exercises/security-privacy/test_security_privacy_exercise_06.py
```

`SECURITY_PRIVACY_REFERENCE=1` checks the reference solutions instead.

## What it leaves out

- Code in this Python process is trusted. Operating-system isolation, network
  and file confinement, a real identity provider and retention enforcement are
  deployment requirements; the companion builds no compute sandbox and erases
  no append-only data.
- The core release fixture supplies no authenticated security runtime, so the
  tests run the release handler with explicitly constructed authenticated runs,
  and no assembled passing release record exists.
- `integration-01` is a refusing preflight: the assembled start, the correction
  pause and resume, and the time-box end are not run, and no cohort payment or
  successful start is claimed. Chapter 32's start checks run only for selected
  components, so they cannot notice a missing scope component.
- A tier 3 request with an unauthorized order reference opens a request-only
  named-approver review with reason `order_not_found`, discloses no order, and
  cannot be approved for payment.
