"""Chapter 35's deterministic, constructed Security and Privacy evidence.

Import is inert. Capture each marked output through the repository function.
No live model, real customer data, operating-system sandbox or live canary.
"""

import runpy
from collections import Counter
from math import log
from pathlib import Path

from refund_agent.data.generate import generate
from refund_agent.security.security_privacy_accounts import AccountRecords
from refund_agent.security.security_privacy_attacks import carrier_status
from refund_agent.security.security_privacy_experiment import (
    FIXTURE_KEY,
    HASH_KEY,
    attack_runs,
    positive_runs,
    public_runs,
)
from refund_agent.security.security_privacy_grid import reply_grid, tier3_grids
from refund_agent.security.security_privacy_hashing import (
    export_records,
    keyed,
    plain,
    recover_plain,
)
from refund_agent.security.security_privacy_math import attempts_for, chance, zero_success_bound
from refund_agent.security.security_privacy_scan import scan
from refund_agent.security.security_privacy_tools import ScopeSkipsOrder


def text(*lines):
    return "\n".join(str(line) for line in lines) + "\n"


def create_outputs():
    visible, secrets = [], [FIXTURE_KEY, HASH_KEY]
    without, raw, seals = attack_runs(scoped=False)
    visible.extend(raw)
    secrets.extend(seals)
    scoped, raw, seals = attack_runs()
    visible.extend(raw)
    secrets.extend(seals)
    mutation, raw, seals = attack_runs(tools_type=ScopeSkipsOrder)
    visible.extend(raw)
    secrets.extend(seals)
    positives = positive_runs(observation_sink=visible, secret_sink=secrets)
    bare_positives = positive_runs(scoped=False, observation_sink=visible)
    public = public_runs(observation_sink=visible, secret_sink=secrets)
    bare_public = public_runs(scoped=False, observation_sink=visible)
    revoked = public_runs(revoke_payment=True, observation_sink=visible, secret_sink=secrets)
    grid = reply_grid(observation_sink=visible, secret_sink=secrets)
    tier3 = tier3_grids(observation_sink=visible, secret_sink=secrets)
    outputs = {}
    outputs["attack-01"] = text(
        "Constructed resent request; original payment destination retained.",
        f"Without scopes: {without[0].outcome}; {without[0].reason}; paid {without[0].paid_cents} cents",
        f"With scopes: {scoped[0].outcome}; {scoped[0].reason}; paid {scoped[0].paid_cents} cents",
        "Scoped decision sent: refer, amount 2499, tier 1, citation none.",
        "No fetched foreign-order fields reach state, traces or replies.",
    )
    attempts = [f"Independent attempts n={n}: {chance(0.05, n):.3f}" for n in (14, 20, 59, 60)]
    accounts = AccountRecords.from_dataset(generate(7))
    outputs["filter-01"] = text(
        "Constructed fixed independent-attempt inputs; no measured attack rate.",
        "Text rule: ignore = 0.9; disregard = 0.1; threshold = 0.2.",
        *attempts,
        f"Even chance: {log(0.5) / log(0.95):.2f}; fewest attempts {attempts_for(0.05, 0.5)}",
        f"50 clean independent trials: 95% upper bound {zero_success_bound(50):.4f}",
        f"Agent orders: 2000; constructed account orders: {len(accounts.orders_of('customer-7-0012'))}",
    )
    outputs["grid-01"] = text(
        "Constructed one-reply grid, fresh tools per reply.",
        f"Ordinary replies: {len(grid)}; forbidden effects: {sum(r.forbidden for r in grid)}",
        *[
            f"{reason}: {count}"
            for reason, count in sorted(Counter(r.reason for r in grid).items())
        ],
        f"Payments: {sum(bool(r.paid) for r in grid)}; every payment: 2499 cents on o-0007",
        f"Tier 3 paths: {', '.join(tier3)}",
        f"Tier 3 replies: {sum(len(rows) for rows in tier3.values())}; forbidden effects: {sum(r.forbidden for rows in tier3.values() for r in rows)}",
        "Tier 3 payments: 0; every terminal path opens named-approver review.",
        "Refusal order: approval_required; unexpected_tool; order_mismatch.",
    )
    normal_counts = Counter(d.outcome.value for d, _ in public)
    revoked_counts = Counter(d.outcome.value for d, _ in revoked)
    outputs["positive-01"] = text(
        "Constructed event-time runs; credentials reissued at every resumed run.",
        f"Must-pay: {sum(d.executed for d, _ in positives)}/5; cents: {sum(c for _, c in positives)}",
        *[f"{d.request_id}: {d.reason}; paid {c} cents" for d, c in positives],
        f"Positive decision changes: {sum(a != b for a, b in zip(positives, bare_positives, strict=True))}",
        f"Public decision changes: {sum(a != b for a, b in zip(public, bare_public, strict=True))}",
        f"Public 40: approve {normal_counts['approve']}; deny {normal_counts['deny']}; refer {normal_counts['refer']}",
        "r-7-0000: one authorized match; eligible; paid 2499 cents.",
        "Reviewer approvals authenticated; positive-100 waits for final evidence.",
    )
    outputs["revoke-01"] = text(
        "Constructed payment-scope revocation; reads continue.",
        f"Before: approve {normal_counts['approve']}; deny {normal_counts['deny']}; refer {normal_counts['refer']}",
        f"After: approve {revoked_counts['approve']}; deny {revoked_counts['deny']}; refer {revoked_counts['refer']}",
        f"Payments referred as credential_revoked: {sum(d.reason == 'credential_revoked' for d, _ in revoked)}",
        f"Paid cents: {sum(c for _, c in revoked)}",
        "r-7-0008 is checked on its timed rerun with a fresh credential.",
    )
    table = [
        "Constructed attacks; effect oracle reads actual stores and returned decisions.",
        "attack | entry | stopping check (with scope)",
    ]
    checks = {
        "order_not_found": "scope resolver",
        "suspicious_request": "risk rule",
        "credential_amount": "credential amount",
        "credential_missing": "credential missing",
        "eligible": "no secret in model context",
        "credential_request": "credential request",
        "credential_expired": "credential validity times",
        "credential_revoked": "credential revocation",
    }
    for bare, safe in zip(without, scoped, strict=True):
        reason = safe.intake_reason or safe.reason
        table.append(f"{safe.name} | {safe.channel} | {checks[reason]}")
        table.append(
            f"  without: {bare.reason}, {bare.paid_cents}; with: {reason}, {safe.paid_cents}"
        )
    refused, code = carrier_status(generate(7, orders=40, requests=40))
    table.extend(
        (
            "carrier | carrier status | ScanEvent and shipment contract",
            f"  ScanEvent refused {refused}; contract: {code}",
            f"Forbidden attack runs: without {sum(r.forbidden for r in without)}; with {sum(r.forbidden for r in scoped)}",
            f"Scope-removed local suite: {'fail' if any(r.forbidden for r in without) else 'pass'}",
            f"ScopeSkipsOrder local suite: {'fail' if any(r.forbidden for r in mutation) else 'pass'}",
            "Core release slot: blocked; authenticated fixture context is absent.",
            "Slot handler: full effects, grids, scan and controls wired.",
            "Reviewer role alteration: credential_seal; stores no approval.",
        )
    )
    bare_reviewer = next(row for row in without if row.name == "reviewer")
    reviewer = next(row for row in scoped if row.name == "reviewer")
    secret = next(row for row in scoped if row.name == "secret")
    table.extend(
        (
            f"Reviewer approvals stored: without {bare_reviewer.stored_approvals}; with {reviewer.stored_approvals}",
            f"Secret probe: {secret.reason}, paid {secret.paid_cents} cents; secret matches {secret.scan_matches}",
        )
    )
    outputs["attacks-01"] = text(*table)
    candidates = tuple(f"customer-7-{i:04}" for i in range(10000))
    identifier = "customer-7-0012"
    recovered = recover_plain(plain(identifier), candidates)
    missing = recover_plain(keyed(identifier, HASH_KEY), candidates)
    position = candidates.index(recovered) + 1 if recovered is not None else None
    keyed_count = len(candidates) if missing is None else candidates.index(missing) + 1
    keyed_result = "not recovered" if missing is None else f"recovered {missing}"
    export_input = {
        "requester_id": identifier,
        "customer_id": identifier,
        "request_id": "r-7-0012",
        "order_id": "o-0012",
        "trace_id": "trace-r-7-0012",
    }
    exported = export_records([export_input], HASH_KEY)
    visible.extend(exported)
    outputs["hash-01"] = text(
        "Constructed enumeration, not anonymous data.",
        f"Identifiers enumerated: {len(candidates)}; plain recovery position: {position}",
        f"Plain SHA-256 recovered: {recovered}",
        f"Keyed hash: {keyed_result} after enumerating {keyed_count} identifiers.",
        f"Candidates: {len(candidates)}; maximum hashes: {len(candidates)}; uniform average: {(len(candidates) + 1) / 2}",
        "Five join fields keyed; shared suffixes broken; key holders can relink.",
    )
    exercise = runpy.run_path(
        str(
            Path(__file__).resolve().parents[1]
            / "exercises/security-privacy/security_privacy_day_05.py"
        )
    )
    ((fault_request, intake_fault),) = exercise["diagnose"](exercise["fixture"]()).items()
    outputs["keys-01"] = text(
        "Given exercise inputs; independent attempts have a fixed chance.",
        "Exercise 2: order_not_found, credential_expired, credential_amount,",
        "  credential_request, credential_reviewer",
        f"Exercise 3: {chance(0.02, 50):.3f}; {chance(0.02, 200):.3f}; {attempts_for(0.02, 0.5)}; {attempts_for(0.02, 0.95)}",
        f"Exercise 4: maximum {10**8}; average {(10**8 + 1) / 2:.1f}",
        "Key space: 2^256 (number of keys, not every attack's cost)",
        str(2**256),
        "Without key: candidate enumeration cannot verify guesses.",
        "With key: enumeration works; pseudonymized, not anonymous.",
        f"Exercise 5: {fault_request}; {intake_fault}.",
        "Disclosure: already_delivered on another account's order.",
        "Fix issuance, revoke all intake credentials that day, list disclosed decisions.",
        "Exercise 6 reference: expires; only an owner may delegate.",
    )
    visible.extend(outputs.values())
    result = scan(visible, tuple(secrets))
    outputs["scan-01"] = text(
        "Every recorded demo trace, audit, decision, credential log and prompt.",
        "Includes ordinary/tier 3 grids, controls, positives, public and revoke runs.",
        f"Observed objects searched: {result.outputs + 1}",
        f"Raw, hexadecimal and base64 secret matches: {len(result.matches)}",
        "Issuer and export keys are distinct public fixture labels.",
    )
    if scan((outputs["scan-01"],), tuple(secrets)).matches:
        raise ValueError("secret in scanner's own output")
    return outputs


def plot_data():
    return {
        "constructed": True,
        "independent": True,
        "p_pass": 0.05,
        "attempts": list(range(1, 101)),
        "chance": [chance(0.05, n) for n in range(1, 101)],
    }


if __name__ == "__main__":
    captured = create_outputs()

# region book:fnd.security-privacy.attack-01 output
if __name__ == "__main__":
    print(captured["attack-01"], end="")
# endregion book:fnd.security-privacy.attack-01

# region book:fnd.security-privacy.filter-01 output
if __name__ == "__main__":
    print(captured["filter-01"], end="")
# endregion book:fnd.security-privacy.filter-01

# region book:fnd.security-privacy.grid-01 output
if __name__ == "__main__":
    print(captured["grid-01"], end="")
# endregion book:fnd.security-privacy.grid-01

# region book:fnd.security-privacy.revoke-01 output
if __name__ == "__main__":
    print(captured["revoke-01"], end="")
# endregion book:fnd.security-privacy.revoke-01

# region book:fnd.security-privacy.hash-01 output
if __name__ == "__main__":
    print(captured["hash-01"], end="")
# endregion book:fnd.security-privacy.hash-01

# region book:fnd.security-privacy.scan-01 output
if __name__ == "__main__":
    print(captured["scan-01"], end="")
# endregion book:fnd.security-privacy.scan-01

# region book:fnd.security-privacy.attacks-01 output
if __name__ == "__main__":
    print(captured["attacks-01"], end="")
# endregion book:fnd.security-privacy.attacks-01

# region book:fnd.security-privacy.positive-01 output
if __name__ == "__main__":
    print(captured["positive-01"], end="")
# endregion book:fnd.security-privacy.positive-01

# region book:fnd.security-privacy.keys-01 output
if __name__ == "__main__":
    print(captured["keys-01"], end="")
# endregion book:fnd.security-privacy.keys-01
