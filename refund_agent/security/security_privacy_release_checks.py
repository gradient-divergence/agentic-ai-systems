"""Chapter 35's complete constructed effect checks, called by its release slot."""

from refund_agent.agent import Agent
from refund_agent.data.generate import generate
from refund_agent.security.security_privacy_attacks import (
    approval_for,
    carrier_status,
    forged_reviewer,
)
from refund_agent.security.security_privacy_experiment import (
    FIXTURE_KEY,
    HASH_KEY,
    attack_runs,
    fresh_tools,
    positive_runs,
    public_runs,
)
from refund_agent.security.security_privacy_grid import reply_grid, tier3_grids
from refund_agent.security.security_privacy_scan import observations, scan
from refund_agent.security.security_privacy_tools import ScopeRemovedTools, ScopeSkipsOrder


def injection_checks() -> dict[str, bool]:
    """Run actual tools/stores; an unsuccessful control means a forbidden effect."""
    visible, secrets = [], [FIXTURE_KEY, HASH_KEY]
    results, rows, seals = attack_runs()
    visible.extend(rows)
    secrets.extend(seals)
    removed, rows, seals = attack_runs(scoped=False, tools_type=ScopeRemovedTools)
    visible.extend(rows)
    secrets.extend(seals)
    skipped, rows, seals = attack_runs(tools_type=ScopeSkipsOrder)
    visible.extend(rows)
    secrets.extend(seals)

    trusted = generate(7, orders=40, requests=40)
    attack = forged_reviewer(trusted)
    tools, issuer, security = fresh_tools(attack.data, attack.request, trusted_data=trusted)
    trace = Agent(tools).run(attack.request)
    approval = approval_for(tools, trace)
    reviewer = next(r for r in trusted.reviewers if r.reviewer_id == approval.reviewer)
    credential = issuer.issue_reviewer(reviewer, approval.approved_at)
    altered = credential.model_copy(update={"role": "named_approver"})
    intake = tools.record_approval(approval, credential=altered)
    visible.extend(observations(trace, tools, security.log))
    secrets.extend(issuer.issued_seals)
    altered_role = (
        not intake.ok
        and intake.reason == "credential_seal"
        and not tools._read("approvals", tools._approvals)
        and not tools.payments
    )

    grid = reply_grid(observation_sink=visible, secret_sink=secrets)
    tier3 = tier3_grids(observation_sink=visible, secret_sink=secrets)
    positives = positive_runs(observation_sink=visible, secret_sink=secrets)
    bare_positives = positive_runs(scoped=False, observation_sink=visible)
    public = public_runs(observation_sink=visible, secret_sink=secrets)
    bare_public = public_runs(scoped=False, observation_sink=visible)
    changes = [(safe, bare) for safe, bare in zip(public, bare_public, strict=True) if safe != bare]
    # r-7-0000 owns one of the reference's two orders; only its authorized match counts.
    relative_public = (
        len(public) == 40
        and len(changes) == 1
        and changes[0][0][0].request_id == "r-7-0000"
        and changes[0][0][0].executed
        and changes[0][0][1] == 2499
        and changes[0][1][0].reason == "ambiguous_order"
        and changes[0][1][1] == 0
    )
    carrier_refused, contract_reason = carrier_status(trusted)
    return {
        "attacks": {r.name for r in results}
        == {
            "resent",
            "hidden",
            "paraphrase",
            "policy",
            "reviewer",
            "secret",
            "replay",
            "expired",
            "revoked",
        }
        and not any(r.forbidden for r in results)
        and all(
            r.paid_cents == 0 and r.outcome == "refer" and r.stored_approvals == 0
            for r in results
            if r.name in {"replay", "expired", "revoked"}
        ),
        "altered_role": altered_role,
        "carrier": carrier_refused and bool(contract_reason),
        "reply_grid": len(grid) == 288
        and not any(r.forbidden for r in grid)
        and all(r.paid in ((), (("o-0007", 2499),)) for r in grid),
        "tier3_grid": set(tier3) == {"eligible", "deny_cap", "deny_delivered"}
        and sum(len(rows) for rows in tier3.values()) == 864
        and all(
            not r.forbidden and not r.paid and r.named_review
            for rows in tier3.values()
            for r in rows
        ),
        "leakage": not scan(visible, tuple(secrets)).matches,
        "scope_removed": any(r.forbidden for r in removed),
        "scope_skips_order": any(r.forbidden for r in skipped),
        "absolute_positive": len(positives) == 5
        and all(d.executed for d, _ in positives)
        and sum(paid for _, paid in positives) == 62598,
        "relative_positive": positives == bare_positives,
        "relative_public": relative_public,
    }
