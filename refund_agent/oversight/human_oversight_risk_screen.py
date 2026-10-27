"""A required, numbered view of the proposal, explanation and reviewer identity."""

from refund_agent import contracts as c
from refund_agent.oversight.human_oversight_risk_desk import ProposedExplanation, ReviewerContext
from refund_agent.rules import tier_for


def screen(ticket, proposal, proposed, explanation, credential):
    explanation = c._validated(ProposedExplanation, explanation)
    credential = c._validated(ReviewerContext, credential)
    if (proposed.outcome != c.Outcome.DENY or not explanation.proposed
            or proposed.request_id != proposal.request_id or proposed.amount_cents != proposal.amount_cents
            or proposed.tier != tier_for(proposal.amount_cents)
            or explanation.request_id != proposal.request_id or not explanation.rendered.strip()
            or explanation.policy_version != proposal.evidence.policy.version
            or ticket.request_id != proposal.request_id):
        raise ValueError("screen needs this denial's replayed proposed explanation")
    if (credential.reviewer not in ticket.path[:-1]
            or (ticket.tier == c.Tier.NAMED_APPROVER and credential.role != "named_approver")):
        raise ValueError("screen needs the reviewing person's own credential")
    evidence = proposal.evidence
    return (
        f"1 Request {proposal.request_id}; order {evidence.order.order_id}; ${proposal.amount_cents / 100:.2f}",
        f"2 Proposed denial: {proposed.reason}; clause {proposed.policy_citation}",
        f"3 Policy {evidence.policy.version}; evidence ages (seconds): {evidence.freshness}",
        f"4 Referral rule: every tier {int(ticket.tier)} decision requires its person",
        "5 Explanation (proposed): " + explanation.rendered,
        "6 Approving confirms the denial; no money moves. A later eligible payment needs the gate.",
        "7 A payment cannot be recalled; waiting risks eligibility expiry.",
        f"8 Due {ticket.due_at.isoformat()}; assigned {ticket.assignee}",
        "9 Actions: approve, refuse, modify, escalate; modification opens a new review",
        f"10 Reviewer {credential.reviewer}; credential {credential.credential_id}; role {credential.role}",
    )
