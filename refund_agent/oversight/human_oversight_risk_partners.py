"""Chapter-owned readers of the declared observability and appeals interfaces."""

from refund_agent import contracts as c
from refund_agent.oversight.human_oversight_risk_simulate import QueueArrival


def observability_referrals(spans, requests, reason_classes):
    """Root-span refusals enter the desk once; in-flight holds do not.

    The caller supplies chapter 34's REASON_CLASS. No local reason fallback is
    permitted: an unknown code refuses the join. Parent/tool spans cannot create
    duplicate human work. The human clock starts when the referral is returned.
    """
    by_trace = {"trace-" + r.request_id: r for r in requests}
    result = {}
    for span in sorted(spans, key=lambda s: (s.end, s.run_id)):
        if span.name != "run" or span.status not in ("held", "refused"):
            continue
        if span.reason not in reason_classes:
            raise ValueError("unclassified observability reason")
        if reason_classes[span.reason] == "hold":
            continue
        request = by_trace[span.trace_id]
        if request.requester_id != span.requester_id:
            raise ValueError("span requester does not match the referred request")
        result.setdefault(request.request_id, QueueArrival(
            "human-oversight-risk-" + request.request_id, "referral", c._utc_time(span.end)))
    return tuple(result.values())


def appeal_route(*, request_id, tier, first_reviewer, reviewers, policy,
                 appeals_approver=None, independent_assessment=None):
    if tier == c.Tier.NAMED_APPROVER:
        if (not appeals_approver or appeals_approver == first_reviewer
                or not independent_assessment or independent_assessment["request_id"] != request_id
                or independent_assessment["reviewer"] in (first_reviewer, appeals_approver)
                or not independent_assessment["text"].strip()):
            raise ValueError("tier3 appeal needs a preappointed approver and independent written assessment")
        person = next((r for r in reviewers if r.reviewer_id == appeals_approver), None)
        if person is None or person.role != "named_approver" or policy.named_approver != appeals_approver:
            raise ValueError("appeals appointment needs the corresponding policy version and new proposal")
        return appeals_approver
    people = [r.reviewer_id for r in reviewers if r.reviewer_id != first_reviewer]
    if not people:
        raise ValueError("an independent reviewer is required")
    return sorted(people)[0]
