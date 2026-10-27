"""A clock can only escalate. This module has no payment or approval interface."""

from refund_agent import contracts as c
from refund_agent.oversight.human_oversight_risk_lifecycle import (
    Notice,
    ReviewEvent,
    ReviewTicket,
    add_desk_hours,
)


def deadline_counts(ticket, tools):
    approvals = sum(event.kind == "approval" for event in ticket.events)
    denials = sum(event.kind in ("refusal", "denial_confirmation") for event in ticket.events)
    receipts = sum(receipt.request is not None and
                   receipt.request.request_id == ticket.request_id
                   for receipt in tools.ledger.values())
    return approvals, denials, receipts


def print_timeout():
    from datetime import datetime, timezone

    from refund_agent.data.generate import Dataset
    from refund_agent.tools import Tools

    tools = Tools(Dataset(orders=[], requests=[], shipments=[],
                          policies=[], retrievals=[], correct=[]))
    ticket = ReviewTicket(
        review_id="human-oversight-risk-timeout", request_id="human-oversight-risk-120",
        requester_id="customer-120", work_class="tier2", tier=c.Tier.REVIEWER,
        opened_at=datetime(2026, 10, 2, 18, 20, tzinfo=timezone.utc),
        due_at=datetime(2026, 10, 5, 17, tzinfo=timezone.utc), assignee="alex",
        path=("alex", "morgan", "decision-owner"), queue_length=1,
    )
    print("Constructed $120.00 tier 2 request; reviewer alex away Monday.")
    print("Arrival:", ticket.opened_at.isoformat())
    print("Due:", ticket.due_at.isoformat())
    ticket = on_deadline(ticket, ticket.due_at)
    print("Escalated to:", ticket.assignee, "at", ticket.events[-1].at.isoformat())
    print("New due:", ticket.due_at.isoformat())
    print("Notice to:", ticket.notices[-1].requester_id)
    print("Deadline events:", ", ".join(e.kind for e in ticket.events))
    approvals, denials, receipts = deadline_counts(ticket, tools)
    status = "open" if ticket.closed_at is None else "closed"
    print(f"Deadline approvals={approvals} denials={denials} receipts={receipts}; item remains {status}.")
    print("Tuesday 10:15 approval, credential and fresh-gate run:")
    print("  integration pending: chapter 35's authenticated review intake.")
    print("Constructed execution-refusal/expiry/deputy records: review-records-01.")
    print("Required partner method: ScopedTools.record_approval with credential.")
    print("The authorized-review gate, stale-read and expiry tests use")
    print("test-constructed partner records; no substitute issuer is shipped.")


# region book:fnd.human-oversight-risk.timeout-01 output
def on_deadline(ticket: ReviewTicket, at) -> ReviewTicket:
    at = c._utc_time(at)
    if (
        ticket.closed_at is not None or not ticket.blocking
        or at < ticket.due_at
    ):
        return ticket
    if ticket.step == len(ticket.path) - 1:
        return ticket  # The decision owner must arrange authorized capacity.
    step = ticket.step + 1
    owner = ticket.path[step]
    due = add_desk_hours(at, 4)
    event = ReviewEvent(
        kind="escalation", owner=owner, at=at,
        due_at=due, reason="review_deadline",
    )
    notice = Notice(
        requester_id=ticket.requester_id, owner=owner, at=at,
        due_at=due, reason="review_deadline",
    )
    return ticket.model_copy(update={
        "step": step, "assignee": owner, "due_at": due,
        "outcome": "escalated", "events": ticket.events + (event,),
        "notices": ticket.notices + (notice,),
    })


if __name__ == "__main__":
    print_timeout()
# endregion book:fnd.human-oversight-risk.timeout-01
