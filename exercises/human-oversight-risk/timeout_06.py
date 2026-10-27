"""Exercise 6 starter: intentionally unsafe; never use as a real handler."""

from refund_agent.oversight.human_oversight_risk_lifecycle import ReviewEvent


def on_deadline(ticket, at):
    if ticket.closed_at is not None or at < ticket.due_at:
        return ticket
    approval = ReviewEvent(kind="approval", owner=ticket.assignee, at=at,
                           reason="keep_queue_moving")
    return ticket.model_copy(update={
        "outcome": "approved", "closed_at": at, "events": ticket.events + (approval,),
    })
