"""Questions from request-time facts; a label supplies only the relevant version."""

from dataclasses import dataclass
from datetime import datetime

from refund_agent.contracts import Request


@dataclass(frozen=True)
class QuestionRequest:
    request: Request
    days_since_delivery: float | None


@dataclass(frozen=True)
class Question:
    request_id: str
    section: str
    text: str
    as_of: datetime
    relevant: str


def question_requests(data, requests) -> list[QuestionRequest]:
    rows = []
    shipments = {s.order_id: s for s in data.shipments}
    for r in requests:
        orders = [o for o in data.orders if r.order_reference in (o.order_id, o.reference)]
        shipment = shipments.get(orders[0].order_id) if len(orders) == 1 else None
        delivered = (
            [e.at for e in shipment.events if e.status == "delivered" and e.at <= r.at]
            if shipment
            else []
        )
        days = (r.at - max(delivered)).total_seconds() / 86400 if delivered else None
        rows.append(QuestionRequest(r, days))
    return rows


def policy_questions(requests, labels, wording="policy") -> list[Question]:
    if wording not in ("policy", "customer"):
        raise ValueError("wording must be policy or customer")
    by_id = {label.request_id: label for label in labels}
    questions = []
    for row in requests:
        r, days = row.request, row.days_since_delivery
        citation = by_id[r.request_id].policy_citation
        if not citation:
            raise ValueError("question needs a dated relevance label")
        version = citation.split("#")[0]
        age = f"{days:g}" if days is not None else "unknown"
        if wording == "policy":
            eligibility = (
                "not_delivered if no delivery"
                if r.reason == "not_delivered"
                else f"{r.reason} {age} days after delivery"
            )
            cap = f"at most {r.amount_cents} cents and the unrefunded payment balance"
        else:
            reason = {
                "damaged": "it arrived broken",
                "changed_mind": "I no longer want it",
                "not_delivered": "my parcel has not arrived",
            }[r.reason]
            eligibility = f"{reason}; {age} days later can I get my money back"
            cap = f"can you pay back {r.amount_cents} from what I paid"
        for section, text in (("eligibility", eligibility), ("cap", cap)):
            questions.append(Question(r.request_id, section, text, r.at, f"{version}#{section}"))
    return questions
