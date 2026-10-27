"""Six categorical tokens; the first five describe a case, not an agent action."""

from itertools import product

FIELDS = ("reason", "delivered", "days", "amount", "earlier_refund", "decision")
VALUES = dict(
    zip(
        FIELDS,
        (
            ("damaged", "changed_mind", "not_delivered"),
            (False, True),
            (None, "0-14", "15-21", "22-30", "31-45", "46+"),
            (2499, 4999, 5000, 50000, 50001, 65000, 120000),
            ("none", "one"),
            ("approve", "deny", "refer"),
        ),
    )
)
BANDS = (
    (0, 14, "0-14"),
    (15, 21, "15-21"),
    (22, 30, "22-30"),
    (31, 45, "31-45"),
    (46, None, "46+"),
)


def case_tokens(request, order, shipment, decision):
    deliveries = [e.at for e in shipment.events if e.status == "delivered"]
    band = None
    if deliveries:
        days = (request.at - max(deliveries)).total_seconds() / 86400
        if days < 0:
            raise ValueError("delivery is after the request")
        band = next(name for _, high, name in BANDS if high is None or days <= high)
    return (
        request.reason,
        bool(deliveries),
        band,
        request.amount_cents,
        "one" if order.prior_refunds else "none",
        decision.outcome.value,
    )


def valid_cases():
    """List all 252 cases, including reasons the policy will deny or refer."""
    return tuple(
        case
        for case in product(*(VALUES[field] for field in FIELDS[:5]))
        if case[1] == (case[2] is not None)
    )
