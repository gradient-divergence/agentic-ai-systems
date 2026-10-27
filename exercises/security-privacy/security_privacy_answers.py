"""Reference solutions computed from the given inputs, not imported by starters."""

from refund_agent.security.security_privacy_math import attempts_for, chance

ANSWERS = {
    1: {
        "request": "customer",
        "order": "customer",
        "shipment": "carrier",
        "policy": "policy owner",
        "approval": "authenticated reviewer",
    },
    2: (
        "order_not_found",
        "credential_expired",
        "credential_amount",
        "credential_request",
        "credential_reviewer",
    ),
    3: (
        round(chance(0.02, 50), 3),
        round(chance(0.02, 200), 3),
        attempts_for(0.02, 0.5),
        attempts_for(0.02, 0.95),
    ),
    4: (10**8, (10**8 + 1) / 2, 2**256, "key required for enumeration", "pseudonymized"),
    5: {
        "request": "security-privacy-day-05",
        "cause": "corrupted trusted grant input",
        "disclosure": "already_delivered on another account's order",
        "reply": "Fix issuance; revoke all intake credentials that day; list disclosed decisions.",
    },
}
