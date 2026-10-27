"""Plain hashes are enumerable; a keyed mapping breaks every join identifier."""

import hmac
from hashlib import sha256


def plain(identifier: str) -> str:
    return sha256(identifier.encode()).hexdigest()


def keyed(identifier: str, key: bytes, *, field: str = "identifier") -> str:
    if type(key) is not bytes or len(key) < 32:
        raise ValueError("pseudonymization key needs at least 32 bytes")
    return hmac.new(key, (field + "\0" + identifier).encode(), sha256).hexdigest()


def recover_plain(target: str, candidates) -> str | None:
    return next((candidate for candidate in candidates if plain(candidate) == target), None)


IDENTIFYING_FIELDS = frozenset(
    {
        "requester_id",
        "customer_id",
        "request_id",
        "order_id",
        "order_reference",
        "trace_id",
        "run_id",
        "span_id",
        "parent_id",
        "credential_id",
        "subject",
        "preparer_id",
        "payment_id",
        "receipt_id",
        "proposal_id",
        "review_id",
        "approval_id",
        "audit_id",
        "event_id",
    }
)
IDENTIFYING_LISTS = frozenset({"order_ids", "approval_ids", "reads", "writes"})


def export_records(records, key: bytes):
    """Export a closed schema of telemetry, not arbitrary free-text payloads.

    Raw prompts, item names, request text and diagnostic payloads are refused.
    Keyed namespaces remove the generator's shared suffixes, preserving only
    same-field joins. Retention and role/purpose checks belong to deployment.
    """
    permitted = (
        IDENTIFYING_FIELDS
        | IDENTIFYING_LISTS
        | frozenset(
            {
                "kind",
                "tool",
                "outcome",
                "reason",
                "at",
                "start",
                "end",
                "status",
                "name",
                "attempt",
                "amount_cents",
                "paid_amount_cents",
                "money_moved",
                "executed",
                "tier",
                "policy_citation",
                "policy_version",
                "scopes_version",
                "event",
                "freshness",
                "risk_score",
                "decision",
                "sources",
                "tokens",
                "links",
            }
        )
    )

    identifiers: set[str] = set()

    def collect(value, field=""):
        if field == "links":
            for span, _relation in value:
                identifiers.add(span)
            return
        if isinstance(value, dict):
            for name, item in value.items():
                collect(item, name)
        elif isinstance(value, (tuple, list)):
            for item in value:
                collect(item, field)
        elif value is not None and field in IDENTIFYING_FIELDS | IDENTIFYING_LISTS:
            identifiers.add(str(value))

    for record in records:
        collect(record)

    def transform(value, field=""):
        if field == "links":
            if any(len(pair) != 2 or pair[1] not in ("released_by", "corrects") for pair in value):
                raise ValueError("unapproved span link")
            return [[keyed(span, key, field="span_id"), relation] for span, relation in value]
        if isinstance(value, dict):
            if field in ("freshness", "sources", "tokens"):
                if set(value) - {"order", "shipment", "policy", "ledger"}:
                    raise ValueError("unapproved source or token telemetry")
            elif set(value) - permitted:
                raise ValueError("export contains unapproved telemetry fields")
            return {name: transform(item, name) for name, item in value.items()}
        if isinstance(value, (tuple, list)):
            namespace = (
                "order_id"
                if field == "order_ids"
                else "approval_id"
                if field == "approval_ids"
                else field
            )
            return [transform(item, namespace) for item in value]
        if value is None:
            return None
        if field in IDENTIFYING_FIELDS or field in IDENTIFYING_LISTS:
            return keyed(str(value), key, field=field)
        if type(value) not in (str, bool, int, float):
            raise ValueError("export values must be JSON scalars")
        if type(value) is str:
            for identifier in sorted(identifiers, key=lambda text: (-len(text), text)):
                if identifier:
                    value = value.replace(identifier, keyed(identifier, key, field="embedded"))
        return value

    return [transform(record) for record in records]
