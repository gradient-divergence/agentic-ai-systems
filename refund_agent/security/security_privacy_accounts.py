"""Constructed account records, frozen before any attack data is built."""

from dataclasses import dataclass

from refund_agent.data.generate import Dataset


@dataclass(frozen=True)
class AccountRecords:
    grants: tuple[tuple[str, str], ...]

    @classmethod
    def from_dataset(cls, data: Dataset) -> "AccountRecords":
        references: dict[str, set[str]] = {}
        for order in data.orders:
            for reference in (order.reference, order.order_id):
                references.setdefault(reference, set()).add(order.order_id)
        grants: set[tuple[str, str]] = set()
        for request in data.requests:
            ids = references.get(request.order_reference, set())
            if len(ids) == 1:
                grants.add((request.requester_id, next(iter(ids))))
        assigned = {order_id for _, order_id in grants}
        for order in data.orders:
            if order.order_id not in assigned:
                prefix = "instructor-" if order.order_id.startswith("instructor-") else ""
                number = order.order_id.removeprefix(prefix).removeprefix("o-")
                grants.add((f"{prefix}customer-7-{number}", order.order_id))
        return cls(tuple(sorted(grants)))

    def may_act(self, subject: str, order_id: str) -> bool:
        return (subject, order_id) in self.grants

    def orders_of(self, subject: str) -> tuple[str, ...]:
        return tuple(order for owner, order in self.grants if owner == subject)


@dataclass(frozen=True)
class ScopeResolver:
    """Trusted reference-to-ID metadata; never fetches an order's payload."""

    references: tuple[tuple[str, tuple[str, ...]], ...]

    @classmethod
    def from_dataset(cls, data: Dataset) -> "ScopeResolver":
        index: dict[str, set[str]] = {}
        for order in data.orders:
            for reference in (order.order_id, order.reference):
                index.setdefault(reference, set()).add(order.order_id)
        return cls(tuple((ref, tuple(sorted(ids))) for ref, ids in sorted(index.items())))

    def resolve(self, reference: str, allowed: tuple[str, ...]) -> tuple[str | None, str | None]:
        ids = tuple(o for o in dict(self.references).get(reference, ()) if o in allowed)
        if len(ids) > 1:
            return None, "ambiguous_order"
        if not ids:
            return None, "order_not_found"
        return ids[0], None
