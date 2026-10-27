"""A constructed tier-1 mix, with truth sealed separately from every runtime input."""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from refund_agent import contracts as c
from refund_agent.contracts import _detach
from refund_agent.data.generate import Dataset, generate, number
from refund_agent.selection import fingerprint
from refund_agent.split import GoldenSet
from refund_agent.split import fingerprint as label_fingerprint

SEED = 7
AT = datetime(2026, 6, 1, 12, tzinfo=timezone.utc)
GROUP_COUNTS = {"contradiction": 24, "false_date": 16, "phrase_trip": 8, "plain": 352}
PHRASE_TRIP = "The first lamp never arrived, so I ordered again; this one came cracked."


class TruthRow(c.Record):
    request_id: str
    true_delivery_at: datetime | None
    owed: bool
    label: c.Decision


class TruthTable:
    """Frozen truth; every consultation shares GoldenSet's append-only journal semantics."""

    def __init__(self, requests, rows):
        self.__rows = tuple(_detach(rows))
        labels = [row.label for row in rows]
        manifest = {
            "name": "multi-agent-constructed-truth-v1",
            "cases": len(rows),
            "sha256": label_fingerprint(requests, labels),
            "truth_sha256": fingerprint([row.model_dump(mode="json") for row in rows]),
        }
        self.__sealed = GoldenSet(manifest, tuple(_detach(requests)), tuple(_detach(labels)))

    @property
    def manifest(self):
        return _detach(self.__sealed.manifest)

    @property
    def log(self):
        return tuple(_detach(self.__sealed.log))

    def intact(self):
        return self.__sealed.intact() and self.manifest["truth_sha256"] == fingerprint(
            [row.model_dump(mode="json") for row in self.__rows]
        )

    def persist(self, directory):
        if not self.intact():
            raise ValueError("truth table fingerprint changed")
        self.__sealed.persist(directory)

    def consult(self, reader, purpose):
        if not self.intact() or not reader or not purpose:
            raise ValueError("intact truth and named read required")
        self.__sealed.consult(reader, purpose)
        return tuple(_detach(self.__rows))


@dataclass(frozen=True)
class Fixture:
    data: Dataset
    truth: TruthTable
    groups: dict
    manifest: dict

    def intact(self):
        payload = {key: value for key, value in self.manifest.items() if key != "sha256"}
        return (
            self.truth.intact()
            and fingerprint(payload) == self.manifest["sha256"]
            and fingerprint(self.data.model_dump(mode="json")) == self.manifest["inputs_sha256"]
        )


def _order(at, oid, purchased_at):
    return c.Order(
        order_id=oid,
        reference=oid,
        purchased_at=purchased_at,
        observed_at=at - timedelta(seconds=60),
        items=[c.Item(name="physical desk lamp", quantity=1, unit_cents=2499)],
        paid_cents=2499,
        payment_id=f"payment-{oid}",
    )


def _truth(rid, group, true_delivery, policy):
    outcome = (
        c.Outcome.REFER
        if group == "contradiction"
        else (c.Outcome.DENY if group == "false_date" else c.Outcome.APPROVE)
    )
    label = c.Decision(
        request_id=rid,
        outcome=outcome,
        amount_cents=2499,
        tier=c.Tier.AGENT,
        reason={
            "contradiction": "words_contradict_record",
            "false_date": "outside_return_window",
        }.get(group, "eligible"),
        policy_citation=f"refund_cap_v4@{policy.version}#eligibility",
    )
    return TruthRow(
        request_id=rid,
        true_delivery_at=None if group == "contradiction" else true_delivery,
        owed=group not in ("contradiction", "false_date"),
        label=label,
    )


def _case(seed, index, group, policy):
    at = AT + timedelta(seconds=number(seed, index, "second", 3600))
    delivery = at - timedelta(days=number(seed, index, "age", 10) + 1)
    true_delivery = delivery - timedelta(days=60) if group == "false_date" else delivery
    # A separate seed field, common to all groups, predates even the oldest delivery.
    purchased_at = AT - timedelta(days=71 + number(seed, index, "purchase_age", 30))
    oid, rid = f"verifier-o-{seed}-{index:04d}", f"verifier-r-{seed}-{index:04d}"
    text = {
        "contradiction": "The lamp never arrived. Please review this refund request.",
        "phrase_trip": PHRASE_TRIP,
    }.get(group, "This lamp came cracked. Please review this refund request.")
    order = _order(at, oid, purchased_at)
    request = c.Request(
        request_id=rid,
        requester_id=f"customer-{rid}",
        order_reference=oid,
        at=at,
        amount_cents=2499,
        reason="damaged",
        text=text,
    )
    shipment = c.Shipment(
        order_id=oid,
        observed_at=at - timedelta(seconds=120),
        events=[c.ShipmentEvent(at=delivery, status="delivered")],
    )
    snapshot = c.PolicySnapshot(
        request_id=rid, version=policy.version, observed_at=at - timedelta(hours=1)
    )
    return order, request, shipment, snapshot, _truth(rid, group, true_delivery, policy)


def build_fixture(seed=SEED):
    base = generate(seed, orders=5, requests=5)
    policy = base.policy_at(AT)
    indices = sorted(range(400), key=lambda i: number(seed, i, "group", 2**128))
    groups, assignments, start = {}, {}, 0
    for name, count in GROUP_COUNTS.items():
        assignments.update((index, name) for index in indices[start : start + count])
        start += count
        groups[name] = []
    rows = [_case(seed, i, assignments[i], policy) for i in range(400)]
    for i, row in enumerate(rows):
        groups[assignments[i]].append(row[1].request_id)
    data = Dataset(
        orders=[r[0] for r in rows],
        requests=[r[1] for r in rows],
        shipments=[r[2] for r in rows],
        retrievals=[r[3] for r in rows],
        policies=base.policies,
        reviewers=base.reviewers,
        correct=[],
    )
    truth = TruthTable(data.requests, [r[4] for r in rows])
    manifest = {
        "name": "multi-agent-constructed-fixture-v1",
        "seed": seed,
        "cases": 400,
        "groups": dict(GROUP_COUNTS),
        "inputs_sha256": fingerprint(data.model_dump(mode="json")),
        "truth": truth.manifest,
    }
    manifest["sha256"] = fingerprint(manifest)
    return Fixture(data, truth, {k: tuple(v) for k, v in groups.items()}, manifest)


def proof_of_delivery(seed=SEED):
    """Exercise 6's explicitly supplied second source, built without consulting sealed truth."""
    fixture = build_fixture(seed)
    false = set(fixture.groups["false_date"])
    rows = []
    for request, shipment in zip(fixture.data.requests, fixture.data.shipments):
        delivered = shipment.events[0].at
        rows.append(
            {
                "request_id": request.request_id,
                "delivered_at": (
                    delivered - timedelta(days=60) if request.request_id in false else delivered
                ).isoformat(),
            }
        )
    return {"constructed": True, "source": "proof-of-delivery-v1", "seed": seed, "rows": rows}
