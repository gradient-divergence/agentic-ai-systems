"""Append-only announced contracts and the versions the candidate pins."""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from refund_agent.contracts import Record
from refund_agent.memory.memory import NOTICE
from refund_agent.sources.data_systems_budget import budget_record
from refund_agent.sources.data_systems_contract import (
    FieldMeaning,
    NoticeOfChange,
    SourceContract,
)

AT = datetime(2026, 10, 12, tzinfo=timezone.utc)
EFFECTIVE = datetime(2026, 1, 1, tzinfo=timezone.utc)
VERSION = "data-systems.registry@1"


class ContractVersion(Record):
    contract: SourceContract
    announced_at: datetime
    effective_at: datetime


class Registry:
    def __init__(
        self, entries: tuple[ContractVersion, ...], pins: dict[str, str], version: str = VERSION
    ):
        self._entries: tuple[ContractVersion, ...] = ()
        self.version = version
        for entry in entries:
            self.announce(entry)
        self._pins = dict(pins)
        for source in self._pins:
            self.contract(source)

    @property
    def entries(self):
        return tuple(e.model_copy(deep=True) for e in self._entries)

    @property
    def pins(self) -> dict[str, str]:
        return dict(self._pins)

    def announce(self, entry: ContractVersion) -> None:
        entry = ContractVersion.model_validate(entry.model_dump())
        if any(e.contract.version == entry.contract.version for e in self._entries):
            raise ValueError("contract version already recorded")
        notice = timedelta(days=entry.contract.notice.days)
        if entry.announced_at + notice > entry.effective_at:
            raise ValueError("breaking contract announced too late")
        self._entries += (entry,)

    def contract(self, source: str) -> SourceContract:
        version = self._pins[source]
        return next(
            e.contract.model_copy(deep=True)
            for e in self._entries
            if e.contract.source == source and e.contract.version == version
        )

    def record(self) -> dict:
        return {
            "version": self.version,
            "pins": self.pins,
            "entries": [e.model_dump(mode="json") for e in self._entries],
        }


def default_registry() -> Registry:
    owners = Path(__file__).resolve().parents[1] / "release/owners.json"
    roles = json.loads(owners.read_text())["sources"]["roles"]
    budgets = {
        "order": budget_record(
            1800,
            "constructed quiet log: 0 changes in 6000 order-hours; spec limit retained",
            roles["order"],
            AT,
            n=0,
            hours=6000,
        ),
        "shipment": budget_record(
            1800,
            "constructed: 48 first-delivery scans in 2400 transit parcel-hours",
            roles["shipment"],
            AT,
            n=48,
            hours=2400,
        ),
        "policy": budget_record(
            2592000,
            "specification judgment, no measured rate; effective dates use the register",
            roles["policy"],
            AT,
        ),
        "ledger": budget_record(
            0, "execution-time authoritative read; no cached reuse", roles["ledger"], AT
        ),
    }
    meanings = {
        "order": (("amounts", "USD cents", "one currency; prior refunds copy the ledger"),),
        "shipment": (
            ("status", "code", "shipped or delivered to the customer"),
            ("at", "UTC instant", "when the carrier made the scan"),
            ("observed_at", "UTC instant", "when the source observed the record"),
            ("events", "scan history", "complete: every scan the carrier made has been sent"),
        ),
        "policy": (("version", "date", "effective date of the policy content"),),
        "ledger": (("amount_cents", "USD cents", "money already paid"),),
    }
    invariants = {
        "order": ("paid_equals_items", "prior_within_paid"),
        "shipment": ("time_order", "delivery_after_shipping", "not_after_observation"),
        "policy": ("version_equals_effective_date",),
        "ledger": ("positive_integer_cents", "unique_idempotency_key"),
    }
    entries = []
    for source, role in roles.items():
        contract = SourceContract(
            source=source,
            version=f"{source}@1",
            owner_role=role,
            fields=tuple(FieldMeaning(field=f, unit=u, meaning=m) for f, u, m in meanings[source]),
            vocabularies={"status": ("shipped", "delivered")} if source == "shipment" else {},
            invariants=invariants[source],
            budget=budgets[source],
            notice=NoticeOfChange(
                days=NOTICE.days if source == "policy" else 7,
                basis="policy register"
                if source == "policy"
                else "constructed seven-day source-owner agreement",
            ),
        )
        entries.append(
            ContractVersion(
                contract=contract,
                announced_at=EFFECTIVE - timedelta(days=7),
                effective_at=EFFECTIVE,
            )
        )
    return Registry(tuple(entries), {s: f"{s}@1" for s in roles})
