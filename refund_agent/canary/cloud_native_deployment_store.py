"""One append store, explicit namespaces, checked records and a hash chain.

The synchronous file ledger is a teaching stand-in, not a payment provider.
Readers validate their own contracts before projecting into the core contracts.
"""

import fcntl
import json
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from threading import RLock

from refund_agent import contracts as c
from refund_agent.release.journal import Journal
from refund_agent.selection import fingerprint
from refund_agent.tools import Tools

KEYS = {
    "ledger": "idempotency_key", "audits": "audit_id", "proposals": "proposal_id",
    "reviews": "review_id", "approvals": "approval_id",
}
MODELS = {
    "ledger": c.Receipt, "audits": c.AuditRecord, "proposals": c.Proposal,
    "reviews": c.ReviewItem, "approvals": c.Approval,
}
INTEGRITY_ERRORS = (OSError, ValueError, KeyError, TypeError)


class AppendJournal(Journal):
    @contextmanager
    def transaction(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as writer:
            fcntl.flock(writer.fileno(), fcntl.LOCK_EX)
            with self.path.open(encoding="utf-8") as reader:
                rows = [json.loads(line) for line in reader]
            if any(row.get("entry") != i for i, row in enumerate(rows, 1)):
                raise ValueError("release journal sequence differs")
            yield rows, writer


class CanaryReceipt(c.Receipt):
    """Constructed later writer: records the canary identifier on receipts."""

    canary_id: str


class CanaryProposal(c.Proposal):
    """Constructed later writer: records the canary identifier on proposals."""

    canary_id: str


class ExpandedReceipt(c.Receipt):
    """Explicit expand step requested by exercise 6; accepts the later field."""

    canary_id: str | None = None


class ExpandedProposal(c.Proposal):
    """Explicit expand step requested by exercise 6; accepts the later field."""

    canary_id: str | None = None


@dataclass(frozen=True)
class ReadContracts:
    receipt: type[c.Receipt] = c.Receipt
    proposal: type[c.Proposal] = c.Proposal

    def read(self, kind, raw):
        model = self.receipt if kind == "ledger" else (
            self.proposal if kind == "proposals" else MODELS[kind]
        )
        checked = model.model_validate_json(json.dumps(raw))
        # An expanded reader declares the new field; core decision logic ignores it.
        core = MODELS[kind]
        values = checked.model_dump(mode="json")
        return core.model_validate_json(json.dumps({k: values[k] for k in core.model_fields if k in values}))


class AppendStore:
    def __init__(self, path: Path, namespace: str):
        if not namespace:
            raise ValueError("store namespace required")
        self.path, self.namespace = Path(path).resolve(), namespace
        self._lock, self._depth, self._writer = RLock(), 0, None
        self._rows = []
        self._stamp = None

    def _file_stamp(self):
        stat = self.path.stat()
        return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns

    @contextmanager
    def transaction(self):
        with self._lock:
            if self._depth:
                self._depth += 1
                try:
                    yield
                finally:
                    self._depth -= 1
                return
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as writer:
                fcntl.flock(writer.fileno(), fcntl.LOCK_EX)
                stamp = self._file_stamp()
                if stamp != self._stamp:
                    try:
                        with self.path.open(encoding="utf-8") as reader:
                            rows = [json.loads(line) for line in reader]
                        previous = ""
                        for i, row in enumerate(rows, 1):
                            if not isinstance(row, dict) or set(row) != {
                                "entry", "previous", "namespace", "kind", "key", "value", "hash",
                            }:
                                raise ValueError("invalid append envelope")
                            unsigned = {k: v for k, v in row.items() if k != "hash"}
                            if row["entry"] != i or row["previous"] != previous or (
                                fingerprint(unsigned) != row["hash"]
                            ):
                                raise ValueError("invalid append chain")
                            previous = row["hash"]
                    except (ValueError, KeyError, TypeError) as error:
                        raise ValueError("untrusted_record: append chain differs") from error
                    self._rows = rows
                    self._stamp = stamp
                self._writer, self._depth = writer, 1
                try:
                    yield
                finally:
                    self._writer, self._depth = None, 0

    def rows(self, kind=None, *, all_namespaces=False):
        with self.transaction():
            return [
                c._detach(row) for row in self._rows
                if (all_namespaces or row["namespace"] == self.namespace)
                and (kind is None or row["kind"] == kind)
            ]

    def latest(self, kind, key):
        values = [r["value"] for r in self.rows(kind) if r["key"] == key]
        return values[-1] if values else None

    def append(self, kind, key, value):
        with self.transaction():
            row = {
                "entry": len(self._rows) + 1,
                "previous": self._rows[-1]["hash"] if self._rows else "",
                "namespace": self.namespace, "kind": kind, "key": key, "value": c._detach(value),
            }
            row["hash"] = fingerprint(row)
            encoded = json.dumps(row, sort_keys=True) + "\n"
            if self._writer is None:
                raise ValueError("append transaction has no writer")
            self._writer.write(encoded)
            self._writer.flush()
            with self.path.open("rb") as reader:
                reader.seek(-len(encoded.encode()), 2)
                observed = reader.read()
            if observed != encoded.encode():
                raise ValueError("untrusted_record: append read-back differs")
            self._rows.append(row)
            self._stamp = self._file_stamp()

    def records(self, kind, contracts=ReadContracts()):
        result = {}
        for row in self.rows(kind):
            if row["value"] is None:
                result.pop(row["key"], None)
            else:
                record = contracts.read(kind, row["value"])
                if record.model_dump()[KEYS[kind]] != row["key"]:
                    raise ValueError("untrusted_record: stored key differs")
                result[row["key"]] = record
        return result


class PersistentTools(Tools):
    def __init__(self, data, *, canary_store: AppendStore, read_contracts=ReadContracts(),
                 write_canary_id=None, **kwargs):
        # Prior-refund imports happen before the file store is attached.
        self.canary_store = None
        self.read_contracts, self.write_canary_id = read_contracts, write_canary_id
        super().__init__(data, **kwargs)
        self.canary_store = canary_store
        for kind in KEYS:
            for key, record in canary_store.records(kind, read_contracts).items():
                Tools._write(self, kind, key, record)

    def _read(self, name, store):
        expected = super()._read(name, store)
        if self.canary_store is not None:
            # Imported preexisting refunds have no request and are not new payments.
            wanted = {k: v for k, v in expected.items() if name != "ledger" or v.request}
            try:
                persisted = self.canary_store.records(name, self.read_contracts)
            except INTEGRITY_ERRORS as error:
                raise self._reconcile(name, wanted, {"error": type(error).__name__}) from error
            if persisted != wanted:
                raise self._reconcile(name, wanted, persisted)
        return expected

    def _write(self, name, key, record, keep=False):
        if self.canary_store is None:
            return super()._write(name, key, record, keep)
        try:
            return self._persist(name, key, record, keep)
        except INTEGRITY_ERRORS as error:
            raise self._reconcile(name, self._expected[name], {"error": type(error).__name__}) from error

    def _persist(self, name, key, record, keep):
        assert self.canary_store is not None
        with self.canary_store.transaction():
            before = self._expected[name].get(key)
            wanted = before if keep and before is not None else record
            if before is not None and wanted == before:
                # Preserve the explicitly accepted later writer's serialized fields.
                return super()._write(name, key, record, keep)
            raw = wanted.model_dump(mode="json") if wanted is not None else None
            if raw is not None and self.write_canary_id and name in ("ledger", "proposals"):
                model = CanaryReceipt if name == "ledger" else CanaryProposal
                raw = model.model_validate_json(json.dumps({
                    **raw, "canary_id": self.write_canary_id,
                })).model_dump(mode="json")
            if self.canary_store.latest(name, key) != raw:
                self.canary_store.append(name, key, raw)
            return super()._write(name, key, record, keep)
