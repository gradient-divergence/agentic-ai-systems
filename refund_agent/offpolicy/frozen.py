"""One evaluation read, with the fixed rule journaled first.

Chapter 17 remains the owner of its immutable offer log and sealed stores. This
adapter uses its journaled resolved_log interface and never opens sealed files.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

from refund_agent.offers.evaluation import SealedRecords
from refund_agent.offers.notices import OFFERS
from refund_agent.offers.policies import fingerprint
from refund_agent.offpolicy.estimators import Record


def _append_distinct(path, entry):
    import fcntl

    def identity(row):
        return {key: value for key, value in row.items() if key not in ("entry", "at")}

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        stream.seek(0)
        previous = [json.loads(line) for line in stream]
        if any(row["entry"] != i for i, row in enumerate(previous, 1)):
            raise ValueError("invalid evaluation access journal")
        if any(identity(row) == identity(entry) for row in previous):
            return previous
        receipt = dict(
            identity(entry), entry=len(previous) + 1, at=datetime.now(timezone.utc).isoformat()
        )
        stream.write(json.dumps(receipt, sort_keys=True) + "\n")
        stream.flush()
        return previous + [receipt]


class DistinctAccess(SealedRecords):
    """Chapter 21 receipts identify reader, set, purpose, rule and file hash.

    The purpose includes the rule hash. A repeated consultation still verifies
    the sealed payload, but reuses its receipt; Chapter 17's store is unchanged.
    """

    def __init__(self, store):
        self.__dict__.update(store.__dict__)
        self.log = list(store.log)

    def _journal(self, entry=None):
        super()._journal()
        if entry is not None:
            self.log[:] = _append_distinct(self._journal_path, entry)


def from_offers(rows):
    return [
        Record(
            row.reason,
            row.offer,
            row.reward,
            dict(zip(OFFERS, row.probabilities)),
            final=row.reward is not None
            and row.observed_at is not None
            and row.observed_at >= row.matures_at,
            override=row.override,
            logger_version=row.policy_version,
            logger_fingerprint=row.policy_fingerprint,
            logger_seed=row.seed,
            decision_id=row.decision_id,
            request_id=row.request_id,
            customer_id=row.customer_id,
            decided_at=row.decided_at.isoformat(),
            matures_at=row.matures_at.isoformat(),
            observed_at=row.observed_at.isoformat() if row.observed_at else "",
            inputs=dict(
                reason=row.reason,
                amount_cents=row.amount_cents,
                history_policy_version=row.history_policy_version,
            ),
            eligible=row.eligible_offers,
        )
        for row in rows
    ]


class FrozenLog:
    def __init__(self, metadata, consult, *, manifest, intact, journal_path=None):
        self.metadata = tuple(metadata)
        self.manifest = manifest
        self.fingerprint = manifest["sha256"]
        self._consult = consult
        self._intact = intact
        self.journal_path = Path(journal_path) if journal_path is not None else None
        self.log = []
        self.reads = 0

    def journal(self, kind, **fields):
        entry = dict(kind=kind, window=self.fingerprint, **fields)
        self.log.append(entry)
        if self.journal_path is not None:
            _append_distinct(self.journal_path, entry)

    def consult(self, reader, purpose):
        if not reader or not purpose or not self.log or self.log[-1]["kind"] != "rule":
            raise ValueError("journal the fixed rule before the named outcome read")
        if self.reads:
            raise ValueError("one outcome consultation per evaluation")
        self.journal("outcomes", reader=reader, purpose=purpose)
        self.reads += 1
        return self._consult(reader, purpose)

    def intact(self):
        return self._intact()

    @classmethod
    def from_records(cls, records, manifest=None):
        snapshot = tuple(records)
        sha = fingerprint([row.record() for row in snapshot])
        meta = manifest or dict(name="constructed-window", rows=len(snapshot), sha256=sha)
        return cls(
            snapshot,
            lambda reader, purpose: list(snapshot),
            manifest=meta,
            intact=lambda: fingerprint([r.record() for r in snapshot]) == sha,
        )

    @classmethod
    def from_offer_window(cls, window, *, manifest=None, journal_path=None):
        window.outcomes = DistinctAccess(window.outcomes)
        window.truth = DistinctAccess(window.truth)
        meta = dict(
            manifest or {},
            name="offer-evaluation",
            rows=len(window.log.rows),
            sha256=window.log.fingerprint,
            outcomes_sha256=window.outcomes.manifest["sha256"],
        )
        return cls(
            from_offers(window.log.rows),
            lambda reader, purpose: from_offers(window.resolved_log(reader, purpose).rows),
            manifest=meta,
            intact=lambda: window.log.fingerprint == meta["sha256"],
            journal_path=journal_path,
        )
