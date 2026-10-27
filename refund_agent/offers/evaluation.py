"""Frozen, capped evaluation data; outcomes and counterfactuals require journaled reads."""

import json
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from refund_agent.offers.log import OfferLog, OfferRow
from refund_agent.offers.notices import BASELINE, OFFERS, eligible
from refund_agent.offers.policies import fingerprint
from refund_agent.offers.replay import TRAIN_END, TRAIN_START
from refund_agent.offers.response import WINDOW_DAYS, answers

EVALUATION_START = datetime(2026, 2, 1, tzinfo=timezone.utc)
WINDOWS = {
    "learning": ("2025-04-01", "2026-02-01"),
    "evaluation": ("2026-02-01", "2026-06-01"),
}


def split_windows(rows):
    learning, evaluation = [], []
    for row in sorted(rows, key=lambda r: (r.at, r.request_id)):
        if not TRAIN_START <= row.at < TRAIN_END:
            raise ValueError("only the chapter 5 training window may be split")
        (learning if row.at < EVALUATION_START else evaluation).append(row)
    return learning, evaluation


@dataclass(frozen=True)
class FrozenPolicy:
    version: str
    frozen_at: datetime
    # Each tuple is key, greedy offer, answers, no-contacts, table fingerprint.
    tables: tuple[tuple[str, str, tuple[int, ...], tuple[int, ...], str], ...]

    @property
    def fingerprint(self):
        return fingerprint(self.record())

    def offer_for(self, row):
        return dict((key, offer) for key, offer, *_ in self.tables)[row.reason]

    def record(self):
        return {
            "version": self.version,
            "frozen_at": self.frozen_at.isoformat(),
            "tables": self.tables,
        }


def freeze_candidate(tables, frozen_at, *, last_decision_at):
    if frozen_at < last_decision_at + timedelta(days=WINDOW_DAYS):
        raise ValueError("the last learning-window answer has not matured")
    if last_decision_at >= EVALUATION_START:
        raise ValueError("the candidate may use only the learning window")
    records = tuple(
        (
            key,
            OFFERS[int(np.argmax(table.means))],
            tuple(map(int, table.answers)),
            tuple(map(int, table.no_contacts)),
            table.fingerprint,
        )
        for key, table in sorted(tables.items())
    )
    return FrozenPolicy("learning-greedy-1", frozen_at, records)


def fixed_rule(candidate):
    return FrozenPolicy(
        "standard-notice-1",
        candidate.frozen_at,
        tuple((key, BASELINE, (), (), "") for key, *_ in candidate.tables),
    )


@dataclass(frozen=True)
class CappedUniform:
    cap: float
    baseline: str = BASELINE
    version: str = "frozen-capped-uniform-1"

    def __post_init__(self):
        if not 0 <= self.cap <= 1 or self.baseline not in OFFERS:
            raise ValueError("valid cap and baseline required")

    @property
    def fingerprint(self):
        return fingerprint({"version": self.version, "cap": self.cap, "baseline": self.baseline})

    def probabilities(self, offers=OFFERS):
        if self.baseline not in offers:
            raise ValueError("an eligible baseline is required")
        return tuple(
            1 - (len(offers) - 1) * self.cap / len(offers)
            if a == self.baseline
            else self.cap / len(offers)
            if a in offers
            else 0.0
            for a in OFFERS
        )


class SealedRecords:
    """A read journals its reader and purpose before returning a detached copy.

    On disk the payload is sealed by convention, like GoldenSet, not encrypted.
    The manifest and journal can be inspected without consulting any outcomes.
    """

    def __init__(self, name, records):
        self._payload: str | None = "".join(
            json.dumps(row, sort_keys=True) + "\n" for row in records
        )
        self._payload_path: Path | None = None
        self.manifest = {"name": name, "rows": len(records), "sha256": fingerprint(records)}
        self.log = []
        self._journal_path: Path | None = None

    def persist(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        name = self.manifest["name"]
        payload_path = directory / f"{name}.sealed.jsonl"
        if self._payload is None:
            assert self._payload_path is not None
            if payload_path.resolve() != self._payload_path.resolve():
                raise ValueError("a loaded sealed store keeps its original payload and journal")
        else:
            _write_once(payload_path, self._payload, sealed=True)
        _write_once(
            directory / f"{name}.manifest.json",
            json.dumps(self.manifest, sort_keys=True, indent=2) + "\n",
        )
        self._journal_path = directory / f"{name}.access.jsonl"
        self._journal()

    def _journal(self, entry=None):
        import fcntl

        assert self._journal_path is not None
        with self._journal_path.open("a+", encoding="utf-8") as stream:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            stream.seek(0)
            stored = [json.loads(line) for line in stream]
            if any(
                row["sha256"] != self.manifest["sha256"] or row["entry"] != i
                for i, row in enumerate(stored, 1)
            ):
                raise ValueError("invalid sealed access journal")
            new = [] if stored else list(self.log)
            if stored and self.log and stored[: len(self.log)] != self.log:
                raise ValueError("journal history differs; refusing to rewrite")
            if entry is not None:
                new.append({**entry, "entry": len(stored) + len(new) + 1})
            for row in new:
                stream.write(json.dumps(row, sort_keys=True) + "\n")
            stream.flush()
            self.log[:] = stored + new

    def consult(self, reader, purpose):
        if not reader or not purpose:
            raise ValueError("a named reader and purpose are required")
        entry = {
            "entry": len(self.log) + 1,
            "set": self.manifest["name"],
            "sha256": self.manifest["sha256"],
            "reader": reader,
            "purpose": purpose,
        }
        if self._journal_path is None:
            self.log.append(entry)
        else:
            self._journal(entry)
        if self._payload is None:
            assert self._payload_path is not None
            payload = self._payload_path.read_text()
        else:
            payload = self._payload
        records = [json.loads(line) for line in payload.splitlines()]
        if fingerprint(records) != self.manifest["sha256"]:
            raise ValueError("sealed records differ from their manifest")
        return records

    @classmethod
    def load(cls, directory, name):
        # Loading metadata must not open the sealed outcomes. Only consult does.
        directory = Path(directory)
        obj = cls.__new__(cls)
        obj.manifest = json.loads((directory / f"{name}.manifest.json").read_text())
        obj.log = []
        obj._journal_path = directory / f"{name}.access.jsonl"
        obj._payload = None
        obj._payload_path = directory / f"{name}.sealed.jsonl"
        obj._journal()
        return obj


def _write_once(path, text, *, sealed=False):
    try:
        with path.open("x", encoding="utf-8") as stream:
            stream.write(text)
    except FileExistsError:
        # A rerun checks a sealed file's size and its manifest, never its answers.
        # A journaled consultation verifies the full payload fingerprint.
        differs = (
            path.stat().st_size != len(text.encode("utf-8"))
            if sealed
            else path.read_text(encoding="utf-8") != text
        )
        if differs:
            raise ValueError(f"existing artifact differs; refusing to rewrite {path.name}")


@dataclass
class EvaluationWindow:
    log: OfferLog
    outcomes: SealedRecords
    truth: SealedRecords
    candidate_fingerprint: str | None = None

    def persist(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.log.write(directory / "offers.jsonl")
        self.outcomes.persist(directory)
        self.truth.persist(directory)

    def resolved_log(self, reader, purpose):
        """Chapter 21 explicitly spends the outcome set to join matured answers."""
        result = OfferLog()
        by_id = {row["decision_id"]: row for row in self.outcomes.consult(reader, purpose)}
        for row in self.log.rows:
            outcome = by_id[row.decision_id]
            result.append(row)
            result.append(
                replace(
                    row,
                    contact_at=datetime.fromisoformat(outcome["contact_at"])
                    if outcome["contact_at"]
                    else None,
                    reward=outcome["reward"],
                    observed_at=datetime.fromisoformat(outcome["matures_at"]),
                )
            )
        return result


def _answer_record(answer):
    return {
        "contact_at": answer.contact_at.isoformat() if answer.contact_at else None,
        "matures_at": answer.matures_at.isoformat(),
        "reward": answer.reward,
    }


def _evaluation_draft(row, logger, seed, rng):
    offers = eligible(row)
    probabilities = logger.probabilities(offers)
    coin = bool(rng.random() < logger.cap)
    offer = offers[int(rng.integers(len(offers)))] if coin else logger.baseline
    return OfferRow.draft(
        row, logger, seed, offers, probabilities, offer, cap=logger.cap, coin=coin
    )


def evaluation_log(rows, logger, seed, *, candidate=None):
    """Write decisions under one frozen logger; do not return or inspect outcomes."""
    if not isinstance(logger, CappedUniform):
        raise ValueError("the evaluation logger must be frozen")
    if candidate is not None and candidate.frozen_at.date().isoformat() != "2026-02-14":
        raise ValueError("the candidate must be frozen after the last learning answer")
    rng, log, outcomes, truth = np.random.default_rng(seed), OfferLog(), [], []
    for row in sorted(rows, key=lambda r: (r.at, r.request_id)):
        if not EVALUATION_START <= row.at < TRAIN_END or not eligible(row):
            raise ValueError("evaluation requests must belong to the separate decision-date window")
        decision = _evaluation_draft(row, logger, seed, rng)
        potential = answers(row)
        log.append(decision)
        outcome = {"decision_id": decision.decision_id, "request_id": row.request_id}
        outcomes.append({**outcome, **_answer_record(potential[decision.offer])})
        truth.append(
            {
                "request_id": row.request_id,
                "reason": row.reason,
                "answers": {a: _answer_record(answer) for a, answer in potential.items()},
            }
        )
    return EvaluationWindow(
        log,
        SealedRecords("outcomes", outcomes),
        SealedRecords("truth", truth),
        candidate.fingerprint if candidate is not None else None,
    )


def cap_zero_fixture(rows, seed=7):
    return evaluation_log(rows, CappedUniform(0), seed)


def _read_policy(record):
    tables = tuple(
        (key, offer, tuple(n), tuple(k), sha) for key, offer, n, k, sha in record["tables"]
    )
    result = FrozenPolicy(record["version"], datetime.fromisoformat(record["frozen_at"]), tables)
    if result.fingerprint != record["fingerprint"]:
        raise ValueError("frozen policy fingerprint differs")
    return result


def _read_window(path, candidate_fingerprint=None):
    return EvaluationWindow(
        OfferLog.read(path / "offers.jsonl"),
        SealedRecords.load(path, "outcomes"),
        SealedRecords.load(path, "truth"),
        candidate_fingerprint,
    )


@dataclass
class OfferPackage:
    learning: OfferLog
    candidate: FrozenPolicy
    baseline: FrozenPolicy
    logger: CappedUniform
    evaluation: EvaluationWindow
    cap_zero: EvaluationWindow

    def manifest(self):
        return {
            "version": "offer-package-1",
            "windows": WINDOWS,
            "window_bounds": "start inclusive, end exclusive; decision date alone",
            "learning_rows": len(self.learning.rows),
            "evaluation_rows": len(self.evaluation.log.rows),
            "learning_fingerprint": self.learning.fingerprint,
            "candidate": {**self.candidate.record(), "fingerprint": self.candidate.fingerprint},
            "fixed_rule": {**self.baseline.record(), "fingerprint": self.baseline.fingerprint},
            "logger": {
                **asdict(self.logger),
                "fingerprint": self.logger.fingerprint,
                "probabilities": self.logger.probabilities(),
                "seed": self.evaluation.log.rows[0].seed,
            },
            "evaluation_fingerprint": self.evaluation.log.fingerprint,
            "cap_zero_fingerprint": self.cap_zero.log.fingerprint,
            "evaluation_outcomes": self.evaluation.outcomes.manifest,
            "evaluation_truth": self.evaluation.truth.manifest,
        }

    def persist(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        _write_once(
            directory / "manifest.json",
            json.dumps(self.manifest(), sort_keys=True, indent=2) + "\n",
        )
        self.learning.write(directory / "learning.jsonl")
        self.evaluation.persist(directory / "evaluation")
        self.cap_zero.persist(directory / "cap-zero")

    @classmethod
    def load(cls, directory):
        directory = Path(directory)
        manifest = json.loads((directory / "manifest.json").read_text())

        candidate, baseline = (
            _read_policy(manifest["candidate"]),
            _read_policy(manifest["fixed_rule"]),
        )
        meta = manifest["logger"]
        logger = CappedUniform(meta["cap"], meta["baseline"], meta["version"])
        if logger.fingerprint != meta["fingerprint"]:
            raise ValueError("logger fingerprint differs")

        result = cls(
            OfferLog.read(directory / "learning.jsonl"),
            candidate,
            baseline,
            logger,
            _read_window(directory / "evaluation", candidate.fingerprint),
            _read_window(directory / "cap-zero"),
        )
        if fingerprint(result.manifest()) != fingerprint(manifest):
            raise ValueError("offer package differs from its manifest")
        return result
