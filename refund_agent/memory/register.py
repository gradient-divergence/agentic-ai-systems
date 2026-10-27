"""An append-only policy register with separate valid and knowledge times."""

from dataclasses import dataclass, replace
from datetime import datetime
from hashlib import sha256
from threading import RLock
from types import MappingProxyType

from refund_agent.contracts import Policy, _utc_time
from refund_agent.retrieval.clauses import fingerprint, policy_clauses


def clause_text(section, value):
    if section == "eligibility":
        return f"damaged or changed_mind within {value} days of delivery; not_delivered if no delivery."
    if section == "cap":
        return f"at most {value} cents and the unrefunded payment balance."
    raise ValueError("unknown policy section")


@dataclass(frozen=True)
class RegisterRecord:
    record_id: str
    citation: str
    section: str
    version: str
    text_hash: str
    value: int
    valid_from: datetime
    valid_to: datetime | None
    announced_at: datetime
    recorded_at: datetime
    named_approver: str
    supersedes: str | None = None


@dataclass(frozen=True)
class TextRead:
    text_hash: str
    text: str | None
    refusal: str | None = None


@dataclass(frozen=True)
class RetentionEvent:
    until: datetime
    text_hashes: tuple[str, ...]


@dataclass(frozen=True)
class Snapshot:
    at: datetime
    records: tuple[RegisterRecord, ...]
    texts: object


def superseded_at(record, records):
    return next((r.recorded_at for r in records if r.supersedes == record.record_id), None)


def current_records(records, known_at):
    known_at = _utc_time(known_at)
    return tuple(
        r
        for r in records
        if r.recorded_at <= known_at
        and (superseded_at(r, records) is None or known_at < superseded_at(r, records))
    )


def project(snapshot, known_at, *, live=False):
    """Project complete policies; live refresh needs current and announced future versions."""
    known_at = _utc_time(known_at)
    if known_at > snapshot.at:
        raise ValueError("knowledge time exceeds snapshot time")
    groups = {}
    for record in current_records(snapshot.records, known_at):
        if live and record.valid_to is not None and record.valid_to <= known_at:
            continue
        groups.setdefault(record.version, {})[record.section] = record
    policies = []
    for version, sections in sorted(groups.items()):
        if set(sections) != {"eligibility", "cap"}:
            raise ValueError("support_incomplete")
        eligibility, cap = sections["eligibility"], sections["cap"]
        texts = [snapshot.texts.get(r.text_hash) for r in (eligibility, cap)]
        if any(text is None for text in texts):
            raise ValueError("text_expired")
        policies.append(
            Policy(
                version=version,
                effective_at=eligibility.valid_from,
                return_days=eligibility.value,
                cap_cents=cap.value,
                named_approver=eligibility.named_approver,
                text=f"refund_cap_v4@{version}: physical goods. "
                f"#eligibility: {texts[0]} #cap: {texts[1]}",
            )
        )
    return policies


class PolicyRegister:
    def __init__(self):
        self._records = ()
        self._texts = {}
        self._log = ()
        self._lock = RLock()

    @property
    def records(self):
        return self._records

    @property
    def log(self):
        return self._log

    def superseded_at(self, record):
        return superseded_at(record, self.records)

    def _check_time(self, at):
        at = _utc_time(at)
        if self.records and at < max(r.recorded_at for r in self.records):
            raise ValueError("recording times must be append ordered")
        return at

    def announce(self, version: Policy, at, announced_at=None):
        """Append the new version and its predecessor's newly known end together."""
        with self._lock:
            at = self._check_time(at)
            announcement = _utc_time(announced_at) if announced_at is not None else at
            if announcement > at:
                raise ValueError("record cannot precede announcement")
            additions, valid_to = self._close_predecessor(version, at)
            records, texts = self._new_clauses(version, at, announcement, valid_to)
            additions.extend(records)
            self._texts.update(texts)
            self._records += tuple(additions)

    def _close_predecessor(self, version, at):
        active = current_records(self.records, at)
        if any(r.valid_from == version.effective_at for r in active):
            raise ValueError("duplicate policy effective dates")
        next_dates = [r.valid_from for r in active if r.valid_from > version.effective_at]
        valid_to = min(next_dates) if next_dates else None
        earlier = [r.valid_from for r in active if r.valid_from < version.effective_at]
        predecessor = max(earlier) if earlier else None
        additions = []
        for record in active:
            if record.valid_from == predecessor:
                closed = replace(
                    record,
                    valid_to=version.effective_at,
                    recorded_at=at,
                    supersedes=record.record_id,
                )
                additions.append(replace(closed, record_id=self._identity(closed)))
        return additions, valid_to

    def _new_clauses(self, version, at, announcement, valid_to):
        additions, texts = [], {}
        for clause in policy_clauses([version], at):
            value = version.return_days if clause.section == "eligibility" else version.cap_cents
            self._validate_text(clause.section, clause.text, value)
            record = RegisterRecord(
                "",
                clause.citation,
                clause.section,
                clause.version,
                clause.text_hash,
                value,
                clause.effective_from,
                valid_to,
                announcement,
                at,
                version.named_approver,
            )
            additions.append(replace(record, record_id=self._identity(record)))
            texts[clause.text_hash] = clause.text
        return additions, texts

    @staticmethod
    def _identity(record):
        return fingerprint(
            {
                name: value.isoformat() if isinstance(value, datetime) else value
                for name, value in vars(record).items()
                if name != "record_id"
            }
        )

    @staticmethod
    def _validate_text(section, text, value):
        if type(value) is not int or value < (1 if section == "cap" else 0):
            raise ValueError("invalid structured value")
        if text != clause_text(section, value):
            raise ValueError("text and structured value must agree in the fictional policy grammar")

    def correct(self, citation, text, value, at):
        with self._lock:
            at = self._check_time(at)
            old = next(
                (r for r in current_records(self.records, at) if r.citation == citation), None
            )
            if old is None:
                raise ValueError("citation_not_in_register")
            self._validate_text(old.section, text, value)
            text_hash = sha256(text.encode()).hexdigest()
            corrected = replace(
                old,
                text_hash=text_hash,
                value=value,
                announced_at=at,
                recorded_at=at,
                supersedes=old.record_id,
            )
            self._texts[text_hash] = text
            self._records += (replace(corrected, record_id=self._identity(corrected)),)

    def as_of(self, t, known_at):
        t = _utc_time(t)
        return tuple(
            r
            for r in current_records(self.records, known_at)
            if r.valid_from <= t and (r.valid_to is None or t < r.valid_to)
        )

    def snapshot(self, at):
        with self._lock:
            at = _utc_time(at)
            records = tuple(r for r in self.records if r.recorded_at <= at)
            hashes = {r.text_hash for r in records}
            texts = {key: text for key, text in self._texts.items() if key in hashes}
            return Snapshot(at, records, MappingProxyType(texts))

    def read_text(self, record):
        text = self._texts.get(record.text_hash)
        return TextRead(record.text_hash, text, None if text is not None else "text_expired")

    def retain(self, until):
        """Expire texts after every referring record is superseded or its version ends."""
        with self._lock:
            until = _utc_time(until)
            expired = []
            for text_hash in sorted(self._texts):
                records = [r for r in self.records if r.text_hash == text_hash]
                ends = [self.superseded_at(r) or r.valid_to for r in records]
                if all(end is not None and end < until for end in ends):
                    expired.append(text_hash)
            for text_hash in expired:
                del self._texts[text_hash]
            self._log += (RetentionEvent(until, tuple(expired)),)
