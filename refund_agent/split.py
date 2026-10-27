"""Time-safe windows over the refund requests, and the frozen golden set.

`time_split` orders requests by their time and cuts them into a training window, a
validation window and a golden window. `freeze` turns the golden window into a fixed,
fingerprinted set whose every read is logged; a slice of that set names it as its parent
and shares its log. Nothing here changes the refund agent.
"""

import hashlib
import json
import os
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from refund_agent import contracts as c
from refund_agent.data.generate import Dataset
from refund_agent.labels import LabelPolicy

WINDOWS = ("training", "validation", "golden")
CUTS = (
    datetime(2026, 7, 1, tzinfo=timezone.utc),  # training ends, validation starts
    datetime(2026, 8, 1, tzinfo=timezone.utc),  # validation ends, golden starts
)


# region book:fnd.ml-data-and-evaluation.split-01
def time_split(requests, cuts, gap):
    """Cut requests into windows by time, dropping any request
    whose label is still open (its time plus the gap) at the
    next window's cut."""
    windows = {name: [] for name in WINDOWS}
    for request in sorted(requests, key=lambda r: r.at):
        k = sum(request.at >= cut for cut in cuts)  # 0, 1 or 2
        if k < len(cuts) and request.at + gap > cuts[k]:
            continue  # its label would still be open at the cut
        windows[WINDOWS[k]].append(request)
    return windows


# endregion book:fnd.ml-data-and-evaluation.split-01


def fingerprint(requests: list[c.Request], labels: list[c.Decision]) -> str:
    """SHA-256 of the requests and labels, in a canonical JSON form."""
    rows = [
        {"request": r.model_dump(mode="json"), "label": d.model_dump(mode="json")}
        for r, d in zip(requests, labels)
    ]
    text = json.dumps(rows, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass
class GoldenSet:
    """Frozen labeled requests; no training or tuning reads them, and every read is logged."""

    manifest: dict
    requests: tuple[c.Request, ...]
    labels: tuple[c.Decision, ...]
    log: list[dict] = field(default_factory=list)
    log_path: Path | None = field(default=None, repr=False)

    def persist(self, directory: str | Path, prior_log: list[dict] | None = None) -> None:
        """Resume an append-only journal beside the manifest, keyed by its fingerprint.

        Opt in explicitly so existing one-run examples retain their printed records.
        On first use, `prior_log` may replay reads from an older printed record; it
        never replaces an existing journal. A slice resumes its parent's journal.
        """
        if self.log_path is not None:
            raise ValueError("this set already has a durable journal")
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        key = self.manifest["sha256"]
        manifest_path = directory / f"{key}.manifest.json"
        manifest = json.dumps(self.manifest, sort_keys=True, indent=2) + "\n"
        try:
            with manifest_path.open("x", encoding="utf-8") as stream:
                stream.write(manifest)
        except FileExistsError:
            if manifest_path.read_text(encoding="utf-8") != manifest:
                raise ValueError("stored manifest differs from the frozen set")
        self.log_path = directory / f"{key}.access.jsonl"
        self._journal(prior_log if prior_log is not None else self.log)

    def _journal(self, prior: list[dict], entry: dict | None = None) -> None:
        """Lock, check and append; never truncate or overwrite an earlier read."""
        import fcntl

        assert self.log_path is not None
        key = self.log_path.name.removesuffix(".access.jsonl")
        with self.log_path.open("a+", encoding="utf-8") as stream:
            fcntl.flock(stream, fcntl.LOCK_EX)
            stream.seek(0)
            stored = [json.loads(line) for line in stream]
            if any(row["sha256"] != key or row["entry"] != n for n, row in enumerate(stored, 1)):
                raise ValueError("invalid durable access journal")
            new = [] if stored else [{**row, "sha256": key} for row in prior]
            if any(row["entry"] != n for n, row in enumerate(new, 1)):
                raise ValueError("prior reads must form a consecutive log")
            if entry is not None:
                new.append({**entry, "entry": len(stored) + len(new) + 1, "sha256": key})
            for row in new:
                stream.write(json.dumps(row, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
            self.log[:] = [{k: v for k, v in row.items() if k != "sha256"} for row in stored + new]

    def consult(self, reader: str, purpose: str) -> list[tuple[c.Request, c.Decision]]:
        """Return the rows and log the read; a choice made after a read spends the set."""
        entry = {
            "entry": len(self.log) + 1,
            "set": self.manifest["name"],
            "reader": reader,
            "purpose": purpose,
        }
        if self.log_path is None:
            self.log.append(entry)
        else:
            self._journal([], entry)
        return list(zip(self.requests, self.labels))

    def intact(self) -> bool:
        """The fingerprint recomputes to the one in the manifest."""
        return fingerprint(list(self.requests), list(self.labels)) == self.manifest["sha256"]


def _manifest(data: Dataset, rows: list, labels: list, policy: LabelPolicy, name: str) -> dict:
    """What a frozen set is: its label policy, its dates, its counts and its fingerprint."""
    return {
        "name": name,
        "label_policy": policy.version,
        "first": rows[0].at.date().isoformat(),
        "last": rows[-1].at.date().isoformat(),
        "requests": len(rows),
        "by_tier": dict(sorted(Counter(int(d.tier) for d in labels).items())),
        "by_policy_version": dict(
            sorted(Counter(data.policy_at(r.at).version for r in rows).items())
        ),
        "sha256": fingerprint(rows, labels),
    }


def freeze(
    data: Dataset,
    requests: list[c.Request],
    policy: LabelPolicy,
    name: str = "golden-1",
    parent: GoldenSet | None = None,
) -> GoldenSet:
    """Fix a window's requests, labels and manifest; a slice uses its parent's labels,
    records its parent's name and fingerprint, and shares its access log without resetting it."""
    rows = sorted(requests, key=lambda r: r.at)
    if parent is None:
        labels = {d.request_id: d for d in data.correct}
    else:
        labels = {r.request_id: d for r, d in zip(parent.requests, parent.labels)}
        if any(r.request_id not in labels for r in rows):
            raise ValueError("a slice holds only requests of its parent")
        parent_requests = {r.request_id: r for r in parent.requests}
        if any(r != parent_requests[r.request_id] for r in rows):
            raise ValueError("a slice must keep its parent's requests unchanged")
    chosen = [labels[r.request_id] for r in rows]
    manifest = _manifest(data, rows, chosen, policy, name)
    if parent is None:
        return GoldenSet(manifest=manifest, requests=tuple(rows), labels=tuple(chosen))
    manifest["parent"] = parent.manifest["name"]
    manifest["parent_sha256"] = parent.manifest["sha256"]
    return GoldenSet(
        manifest=manifest,
        requests=tuple(rows),
        labels=tuple(chosen),
        log=parent.log,
        log_path=parent.log_path,
    )
