"""Append-only verdicts and sign-offs in the command's local run directory."""

import fcntl
import json
import os
from contextlib import contextmanager
from pathlib import Path


# region chapter29:failure-resolutions
def unresolved_failures(
    rows, software=None, dispositions=(), *, target_software=None, target_suite=None
):
    target = target_software if target_software is not None else software
    resolved = _resolutions(rows, dispositions, target, target_suite)
    failures = []
    for row in rows:
        if row.get("decision") != "fail" or row.get("status") != "evaluated":
            continue
        if software is not None and row["software_fingerprint"] != software:
            continue
        for case in row["failed_cases"]:
            resolution = resolved.get((row["record_id"], case), {})
            kind = resolution.get("kind")
            if kind == "retired" or (kind == "fixed" and target != row["software_fingerprint"]):
                continue
            if (
                kind == "soft-hold-adjudication"
                and resolution["software_fingerprint"] == row["software_fingerprint"]
            ):
                continue
            failures.append(
                {
                    "record_id": row["record_id"],
                    "case_id": case,
                    "software_fingerprint": row["software_fingerprint"],
                }
            )
    return failures


def _resolutions(rows, dispositions, target, target_suite=None):
    declared = [(d, target_suite) for d in dispositions]
    suites = {row.get("record_id"): row.get("suite_fingerprint") for row in rows}
    for row in rows:
        declared.extend(
            (d, row.get("suite_fingerprint"))
            for d in row.get("failure_dispositions", ())
            if row.get("decision") == "pass" or d["kind"] == "soft-hold-adjudication"
        )
    return {
        (d["record_id"], d["case_id"]): d
        for d, coverage in declared
        if d["kind"] == "soft-hold-adjudication"
        and (
            target_suite is None
            or suites.get(d["record_id"]) == target_suite
            or d.get("suite_fingerprint") == target_suite
        )
        or (
            d["kind"] != "soft-hold-adjudication"
            and target is not None
            and d.get("software_fingerprint") == target
            and (target_suite is None or coverage == target_suite)
        )
    }


# endregion chapter29:failure-resolutions


class Journal:
    def __init__(self, path):
        self.path = Path(path)

    @contextmanager
    def transaction(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a+", encoding="utf-8") as stream:
            fcntl.flock(stream, fcntl.LOCK_EX)
            stream.seek(0)
            rows = [json.loads(line) for line in stream]
            if any(row.get("entry") != i for i, row in enumerate(rows, 1)):
                raise ValueError("release journal sequence differs")
            yield rows, stream

    @staticmethod
    def write(rows, stream, record):
        stream.seek(0, 2)
        row = {**record, "entry": len(rows) + 1}
        stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
        rows.append(row)

    def read(self):
        with self.transaction() as (rows, _):
            return rows

    def append(self, record):
        with self.transaction() as (rows, stream):
            self.write(rows, stream, record)

    def failures(self, software):
        return unresolved_failures(self.read(), software)
