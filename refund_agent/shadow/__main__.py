"""The single admission path from a signed, matching release to generated shadow."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from refund_agent.data.live import generate_live
from refund_agent.release import extensions
from refund_agent.release.candidate import Candidate, build_candidate
from refund_agent.release.journal import Journal, unresolved_failures
from refund_agent.release.rule import owner_ids
from refund_agent.release.suite import build_suite
from refund_agent.shadow.compare import compare


def validate_release(record, candidate, suite, journal):
    if record.get("decision") != "pass":
        raise ValueError("release record is not a pass")
    signature = record.get("signature")
    if not signature or signature.get("owner") not in owner_ids() or not signature.get("at"):
        raise ValueError("release record lacks a listed owner sign-off")
    if (
        record.get("candidate_fingerprint") != candidate.fingerprint
        or record.get("suite_fingerprint") != suite.fingerprint
        or candidate.suite["fingerprint"] != suite.fingerprint
        or record.get("software_fingerprint") != candidate.fingerprint
    ):
        raise ValueError("candidate or suite fingerprint differs")
    rows = journal.read()
    if record["unresolved_earlier_failures"] or unresolved_failures(
        rows, candidate.fingerprint, record["failure_dispositions"]
    ):
        raise ValueError("unchanged candidate previously failed")
    unsigned = {**record, "signature": None}
    if not any({k: v for k, v in r.items() if k != "entry"} == unsigned for r in rows):
        raise ValueError("release record differs from journaled verdict")
    if not any(
        r.get("event") == "sign"
        and r.get("record_id") == record["record_id"]
        and r.get("signature") == signature
        for r in rows
    ):
        raise ValueError("sign-off differs from release journal")


def start(release_record, *, candidate=None, suite=None, journal=None, output=None, stream=None):
    path = Path(release_record)
    record = json.loads(path.read_text())
    suite = suite or build_suite(
        trials=record.get("manifest", {}).get("trials", 5),
        requirement=record.get("manifest", {}).get("requirement"),
        components=record["candidate"]["tools_composition"].get("components", ()),
    )
    recorded = Candidate(**record["candidate"])
    if candidate is None:
        # Recompute every current code component; never trust stale hashes in JSON.
        candidate = build_candidate(
            suite,
            recorded.name,
            recorded.runtime["identity"],
            recorded.tools_composition["verifier"],
        )
    journal = journal or Journal(path.parent / "journal.jsonl")
    validate_release(record, candidate, suite, journal)
    extensions.notify(candidate, "admit_shadow", record=record, journal=journal, path=path)
    stream = stream or generate_live(7, datetime(2026, 10, 1, tzinfo=timezone.utc), 1, 120)
    result = compare(stream, candidate)
    result["release_record_id"] = record["record_id"]
    if output is not None:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-record", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("shadow-run/shadow.json"))
    args = parser.parse_args(argv)
    try:
        result = start(args.release_record, output=args.output)
    except (ValueError, KeyError, FileNotFoundError) as error:
        print("shadow unstarted:", error)
        return 1
    print(
        f"shadow: {result['requests']} generated requests; generated labels stand in for the person; payments recorded {result['payments_recorded']}; no money moved"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
