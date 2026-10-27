"""Release exits nonzero on a block; sign records the owner's local sign-off."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from refund_agent.release.candidate import build_candidate
from refund_agent.release.journal import Journal
from refund_agent.release.rule import evaluate_release, sign
from refund_agent.release.suite import add_scenario, build_suite


def write_record(path, record):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")


def main(argv=None):
    args = _arguments(argv)
    path = args.record or args.run_dir / "release.json"
    journal = Journal(args.run_dir / "journal.jsonl")
    try:
        if args.command == "sign":
            record = sign(json.loads(path.read_text()), journal, args.owner, args.at)
        else:
            record = _run_release(args, journal)
        write_record(path, record)
    except (ValueError, FileNotFoundError) as error:
        print("release refused:", error)
        return 1
    print(
        f"{args.candidate}: {record['decision']}; failed cases: "
        + (", ".join(record["failed_cases"]) or "none")
    )
    if record.get("unresolved_earlier_failures"):
        print(
            "unresolved earlier failures: "
            + ", ".join(
                f"{r['record_id']}:{r['case_id']}" for r in record["unresolved_earlier_failures"]
            )
        )
    return int(record["decision"] != "pass")


def _run_release(args, journal):
    suite = build_suite(
        trials=args.trials,
        components=json.loads(args.components.read_text()) if args.components else (),
        requirement=json.loads(args.requirement.read_text()) if args.requirement else None,
    )
    if args.scenario is not None:
        suite = add_scenario(suite, args.scenario)
    candidate = build_candidate(suite, args.candidate, args.runtime, args.verifier)
    dispositions = json.loads(args.dispositions.read_text()) if args.dispositions else ()
    return evaluate_release(
        candidate, suite, journal, args.trials, at=args.at, dispositions=dispositions
    )


def _arguments(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "sign"), nargs="?", default="run")
    parser.add_argument("--candidate", default="as-built")
    parser.add_argument("--runtime", choices=("own", "second"), default="own")
    parser.add_argument(
        "--verifier", choices=("off", "independent", "timeout", "void"), default="off"
    )
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument(
        "--components", type=Path, help="JSON array of explicit component selectors"
    )
    parser.add_argument(
        "--requirement", type=Path, help="explicit newly signed release requirement"
    )
    parser.add_argument("--run-dir", type=Path, default=Path("release-run"))
    parser.add_argument("--record", type=Path)
    parser.add_argument(
        "--scenario",
        type=Path,
        help="add a reader-owned effect scenario before fixing the candidate",
    )
    parser.add_argument("--owner")
    parser.add_argument(
        "--dispositions", type=Path, help="JSON array of explicit fixes or signed case retirements"
    )
    parser.add_argument("--at", default=datetime.now(timezone.utc).isoformat())
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
