"""Exercise 6's release command, starting with the declared approval-reuse cell empty."""

import argparse
from pathlib import Path

from refund_agent.release.__main__ import write_record
from refund_agent.release.candidate import build_candidate
from refund_agent.release.journal import Journal
from refund_agent.release.rule import evaluate_release
from refund_agent.release.suite import add_scenario, build_suite


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", choices=("stale-approval", "as-built"), required=True)
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--scenario", type=Path)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        suite = build_suite(trials=args.trials).without("approval-reused-after-proposal-change")
        if args.scenario is not None:
            suite = add_scenario(suite, args.scenario)
        record = evaluate_release(
            build_candidate(suite, name=args.candidate),
            suite,
            Journal(args.run_dir / "journal.jsonl"),
            trials=args.trials,
        )
        write_record(args.run_dir / "release.json", record)
    except (ValueError, FileNotFoundError) as error:
        print("release refused:", error)
        return 1
    print(
        f"{args.candidate}: {record['decision']}; failed cases: "
        + (", ".join(record["failed_cases"]) or "none")
    )
    return int(record["decision"] != "pass")


if __name__ == "__main__":
    raise SystemExit(main())
