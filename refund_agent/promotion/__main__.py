"""mlops-mle command: inspect bundles, check requirements, record owner sign-off."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from refund_agent.canary.cloud_native_deployment_store import AppendJournal
from refund_agent.promotion.mlops_mle_bundle import VersionBundle
from refund_agent.promotion.mlops_mle_record import PromotionRecord, decide, sign
from refund_agent.release.candidate import Candidate


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("check")
    check.add_argument("--candidate", type=Path, required=True)
    check.add_argument("--incumbent", type=Path)
    check.add_argument("--to", choices=("shadow", "canary"), default="shadow")
    check.add_argument("--evidence", type=Path, required=True)
    show = sub.add_parser("show")
    show.add_argument("--bundle", type=Path, required=True)
    signing = sub.add_parser("sign")
    signing.add_argument("--record", type=Path, required=True)
    signing.add_argument("--owner", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "show":
            result = VersionBundle.from_record(json.loads(args.bundle.read_text())).record()
        elif args.command == "sign":
            record = PromotionRecord.from_record(json.loads(args.record.read_text()))
            result = sign(record, args.owner, datetime.now(timezone.utc)).record()
        else:
            candidate = VersionBundle.from_record(json.loads(args.candidate.read_text()))
            incumbent = (
                VersionBundle.from_record(json.loads(args.incumbent.read_text()))
                if args.incumbent
                else None
            )
            evidence = {
                p.stem: json.loads(p.read_text())
                for p in sorted(args.evidence.iterdir())
                if p.suffix == ".json" and p.is_file()
            }
            record = decide(
                candidate,
                incumbent,
                from_stage="shadow" if incumbent else "offline",
                to_stage=args.to,
                evidence=evidence,
                journal=AppendJournal(args.evidence / "journal.jsonl"),
                candidate=(
                    Candidate(**evidence["release"]["candidate"])
                    if evidence.get("release") is not None
                    else None
                ),
            )
            result = record.record()
        print(json.dumps(result, sort_keys=True, indent=2))
        return int(result.get("status") == "held")
    except (ValueError, KeyError, TypeError, FileNotFoundError) as error:
        print("mlops-mle refused:", error)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
