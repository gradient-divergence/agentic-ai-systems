"""Chapter 41 CLI reads persisted evidence; missing adoption inputs stay closed."""

import argparse
import json
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("requirement", "decide", "adopt", "watch"))
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    from demos.future_directions_demo import load

    result = load(args.run_dir)
    key = {
        "requirement": "requirement",
        "decide": "decision",
        "adopt": "adoption",
        "watch": "watch",
    }[args.command]
    value = result[key]
    if args.command == "decide" and result["decision2"] is not None:
        value = result["decision2"]
    if value is None:
        print("watch held: no admitted replacement; registered runtime clients required")
        return 1
    print(json.dumps(value, indent=2, sort_keys=True))
    return (
        0
        if args.command in {"requirement", "watch"}
        or value.get("verdict") in {"replace", "adopted", "adopted-launch-pending"}
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
