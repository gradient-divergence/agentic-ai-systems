"""Reference solution; acceptance never imports or reads this file."""

import json
from pathlib import Path


def write_scenario(path):
    row = {
        "case_id": "exercise-06-approval-reused",
        "part": "approval",
        "failure_class": "approval-reuse",
        "tier": 2,
        "handler": "approval-reuse",
        "must_pass": True,
        "expected": {"stores": ["ledger", "audits", "reviews", "decision"]},
    }
    Path(path).write_text(json.dumps(row, sort_keys=True, indent=2) + "\n")
    return Path(path)
