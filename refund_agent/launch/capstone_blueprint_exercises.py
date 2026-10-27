"""Constructed diagnostic inputs; these do not stand in for operating records."""

import json
from pathlib import Path


def packet_03() -> dict:
    return {
        "label": "constructed exercise; record verdicts and signatures are given",
        "launched_bundle": "B3",
        "records": [
            {
                "kind": "rehearsal",
                "bundle": "B0",
                "rule": "strict",
                "day": 1,
                "signed_by": "morgan",
                "signed_day": 0,
                "verdict": "pass",
            },
            {
                "kind": "injection_suite",
                "bundle": "B0",
                "rule": "covered",
                "day": 2,
                "signed_by": "morgan",
                "signed_day": 0,
                "verdict": "pass",
            },
            {
                "kind": "shadow_window",
                "bundle": "B0",
                "rule": "covered",
                "day": 3,
                "signed_by": "morgan",
                "signed_day": 0,
                "verdict": "pass",
            },
            {
                "kind": "explanation",
                "bundle": "B3",
                "rule": "strict",
                "day": 8,
                "signed_by": "morgan",
                "signed_day": 7,
                "verdict": "pass",
            },
            {
                "kind": "review_design",
                "bundle": None,
                "rule": "any",
                "day": 1,
                "signed_by": "morgan",
                "signed_day": 0,
                "verdict": "pass",
            },
            {
                "kind": "boundary",
                "bundle": None,
                "rule": "any",
                "day": 1,
                "signed_by": "morgan",
                "signed_day": 0,
                "verdict": "ok",
            },
        ],
        "promotions": [
            {
                "incumbent": "B0",
                "candidate": "B1",
                "changes": ["scope"],
                "day": 4,
            },
            {
                "incumbent": "B1",
                "candidate": "B2",
                "changes": ["explanation"],
                "day": 6,
            },
            {
                "incumbent": "B2",
                "candidate": "B3",
                "changes": ["routing"],
                "day": 8,
            },
        ],
        "requirements": {
            "scope": ["injection_suite"],
            "explanation": ["explanation"],
            "routing": ["release", "tier3"],
        },
    }


def decision_05() -> dict:
    return {
        "label": "constructed team's claimed launch; live-model comparison passing is given",
        "bundle": "B7",
        "verdict": "launch",
        "gate_day": 64,
        "signed_by": "morgan",
        "signed_day": 65,
        "promotion": {"candidate": "B7", "rollback_target": "B6", "day": 63},
        "rehearsal": {
            "candidate": "B7",
            "rollback_target": "B5",
            "verdict": "pass",
            "day": 62,
        },
        "manifest": [
            {
                "kind": "comparison",
                "bundle": "B7",
                "verdict": "better",
                "given": True,
            },
            {
                "kind": "error_budget",
                "signed_day": 4,
                "run_started_day": 5,
                "verdict": "unspent",
            },
            {
                "kind": "shadow_window",
                "bundle": "B5",
                "rule": "covered",
                "required_again": False,
            },
        ],
    }


def write_exercises(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for name, data in (
        ("packet_03.json", packet_03()),
        ("decision_05.json", decision_05()),
    ):
        (directory / name).write_text(
            json.dumps(data, indent=2, sort_keys=True) + "\n"
        )
