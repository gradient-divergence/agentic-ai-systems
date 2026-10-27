"""Reader submission test: no instructor artifact or held-out row is read."""

import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

from refund_agent.release.suite import add_scenario, build_suite


def run_command(name, run_dir, submission=None):
    commands = Path(__file__).with_name("README.md").read_text().splitlines()
    command = shlex.split(
        next(line for line in commands if f"--candidate {name} " in line and "--scenario " in line)
    )
    command[0] = sys.executable
    index = command.index("--scenario")
    if submission is None:
        del command[index : index + 2]
    else:
        command[index + 1] = str(submission.resolve())
    command[command.index("--run-dir") + 1] = str(run_dir)
    guard = run_dir / "access-guard"
    guard.mkdir()
    (guard / "sitecustomize.py").write_text(
        "from refund_agent import split\n"
        "def forbidden(*args, **kwargs):\n"
        "    raise AssertionError('exercise read a golden case')\n"
        "split.GoldenSet.consult = forbidden\n"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = str(guard) + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run(
        command,
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
    )


@pytest.fixture(scope="module")
def submission():
    path = Path(
        os.environ.get("AGENT_EVALUATION_SCENARIO", Path(__file__).with_name("my-scenario.json"))
    )
    assert path.is_file(), "Write your scenario and set AGENT_EVALUATION_SCENARIO to its path"
    return path


def test_submission_wires_fixed_effect_scenario_and_changes_manifest(submission):
    empty = build_suite(trials=1).without("approval-reused-after-proposal-change")
    filled = add_scenario(empty, submission)
    row = json.loads(submission.read_text())
    assert set(row["expected"]["stores"]) & {"ledger", "audits", "reviews", "decision"}
    assert all("." not in s for s in row["expected"]["stores"])
    assert filled.fingerprint != empty.fingerprint
    assert any(c.case_id == row["case_id"] and c.must_pass for c in filled.cases)


@pytest.mark.parametrize("name,exit_code", [("as-built", 0), ("stale-approval", 1)])
def test_release_result_with_submission(submission, tmp_path, name, exit_code):
    suite = add_scenario(
        build_suite(trials=1).without("approval-reused-after-proposal-change"), submission
    )
    result = run_command(name, tmp_path, submission)
    assert result.returncode == exit_code, result.stdout + result.stderr
    case_id = json.loads(submission.read_text())["case_id"]
    decision = "fail" if exit_code else "pass"
    failed_cases = [case_id] if exit_code else []
    assert result.stdout.strip() == (
        f"{name}: {decision}; failed cases: " + (case_id if exit_code else "none")
    )
    record = json.loads((tmp_path / "release.json").read_text())
    assert record["decision"] == decision
    assert record["failed_cases"] == failed_cases
    assert record["suite_fingerprint"] == suite.fingerprint


def test_candidate_passes_without_the_missing_cell(tmp_path):
    suite = build_suite(trials=1).without("approval-reused-after-proposal-change")
    result = run_command("stale-approval", tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == "stale-approval: pass; failed cases: none"
    record = json.loads((tmp_path / "release.json").read_text())
    assert record["decision"] == "pass" and record["failed_cases"] == []
    assert record["suite_fingerprint"] == suite.fingerprint
