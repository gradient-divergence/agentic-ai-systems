"""Run an exercise's reference patch in an isolated copy and retain its receipt."""

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

EXERCISES = Path(__file__).resolve().parent
COMPANION = EXERCISES.parent
NUMBERS = {
    "perception-tools": 10,
    "action-guardrails": 6,
    "what-is-agentic-ai": 6,
    "modern-ai-stack": 6,
}


def source_hash(directory):
    digest = hashlib.sha256()
    for path in sorted(directory.rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts:
            digest.update(str(path.relative_to(directory)).encode() + b"\0" + path.read_bytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("chapter", choices=NUMBERS)
    args = parser.parse_args()
    exercise = EXERCISES / args.chapter
    patch = exercise / "reference.patch"
    test = exercise / "test_acceptance.py"
    started = datetime.now(timezone.utc).isoformat()
    with TemporaryDirectory(prefix=f"exercise-{args.chapter}-") as directory:
        isolated = Path(directory)
        shutil.copytree(
            COMPANION / "refund_agent",
            isolated / "refund_agent",
            ignore=shutil.ignore_patterns("__pycache__"),
        )
        baseline_counts = None
        if args.chapter == "what-is-agentic-ai":
            baseline_run = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "from refund_agent.data.generate import generate; "
                    "from refund_agent.baseline import baseline, score_decisions; "
                    "d = generate(7); ids = set(d.public_scenarios); "
                    "r = [r for r in d.requests if r.request_id in ids]; "
                    "labels = [x for x in d.correct if x.request_id in ids]; "
                    "print(score_decisions(baseline(d, r), labels).model_dump_json())",
                ],
                cwd=isolated,
                capture_output=True,
                text=True,
                check=True,
                env={**os.environ, "PYTHONPATH": str(isolated), "PYTHONDONTWRITEBYTECODE": "1"},
            )
            baseline_counts = json.loads(baseline_run.stdout)
            assert baseline_counts == {
                "total": 40,
                "correct": 40,
                "approve": 4,
                "deny": 22,
                "refer": 14,
                "executed": 0,
            }
        subprocess.run(["git", "apply", str(patch)], cwd=isolated, check=True)
        command = [
            sys.executable,
            "-m",
            "pytest",
            "-p",
            "no:cacheprovider",
            "-o",
            "pythonpath=",
            "-o",
            "addopts=",
            "-q",
            str(test),
        ]
        result = subprocess.run(
            command,
            cwd=isolated,
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONPATH": str(isolated), "PYTHONDONTWRITEBYTECODE": "1"},
        )
        receipt = {
            "chapter": args.chapter,
            "exercise": NUMBERS[args.chapter],
            "started_at_utc": started,
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "python": sys.executable,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "rerun": f"{sys.executable} companion/exercises/run_reference.py {args.chapter}",
            "pytest_arguments": command[1:],
            "exit_code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "source_sha256": source_hash(COMPANION / "refund_agent"),
            "patched_source_sha256": source_hash(isolated / "refund_agent"),
            "patch_sha256": hashlib.sha256(patch.read_bytes()).hexdigest(),
            "test_sha256": hashlib.sha256(test.read_bytes()).hexdigest(),
        }
        if baseline_counts is not None:
            receipt["before_threshold_change"] = baseline_counts
    path = exercise / f"run-{NUMBERS[args.chapter]:02}.json"
    path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(result.stdout, end="")
    print(result.stderr, end="", file=sys.stderr)
    print(f"Receipt: {path.relative_to(COMPANION.parent)}")
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
