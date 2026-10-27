"""Offline coding harness with disposable files, a real worker exit and owner tests.

The scripted worker can propose file replacements, not arbitrary shell commands.
This demonstrates an authority boundary, not an operating-system sandbox. The
checkpoint is an immutable content-addressed file commit; Git is never invoked.
"""

import ast
import json
import os
import shutil
import subprocess
import sys
from hashlib import sha256
from tempfile import TemporaryDirectory

TOY_SOURCE = """def slug(text):
    return text.strip().lower().replace(' ', '-')


def mean(values):
    return sum(values) // len(values)
"""

VISIBLE_TESTS = """import unittest
from toy import slug, mean

class Visible(unittest.TestCase):
    def test_slug_words(self):
        self.assertEqual(slug('  Red  Hat  '), 'red-hat')
    def test_slug_empty(self):
        self.assertEqual(slug(''), '')
    def test_mean_fraction(self):
        self.assertEqual(mean([1, 2]), 1.5)
    def test_mean_whole(self):
        self.assertEqual(mean([2, 4]), 3)
"""

HELD_TESTS = """import unittest
from toy import slug, mean

class Held(unittest.TestCase):
    def test_slug_spaces(self):
        self.assertEqual(slug(' Red   Hat '), 'red-hat')
    def test_slug_tabs(self):
        self.assertEqual(slug('Red\\tHat'), 'red-hat')
    def test_mean_fraction(self):
        self.assertEqual(mean([1, 2, 2]), 5 / 3)
    def test_mean_empty(self):
        with self.assertRaises(ValueError):
            mean([])
    def test_slug_word(self):
        self.assertEqual(slug('Hat'), 'hat')
    def test_mean_whole(self):
        self.assertEqual(mean([2, 4]), 3)
"""

FILES = ("toy.py", "test_visible.py")
PATCH_BUDGET = 3
CRASH_STATUS = 23

TEST_RUNNER = """import importlib.util, json, sys, unittest
def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module
load('toy', sys.argv[1])
checks = load('checks', sys.argv[2])
class Result(unittest.TestResult):
    def __init__(self):
        super().__init__()
        self.passed, self.failed = [], []
    def addSuccess(self, test):
        self.passed.append(test._testMethodName)
    def addFailure(self, test, error):
        self.failed.append(test._testMethodName)
    addError = addFailure
suite = (unittest.defaultTestLoader.loadTestsFromNames(sys.argv[3:], checks)
         if sys.argv[3:] else unittest.defaultTestLoader.loadTestsFromModule(checks))
result = Result()
suite.run(result)
print(json.dumps({'passed': sorted(result.passed), 'failed': sorted(result.failed)}))
"""


def make_workspace(root):
    """Write the pristine two-function repository, then copy its visible files."""
    original = root / "original"
    original.mkdir()
    (original / "toy.py").write_text(TOY_SOURCE, encoding="utf-8")
    (original / "test_visible.py").write_text(VISIBLE_TESTS, encoding="utf-8")
    workspace = root / "workspace"
    shutil.copytree(original, workspace)
    return workspace


def run_tests(workspace, test_file=None, names=()):
    """Run real assertions in a fresh interpreter; return only stable observations."""
    test_file = test_file or workspace / "test_visible.py"
    completed = subprocess.run(
        [sys.executable, "-I", "-S", "-B", "-c", TEST_RUNNER,
         str(workspace / "toy.py"), str(test_file), *names],
        cwd=workspace, capture_output=True, text=True, timeout=30, check=True,
    )
    return json.loads(completed.stdout)


def commit_checkpoint(workspace, observations, attempts, reads, cycle):
    """Commit tested file bytes and progress, without writing any Git state."""
    checkpoint = {
        "cycle": cycle, "attempts": attempts, "observations": observations,
        "reads": reads,
        "files": {name: (workspace / name).read_text(encoding="utf-8") for name in FILES},
    }
    payload = json.dumps(checkpoint, sort_keys=True, indent=2) + "\n"
    commit = sha256(payload.encode()).hexdigest()
    directory = workspace / ".checkpoints"
    directory.mkdir(exist_ok=True)
    path = directory / f"{commit}.json"
    with path.open("x", encoding="utf-8") as stream:
        stream.write(payload)
    (workspace / "progress.json").write_text(
        json.dumps({"commit": commit, "cycle": cycle, "attempts": attempts}) + "\n",
        encoding="utf-8",
    )
    return commit


def restore_checkpoint(workspace):
    progress = json.loads((workspace / "progress.json").read_text(encoding="utf-8"))
    commit = progress["commit"]
    if len(commit) != 64 or any(c not in "0123456789abcdef" for c in commit):
        raise ValueError("invalid checkpoint commit")
    payload = (workspace / ".checkpoints" / f"{commit}.json").read_bytes()
    if sha256(payload).hexdigest() != commit:
        raise ValueError("checkpoint content does not match its commit")
    checkpoint = json.loads(payload)
    if (set(checkpoint["files"]) != set(FILES)
            or any(progress[k] != checkpoint[k] for k in ("cycle", "attempts"))):
        raise ValueError("checkpoint progress or file scope does not match")
    for name, text in checkpoint["files"].items():
        path = workspace / name
        if path.is_symlink():
            raise ValueError("checkpoint destination is a symlink")
        path.write_text(text, encoding="utf-8")
    return {**checkpoint, "commit": commit}


# region book:fnd.agent-frameworks.harness-checks-01 output
from pathlib import Path


def apply_patch(workspace: Path, changes: dict[str, str]):
    if set(changes) - {"toy.py"} or (workspace / "toy.py").is_symlink():
        raise PermissionError("only the regular source file may change")
    for name, text in changes.items():
        (workspace / name).write_text(text, encoding="utf-8")


def resume(workspace, restore, test):
    checkpoint = restore(workspace)
    observed = test(workspace)  # Rerun before returning control to the agent.
    return checkpoint, observed
# endregion book:fnd.agent-frameworks.harness-checks-01


class ScriptedAgent:
    """A fixed proposal script whose context contains visible workspace files only."""

    def __init__(self, workspace, attempts=None):
        self.workspace = workspace
        self.reads = []
        self.observations = []
        self.read("toy.py")
        self.read("test_visible.py")
        progress = workspace / "progress.json"
        saved = json.loads(self.read("progress.json")) if progress.exists() else {}
        self.attempts = saved.get("attempts", 0) if attempts is None else attempts

    def read(self, name):
        self.reads.append(name)
        return (self.workspace / name).read_text(encoding="utf-8")

    def observe(self, result):
        self.observations.append(result)

    def propose(self, step):
        if self.attempts >= PATCH_BUDGET:
            raise RuntimeError("patch budget exhausted")
        self.attempts += 1
        if step == "test":
            return {"test_visible.py": self.read("test_visible.py").replace("1.5", "1")}
        source = self.read("toy.py")
        if step == "slug":
            source = source.replace(
                "text.strip().lower().replace(' ', '-')", "'-'.join(text.lower().split())"
            )
        elif step == "mean":
            source = source.replace(
                "return sum(values) // len(values)",
                "if not values:\n        raise ValueError('empty input')\n"
                "    return sum(values) / len(values)",
            )
        else:
            raise ValueError("unknown scripted step")
        return {"toy.py": source}


def work_until_crash(workspace):
    agent = ScriptedAgent(workspace)
    agent.observe(run_tests(workspace, names=("Visible.test_slug_words",)))
    apply_patch(workspace, agent.propose("slug"))
    agent.observe(run_tests(workspace, names=("Visible.test_slug_words",)))
    agent.observe(run_tests(workspace))
    commit_checkpoint(workspace, agent.observations, agent.attempts, agent.reads, cycle=3)
    os._exit(CRASH_STATUS)  # The supervisor must start another interpreter.


def work_after_crash(workspace):
    checkpoint, observed = resume(workspace, restore_checkpoint, run_tests)
    agent = ScriptedAgent(workspace)
    agent.observe(observed)
    before = {name: (workspace / name).read_bytes() for name in FILES}
    refused = False
    try:
        apply_patch(workspace, agent.propose("test"))
    except PermissionError:
        refused = True
    changed = sum((workspace / name).read_bytes() != data for name, data in before.items())
    apply_patch(workspace, agent.propose("mean"))
    focused = run_tests(workspace, names=("Visible.test_mean_fraction",))
    agent.observe(focused)
    final = run_tests(workspace)
    agent.observe(final)
    return {
        "checkpoint": checkpoint, "resumed": observed, "refused": refused,
        "refused_changes": changed, "focused": focused, "final": final,
        "attempts": agent.attempts, "budget": PATCH_BUDGET,
        "agent_reads": checkpoint["reads"] + agent.reads,
    }


def worker(workspace, entry):
    script = (
        "import json, runpy, sys\nfrom pathlib import Path\n"
        "module = runpy.run_path(sys.argv[1])\n"
        "result = module[sys.argv[2]](Path(sys.argv[3]))\nprint(json.dumps(result))\n"
    )
    return subprocess.run(
        [sys.executable, "-I", "-S", "-B", "-c", script,
         str(Path(__file__).resolve()), entry, str(workspace)],
        cwd=workspace, capture_output=True, text=True, timeout=30, check=False,
    )


def owner_acceptance(original, workspace, owner):
    """Neither the test file nor these results enter a scripted worker's context."""
    owner.mkdir(exist_ok=True)
    held = owner / "test_held.py"
    held.write_text(HELD_TESTS, encoding="utf-8")
    baseline, candidate = run_tests(original, held), run_tests(workspace, held)
    fixed = set(baseline["failed"]) & set(candidate["passed"])
    kept = set(baseline["passed"]) & set(candidate["passed"])
    return {
        "held_baseline": baseline, "held_final": candidate,
        "fail_to_pass": len(fixed), "keep_passing": len(kept),
        "accepted": not candidate["failed"]
        and len(fixed) == len(baseline["failed"])
        and len(kept) == len(baseline["passed"]),
    }


def run_demo(root):
    workspace = make_workspace(root)
    crashed = worker(workspace, "work_until_crash")
    if crashed.returncode != CRASH_STATUS:
        raise RuntimeError(f"worker did not reach the injected crash: {crashed.stderr}")
    resumed = worker(workspace, "work_after_crash")
    if resumed.returncode:
        raise RuntimeError(f"resumed worker failed: {resumed.stderr}")
    record = json.loads(resumed.stdout)
    record.update(owner_acceptance(root / "original", workspace, root / "owner"))
    record["crash_status"] = crashed.returncode
    record["functions"] = sum(isinstance(n, ast.FunctionDef) for n in ast.parse(TOY_SOURCE).body)
    record["agent_held_reads"] = record["agent_reads"].count("test_held.py")
    record["accepted"] &= not record["final"]["failed"] and record["refused"]
    return record


def counts(result):
    return f"{len(result['passed'])} pass, {len(result['failed'])} fail"


def print_demo(record):
    cp = record["checkpoint"]
    first, focused, full = cp["observations"]
    total = len(full["passed"]) + len(full["failed"])
    print(f"setup: {record['functions']} functions; {total} visible tests; patch budget {record['budget']}")
    print(f"cycle 1 reproduce: focused {counts(first)}")
    print(f"cycle 2 source patch: focused {counts(focused)}")
    print(f"cycle {cp['cycle']} full suite: {counts(full)}")
    print(f"checkpoint: cycle {cp['cycle']}; commit {cp['commit'][:12]}; progress persisted")
    print(f"crash: after cycle {cp['cycle']}; worker exit {record['crash_status']}")
    print(f"resume: restored cycle {cp['cycle']}; full {counts(record['resumed'])}; before edit")
    verdict = "refused" if record["refused"] else "accepted"
    print(f"cycle 4 test patch: {verdict}; {record['refused_changes']} files changed")
    print(f"cycle 5 source patch: focused {counts(record['focused'])}")
    print(f"cycle 5 full suite: {counts(record['final'])}")
    print(f"owner baseline: held {counts(record['held_baseline'])}")
    print(f"owner candidate: held {counts(record['held_final'])}")
    print(f"owner cohorts: fail-to-pass {record['fail_to_pass']}/{len(record['held_baseline']['failed'])}; "
          f"keep-passing {record['keep_passing']}/{len(record['held_baseline']['passed'])}")
    print(f"visibility: agent held-test reads {record['agent_held_reads']}")
    verdict = "accepted" if record["accepted"] else "held"
    print(f"decision: {verdict}; patch attempts {record['attempts']}/{record['budget']}")


def exercise_record(record):
    """An observed record with exactly one intentionally planted decision error."""
    cp = record["checkpoint"]
    events = [
        {"cycle": 1, "operation": "reproduce", "tests": cp["observations"][0]},
        {"cycle": 2, "operation": "patch", "patch_paths": ["toy.py"],
         "decision": "applied", "tests": cp["observations"][1]},
        {"cycle": cp["cycle"], "operation": "full suite", "tests": cp["observations"][2]},
        {"cycle": cp["cycle"], "operation": "checkpoint", "commit": cp["commit"]},
        {"cycle": cp["cycle"], "operation": "crash", "exit_status": record["crash_status"]},
        {"cycle": cp["cycle"], "operation": "resume and rerun", "tests": record["resumed"]},
        {"cycle": 4, "operation": "patch", "patch_paths": ["test_visible.py"],
         "decision": "accepted"},
        {"cycle": 5, "operation": "source patch and full suite", "patch_paths": ["toy.py"],
         "tests": record["final"]},
    ]
    return {
        "exercise": 7, "level": "analyze", "stars": 2,
        "record_type": "observed run with one planted decision error for analysis",
        "task": "Identify the invalid event, explain its effect on acceptance, and repair the rule.",
        "allowed_patch_paths": ["toy.py"], "patch_budget": record["budget"], "events": events,
    }


def print_exercise_key():
    with TemporaryDirectory(prefix="harness-key-") as directory:
        record = run_demo(Path(directory))
    print("Exercise 7: cycle 4 test-editing patch must be refused; "
          f"{record['refused_changes']} files changed in the actual run.")
    print(f"Owner-held acceptance: {record['fail_to_pass']}/{len(record['held_baseline']['failed'])} "
          f"fail-to-pass; {record['keep_passing']}/{len(record['held_baseline']['passed'])} "
          f"keep-passing; agent held-test reads {record['agent_held_reads']}.")


# region book:fnd.agent-frameworks.harness-01 output
if __name__ == "__main__":
    from pathlib import Path
    from tempfile import TemporaryDirectory

    from demos.harness_demo import print_demo, run_demo

    with TemporaryDirectory(prefix="harness-demo-") as directory:
        print_demo(run_demo(Path(directory)))
# endregion book:fnd.agent-frameworks.harness-01
