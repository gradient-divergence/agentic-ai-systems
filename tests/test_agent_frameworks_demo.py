"""Chapter-order listings and byte-identical runtime comparison outputs."""

import ast
import importlib
import inspect
import json
import os
import subprocess
import sys
from datetime import timedelta
from hashlib import sha256
from pathlib import Path

import pytest

from demos import regions as _regions

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "demos/agent_frameworks_demo.py"
TOOLS = vars(_regions)
PREFIX = "fnd.agent-frameworks."
OUTPUTS = ("swap-01", "swap-02", "resume-01", "keys-01", "replay-01")
EXERCISE = ROOT / "exercises/agent-frameworks"


@pytest.mark.parametrize("style", ["from runner_c import RunnerC", "import runner_c"])
def test_exercise_grader_accepts_both_correct_import_styles(style, tmp_path):
    runner = "RunnerC" if style.startswith("from ") else "runner_c.RunnerC"
    solution = tmp_path / "solution.py"
    solution.write_text(
        f"{style}\n"
        "from refund_agent.runtimes.adapter import AdapterSettings, RefundAdapter\n"
        "def mount(tools, model=None, settings=None):\n"
        "    return RefundAdapter(tools, model, settings=settings or AdapterSettings(), "
        f"runner_type={runner})\n"
    )
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            str(EXERCISE / "test_exercise_06.py"),
            "-p",
            "no:cacheprovider",
        ],
        env={**os.environ, "AGENT_FRAMEWORKS_SOLUTION": str(solution)},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "4 passed" in result.stdout


def test_exercise_grader_requires_an_explicit_solution():
    environment = dict(os.environ)
    environment.pop("AGENT_FRAMEWORKS_SOLUTION", None)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            str(EXERCISE / "test_exercise_06.py"),
            "-p",
            "no:cacheprovider",
            "-k",
            "matches_the_entire_suite",
        ],
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode != 0
    assert "set AGENT_FRAMEWORKS_SOLUTION" in result.stdout


def test_student_starter_fails_on_retry_semantics():
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            str(EXERCISE / "test_exercise_06.py"),
            "-p",
            "no:cacheprovider",
        ],
        env={**os.environ, "AGENT_FRAMEWORKS_SOLUTION": str(EXERCISE / "starter.py")},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 1, result.stdout + result.stderr
    assert "1 failed, 3 passed" in result.stdout
    assert 'result["different_cases"]' in result.stdout


def regions():
    assert DEMO.is_file(), "the chapter's runnable demo must exist"
    return TOOLS["extract_snippets_from_text"](DEMO.read_text(), DEMO)


def test_printed_listing_runs_alone_and_in_chapter_order():
    snippets = {s.snippet_id: s for s in regions()}
    namespace = {"__name__": "__main__"}
    for name in ("adapter-01",):
        exec(compile(snippets[PREFIX + name].content, name, "exec"), namespace)
    from refund_agent.data.generate import generate
    from refund_agent.tools import Tools

    data = generate(7, orders=40, requests=40)
    tools = Tools(data)
    adapter = namespace["mount"](tools)
    request = next(r for r in data.requests if r.request_id == "r-7-0007")
    trace = adapter.run(request)
    assert adapter.tools is tools
    assert trace.decision.executed and tools.payments[0].amount_cents == 2499


def test_regions_follow_the_outline():
    assert [s.snippet_id for s in regions()] == [PREFIX + n for n in ("adapter-01", *OUTPUTS)]


def test_callbacks_are_source_excerpts_without_behavior_changes():
    source = ROOT / "refund_agent/runtimes/adapter.py"
    snippets = TOOLS["extract_snippets_from_text"](source.read_text(), source)
    assert [s.snippet_id for s in snippets] == [PREFIX + f"callbacks-0{i}" for i in (1, 2, 3)]
    assert "def before_call" in snippets[0].content and "unexpected_tool" in snippets[0].content
    assert "refer_on_failure" in snippets[1].content and "self._refer" in snippets[1].content
    assert "def on_error" in snippets[2].content and "self._refer" in snippets[2].content
    assert sum(len(s.content.splitlines()) for s in snippets) <= 60
    for snippet in snippets:
        ast.parse(snippet.content)  # Each excerpt can be included as a Python listing.
    from refund_agent.runtimes.swap import build_record

    # Check the callbacks' effects across payments, refusals, faults and continuations.
    full = build_record(seed=7)["configurations"][-1]
    assert full["decision"] == "eligible for the shadow run"
    assert full["different_cases"] == full["money_differences"] == full["unauthorized_cents"] == 0
    assert all(not row["different_fields"] for row in full["cases"])


def test_cumulative_replay_keeps_memory_beside_both_runtime_compositions(capsys):
    from demos import agent_frameworks_demo as demo

    assert PREFIX + "replay-01" in {s.snippet_id for s in regions()}
    replay = demo.cumulative_replay()
    assert replay["request"].request_id == "r-7-0085"
    assert replay["memory"]["old_refusal"] == "beyond_notice"
    assert replay["memory"]["old_draft"].outcome.value == "deny"
    assert replay["memory"]["old_draft"].reason == "outside_return_window"
    assert replay["memory"]["fresh_draft"].outcome.value == "approve"
    assert replay["memory"]["fresh_draft"].policy_citation == "refund_cap_v4@2026-05-01#eligibility"
    assert replay["memory"]["assembly"].words == 30
    assert replay["memory"]["assembly"].facts["snapshot_at"] == replay["request"].at - timedelta(
        days=7
    )
    assert replay["memory"]["support"]["status"] == "supported"
    for run in replay["runs"].values():
        assert run["record"]["outcome"] == "approve" and run["record"]["tier"] == 1
        assert len(run["record"]["ledger_receipts"]) == 1
        assert len(run["record"]["audits"]) == 2
        assert run["record"]["ledger_receipts"][0]["amount_cents"] == 4999
    assert replay["different_fields"] == []
    from refund_agent.runtimes.adapter import RefundAdapter
    from refund_agent.verifier.independent import VerifiedTools

    second = replay["runs"]["second verified"]
    assert isinstance(second["runtime"], RefundAdapter)
    assert isinstance(second["runtime"].tools.inner, VerifiedTools)
    assert second["runtime"].tools.inner is second["tools"]
    for name in ("own verified", "second verified"):
        (verdict,) = replay["runs"][name]["verdicts"]
        assert verdict.answer == "agree" and verdict.objection_reason is None
    demo.print_replay(replay)
    text = capsys.readouterr().out
    assert "separate memory experiment beside the loop" in text
    assert "composition: VerifiedTools under RefundAdapter (full adapter)" in text
    assert "runtime fields differing: 0/10" in text


@pytest.fixture(scope="module")
def captured():
    snippets = {s.snippet_id: s for s in regions()}
    first = TOOLS["capture_outputs"](snippets, sys.executable)
    assert first == TOOLS["capture_outputs"](snippets, sys.executable)
    return first


@pytest.mark.parametrize("name", OUTPUTS)
def test_every_stored_output_regenerates_byte_for_byte(name, captured):
    path = ROOT / "book_outputs" / f"_{PREFIX}{name}.txt"
    committed = subprocess.check_output(["git", "show", f"HEAD:{path.relative_to(ROOT)}"], cwd=ROOT)
    fresh = captured[PREFIX + name].encode()
    assert fresh == path.read_bytes()
    if name == "keys-01":
        # The enrichment appends exercise 7; every earlier key byte is frozen.
        boundary = b"Exercise 7:"
        assert fresh.partition(boundary)[0] == committed.partition(boundary)[0]
        assert fresh.count(boundary) == 1
        assert len(fresh.partition(boundary)[2].splitlines()) == 2
    elif name in {"resume-01", "replay-01"}:
        assert all(len(line) <= 79 for line in fresh.splitlines())
        # Only field and audit-record continuations may differ from HEAD.
        assert fresh.replace(b";\n  ", b"; ").replace(b",\n    ", b", ") == (
            committed.replace(b";\n  ", b"; ").replace(b",\n    ", b", ")
        )
    else:
        assert fresh == committed


def current_source_fingerprint(*names, directory=None):
    directory = directory or ROOT / "refund_agent/runtimes"
    hashes = {
        Path(name).name: sha256((directory / name).read_bytes()).hexdigest() for name in names
    }
    return sha256(json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def test_swap_record_and_exercise_fixtures_regenerate(tmp_path, monkeypatch):
    demo = importlib.import_module("demos.agent_frameworks_demo")
    monkeypatch.setattr(demo, "HERE", tmp_path / "demos")
    monkeypatch.setattr(demo, "BOOK", tmp_path / "book_outputs")
    (tmp_path / "demos/data").mkdir(parents=True)
    demo.main(["--write-data"])
    for name in ("03", "05"):
        relative = f"demos/data/agent_frameworks_swap_{name}.json"
        assert (tmp_path / relative).read_bytes() == (ROOT / relative).read_bytes()
    path = "book_outputs/_fnd.agent-frameworks.swap-record.json"
    fresh = json.loads((tmp_path / path).read_text())
    stored = json.loads((ROOT / path).read_text())
    # Preserve the historical source hashes; verify current hashes and every other field.
    assert fresh["runtimes"][0]["fingerprint"] == current_source_fingerprint(
        "__init__.py", directory=ROOT / "refund_agent/agent"
    )
    assert fresh["runtimes"][1]["fingerprint"] == current_source_fingerprint("runner.py")
    assert fresh["adapter"]["fingerprint"] == current_source_fingerprint("adapter.py")
    assert fresh["suite"]["source_fingerprint"] == current_source_fingerprint(
        "swap.py", "faults.py"
    )
    assert fresh["agent_fingerprint"] == current_source_fingerprint(
        "agent/__init__.py", "tools.py", "contracts.py", directory=ROOT / "refund_agent"
    )
    from refund_agent.tools import Tools

    assert (
        fresh["tools_composition"]["fingerprint"]
        == sha256(inspect.getsource(Tools).encode()).hexdigest()
    )
    stored["runtimes"][0]["fingerprint"] = fresh["runtimes"][0]["fingerprint"]
    stored["runtimes"][1]["fingerprint"] = fresh["runtimes"][1]["fingerprint"]
    stored["adapter"]["fingerprint"] = fresh["adapter"]["fingerprint"]
    stored["suite"]["source_fingerprint"] = fresh["suite"]["source_fingerprint"]
    stored["agent_fingerprint"] = fresh["agent_fingerprint"]
    stored["tools_composition"]["fingerprint"] = fresh["tools_composition"]["fingerprint"]
    assert (tmp_path / path).read_text() == json.dumps(stored, indent=2, sort_keys=True) + "\n"
    fixture = json.loads((tmp_path / "demos/data/agent_frameworks_swap_03.json").read_text())
    assert len(fixture["cases"]) == 6
    assert {r["kind"] for r in fixture["cases"]} >= {"public", "refusal", "fault"}
    assert any(r["different_fields"] for r in fixture["cases"])
