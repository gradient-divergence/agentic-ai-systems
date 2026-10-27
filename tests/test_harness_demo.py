"""Chapter 28's coding harness: real workers, immutable tests and owner acceptance."""

import json
import sys
from pathlib import Path

import pytest

from demos import regions as _regions

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "demos/harness_demo.py"
EXERCISE_FIXTURE = ROOT / "exercises/agent-frameworks/harness_07.json"


def demo():
    assert SOURCE.is_file(), "the chapter's coding harness is missing"
    from demos import harness_demo

    return harness_demo


def test_crashed_worker_resumes_from_files_then_passes_owner_tests(tmp_path):
    h = demo()
    record = h.run_demo(tmp_path)
    assert record["crash_status"] == 23
    assert record["checkpoint"]["cycle"] == 3
    assert record["checkpoint"]["observations"][-1]["passed"] == [
        "test_mean_whole", "test_slug_empty", "test_slug_words"
    ]
    assert record["checkpoint"]["observations"][-1]["failed"] == ["test_mean_fraction"]
    assert record["resumed"]["failed"] == ["test_mean_fraction"]
    assert record["refused"] and record["refused_changes"] == 0
    assert len(record["final"]["passed"]) == 4 and not record["final"]["failed"]
    assert len(record["held_baseline"]["passed"]) == 2
    assert len(record["held_baseline"]["failed"]) == 4
    assert len(record["held_final"]["passed"]) == 6 and not record["held_final"]["failed"]
    assert record["fail_to_pass"] == 4 and record["keep_passing"] == 2
    assert record["accepted"] and record["attempts"] == record["budget"] == 3
    assert set(record["agent_reads"]) == {"toy.py", "test_visible.py", "progress.json"}
    assert not (tmp_path / "workspace/test_held.py").exists()
    assert (tmp_path / "workspace/test_visible.py").read_text() == h.VISIBLE_TESTS
    assert (tmp_path / "original/toy.py").read_text() == h.TOY_SOURCE


@pytest.mark.parametrize("path", ["test_visible.py", "progress.json", "../toy.py", "/tmp/toy.py"])
def test_patch_allowlist_refuses_the_whole_batch_before_writing(tmp_path, path):
    h = demo()
    workspace = h.make_workspace(tmp_path)
    before = (workspace / "toy.py").read_bytes()
    with pytest.raises(PermissionError):
        h.apply_patch(workspace, {"toy.py": "broken", path: "tampered"})
    assert (workspace / "toy.py").read_bytes() == before
    assert (workspace / "test_visible.py").read_text() == h.VISIBLE_TESTS


def test_patch_refuses_a_source_symlink(tmp_path):
    h = demo()
    workspace = h.make_workspace(tmp_path)
    target = tmp_path / "outside.py"
    target.write_text("owner data")
    (workspace / "toy.py").unlink()
    (workspace / "toy.py").symlink_to(target)
    with pytest.raises(PermissionError):
        h.apply_patch(workspace, {"toy.py": "tampered"})
    assert target.read_text() == "owner data"


def test_resume_restores_the_snapshot_and_retests_before_returning(tmp_path):
    h = demo()
    workspace = h.make_workspace(tmp_path)
    observed = h.run_tests(workspace)
    h.commit_checkpoint(workspace, [observed], attempts=0, reads=[], cycle=3)
    (workspace / "toy.py").write_text("raise RuntimeError('uncommitted change')")
    checkpoint, fresh = h.resume(workspace, h.restore_checkpoint, h.run_tests)
    assert checkpoint["cycle"] == 3
    assert (workspace / "toy.py").read_text() == h.TOY_SOURCE
    assert set(fresh["failed"]) == {"test_mean_fraction", "test_slug_words"}
    assert fresh == observed


def test_resume_uses_fresh_results_instead_of_checkpoint_claims(tmp_path):
    h = demo()
    workspace = h.make_workspace(tmp_path)
    stale = {"passed": ["test_slug_words"], "failed": []}
    h.commit_checkpoint(workspace, [stale], attempts=0, reads=[], cycle=3)
    _, observed = h.resume(workspace, h.restore_checkpoint, h.run_tests)
    assert "test_slug_words" in observed["failed"]
    assert "test_mean_fraction" in observed["failed"]


def test_corrupt_checkpoint_is_refused(tmp_path):
    h = demo()
    workspace = h.make_workspace(tmp_path)
    commit = h.commit_checkpoint(workspace, [h.run_tests(workspace)], 0, [], 3)
    path = workspace / ".checkpoints" / f"{commit}.json"
    path.write_text(path.read_text().replace("test_mean_fraction", "changed_test"))
    with pytest.raises(ValueError, match="checkpoint"):
        h.restore_checkpoint(workspace)


def test_patch_budget_survives_a_restart(tmp_path):
    h = demo()
    workspace = h.make_workspace(tmp_path)
    h.commit_checkpoint(workspace, [h.run_tests(workspace)], attempts=3, reads=[], cycle=3)
    h.resume(workspace, h.restore_checkpoint, h.run_tests)
    agent = h.ScriptedAgent(workspace)
    with pytest.raises(RuntimeError, match="budget"):
        agent.propose("mean")
    assert (workspace / "toy.py").read_text() == h.TOY_SOURCE


def test_owner_refuses_a_visible_only_repair(tmp_path):
    h = demo()
    workspace = h.make_workspace(tmp_path)
    agent = h.ScriptedAgent(workspace)
    h.apply_patch(workspace, agent.propose("slug"))
    # This passes visible fractional means but misses the owner's empty case.
    source = (workspace / "toy.py").read_text().replace("// len(values)", "/ len(values)")
    h.apply_patch(workspace, {"toy.py": source})
    assert not h.run_tests(workspace)["failed"]
    verdict = h.owner_acceptance(tmp_path / "original", workspace, tmp_path / "owner")
    assert not verdict["accepted"]
    assert verdict["held_final"]["failed"] == ["test_mean_empty"]


def test_exercise_fixture_does_not_reveal_the_answer():
    exercise = json.loads(EXERCISE_FIXTURE.read_text())
    assert set(exercise) == {
        "allowed_patch_paths", "events", "exercise", "level", "patch_budget",
        "record_type", "stars", "task",
    }, "student fixture must contain only run data and the generic exercise prompt"
    event_fields = {
        "commit", "cycle", "decision", "exit_status", "operation", "patch_paths", "tests",
    }
    for event in exercise["events"]:
        assert set(event) <= event_fields, "event metadata must not identify the answer"
        if "tests" in event:
            assert set(event["tests"]) == {"passed", "failed"}
    assert exercise["record_type"] == "observed run with one planted decision error for analysis"
    assert exercise["task"] == (
        "Identify the invalid event, explain its effect on acceptance, and repair the rule."
    )


def test_exercise_has_one_derived_planted_error(tmp_path):
    h = demo()
    record = h.run_demo(tmp_path)
    exercise = h.exercise_record(record)
    assert json.loads(EXERCISE_FIXTURE.read_text()) == exercise
    actual = exercise["events"][6]
    assert actual["cycle"] == 4 and actual["patch_paths"] == ["test_visible.py"]
    assert actual["decision"] == "accepted"
    assert record["refused"] and record["refused_changes"] == 0


def test_regions_are_independent_and_captured_deterministically(tmp_path):
    h = demo()
    tools = vars(_regions)
    snippets = {s.snippet_id: s for s in tools["extract_snippets_from_text"](
        SOURCE.read_text(), SOURCE
    )}
    assert set(snippets) == {
        "fnd.agent-frameworks.harness-checks-01", "fnd.agent-frameworks.harness-01"
    }
    for snippet in snippets.values():
        assert snippet.output and len(snippet.content.splitlines()) <= 15
        assert max(map(len, snippet.content.splitlines())) <= 79
        namespace = {"__name__": "__main__"}
        exec(compile(snippet.content, snippet.snippet_id, "exec"), namespace)
        if "apply_patch" in namespace:
            workspace = h.make_workspace(tmp_path)
            with pytest.raises(PermissionError):
                namespace["apply_patch"](workspace, {"test_visible.py": "tampered"})
            _, fresh = namespace["resume"](workspace, lambda _: {}, h.run_tests)
            assert "test_slug_words" in fresh["failed"]
    first = tools["capture_outputs"](snippets, sys.executable)
    assert first == tools["capture_outputs"](snippets, sys.executable)
    assert len(first["fnd.agent-frameworks.harness-01"].splitlines()) == 15
    for key, output in first.items():
        assert (ROOT / f"book_outputs/_{key}.txt").read_text() == output
