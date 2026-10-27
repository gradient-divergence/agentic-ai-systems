"""Chapter 29's printed regions must run with only their own setup."""

import ast
import contextlib
import io
import json
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from demos import regions as _regions

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "demos/agent_evaluation_release_demo.py"
EXTRACTOR = vars(_regions)
PREFIX = "fnd.agent-evaluation."
NAMES = ("suite-01", "release-01", "trials-01", "judge-01", "shadow-01", "keys-01", "cli-01")
BASE = "668a3b4e"


@pytest.fixture(autouse=True)
def temporary_io(tmp_path, monkeypatch):
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))


def snippets():
    assert DEMO.exists(), "Chapter 29 release demo is missing"
    sources = (DEMO, ROOT / "refund_agent/release/trials.py")
    return {
        s.snippet_id: s
        for source in sources
        for s in EXTRACTOR["extract_snippets_from_text"](source.read_text(), source)
    }


def test_printed_listings_run_in_chapter_order_on_their_own():
    found = snippets()
    namespace = {"__name__": "__main__"}
    with contextlib.redirect_stdout(io.StringIO()) as printed:
        for name in ("suite-01", "trials-01", "shadow-01"):
            exec(compile(found[PREFIX + name].content, str(DEMO), "exec"), namespace)
    text = " ".join(printed.getvalue().split())
    assert "must-pass" in text
    assert "pass@3" in text
    assert "generated labels" in text


@pytest.fixture(scope="module")
def capture_directory(tmp_path_factory):
    return tmp_path_factory.mktemp("chapter29-capture")


@pytest.fixture(scope="module")
def captured(capture_directory):
    from demos.agent_evaluation_capture import capture_chapter

    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("TMPDIR", str(capture_directory))
        patch.setattr(tempfile, "tempdir", str(capture_directory))
        return capture_chapter(capture_directory)


@pytest.mark.parametrize("name", NAMES)
def test_every_stored_output_regenerates_byte_for_byte(name, captured):
    path = ROOT / "book_outputs" / f"_{PREFIX}{name}.txt"
    assert captured[PREFIX + name].encode() == path.read_bytes()
    assert "companion/" not in captured[PREFIX + name]


def test_import_is_silent():
    result = subprocess.run(
        [sys.executable, "-c", "import demos.agent_evaluation_release_demo"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0 and not result.stdout and not result.stderr


def test_judge_prints_wilson_uncertainty(capsys):
    from demos.agent_evaluation_release_demo import print_judge

    print_judge()
    assert "miss rate 95% Wilson interval: [0.1455, 0.5190]" in capsys.readouterr().out


def test_shadow_prints_disagreement_kinds_and_verifier_counts(capsys):
    from demos.agent_evaluation_release_demo import print_shadow

    path = ROOT / "demos/data/agent_evaluation_shadow.json"
    print_shadow({"shadow": json.loads(path.read_text())})
    text = capsys.readouterr().out
    assert "tier 3 denial referrals 36; stale-evidence referrals 8" in text
    assert "verifier catches 0; false objections 0" in text


def test_release_prints_the_gate_removed_refusal_effect(capsys):
    from demos.agent_evaluation_release_demo import print_release

    path = ROOT / "demos/data/agent_evaluation_release.json"
    print_release({"release": json.loads(path.read_text())})
    text = capsys.readouterr().out
    assert "gate-removed refusal-2-amount_mismatch: executed 999999 cents" in text


def test_printed_output_lines_fit_the_measure():
    for snippet in snippets().values():
        assert max(map(len, snippet.content.splitlines())) <= 79, snippet.snippet_id
    for name in NAMES:
        if name == "keys-01":  # Instructor keys are not printed in this chapter.
            continue
        path = ROOT / "book_outputs" / f"_{PREFIX}{name}.txt"
        assert max(map(len, path.read_text().splitlines())) <= 79, name


def test_release_rule_listing_has_five_ordered_callouts():
    path = ROOT / "refund_agent/release/rule.py"
    found = EXTRACTOR["extract_snippets_from_text"](path.read_text(), path)
    listing = next(s for s in found if s.snippet_id == PREFIX + "rule-01")
    assert list(listing.callouts.values()) == [1, 2, 3, 4, 5]
    assert max(map(len, listing.content.splitlines())) <= 79
    namespace = {"__name__": "__main__"}
    exec(compile(listing.content, str(path), "exec"), namespace)
    assert callable(namespace["evaluate_release"])
    assert callable(namespace["_release_blocks"])


def test_retirement_preserves_old_behavior_and_other_source_map_rows():
    old = ROOT / "demos/agent_evaluation_demo.py"
    original = subprocess.check_output(["git", "show", f"{BASE}:{old.relative_to(ROOT)}"])
    assert ast.dump(ast.parse(old.read_text())) == ast.dump(ast.parse(original))
    for name in ("core-01", "example-01"):
        assert PREFIX + name not in old.read_text()
        assert not (ROOT / "book_outputs" / f"_{PREFIX}{name}.txt").exists()
    path = ROOT / "docs/source-map/section_map.csv"
    original = subprocess.check_output(["git", "show", f"{BASE}:{path.relative_to(ROOT)}"])

    def unchanged(data):
        return [
            line for line in data.splitlines(keepends=True) if not line.startswith(b"CODE-002,")
        ]

    assert unchanged(path.read_bytes()) == unchanged(original)


def test_capture_writes_every_record_byte_for_byte(captured, capture_directory):
    for name in ("figures", "release", "shadow"):
        relative = Path("demos/data") / f"agent_evaluation_{name}.json"
        assert (capture_directory / relative).read_bytes() == (
            ROOT / relative
        ).read_bytes()
    for name in NAMES:
        relative = Path("book_outputs") / f"_{PREFIX}{name}.txt"
        assert (capture_directory / relative).read_bytes() == (
            ROOT / relative
        ).read_bytes()


def test_cli_capture_records_command_outcomes(captured):
    text = captured[PREFIX + "cli-01"]
    assert text.count("exit 0") == 3 and text.count("exit 1") == 3
    assert "release refused:" in text and "shadow unstarted:" in text
    assert "--verifier independent" in text and "--owner morgan" in text
    assert max(map(len, text.splitlines())) <= 79


def test_captured_suite_and_records_bind_current_suite(captured):
    from refund_agent.release.suite import build_suite

    suite = build_suite()
    assert f"manifest cases {len(suite.cases)}; fingerprint {suite.fingerprint}" in " ".join(
        captured[PREFIX + "suite-01"].split()
    )
    release = json.loads((ROOT / "demos/data/agent_evaluation_release.json").read_text())
    assert all(record["suite_fingerprint"] == suite.fingerprint for record in release["records"])
    shadow = json.loads((ROOT / "demos/data/agent_evaluation_shadow.json").read_text())
    assert shadow["suite_fingerprint"] == suite.fingerprint
    assert shadow["release_record_id"] == release["records"][0]["record_id"]


def test_exercise_coverage_removes_only_the_declared_cell():
    from refund_agent.release.suite import build_suite

    matrix = build_suite().coverage()
    assert matrix["approval-reuse"]["2"] == ["approval-reused-after-proposal-change"]
    matrix["approval-reuse"]["2"] = []
    path = ROOT / "exercises/agent-evaluation/coverage-06.json"
    assert json.loads(path.read_text()) == matrix


def test_practice_key_counts_come_from_the_screen_log(captured):
    path = ROOT / "exercises/agent-evaluation/shadow-practice.json"
    fixture = json.loads(path.read_text())
    rows = fixture["review_screen_log"]
    assert sum(row["requests"] for row in rows) == fixture["requests"]
    assert sum(row["agreed"] for row in rows) == fixture["reported_agreement"]
    key = captured[PREFIX + "keys-01"]
    for row in rows:
        fraction = f"{row['agreed']}/{row['requests']}"
        assert fraction + f"={row['agreed'] / row['requests']:.6f}" in key
    probabilities = [1] * 9 + [0.5]
    at = sum(1 - (1 - p) ** 4 for p in probabilities) / len(probabilities)
    every = sum(p**4 for p in probabilities) / len(probabilities)
    assert f"scenario pass@4 {at:.8f}, pass^4 {every:.8f}" in key


def test_exercise_signoff_follows_the_run_it_cites():
    record = json.loads((ROOT / "exercises/agent-evaluation/record-05.json").read_text())
    cited = next(r for r in record["runs"] if r["id"] == record["cited_run"])
    assert datetime.fromisoformat(record["signature"]["at"]) > datetime.fromisoformat(cited["at"])


def test_full_passing_release_admits_constructed_first_shadow_day():
    from demos.agent_evaluation_release_demo import build_experiment

    report = build_experiment()
    first, fault = report["release"]["records"][:2]
    assert first["decision"] == "pass" and not first["failed_cases"]
    assert set(fault["failed_cases"]) == {"continue-order", "continue-shipment", "continue-policy"}
    assert datetime.fromisoformat(first["signature"]["at"]) > datetime.fromisoformat(first["at"])
    shadow = report["shadow"]
    assert shadow["status"] == "started" and shadow["requests"] == 120
    assert shadow["ledger_unchanged"] and shadow["review_queue_unchanged"]
    assert all(r["person_saw_shadow"] is False for r in shadow["rows"])
    assert not shadow["shadow_to_canary"]["window_complete"]


def test_exercise_readme_withholds_the_analyze_answer():
    text = (ROOT / "exercises/agent-evaluation/README.md").read_text()
    assert "Exercise 5's claimed pass is invalid" not in text
    assert "cleared soft" not in text
    assert "Software and suite identities are distinct" in text
