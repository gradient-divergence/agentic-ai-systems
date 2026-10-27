import json
import runpy
from pathlib import Path

from demos import regions as _regions
from demos.capstone_blueprint_capture import capture_chapter

ROOT = Path(__file__).resolve().parents[1]


def test_default_keys_omit_the_diagnostic_exercise(monkeypatch, capsys):
    from demos import capstone_blueprint_demo as demo

    monkeypatch.delenv("CAPSTONE_BLUEPRINT_USE_REFERENCE", raising=False)
    demo.print_keys()
    assert not any(line.startswith("5:") for line in capsys.readouterr().out.splitlines())


def test_exercise_fixture_omits_the_answer_and_matches_shipped_data(tmp_path):
    from refund_agent.launch.capstone_blueprint_exercises import (
        decision_05,
        write_exercises,
    )

    generated = decision_05()
    shipped = json.loads(
        (ROOT / "exercises/capstone-blueprint/decision_05.json").read_text()
    )
    write_exercises(tmp_path)
    written = json.loads((tmp_path / "decision_05.json").read_text())
    assert generated == shipped == written
    for fixture in (generated, shipped, written):
        assert "consequence" not in fixture


def test_tier3_reversal_prints_the_named_manual_route_within_79_columns(
    monkeypatch, capsys
):
    from demos import capstone_blueprint_demo as demo

    monkeypatch.setattr(demo, "saved", lambda _: None)
    demo.print_decision()
    lines = capsys.readouterr().out.splitlines()
    start = next(i for i, line in enumerate(lines)
                 if line.startswith("tier3_without_named_approver |"))
    end = next(i for i in range(start + 1, len(lines))
               if lines[i].startswith("launch_gate_held |"))
    row = lines[start:end]
    assert max(map(len, row)) <= 79
    text = " ".join(line.strip() for line in row)
    assert "a person decides up to $500" in text
    assert "named approver decides every request over $500, denials included" in text
    assert "release_owner | launch.scan:tier3_without_named_approver" in text


def test_all_captures_are_deterministic_and_state_missing_evidence(tmp_path):
    first = capture_chapter(tmp_path / "first", run_dir=tmp_path / "run")
    second = capture_chapter(tmp_path / "second", run_dir=tmp_path / "run")
    assert first == second and len(first) == 8
    for name, text in first.items():
        committed = ROOT / "book_outputs" / f"_{name}.txt"
        assert (
            tmp_path / "first" / f"_{name}.txt"
        ).read_bytes() == committed.read_bytes()
        if name.endswith(("gate-01", "compare-01", "tier3-01", "decision-01")):
            assert "unavailable" in text
        assert not any(
            line.startswith("companion/") for line in text.splitlines()
        )
    summary = json.loads(
        (tmp_path / "run/capstone-blueprint-run.json").read_text()
    )
    assert summary["gate_day"] is None


def test_refused_output_is_the_full_earlier_bundle_gate(
    tmp_path, monkeypatch, capsys
):
    from demos import capstone_blueprint_demo as demo

    monkeypatch.setenv("CAPSTONE_BLUEPRINT_RUN", str(tmp_path / "run"))
    demo.run_state.cache_clear()
    demo.print_refused()
    label, body = capsys.readouterr().out.split("\n", 1)
    assert label == "Constructed earlier-bundle rehearsal packet:"
    verdict = json.loads(body)
    assert verdict["verdict"] == "hold" and len(verdict["rows"]) == 7
    assert [
        (f["row"], f["kind"], f["check"]) for f in verdict["failures"]
    ] == [
        ("7", "rehearsal", "consistent"),
    ]
    assert all(row["checks"]["complete"] for row in verdict["rows"])
    assert "earlier" in verdict["failures"][0]["reason"]
    demo.run_state.cache_clear()


def test_printed_listing_runs_alone_and_has_no_wide_line(tmp_path):
    extractor = vars(_regions)
    path = ROOT / "refund_agent/launch/capstone_blueprint_gate.py"
    regions = extractor["extract_snippets_from_text"](path.read_text(), path)
    listing = next(
        s for s in regions if s.snippet_id == "fnd.capstone-blueprint.check-01"
    )
    code = listing.content
    assert max(map(len, code.splitlines())) <= 79
    assert len(code.splitlines()) <= 25
    namespace = {}
    exec(compile(code, str(path), "exec"), namespace)
    fixtures = runpy.run_path(
        str(
            ROOT
            / "tests/refund_agent/capstone_blueprint_fixtures.py"
        )
    )
    req, packet, registry = fixtures["packet_fixture"](
        tmp_path / "listing-run"
    )
    from refund_agent.launch.capstone_blueprint_scan import LaunchLogs

    assert (
        namespace["check"](
            packet, req, registry, LaunchLogs(), fixtures["OWNERS"], 64
        ).verdict
        == "launch"
    )


def test_importing_demo_writes_and_prints_nothing(
    tmp_path, capsys, monkeypatch
):
    monkeypatch.setenv("CAPSTONE_BLUEPRINT_RUN", str(tmp_path / "untouched"))
    runpy.run_path(
        str(ROOT / "demos/capstone_blueprint_demo.py"),
        run_name="import_check",
    )
    assert not (tmp_path / "untouched").exists()
    assert capsys.readouterr().out == ""
