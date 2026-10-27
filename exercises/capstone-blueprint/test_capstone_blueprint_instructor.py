"""Reference solution diagnosis; opt in with CAPSTONE_BLUEPRINT_USE_REFERENCE=1."""

import json
import os
import runpy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
DIRECTORY = Path(__file__).resolve().parent
pytestmark = pytest.mark.skipif(
    os.environ.get("CAPSTONE_BLUEPRINT_USE_REFERENCE") != "1",
    reason="reference checks require CAPSTONE_BLUEPRINT_USE_REFERENCE=1",
)


@pytest.mark.parametrize(
    "path,disclosure",
    [
        ("tests/test_capstone_blueprint_demo.py", "B5"),
        ("exercises/capstone-blueprint/README.md", "B5"),
        ("exercises/capstone-blueprint/capstone_blueprint_starters.py", "rollback target must match"),
        ("exercises/capstone-blueprint/test_exercise_06.py", "def test_exercise_05("),
        ("book_outputs/_fnd.capstone-blueprint.keys-01.txt", "5:"),
    ],
)
def test_public_material_does_not_supply_the_diagnosis(path, disclosure):
    text = (ROOT / path).read_text()
    assert disclosure not in text


def test_exercise_05_has_one_planted_error(tmp_path):
    from refund_agent.launch.capstone_blueprint_exercises import decision_05, write_exercises

    reference = runpy.run_path(str(DIRECTORY / "capstone_blueprint_reference.py"))
    write_exercises(tmp_path)
    for fixture in (
        decision_05(),
        json.loads((DIRECTORY / "decision_05.json").read_text()),
        json.loads((tmp_path / "decision_05.json").read_text()),
    ):
        assert "consequence" not in fixture
        assert fixture["rehearsal"]["rollback_target"] == "B5"
        assert fixture["promotion"]["rollback_target"] == "B6"
        assert reference["exercise_05"](fixture) == {"verdict": "hold", "target": "B6"}
        fixture["rehearsal"]["rollback_target"] = fixture["promotion"]["rollback_target"]
        assert reference["exercise_05"](fixture) == {"verdict": "launch", "target": "B6"}


def test_reference_opt_in_prints_the_diagnostic_key(monkeypatch, capsys):
    from demos import capstone_blueprint_demo as demo

    monkeypatch.setenv("CAPSTONE_BLUEPRINT_USE_REFERENCE", "1")
    demo.print_keys()
    assert "5: hold; rehearse B7 with B6, not B5" in capsys.readouterr().out
