"""Exercise acceptance: the untouched starter fails; the reference passes."""

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from refund_agent.promotion.mlops_mle_bundle import VersionBundle
from refund_agent.promotion.mlops_mle_diff import diff
from refund_agent.promotion.mlops_mle_record import PromotionRecord
from refund_agent.promotion.mlops_mle_requirements import REQUIREMENTS, required

ROOT = Path(__file__).parent
TABLE = Path(os.environ.get("MLOPS_MLE_SOLUTION", ROOT / "requirements_06.py"))


def run(table, candidate):
    return subprocess.run(
        [
            sys.executable,
            str(ROOT / "mlops_mle_check_06.py"),
            "--requirements",
            str(table),
            "--candidate",
            str(ROOT / candidate),
        ],
        capture_output=True,
        text=True,
        check=False,
    )


def module(path):
    spec = importlib.util.spec_from_file_location("mlops_mle_acceptance", path)
    assert spec is not None and spec.loader is not None
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def test_runtime_change_is_held():
    result = run(TABLE, "candidate_06.json")
    assert result.returncode == 1, result.stdout + result.stderr
    assert json.loads(result.stdout)["unmet"] == ["runtime_swap"]


def test_b1_is_admissible_under_the_stipulated_prerequisites():
    result = run(TABLE, "b1_06.json")
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["unmet"] == []


def test_submitted_table_keeps_every_row_and_matches_each_component():
    submitted = module(TABLE)
    assert set(submitted.REQUIREMENTS) == set(REQUIREMENTS)
    assert set(submitted.required([])) == set(required([]))
    cases = [
        [{"component": row.component, "item": item if item != "*" else "new-field"}]
        for row in REQUIREMENTS
        for item in row.items
    ]
    cases += [
        [{"component": "unknown", "item": "unknown"}],
        [{"component": "tools", "item": "adapter"}] * 2,
        [{"component": "model", "item": "risk_score"}, {"component": "index", "item": "encoder"}],
    ]
    for changes in cases:
        try:
            expected = required(changes)
        except ValueError:
            with pytest.raises(ValueError):
                submitted.required(changes)
            continue
        actual = submitted.required(changes)
        assert set(actual) == set(expected), changes
        assert len(actual) == len(set(actual))


def test_planted_record_omits_an_actual_index_change():
    candidate = VersionBundle.from_record(json.loads((ROOT / "candidate-05.json").read_text()))
    incumbent = VersionBundle.from_record(json.loads((ROOT / "incumbent-05.json").read_text()))
    record = json.loads((ROOT / "record-05.json").read_text())
    assert all(record["requirements"].get(name) for name in required(record["changes"]))
    assert record["changes"] == [{"component": "tools", "item": "adapter"}]
    assert record["candidate_fingerprint"] == candidate.fingerprint
    assert record["rollback_target"] == incumbent.fingerprint
    assert record["refs"]["suite"] != incumbent.refs["suite"]
    assert PromotionRecord.from_record(record).signed_by == "morgan"
    assert any(row["component"] == "index" for row in diff(incumbent, candidate))
    assert not any(row["component"] == "index" for row in record["changes"])
    assert "frozen_questions" not in record["requirements"]
    assert set(required(diff(incumbent, candidate))) - set(record["requirements"]) == {
        "frozen_questions",
        "dated_recall",
    }
