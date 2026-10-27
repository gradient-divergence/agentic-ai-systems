"""Exercise 7 acceptance checks; the unfinished starter fails deliberately.

Run with the companion Python and its PYTHONPATH, explicitly selecting this file.
Set GENERATIVE_MODELS_SOLUTION to a submission's Python file. These checks grant
no release authority and do not read the golden set. test_scenarios.py must also
pass unchanged after the extension.
"""

import importlib.util
import os
from datetime import datetime, timezone
from pathlib import Path

import pytest

from refund_agent.data.generate import FAILURES
from refund_agent.scenarios.checks import policy_label
from refund_agent.scenarios.model import ScenarioModel, training_data
from refund_agent.split import GoldenSet

PATH = Path(os.environ.get("GENERATIVE_MODELS_SOLUTION", Path(__file__).with_name("starter.py")))
SPEC = importlib.util.spec_from_file_location("generative_models_solution", PATH)
SOLUTION = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SOLUTION)
AS_OF = datetime(2026, 9, 15, tzinfo=timezone.utc)


@pytest.fixture(params=[1, 2])
def tau(request):
    return request.param


@pytest.fixture
def suite(tau, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("the exercise read the frozen golden set")

    monkeypatch.setattr(GoldenSet, "consult", forbidden)
    return SOLUTION.build_suite(AS_OF, tau, 200, 7)


def test_seventh_token_is_materialized_and_every_kept_label_is_independent(suite):
    data = training_data()
    for row in suite.scenarios:
        assert len(row.tokens) == 7
        failure = row.tokens[6]
        assert failure in ("none", *FAILURES)
        assert row.records.request.failure == (None if failure == "none" else failure)
        assert row.label == policy_label(row.records, data)
        if failure != "none":
            assert row.label.outcome.value == "refer"
            assert row.label.reason == failure


def test_failure_draws_exist_and_counts_and_training_fingerprint_are_preserved(suite):
    original = ScenarioModel.fit()
    assert suite.model.counts[:6] == original.counts
    assert suite.model.training_manifest == original.training_manifest
    assert suite.manifest["training_sha256"] == original.training_manifest["sha256"]
    assert suite.manifest["golden_reads"] == 0
    assert suite.manifest["human_review"]["status"] == "pending"
    # Both runs need to exercise at least one failure, rather than append none only.
    assert any(row.records.request.failure is not None for row in suite.scenarios)


def test_extended_suite_regenerates_byte_for_byte(suite, tau, tmp_path):
    assert suite.manifest["tau"] == tau
    rebuilt = SOLUTION.regenerate(suite.manifest)
    assert rebuilt.to_json() == suite.to_json()
    assert suite.manifest["number_drawn"] == 200
    assert suite.manifest["as_of"] == AS_OF.isoformat()
    assert suite.manifest["policy_version"] == "2026-09-01"
    assert suite.manifest["seed"] == 7
    path = tmp_path / "suite.json"
    suite.write(path)
    assert path.read_text() == suite.to_json()


def test_regenerate_uses_the_manifest_temperature(suite, tau, monkeypatch):
    assert suite.manifest["tau"] == tau
    build = SOLUTION.build_suite
    calls = []

    def tracked(as_of, tau, n=200, seed=7):
        calls.append((as_of, tau, n, seed))
        return build(as_of, tau, n, seed)

    monkeypatch.setattr(SOLUTION, "build_suite", tracked)
    assert SOLUTION.regenerate(suite.manifest).to_json() == suite.to_json()
    assert calls == [(AS_OF, tau, 200, 7)]
