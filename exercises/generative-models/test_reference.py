"""Reference solution regression checks; run explicitly with test_acceptance.py."""

import json
import runpy
from pathlib import Path

import numpy as np
import pytest

from refund_agent.data.generate import FAILURES
from refund_agent.scenarios.model import ScenarioModel
from refund_agent.split import GoldenSet

DIRECTORY = Path(__file__).parent
REFERENCE = runpy.run_path(str(DIRECTORY / "reference.py"))
AS_OF = REFERENCE["AS_OF"]


@pytest.fixture
def no_golden(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("the reference consulted the frozen set")

    monkeypatch.setattr(GoldenSet, "consult", forbidden)


def test_failure_distribution_uses_training_none_count_and_add_one_smoothing(no_golden):
    model = REFERENCE["FailureModel"].fit()
    n = len(model.training_tokens)
    assert model.counts[6] == {(): [n, 0, 0, 0, 0, 0]}
    np.testing.assert_allclose(model.conditional(6, {}), np.array([n + 1, 1, 1, 1, 1, 1]) / (n + 6))
    assert tuple(REFERENCE["FAILURE_VALUES"]) == ("none", *FAILURES)


@pytest.mark.parametrize("tau", [1, 2])
def test_six_token_draws_and_likelihood_are_preserved(tau, no_golden):
    original, extended = ScenarioModel.fit(), REFERENCE["FailureModel"].fit()
    cases = original.draw(np.random.default_rng(7), tau, 200, AS_OF)
    draws = extended.draw(np.random.default_rng(7), tau, 200, AS_OF)
    assert tuple(tokens[:6] for tokens in draws) == cases
    for tokens in draws[:10]:
        p = extended.conditional(6, {})[REFERENCE["FAILURE_VALUES"].index(tokens[6])]
        assert extended.log_prob(tokens) == pytest.approx(original.log_prob(tokens[:6]) + np.log(p))


@pytest.mark.parametrize("tau", [1, 2])
def test_saved_manifests_and_receipt_match_the_reproducible_runs(tau, no_golden):
    suite = REFERENCE["build_suite"](AS_OF, tau, 200, 7)
    assert json.loads((DIRECTORY / f"manifest-tau{tau}.json").read_text()) == suite.manifest
    recorded = json.loads((DIRECTORY / "run-07.json").read_text())
    run = next(row for row in recorded["runs"] if row["tau"] == tau)
    assert run == REFERENCE["check_report"]([suite])["runs"][0]
    assert suite.to_json() != REFERENCE["build_suite"](AS_OF, tau, 200, 8).to_json()


def test_changed_manifest_is_rejected_and_existing_journal_is_untouched(
    no_golden, tmp_path, monkeypatch
):
    journal = tmp_path / "golden-1.access.jsonl"
    prior = b'{"entry":1,"reader":"earlier-chapter","set":"golden-1"}\n'
    journal.write_bytes(prior)
    monkeypatch.setenv("BOOK_GOLDEN_DIRECTORY", str(tmp_path))
    suite = REFERENCE["build_suite"](AS_OF, 2, 200, 7)
    for field, value in (("tau", 1), ("model_sha256", "wrong"), ("golden_reads", 1)):
        with pytest.raises(ValueError, match="manifest"):
            REFERENCE["regenerate"](dict(suite.manifest, **{field: value}))
    assert journal.read_bytes() == prior
    assert list(tmp_path.iterdir()) == [journal]


@pytest.mark.parametrize("tau, n", [(0, 200), (-1, 200), (float("inf"), 200), (1, -1), (1, 1.5)])
def test_invalid_sampling_parameters_are_rejected(tau, n, no_golden):
    with pytest.raises(ValueError):
        REFERENCE["build_suite"](AS_OF, tau, n, 7)
