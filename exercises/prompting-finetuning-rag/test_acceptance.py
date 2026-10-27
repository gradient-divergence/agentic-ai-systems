"""Check a submitted build against the chapter's acceptance conditions."""

import os
import runpy
from pathlib import Path

import pytest

PATH = Path(os.environ.get("ADAPTATION_SOLUTION", Path(__file__).with_name("starter.py")))
SOLUTION = runpy.run_path(str(PATH))


@pytest.fixture(scope="module")
def result(tmp_path_factory):
    return SOLUTION["build"](tmp_path_factory.mktemp("exercise"), seed=7)


def test_constructed_policy_prompt_and_forbidden_adapter_drafts(result):
    from demos.ml_data_evaluation_demo import wilson_interval

    policy = result["policy"]
    assert policy.version == "2026-11-01" and policy.return_days == 21
    assert policy.effective_at.isoformat().startswith("2026-11-01")
    assert len(result["requests"]) == 20 and result["prompt_correct"] == 20
    assert all(r.at >= policy.effective_at for r in result["requests"])
    expected = {f"exercise-6-r-7-{i:02d}" for i in (2, 3, 4, 9, 12, 13, 14, 19)}
    assert set(result["forbidden_ids"]) == expected
    assert len(result["forbidden_ids"]) == len(expected)
    assert result["forbidden_interval"] == pytest.approx(
        wilson_interval(len(result["forbidden_ids"]), 20)
    )


def test_extended_suite_preserves_frozen_manifest_and_training(result):
    assert result["before_fingerprint"] != result["after_fingerprint"]
    assert result["before_manifest"] == result["after_manifest"]
    assert result["training_ids_before"] == result["training_ids_after"]
    assert not set(r.request_id for r in result["requests"]) & set(result["training_ids_after"])
    assert all(r.request_id in result["suite_request_ids"] for r in result["requests"])
    assert set(result["policy_change_ids"]) & {r.request_id for r in result["requests"]}
