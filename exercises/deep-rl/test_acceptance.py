"""Exercise 6 target and report checks, including rejected nonfinite estimates."""

import copy
import importlib.util
import json
import os
from pathlib import Path

import pytest
from acceptance import check_report, check_target

from demos.deep_rl.dqn import EVALUATION_SEEDS, TRAINING_SEEDS


def report_fixture():
    return dict(
        training_seeds=list(TRAINING_SEEDS),
        evaluation_seeds=list(EVALUATION_SEEDS),
        settings=dict(
            budget=50000,
            gamma=0.99,
            widths=[4, 64, 64, 2],
            eta=0.001,
            batch_size=64,
            capacity=20000,
            refresh=100,
            epsilon_steps=10000,
        ),
        runs=[
            dict(
                seed=seed,
                original_returns=[10] * 10,
                double_returns=[11] * 10,
                original_starting_estimate=9.0,
                double_starting_estimate=10.0,
            )
            for seed in TRAINING_SEEDS
        ],
    )


@pytest.mark.parametrize("field", ["original_starting_estimate", "double_starting_estimate"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_starting_estimates_are_rejected(field, value):
    report = copy.deepcopy(report_fixture())
    report["runs"][0][field] = value
    with pytest.raises((AssertionError, ValueError)):
        check_report(report)


def test_submitted_target_and_completed_report():
    path = Path(os.environ.get("DEEP_RL_SOLUTION", Path(__file__).with_name("starter.py")))
    spec = importlib.util.spec_from_file_location("double_target_submission", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    check_target(module.double_target)
    report_path = Path(
        os.environ.get("DEEP_RL_REPORT", Path(__file__).with_name("reference_report.json"))
    )
    result = check_report(json.loads(report_path.read_text()))
    json.dumps(result, allow_nan=False)
