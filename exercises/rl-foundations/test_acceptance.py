"""Exercise 6 validates the reader's record; semantic citation review remains human."""

import importlib.util
import os
from pathlib import Path

import pytest
from acceptance import check_submission

from refund_agent.learning_case import LearningCase, threshold_case


def test_submitted_json_round_trip_citations_verdict_render_and_authority():
    source = Path(os.environ.get("RL_FOUNDATIONS_SOLUTION", Path(__file__).with_name("starter.py")))
    spec = importlib.util.spec_from_file_location("learning_case_submission", source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    case, rendered = module.submission()
    check_submission(case, rendered)


@pytest.mark.parametrize(
    "locator",
    ["companion/refund_agent/no-such-file.py:1", "companion/refund_agent/learning_case.py:99999"],
)
def test_structural_checker_rejects_unresolvable_sources(locator):
    fields = threshold_case().model_dump()
    fields["rows"][0]["source"] = locator
    case = LearningCase.model_validate(fields)
    with pytest.raises((AssertionError, FileNotFoundError)):
        check_submission(fields, case.render())
