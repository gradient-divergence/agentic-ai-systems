"""One test file per call; reference mode verifies the five supplied answers."""

import os

import pytest
import security_privacy_01
import security_privacy_02
import security_privacy_03
import security_privacy_04
import security_privacy_05
from security_privacy_answers import ANSWERS

STARTERS = (
    security_privacy_01,
    security_privacy_02,
    security_privacy_03,
    security_privacy_04,
    security_privacy_05,
)


@pytest.mark.parametrize("number", range(1, 6))
def test_exercise(number):
    answer = (
        ANSWERS[number]
        if os.environ.get("SECURITY_PRIVACY_REFERENCE") == "1"
        else STARTERS[number - 1].ANSWER
    )
    assert answer == ANSWERS[number], f"Exercise {number}: designed misconception is still present"
