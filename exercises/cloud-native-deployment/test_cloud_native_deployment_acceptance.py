"""Acceptance checks import only the student answers and rollback reader."""

from cloud_native_deployment_rehearse_06 import run

# isort: split
import cloud_native_deployment_01
import cloud_native_deployment_02
import cloud_native_deployment_03
import cloud_native_deployment_04
import cloud_native_deployment_05
import cloud_native_deployment_target_06
import pytest
from cloud_native_deployment_checks import check

STARTERS = (cloud_native_deployment_01, cloud_native_deployment_02,
            cloud_native_deployment_03, cloud_native_deployment_04, cloud_native_deployment_05)


@pytest.mark.parametrize("number", range(1, 6))
def test_response(number):
    answer = STARTERS[number - 1].ANSWER
    assert check(number, answer), f"Exercise {number}: supplied answer violates its designed check"


def test_rollback_reader():
    reader = cloud_native_deployment_target_06.READER
    result = run(reader)
    assert result.unreadable_records == 0, "The rollback target refuses the writer's canary_id field"
    assert result.verdict == "pass" and all(result.checks.values())
    assert result.paid_twice == 0 and result.checks["receipts_kept"]
    assert result.checks["post_restart"]
