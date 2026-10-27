"""Run the real rollback/restart library with this exercise's reader."""

from datetime import datetime, timezone

from demos.cloud_native_deployment_demo import candidates, plan_for
from refund_agent.canary.cloud_native_deployment_rehearsal import run_rehearsal
from refund_agent.data.live import generate_live


def run(reader):
    _, writer = candidates(planted_field=True)
    _, target = candidates(reader=reader)
    start = datetime(2026, 10, 15, tzinfo=timezone.utc)
    plan = plan_for(writer, target, "exercise-06-rehearsal-only", start,
                    identifier="cloud-native-deployment-exercise-06")
    return run_rehearsal(plan, generate_live(7, start, 14, 233), 14,
                         ["forbidden_payment"], "4h", target, candidate=writer)


if __name__ == "__main__":
    from cloud_native_deployment_target_06 import READER

    print(run(READER).model_dump_json(indent=2))
