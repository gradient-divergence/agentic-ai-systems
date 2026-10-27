"""Chapter 33: mlops-mle captured outputs and the runnable bundle listing."""

from demos.mlops_mle_demo_helpers import (
    experiment,
    print_collision,
    print_keys,
    print_records,
    print_requirements,
    print_rollback,
    print_week,
)

# isort: split
# region book:fnd.mlops-mle.bundle-01 output
from demos.mlops_mle_demo_helpers import (
    START,
    candidates,
    print_bundle,
    register_at,
)
from refund_agent.data.generate import _policies
from refund_agent.promotion.mlops_mle_bundle import from_candidate

register = register_at(START)
snapshot = register.snapshot(START)
suite, candidate = candidates(snapshot, [p.version for p in _policies()])
bundle = from_candidate(
    candidate,
    register_snapshot=snapshot,
    versions=[p.version for p in _policies()],
)
print_bundle(bundle)
# endregion book:fnd.mlops-mle.bundle-01

# region book:fnd.mlops-mle.requirements-01 output
print_requirements()
# endregion book:fnd.mlops-mle.requirements-01

result = experiment()

# region book:fnd.mlops-mle.week-01 output
print_week(result)
# endregion book:fnd.mlops-mle.week-01

# region book:fnd.mlops-mle.records-01 output
print_records(result)
# endregion book:fnd.mlops-mle.records-01

# region book:fnd.mlops-mle.rollback-01 output
print_rollback(result)
# endregion book:fnd.mlops-mle.rollback-01

# region book:fnd.mlops-mle.collision-01 output
print_collision()
# endregion book:fnd.mlops-mle.collision-01

# region book:fnd.mlops-mle.keys-01 output
print_keys()
# endregion book:fnd.mlops-mle.keys-01

if __name__ == "__main__":
    import argparse

    from demos.mlops_mle_demo_helpers import write_data

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-data", action="store_true")
    if parser.parse_args().write_data:
        write_data(result)
