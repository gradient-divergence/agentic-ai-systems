"""Exercise 6's later writer. Keep the records it writes unchanged."""

from demos.cloud_native_deployment_demo import candidates


def candidate():
    return candidates(planted_field=True)[1]
