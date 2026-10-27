"""Install release resource auditing before any component module can read inputs."""

import sys

from refund_agent.release import agent_evaluation_resources as _release_resources

sys.addaudithook(_release_resources._audit_resource_read)
_release_resources.install_import_guards()
