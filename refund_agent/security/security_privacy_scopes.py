"""The source owners' constructed scope table; no credential keys belong here."""

from types import MappingProxyType

from refund_agent.tools import TOOL_NAMES

SCOPES_VERSION = "scopes@1"
SCOPES = MappingProxyType(
    {
        "request": {
            "tools": tuple(TOOL_NAMES),
            "objects": "account orders, shipments, policy, own proposal/review/records",
            "granted_by": ("order team", "carrier integration team", "policy owner", "finance"),
        },
        "reviewer": {
            "tools": ("approval.intake",),
            "objects": "open reviews for the authenticated reviewer's role",
            "granted_by": ("decision_owner",),
        },
        "operator": {
            "tools": ("credential.revoke", "scope.revoke", "scope.reinstate"),
            "objects": "credentials and tool scopes",
            "granted_by": ("release_owner",),
        },
    }
)
