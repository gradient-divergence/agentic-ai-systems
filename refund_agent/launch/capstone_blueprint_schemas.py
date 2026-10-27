"""Chapter 39's producing schemas for a registry's launch-record reader."""

from refund_agent.launch.capstone_blueprint_compare import Comparison
from refund_agent.launch.capstone_blueprint_tier3 import Tier3Record


def validate_owned_record(kind, payload, claims):
    if kind == "comparison":
        record = Comparison.model_validate(payload)
        if (
            record.launched_bundle != claims.bundle
            or not claims.label_dependent
            or record.last_payment_day != claims.last_payment_day
        ):
            raise ValueError(
                "comparison's bundle/label clock differs from indexed claims"
            )
    elif kind == "tier3":
        record = Tier3Record.model_validate(payload)
    else:
        raise ValueError("not a chapter 39 record kind")
    if record.verdict != claims.verdict or record.period != claims.period:
        raise ValueError(
            "owned record's verdict/period differs from indexed claims"
        )
