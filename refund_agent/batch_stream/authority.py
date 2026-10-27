"""Chapter 31's fixed decision authority; describes later owners, builds none."""

from types import MappingProxyType

AUTHORITY = MappingProxyType(
    {
        "pay_absence": (
            "settled",
            "shipment_incomplete; rerun once final",
            "refund policy owner",
            "EventTimeTools",
        ),
        "return_window": (
            "delivery event time",
            "delivery_unknown",
            "refund policy owner",
            "rules.draft",
        ),
        "latest_status": (
            "fast",
            "revise on arrival",
            "support desk",
            "described",
        ),
        "pause_canary": (
            "fast; executed correction",
            "resume on settled, unspent budget",
            "release owner",
            "Deployment and Rollback",
        ),
        "clean_and_widen": (
            "settled",
            "pending; neither clean nor failed",
            "release owner",
            "Deployment and Rollback; widening later",
        ),
        "daily_count_labels": (
            "settled",
            "show provisional",
            "carrier integration / label policy owner",
            "day_counts / label policy",
        ),
    }
)
