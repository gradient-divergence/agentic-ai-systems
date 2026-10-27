"""Exercise acceptance checks; not operational decision code."""

import json
from fractions import Fraction
from pathlib import Path

from refund_agent.canary.cloud_native_deployment_exposure import (
    clean_hours,
    further_requests,
    loss_cents,
)
from refund_agent.canary.cloud_native_deployment_rehearsal import paid_twice


def check(number, answer):
    if number == 1:
        return answer == {"max_cents": 4999, "daily_payments": 75000 // 4999,
                          "signer": "decision_owner", "starter": "release_owner", "resumer": "release_owner"}
    if number == 2:
        return answer == {"next_path": "stop", "scripted_model": "rollback", "receipt": "neither",
                          "notice": "neither", "policy_in_force": "neither", "intent": "neither"}
    if number == 3:
        share, rate = Fraction(1, 20), Fraction(5)
        return answer == {"further_6h": further_requests(share, rate, Fraction(6)),
                          "loss_6h": loss_cents(share, rate, Fraction(6)),
                          "further_24h": further_requests(share, rate, Fraction(24)),
                          "loss_24h": loss_cents(share, rate, Fraction(24)),
                          "clean_hours": clean_hours(share, rate, 299, Fraction(36)), "change": "stop delay"}
    if number == 4:
        return answer == {"rebookings_24h": Fraction(20), "rebookings_2h": Fraction(5, 3),
                          "restored": ("version",)}
    if number == 5:
        raw = json.loads(Path(__file__).with_name("rehearsal-05.json").read_text())
        return paid_twice(raw["receipts"]) == 1 and answer["property"] == "one payment per request across actors and keys" and (
            "fail" in answer["reply"].lower() and "request" in answer["reply"].lower()
        )
    raise ValueError("unknown exercise")
