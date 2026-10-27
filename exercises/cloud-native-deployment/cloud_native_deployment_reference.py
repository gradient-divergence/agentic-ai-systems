"""Reference solutions."""

from fractions import Fraction

ANSWERS = {
    1: {"max_cents": 4999, "daily_payments": 15, "signer": "decision_owner",
        "starter": "release_owner", "resumer": "release_owner"},
    2: {"next_path": "stop", "scripted_model": "rollback", "receipt": "neither",
        "notice": "neither", "policy_in_force": "neither", "intent": "neither"},
    3: {"further_6h": Fraction(3, 2), "loss_6h": Fraction(24995, 2),
        "further_24h": Fraction(6), "loss_24h": Fraction(34993),
        "clean_hours": Fraction(1232), "change": "stop delay"},
    4: {"rebookings_24h": Fraction(20), "rebookings_2h": Fraction(5, 3),
        "restored": ("version",)},
    5: {"property": "one payment per request across actors and keys",
        "reply": "Fail the rehearsal and block start. Derive the desk key from the request; count payments by request."},
}
READER = "expanded"
