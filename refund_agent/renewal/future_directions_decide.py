"""The signed checks determine the verdict; a statistical pass grants nothing."""

from typing import Literal

from refund_agent.contracts import Record
from refund_agent.selection import fingerprint


class Decision(Record):
    requirement_fingerprint: str
    evidence_fingerprint: str
    candidate_bundle_fingerprint: str | None
    candidate_fingerprint: str | None
    attempt: int
    verdict: Literal["replace", "rebuild", "keep"]
    reasons: tuple[str, ...]
    day: float
    first_decision_fingerprint: str | None = None

    @property
    def fingerprint(self):
        return fingerprint(self.model_dump(mode="json"))


def validate_attempt(req, ev, attempt, first):
    if ev.requirement_fingerprint != req.fingerprint or (
        attempt == 1
        and (
            ev.candidate_fingerprint != req.candidate_fingerprint
            or ev.candidate_bundle_fingerprint != req.candidate_bundle_fingerprint
        )
    ):
        raise ValueError("evidence differs from signed requirement")
    if (ev.paid, ev.unique, ev.all_matured, ev.selected) != (
        req.paid_per_window,
        req.paid_per_window,
        True,
        True,
    ):
        raise ValueError("exact declared unique matured paid window required")
    if {c.name for c in ev.checks} != {
        "calibration",
        "cost",
        "wrongful_denials",
        "capacity",
        "release",
        "incumbent_admission",
    }:
        raise ValueError("all required checks must be present")
    if len(ev.checks) != 6 or ev.day < ev.closed_day:
        raise ValueError("unique checks and matured evaluation day required")
    if ev.candidate_fingerprint is None and all(c.passed for c in ev.checks):
        raise ValueError("replacement requires a bound candidate identity")
    if ev.candidate_bundle_fingerprint is None and all(c.passed for c in ev.checks):
        raise ValueError("replacement requires a bound bundle identity")
    if attempt not in (1, 2) or attempt == 1 and first is not None:
        raise ValueError("one or two attempts allowed")
    if attempt == 2 and (
        first is None or first.verdict != "rebuild" or first.reasons != ("calibration",)
    ):
        raise ValueError("second attempt requires the recorded calibration-only failure")
    if attempt == 2 and (
        ev.first_decision_fingerprint != first.fingerprint or not ev.fit_window_fingerprint
    ):
        raise ValueError("rebuilt evidence must retain its first decision and fit window")


# region book:fnd.future-directions.decide-01
def decide(req, ev, attempt, first=None):
    from refund_agent.renewal import future_directions_decide as renewal

    renewal.validate_attempt(req, ev, attempt, first)
    previous = None if first is None else first.fingerprint
    failures = tuple(check.name for check in ev.checks if not check.passed)
    verdict = "replace"
    if failures:
        verdict = "keep"
        if attempt == 1 and failures == ("calibration",):
            verdict = "rebuild"
    return renewal.Decision(
        requirement_fingerprint=req.fingerprint,
        evidence_fingerprint=ev.fingerprint,
        candidate_bundle_fingerprint=ev.candidate_bundle_fingerprint,
        candidate_fingerprint=ev.candidate_fingerprint,
        attempt=attempt,
        verdict=verdict,
        reasons=failures,
        day=ev.day,
        first_decision_fingerprint=previous,
    )


# endregion book:fnd.future-directions.decide-01
