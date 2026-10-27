"""Exercise 7: add a seventh token, none or one of the five injected failures.

Keep the first six token counts and the training-window fingerprint unchanged.
Materialize the failure into Request.failure, and let the independent policy label
make it a referral. Proposed decisions remain scores, never expected answers.
Return a ScenarioSet-compatible object, including a seventh token on every row.
Run test_acceptance.py explicitly; it is outside the default companion test suite.
"""

from datetime import datetime, timezone
from pathlib import Path

from refund_agent.data.generate import FAILURES
from refund_agent.scenarios.model import ScenarioModel

FAILURE_VALUES = ("none", *FAILURES)


def build_suite(as_of, tau, n=200, seed=7):
    """Use ScenarioModel's training-only source and add failure draws to the suite.

    A failure-free training row contributes to the new token's none count. Add-one
    smoothing leaves each planted failure drawable even though it was excluded
    from the six-token fit. Keep the policy checker outside the generator.
    """
    raise NotImplementedError("add the seventh token, then materialize and label every kept row")


def regenerate(manifest):
    """Rebuild the extended suite and reject a manifest that no longer matches."""
    raise NotImplementedError("regenerate from the recorded date, temperature, count and seed")


if __name__ == "__main__":
    at = datetime(2026, 9, 15, tzinfo=timezone.utc)
    original = ScenarioModel.fit().fingerprint
    for tau in (1, 2):
        suite = build_suite(at, tau, 200, 7)
        suite.write(Path(f"scenarios-failures-tau{tau}.json"))
        print(f"temperature {tau}: {suite.manifest['counts']}; {suite.fingerprint}")
    assert ScenarioModel.fit().fingerprint == original
