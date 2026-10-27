"""The chapter's authority disagreement and constructed outcome counts."""

import contextlib
import io
import runpy
from pathlib import Path
from typing import Any

import pytest

from refund_agent import contracts as c
from refund_agent.agent import Agent
from refund_agent.tools import Tools

DEMO = Path(__file__).resolve().parents[1] / "demos" / "agentic_loop_demo.py"


@pytest.fixture(scope="module")
def demo(tmp_path_factory: pytest.TempPathFactory) -> tuple[dict[str, Any], str]:
    printed = io.StringIO()
    with pytest.MonkeyPatch.context() as patch:
        patch.chdir(tmp_path_factory.mktemp("agentic-loop"))
        with contextlib.redirect_stdout(printed):
            namespace = runpy.run_path(str(DEMO), run_name="agentic_loop_demo")
    return namespace, printed.getvalue()


def test_disagreement_is_an_unpaid_tier3_denial(demo: tuple[dict[str, Any], str]) -> None:
    namespace, printed = demo
    assert "request r-7-0005: 65000 cents, tier 3" in printed
    assert "label: deny already_delivered" in printed
    assert "baseline: deny already_delivered" in printed
    assert "agent: refer already_delivered, executed=False" in printed
    assert "review: open=True, required_role=named_approver; payments=0" in printed
    assert "disagreements: tier 3, label deny=14" in printed
    assert len(namespace["disagreements"]) == 14
    for request, label, draft, recorded in namespace["disagreements"]:
        assert label.tier == c.Tier.NAMED_APPROVER and label.outcome == c.Outcome.DENY
        assert draft.outcome == label.outcome
        assert recorded.outcome == c.Outcome.REFER and recorded.reason == label.reason
        assert not recorded.executed
    request = namespace["disagreements"][0][0]
    tools = Tools(namespace["disagreement_data"])
    run = Agent(tools).run(request)
    review = next(iter(tools.reviews.values()))
    assert run.decision == namespace["disagreements"][0][3]
    assert review.open and review.required_role == "named_approver"
    assert review.proposal.request == request and not tools.payments


def test_constructed_outcomes_sum_to_one_thousand(demo: tuple[dict[str, Any], str]) -> None:
    namespace, printed = demo
    assert "without checker: correct=868.126, referred=0.000, wrong=131.874" in printed
    assert "with checker: correct=850.763, referred=136.050, wrong=13.187" in printed
    assert "with checker, every step right: 0.851" in printed
    for arm, expected in (
        ("without_checker", (868.12553324672, 0, 131.87446675328)),
        ("with_checker", (850.7630225817856, 136.0495307428864, 13.187446675328)),
    ):
        counts = namespace[arm]
        assert counts == pytest.approx(expected)
        assert sum(counts) == pytest.approx(1000)
    assert namespace["with_checker"][0] / 1000 == pytest.approx(0.98**8)
