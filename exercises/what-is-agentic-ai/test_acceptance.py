"""Exercise 6: run against the student's patched refund_agent, outside default tests.

The reference runner checks the unchanged 4/22/14/0 baseline before applying its
patch. These tests check the changed submission, not the book's runnable agent.
"""

import pytest

from refund_agent.baseline import baseline, score_decisions
from refund_agent.contracts import Tier
from refund_agent.data.generate import generate
from refund_agent.rules import tier_for
from refund_agent.tools import Tools


@pytest.mark.parametrize(
    "cents,tier",
    [
        (2499, Tier.AGENT),
        (2500, Tier.REVIEWER),
        (50000, Tier.REVIEWER),
        (50001, Tier.NAMED_APPROVER),
    ],
)
def test_changed_tier_boundary(cents, tier):
    assert tier_for(cents) == tier


def test_seed_seven_public_drafts_and_no_write_tool(monkeypatch):
    original = Tools.call

    def reads_only(tools, call, state):
        assert call.name in ("order.lookup", "shipment.status", "policy.retrieve")
        return original(tools, call, state)

    monkeypatch.setattr(Tools, "call", reads_only)
    data = generate(7)
    public = set(data.public_scenarios)
    requests = [r for r in data.requests if r.request_id in public]
    labels = [d for d in data.correct if d.request_id in public]
    card = score_decisions(baseline(data, requests), labels)
    assert (card.total, card.approve, card.deny, card.refer, card.executed) == (40, 2, 22, 16, 0)
