"""Stage-0 drafts and exact outcome scoring against generated labels."""

from collections.abc import Sequence

from refund_agent import contracts
from refund_agent.contracts import Decision, Outcome, Record, Request, State, Tier, ToolCall
from refund_agent.data.generate import Dataset
from refund_agent.rules import assess, decision
from refund_agent.tools import Tools


def baseline(data: Dataset, requests: Sequence[Request] | None = None) -> list[Decision]:
    tools = Tools(data)
    decisions = []
    for request in data.requests if requests is None else requests:
        state = State(request=request, trace_id=f"baseline-{request.request_id}")
        for name in ("order.lookup", "shipment.status", "policy.retrieve"):
            result = tools.call(ToolCall(name=name), state)
            if not result.ok:
                state.decision = decision(state, Outcome.REFER, result.reason)
                break
        chosen = state.decision if state.decision is not None else assess(data, state)
        if chosen.outcome == Outcome.APPROVE and chosen.tier != Tier.AGENT:
            chosen = decision(state, Outcome.REFER, "approval_required")
        decisions.append(chosen)
    return decisions


class Scorecard(Record):
    total: int
    correct: int
    approve: int
    deny: int
    refer: int
    executed: int

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0

    def line(self, name: str) -> str:
        return (
            f"{name}: {self.correct}/{self.total} agreement with generated labels ({self.accuracy:.1%}); "
            f"approve={self.approve} deny={self.deny} refer={self.refer} executed={self.executed}"
        )


def score_decisions(decisions: Sequence[Decision], correct: Sequence[Decision]) -> Scorecard:
    predictions = {d.request_id: contracts._validated(Decision, d) for d in decisions}
    expected = {d.request_id: contracts._validated(Decision, d) for d in correct}
    if (
        len(predictions) != len(decisions)
        or len(expected) != len(correct)
        or predictions.keys() != expected.keys()
    ):
        raise ValueError("score requires exactly one decision per labelled request")
    matches = sum(
        (predictions[key].outcome, predictions[key].amount_cents, predictions[key].tier)
        == (label.outcome, label.amount_cents, label.tier)
        and (
            label.outcome != Outcome.DENY
            or predictions[key].policy_citation == label.policy_citation
        )
        for key, label in expected.items()
    )
    return Scorecard(
        total=len(decisions),
        correct=matches,
        approve=sum(d.outcome == Outcome.APPROVE for d in decisions),
        deny=sum(d.outcome == Outcome.DENY for d in decisions),
        refer=sum(d.outcome == Outcome.REFER for d in decisions),
        executed=sum(d.executed for d in decisions),
    )
