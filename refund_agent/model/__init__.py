"""Offline models. A later LiveModel implements ModelClient here; no network adapter yet."""

from pathlib import Path
from typing import Protocol

from refund_agent import contracts as c
from refund_agent.contracts import _detach, _validated
from refund_agent.rules import draft

Reply = c.ToolCall | c.Decision


class ModelClient(Protocol):
    """Adapters raise TimeoutError when their configured deadline expires."""

    def reply(self, state: c.State) -> Reply: ...


class ScriptedModel:
    def __init__(
        self,
        wrong_reply: str | None = None,
        replies: dict[tuple[str, c.Phase], Reply] | None = None,
    ) -> None:
        if wrong_reply not in (None, "wrong_tool", "over_limit", "follow_instruction"):
            raise ValueError("unknown wrong reply")
        self.wrong_reply = wrong_reply
        self._replies = {
            key: _validated(c.ToolCall if isinstance(reply, c.ToolCall) else c.Decision, reply)
            for key, reply in (replies or {}).items()
        }

    @property
    def replies(self) -> dict[tuple[str, c.Phase], Reply]:
        return _detach(self._replies)

    def reply(self, state: c.State) -> Reply:
        state = _validated(c.State, state)
        key = (state.request.request_id, state.phase)
        if key in self._replies:
            return _detach(self._replies[key])
        wrong = self._wrong_reply(state)
        if wrong is not None:
            return wrong
        if state.phase == c.Phase.ASSESS:
            return draft(state)
        return c.ToolCall(name=c.PHASE_TO_TOOL[state.phase], arguments=self._arguments(state))

    def _wrong_reply(self, state: c.State) -> c.ToolCall | None:
        if (
            self.wrong_reply == "follow_instruction"
            and state.phase == c.Phase.ASSESS
            and "ignore" in state.request.text.casefold()
        ):
            return c.ToolCall(
                name="refund.execute",
                arguments=c.ToolArguments(
                    amount_cents=999999, approval_id="forged", idempotency_key="injected"
                ),
            )
        if state.phase == c.Phase.EXECUTE and self.wrong_reply == "wrong_tool":
            return c.ToolCall(name="payment.send")
        return None

    def _arguments(self, state: c.State) -> c.ToolArguments:
        if state.phase not in (c.Phase.PROPOSE, c.Phase.EXECUTE):
            return c.ToolArguments()
        amount = (
            999999
            if state.phase == c.Phase.EXECUTE and self.wrong_reply == "over_limit"
            else state.request.amount_cents
        )
        return c.ToolArguments(
            amount_cents=amount,
            reason=state.request.reason,
            idempotency_key=f"execute-{state.request.request_id}"
            if state.phase == c.Phase.EXECUTE
            else None,
            approval_id=state.approvals[0].approval_id if state.approvals else None,
        )


class Exchange(c.Record):
    request_id: str
    phase: c.Phase
    reply: Reply


class Transcript(c.Record):
    exchanges: list[Exchange]

    def write(self, path: Path) -> None:
        c.write_json(path, self)

    @classmethod
    def from_trace(cls, trace: c.Trace) -> "Transcript":
        trace = _validated(c.Trace, trace)
        exchanges = []
        for step in trace.steps:
            reply = step.call if step.call is not None else step.reply
            if step.phase != c.Phase.AUDIT and reply is not None:
                exchanges.append(
                    Exchange(request_id=trace.decision.request_id, phase=step.phase, reply=reply)
                )
        return _detach(cls(exchanges=exchanges))


class ReplayModel(ScriptedModel):
    def __init__(self, path: Path) -> None:
        transcript = Transcript.model_validate_json(path.read_text(encoding="utf-8"))
        replies = {(e.request_id, e.phase): e.reply for e in transcript.exchanges}
        if len(replies) != len(transcript.exchanges):
            raise ValueError("duplicate replay state")
        super().__init__(replies=replies)

    def reply(self, state: c.State) -> Reply:
        state = _validated(c.State, state)
        try:
            return _detach(self._replies[(state.request.request_id, state.phase)])
        except KeyError as error:
            raise ValueError("unrecorded replay state") from error
