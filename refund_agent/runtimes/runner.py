"""A small library-shaped runtime; these defaults belong to this demonstration."""

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass(frozen=True)
class Call:
    name: str
    arguments: Any = None


@dataclass(frozen=True)
class Reply:
    calls: list[Call] = field(default_factory=list)
    value: Any = None
    final: bool = True


@dataclass(frozen=True)
class RuntimeSettings:
    stop_on_no_calls: bool = True
    retries: int = 1
    errors_to_model: bool = True
    max_turns: int = 8
    reserve_turns: int = 0

    def __post_init__(self):
        for name in ("retries", "max_turns", "reserve_turns"):
            value = getattr(self, name)
            if type(value) is not int or value < (1 if name == "max_turns" else 0):
                raise ValueError(f"invalid {name}")
        if self.reserve_turns > self.max_turns:
            raise ValueError("reserved turns exceed the turn limit")


@dataclass
class Session:
    trace_id: str
    messages: list[dict] = field(default_factory=list)
    spans: list[dict] = field(default_factory=list)
    turns: int = 0
    stop_reason: str = ""


class Runner:
    def __init__(self, settings: RuntimeSettings | None = None):
        self.settings = settings or RuntimeSettings()

    def invoke(self, context, call, registry, session, fatal_errors=()):
        for attempt in range(self.settings.retries + 1):
            span = {
                "trace_id": session.trace_id,
                "turn": session.turns,
                "tool": call.name,
                "attempt": attempt + 1,
            }
            try:
                result = registry[call.name](context, call)
            except BaseException as error:
                session.spans.append({**span, "status": type(error).__name__})
                if isinstance(error, fatal_errors) or not isinstance(error, Exception):
                    raise
                if attempt == self.settings.retries:
                    raise
            else:
                session.spans.append({**span, "status": "ok"})
                return result

    def _dispatch(self, context, reply, registry, session, fatal_errors):
        return [self.invoke(context, call, registry, session, fatal_errors) for call in reply.calls]

    def run(
        self,
        context,
        model: Callable,
        registry: dict[str, Callable],
        *,
        trace_id="runtime",
        before=None,
        after=None,
        on_error=None,
        stop=None,
        fatal_errors=(),
    ) -> Session:
        session = Session(trace_id)
        limit = self.settings.max_turns - self.settings.reserve_turns
        hooks = (before, after, on_error)
        while session.turns < limit:
            if stop and stop(context):
                session.stop_reason = "application_stop"
                break
            reply, failed = self._turn(context, model, registry, session, hooks, fatal_errors)
            session.turns += 1
            if stop and stop(context):
                session.stop_reason = "application_stop"
                break
            if self._ends_without_calls(reply, failed):
                session.stop_reason = "no_calls"
                break
        if not session.stop_reason:
            session.stop_reason = "turn_limit"
        return session

    def _turn(self, context, model, registry, session, hooks, fatal_errors):
        before, after, on_error = hooks
        reply = None
        failed = False
        try:
            reply = model(context, session)
            session.messages.append(
                {
                    "role": "assistant",
                    "reply": reply.value,
                    "calls": [call.name for call in reply.calls],
                }
            )
            self._reply_effects(context, reply, registry, session, before, after, fatal_errors)
        except BaseException as error:
            if isinstance(error, fatal_errors) or not isinstance(error, Exception):
                raise
            failed = True
            session.messages.append({"role": "error", "error": type(error).__name__})
            handled = on_error(context, reply, error) if on_error else False
            if not handled and not self.settings.errors_to_model:
                raise
        return reply, failed

    def _reply_effects(self, context, reply, registry, session, before, after, fatal_errors):
        results = before(context, reply) if before else None
        if results is None:
            results = self._dispatch(context, reply, registry, session, fatal_errors)
        for call, result in zip(reply.calls, results):
            session.messages.append({"role": "tool", "name": call.name, "result": result})
        if after:
            after(context, reply, results)

    def _ends_without_calls(self, reply, failed):
        return (
            not failed
            and reply is not None
            and not reply.calls
            and reply.final
            and self.settings.stop_on_no_calls
        )
