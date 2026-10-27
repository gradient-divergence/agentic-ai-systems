"""Compositions over the supplied tools class; the gate reads the same view."""

from contextlib import contextmanager

from refund_agent import contracts as c
from refund_agent.batch_stream.log import EventLog
from refund_agent.batch_stream.view import (
    ALLOWED,
    MARGIN,
    complete_through,
    duration,
    shipment_as_of,
)
from refund_agent.rules import decision
from refund_agent.tools import Tools


class LogShipmentTools(Tools):
    def __init__(self, data, *, event_log: EventLog, **kwargs):
        super().__init__(data, **kwargs)
        self.event_log = event_log

    @contextmanager
    def _event_view(self, state):
        """A per-call copy, under the tools lock, never a generated-data write."""
        with self._lock:
            saved = self._data
            at = state.execution_at or state.request.at
            if self.event_log.arrived_by(at):
                shipments = [
                    shipment_as_of(self.event_log, order.order_id, at)
                    for order in saved.orders
                ]
                self._data = saved.model_copy(update={"shipments": shipments})
            else:
                self._data = saved.model_copy(update={"shipments": []})
            try:
                yield
            finally:
                self._data = saved

    def call(self, call, state):
        if call.name == "audit.append" and state.failure_detail:
            # Record an existing source error without trying the broken view again.
            return super().call(call, state)
        with self._event_view(state):
            return super().call(call, state)

    def assess(self, state):
        with self._event_view(state):
            return super().assess(state)


class EventTimeTools(LogShipmentTools):
    version = "batch-stream-1"

    def __init__(
        self,
        data,
        *,
        margin=MARGIN,
        allowed_lateness=ALLOWED,
        batch_stream_fixture=False,
        batch_stream_case=None,
        batch_stream_mode="execute",
        **kwargs,
    ):
        super().__init__(data, **kwargs)
        self.margin = duration(margin)
        self.allowed_lateness = duration(allowed_lateness)
        self.batch_stream_fixture = batch_stream_fixture
        self.batch_stream_case = batch_stream_case
        self.batch_stream_mode = batch_stream_mode
        self.batch_stream_attempts = ()
        self.batch_stream_execution_boundaries = ()

    def _final(self, state):
        return complete_through(
            self.event_log,
            state.execution_at or state.request.at,
            self.margin,
            state.request.at + self.allowed_lateness,
        )

    def assess(self, state):
        chosen = super().assess(state)
        if (
            chosen.outcome == c.Outcome.APPROVE
            and state.request.reason == "not_delivered"
            and not self._final(state)
        ):
            return decision(state, c.Outcome.REFER, "shipment_incomplete")
        return chosen

    def _authorize(self, state, amount, approval):
        if state.request.reason == "not_delivered" and not self._final(state):
            return "shipment_incomplete"
        return super()._authorize(state, amount, approval)
