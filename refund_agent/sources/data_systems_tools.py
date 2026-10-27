"""Contracts compose over the supplied tools, retaining its authority and checks."""

from contextlib import contextmanager

from refund_agent import contracts as c
from refund_agent.rules import decision, evidence_problem, sources
from refund_agent.sources.data_systems_check import contract_problem
from refund_agent.sources.data_systems_lineage import RecordStore, make_lineage
from refund_agent.sources.data_systems_registry import default_registry
from refund_agent.sources.data_systems_releases import SourceRelease
from refund_agent.tools import Tools


class ContractedTools(Tools):
    def __init__(
        self,
        data,
        *,
        candidate_record,
        registry=None,
        served_versions=None,
        code_version=None,
        lineage_mode="executed",
        **kwargs,
    ):
        self.registry = default_registry() if registry is None else registry
        if isinstance(data, SourceRelease):
            if served_versions is not None:
                raise ValueError("source release already names served versions")
            served_versions, data = data.served_versions, data.data
        self.served_versions = dict(
            self.registry.pins if served_versions is None else served_versions
        )
        if set(self.served_versions) != set(self.registry.pins):
            raise ValueError("served versions required for all four sources")
        self.candidate_record = c._detach(candidate_record)
        from refund_agent.release.candidate import Candidate

        self.code_version = code_version or Candidate(**candidate_record).fingerprint
        self.max_steps = candidate_record["adapter"]["settings"]["max_steps"]
        if lineage_mode not in ("executed", "shadow"):
            raise ValueError("unknown lineage mode")
        self.lineage_mode = lineage_mode
        self.record_store = RecordStore()
        self._snapshots, self._read_sources, self._model_exchanges = {}, {}, {}
        self._model_errors = {}
        self._lineage_keys = set()
        self._running = 0
        super().__init__(data, **kwargs)

    @property
    def lineage(self):
        return self.record_store.decisions

    def recording_model(self, model, *, max_steps=None):
        if max_steps is not None:
            self.max_steps = max_steps
        tools = self

        class RecordedModel:
            def reply(self, state):
                try:
                    reply = model.reply(state)
                except BaseException as error:
                    tools._model_errors.setdefault(state.trace_id, []).append(
                        (state.request.request_id, state.phase, type(error).__name__)
                    )
                    raise
                tools._model_exchanges.setdefault(state.trace_id, []).append(
                    (state.request.request_id, state.phase, reply)
                )
                return reply

        return RecordedModel()

    def _contract_problem(self, state):
        order, shipment, snapshot = sources(self._data, state)
        records = {
            "order": order,
            "shipment": shipment,
            "policy": state.policy,
            "ledger": tuple(self._read("ledger", self._ledger).values()),
        }
        for source, record in records.items():
            if record is None:
                continue  # Missing evidence has already been referred by the base.
            age = 0 if source == "ledger" else state.freshness[source]
            problem = contract_problem(
                source, record, self.served_versions[source], age, registry=self.registry
            )
            if problem:
                return problem
        return None

    def assess(self, state):
        checked = c._validated(c.State, state)
        problem = evidence_problem(self._data, checked)
        if problem:
            return super().assess(checked)
        problem = self._contract_problem(checked)
        if problem:
            return decision(checked, c.Outcome.REFER, problem)
        return super().assess(checked)

    def _authorize(self, state, amount, approval):
        problem = evidence_problem(self._data, state)
        if problem:
            return problem
        return self._contract_problem(state) or super()._authorize(state, amount, approval)

    def _dispatch(self, call, state):
        self._read_sources.setdefault(state.trace_id, set())
        source = {
            "order.lookup": "order",
            "shipment.status": "shipment",
            "policy.retrieve": "policy",
            "refund.execute": "ledger",
        }.get(call.name)
        if source:
            self._read_sources[state.trace_id].add(source)
        return super()._dispatch(call, state)

    def _snapshot(self, state):
        orders = [
            o
            for o in self._data.orders
            if state.request.order_reference in (o.order_id, o.reference)
        ]
        ids = {o.order_id for o in orders}
        shipments = [s for s in self._data.shipments if s.order_id in ids]
        retrievals = [s for s in self._data.retrievals if s.request_id == state.request.request_id]
        at = state.execution_at or state.request.at
        receipts = [p for p in self._read("ledger", self._ledger).values() if p.order_id in ids]
        self._read_sources.setdefault(state.trace_id, set()).add("ledger")
        return {
            "order": {"orders": [o.model_dump(mode="json") for o in orders]},
            "shipment": {"shipments": [s.model_dump(mode="json") for s in shipments]},
            "policy": {
                "policies": [p.model_dump(mode="json") for p in self._data.policies],
                "retrievals": [s.model_dump(mode="json") for s in retrievals],
            },
            "ledger": {"receipts": [p.model_dump(mode="json") for p in receipts]},
            "observations": {
                "order": orders[0].observed_at if len(orders) == 1 else None,
                "shipment": shipments[0].observed_at if len(shipments) == 1 else None,
                "policy": retrievals[0].observed_at if len(retrievals) == 1 else None,
                "ledger": at,
            },
        }

    def _commit(self, state, key, amount, approval):
        self._snapshots[state.trace_id] = self._snapshot(state)
        return super()._commit(state, key, amount, approval)

    def _decision(self, state, reason=None):
        if (
            reason is None
            and state.receipt is None
            and state.decision is not None
            and state.decision.outcome == c.Outcome.DENY
        ):
            problem = evidence_problem(self._data, state) or self._contract_problem(state)
            if problem:
                state.decision = decision(state, c.Outcome.REFER, problem)
                state.failure_detail = problem
        if reason is None and state.decision is not None:
            from refund_agent.agent import ensure_named_review

            ensure_named_review(self, state, [])
        if state.receipt is None:
            self._snapshots[state.trace_id] = self._snapshot(state)
        result = super()._decision(state, reason)
        if self._running == 0 and state.decision:
            self._record_lineage(state)
        return result

    def _record_lineage(self, state):
        at = state.execution_at or state.request.at
        key = state.request.model_dump_json(), at, state.decision.model_dump_json()
        if key in self._lineage_keys:
            return
        snapshots = self._snapshots[state.trace_id]
        self.record_store.append(make_lineage(self, state, snapshots))
        self._lineage_keys.add(key)

    @contextmanager
    def _boundary(self, state, operation):
        with super()._boundary(state, operation):
            if operation != "run":
                yield
                return
            self._running += 1
            self._model_exchanges[state.trace_id] = []
            self._model_errors[state.trace_id] = []
            self._snapshots[state.trace_id] = self._snapshot(state)
            try:
                yield
                if state.decision:
                    self._record_lineage(state)
            finally:
                self._running -= 1
