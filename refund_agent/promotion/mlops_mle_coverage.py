"""Policy coverage at assessment, final denial, and execution's live reread."""

from refund_agent import contracts as c
from refund_agent.rules import decision
from refund_agent.tools import Tools


class PolicyCoverage(Tools):
    def __init__(self, data, *, bundle, **kwargs):
        self.bundle = bundle
        super().__init__(data, **kwargs)
        from refund_agent.promotion.mlops_mle_stamp import BundledRecordStore

        self.record_store = BundledRecordStore(bundle)

    def _coverage_problem(self, state):
        at = state.execution_at or state.request.at
        active = [p for p in self._data.policies if p.effective_at <= at]
        if active and not self.bundle.covers(max(active, key=lambda p: p.effective_at).version):
            return "policy_not_in_bundle"
        return None

    def assess(self, state):
        state = c._validated(c.State, state)
        problem = self._coverage_problem(state)
        if problem:
            return decision(state, c.Outcome.REFER, problem)
        return super().assess(state)

    def _authorize(self, state, amount, approval):
        return self._coverage_problem(state) or super()._authorize(state, amount, approval)

    def _decision(self, state, reason=None):
        if state.receipt is None and state.decision is not None:
            problem = self._coverage_problem(state)
            if problem:
                state.decision = decision(state, c.Outcome.REFER, problem)
                state.failure_detail = problem
                reason = None
        return super()._decision(state, reason)
