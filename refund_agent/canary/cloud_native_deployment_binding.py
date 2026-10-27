"""Redirect payment/review calls around the supplied, unchanged tools and gate."""


from refund_agent import contracts as c
from refund_agent.canary.cloud_native_deployment_plan import in_cohort
from refund_agent.canary.cloud_native_deployment_stop import CanarySession
from refund_agent.canary.cloud_native_deployment_store import INTEGRITY_ERRORS, PersistentTools
from refund_agent.rules import decision
from refund_agent.tools import ReconciliationError


class CanaryTools(PersistentTools):
    version = "cloud-native-deployment.canary@1"

    def __init__(self, data, *, canary_session: CanarySession, shadow_type, shadow_kwargs,
                 canary_rollback_only=False,
                 **kwargs):
        self.canary_session = canary_session
        self.canary_rollback_only = canary_rollback_only
        self.person_requests = {}
        self.shadow_recorder = shadow_type(data, **shadow_kwargs)
        self._loading = True
        try:
            super().__init__(data, canary_store=canary_session.store, **kwargs)
        except INTEGRITY_ERRORS:
            canary_session.untrusted_record(data.requests[0].at)
            raise
        self._loading = False

    def _reconcile(self, name, expected, observed):
        state = self._context[0] if self._context else None
        at = (state.execution_at or state.request.at) if state else self.data.requests[0].at
        self.canary_session.untrusted_record(at)
        return super()._reconcile(name, expected, observed)

    @property
    def shadow_payments(self):
        return self.shadow_recorder.shadow_payments

    @property
    def shadow_reviews(self):
        return self.shadow_recorder.shadow_reviews

    def _write(self, name, key, record, keep=False):
        super()._write(name, key, record, keep)
        if name == "ledger" and record is not None and record.request and not self._loading:
            self.canary_session.note_payment(record)

    def _shadow(self, call, state, reason):
        shadow = self.shadow_recorder
        shadow.data = self.data
        if state.proposal_id in self._proposals:
            proposal = self._proposals[state.proposal_id]
            shadow._write("proposals", state.proposal_id, proposal)
        scratch = state.model_copy(deep=True)
        shadow.call(call, scratch)
        self.person_requests[state.request.request_id] = c._detach(state.request)
        return c.ToolResult(ok=False, reason=reason)

    def _review(self, call, state):
        return self._shadow(call, state, "approval_required")

    def _named_reviews(self):
        return {**super()._named_reviews(), **self.shadow_recorder._named_reviews()}

    def _store_review(self, review, state):
        self.shadow_recorder.data = self.data
        result = self.shadow_recorder._store_review(review, state.model_copy(deep=True))
        self.person_requests[state.request.request_id] = c._detach(state.request)
        return result

    def assess(self, state):
        self.canary_session.tick(state.execution_at or state.request.at, self.payments)
        if self.canary_rollback_only or self.canary_session.outcome != "running":
            self.person_requests[state.request.request_id] = c._detach(state.request)
            reason = "canary_rollback" if self.canary_rollback_only else "canary_" + self.canary_session.outcome
            return decision(state, c.Outcome.REFER, reason)
        return super().assess(state)

    def _route_reason(self, call, state, at):
        if self.canary_rollback_only:
            return "canary_rollback"
        plan = self.canary_session.plan
        grant = self.canary_session.grant
        if grant is None or at < grant.started_at or grant.outcome != "running":
            return "canary_" + self.canary_session.outcome
        if state.request.amount_cents >= 5000 or call.arguments.amount_cents is None or (
            call.arguments.amount_cents > plan.max_cents_per_payment
        ):
            return "canary_amount"
        if not in_cohort(plan.canary_id, state.request.requester_id, plan.share):
            return "canary_outside_cohort"
        payments = [p for p in self.payments if p.receipt_id in grant.payment_ids and p.at.date() == at.date()]
        if len(payments) >= plan.daily_cap_count or (
            sum(p.amount_cents for p in payments) + call.arguments.amount_cents > plan.daily_cap_cents
        ):
            return "canary_daily_cap"
        return None

    # region book:fnd.cloud-native-deployment.binding-01
    def _execute(self, call, state):
        at = state.execution_at or state.request.at
        session = self.canary_session
        with session.store.transaction():
            session.tick(at, self.payments)
            reason = self._route_reason(call, state, at)
            if reason:
                return self._shadow(call, state, reason)
            return super()._execute(call, state)
    # endregion book:fnd.cloud-native-deployment.binding-01

    def call(self, call, state):
        try:
            return super().call(call, state)
        except ReconciliationError:
            self.canary_session.untrusted_record(state.execution_at or state.request.at)
            raise


def before_run(*, tools, request, at, runtime_settings, candidate, settings):
    from refund_agent.runtimes.faults import FaultTools

    mounted = tools.inner if isinstance(tools, FaultTools) else tools
    if isinstance(mounted, CanaryTools):
        mounted.canary_session.tick(at or request.at, mounted.payments)
        plan = mounted.canary_session.plan
        if mounted.canary_rollback_only or mounted.canary_session.outcome != "running" or (
            request.amount_cents >= 5000 or not in_cohort(plan.canary_id, request.requester_id)
        ):
            mounted.person_requests[request.request_id] = c._detach(request)
