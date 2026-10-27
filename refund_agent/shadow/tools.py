"""Mixin redirecting only payment and review stores around the candidate's own tools.

The internal receipts preserve idempotency and continuation semantics. Public ledger,
reviews and payments expose the external stores, which never receive shadow writes.
"""

import json
from hashlib import sha256

from refund_agent import contracts as c
from refund_agent.tools import Tools


class ShadowTools(Tools):
    def __init__(self, data, **kwargs):
        super().__init__(data, **kwargs)
        self._live_ledger, self._live_reviews = self._ledger, self._reviews
        self._ledger, self._reviews = c._detach(self._ledger), c._detach(self._reviews)
        self.shadow_log = []
        self.gate_calls = 0

    @property
    def ledger(self):
        return c._detach(self._live_ledger)

    @property
    def reviews(self):
        return c._detach(self._live_reviews)

    @property
    def payments(self):
        return c._detach([p for p in self._live_ledger.values() if p.request])

    @property
    def shadow_payments(self):
        return c._detach([p for p in self._read("ledger", self._ledger).values() if p.request])

    @property
    def shadow_reviews(self):
        return c._detach(self._read("reviews", self._reviews))

    def _authorize(self, state, amount, approval):
        self.gate_calls += 1
        return super()._authorize(state, amount, approval)

    def _commit(self, state, key, amount, approval):
        # Payment binding is core-owned; component payment sinks never run in shadow.
        result = Tools._commit(self, state, key, amount, approval)
        self.shadow_log.append(
            {
                "kind": "payment",
                "request_id": state.request.request_id,
                "proposal_id": state.proposal_id,
                "amount_cents": amount,
                "idempotency_key": key,
                "money_moved": False,
            }
        )
        return result

    def _review(self, call, state):
        # Review storage, like payment storage, is a core shadow binding.
        return Tools._review(self, call, state)

    def _store_review(self, review, state):
        before = set(self._reviews)
        result = Tools._store_review(self, review, state)
        for rid in sorted(set(self._reviews) - before):
            review = self._reviews[rid]
            self.shadow_log.append(
                {
                    "kind": "review",
                    "request_id": state.request.request_id,
                    "proposal_id": review.proposal.proposal_id if review.proposal else None,
                    "required_role": review.required_role,
                    "opened_for_person": False,
                }
            )
        return result

    def _checked_append(self, record):
        would_execute = record.decision.executed
        actual = record.model_copy(
            update={
                "audit_id": "",
                "money_moved": False,
                "paid_amount_cents": 0,
                "receipt_id": None,
                "decision": record.decision.model_copy(update={"executed": False}),
            }
        )
        digest = sha256(
            json.dumps(actual.model_dump(mode="json"), sort_keys=True).encode()
        ).hexdigest()[:24]
        actual = actual.model_copy(update={"audit_id": f"audit-{digest}"})
        result = super()._checked_append(actual)
        self.shadow_log.append(
            {
                "kind": "audit",
                "audit_id": actual.audit_id,
                "request_id": record.decision.request_id,
                "would_execute": would_execute,
                "money_moved": False,
            }
        )
        return result
