"""Chapter 41 runtime closure, separate from its offline training/evidence package."""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from hashlib import sha256
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sklearn.ensemble import HistGradientBoostingClassifier

from refund_agent import contracts as c
from refund_agent.release.extensions import Component, register
from refund_agent.rules import decision, tier_for
from refund_agent.selection import fingerprint
from refund_agent.tools import Tools


def features(rows):
    import numpy as np

    return np.asarray(
        [
            [
                r.amount_cents,
                r.reason == "damaged",
                r.reason == "changed_mind",
                r.delivery_age_days or 0,
                r.prior_refund_count,
                r.paid_cents,
                r.order_evidence_age_seconds,
                r.shipment_evidence_age_seconds,
            ]
            for r in rows
        ],
        dtype=float,
    )


@dataclass(frozen=True)
class RiskComponent:
    model: HistGradientBoostingClassifier
    slope: float
    intercept: float
    threshold: float = 0.080

    def __post_init__(self):
        import numpy as np
        from sklearn.ensemble import HistGradientBoostingClassifier

        if type(self.model) is not HistGradientBoostingClassifier:
            raise ValueError("risk component requires the selected tree implementation")
        if self.threshold != 0.080 or not np.isfinite([self.slope, self.intercept]).all():
            raise ValueError("signed threshold and finite calibration parameters required")

    @property
    def fingerprint(self):
        return fingerprint(
            {
                "trees": sha256(pickle.dumps(self.model, protocol=4)).hexdigest(),
                "slope": self.slope,
                "intercept": self.intercept,
                "threshold": self.threshold,
            }
        )

    def predict(self, rows):
        import numpy as np
        from scipy.special import expit, logit

        raw = np.clip(self.model.predict_proba(features(rows))[:, 1], 1e-12, 1 - 1e-12)
        return expit(self.slope * logit(raw) + self.intercept)


class RiskTools(Tools):
    def __init__(
        self, data, *, future_directions_component, future_directions_policy_version, **kwargs
    ):
        self.future_directions_component = future_directions_component
        self.future_directions_policy_version = future_directions_policy_version
        self.future_directions_fingerprint = future_directions_component.fingerprint
        super().__init__(data, **kwargs)

    def _future_directions_score(self, state):
        if self.future_directions_component.fingerprint != self.future_directions_fingerprint:
            raise ValueError("released risk component changed")
        order, policy = state.order, state.policy
        if order is None or policy is None:
            raise ValueError("risk scoring needs order and policy evidence")
        delivered = [e.at for e in state.shipment_events if e.status == "delivered"]
        at = state.execution_at or state.request.at
        from types import SimpleNamespace

        row = SimpleNamespace(
            amount_cents=state.request.amount_cents,
            reason=state.request.reason,
            delivery_age_days=(at - max(delivered)).days if delivered else None,
            prior_refund_count=len(order.prior_refunds),
            paid_cents=order.paid_cents,
            order_evidence_age_seconds=state.freshness["order"],
            shipment_evidence_age_seconds=state.freshness["shipment"],
        )
        return float(self.future_directions_component.predict([row])[0])

    def assess(self, state):
        original = super().assess(state)
        if original.outcome == c.Outcome.APPROVE and original.tier == c.Tier.AGENT:
            if state.policy.version != self.future_directions_policy_version:
                return decision(state, c.Outcome.REFER, "policy_not_in_bundle")
            score = self._future_directions_score(state)
            if score >= self.future_directions_component.threshold:
                return decision(state, c.Outcome.REFER, f"future_directions_risk:{score:.6f}")
        return original

    def _authorize(self, state, amount, approval):
        problem = super()._authorize(state, amount, approval)
        if problem is not None:
            return problem
        if tier_for(state.request.amount_cents) == c.Tier.AGENT:
            if state.policy.version != self.future_directions_policy_version:
                return "policy_not_in_bundle"
            score = self._future_directions_score(state)
            if score >= self.future_directions_component.threshold:
                return f"future_directions_risk:{score:.6f}"
        return None


def mount(tools, component, *, policy_version):
    """Explicit composition; build_tools selects this through compose below."""
    return type("FutureDirectionsRiskTools", (RiskTools, tools), {}), {
        "future_directions_component": component,
        "future_directions_policy_version": policy_version,
    }


def compose(*, base, kwargs, candidate, settings, mode, context):
    component = context.get("future_directions_component")
    if not isinstance(component, RiskComponent) or component.fingerprint != settings["fingerprint"]:
        raise ValueError("future_directions selected score is absent or differs")
    if component.threshold != settings["threshold"]:
        raise ValueError("released risk threshold differs")
    mounted, owned = mount(base, component, policy_version=settings["policy_version"])
    return mounted, {**kwargs, **owned}


def selector(component, *, policy_version):
    if not policy_version:
        raise ValueError("risk release requires the policy version")
    return {
        "name": "future-directions.risk",
        "module": __name__,
        "settings": {
            "fingerprint": component.fingerprint,
            "threshold": component.threshold,
            "policy_version": policy_version,
        },
    }


def reasons(*, candidate, settings, **context):
    """Chapter 34 classifies the stable code before the score's colon suffix."""
    return {"future_directions_risk": "policy", "policy_not_in_bundle": "system"}


register(
    Component(
        name="future-directions.risk",
        version="future-directions-risk-1",
        module=__name__,
        order=41,
        hooks={"compose": compose, "reasons": reasons},
        slots={"41": "tier 1 risk at assessment and refund.execute"},
    )
)
