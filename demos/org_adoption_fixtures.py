"""Chapter 38's constructed settings and accounting exercises, not admission records."""

import json
from datetime import datetime, timedelta, timezone

from refund_agent import contracts as c
from refund_agent.rollout.org_adoption_plan import (
    EVIDENCE_TYPES,
    ROLES,
    EvidenceRef,
    Owner,
    RolloutPlan,
    Step,
)
from refund_agent.rollout.org_adoption_records import (
    DecisionEntry,
    Intake,
    ProviderPayment,
    Window,
)

START = datetime(2026, 11, 29, tzinfo=timezone.utc)


def constructed_plan(candidate_fingerprint):
    steps = []
    work = {
        "decision_owner": "Sign the plan and each step's evidence",
        "release_owner": "Start and stop the grant, preserving paid refunds",
        "reconciler": "Reconcile daily and record each cause, owner and disposition",
        "reviewer": "Decide every tier 2 proposal, including denials",
        "absence_cover": "Cover the reviewer's named absent days",
        "named_approver": "Decide every request over 500 dollars, including denials",
    }
    first_end = START + timedelta(days=14)
    for i, (share, start, period_start, period_end) in enumerate(
        (
            (
                0.25,
                START,
                datetime(2026, 10, 1, tzinfo=timezone.utc),
                datetime(2026, 10, 29, tzinfo=timezone.utc),
            ),
            (1.0, first_end + timedelta(days=32), START, first_end),
        ),
        1,
    ):
        refs = tuple(
            EvidenceRef(
                record_type=kind,
                record_id=f"org-adoption-step-{i}-{kind}",
                period_start=period_start if kind == "audit" else None,
                period_end=period_end if kind == "audit" else None,
            )
            for kind in EVIDENCE_TYPES
        )
        steps.append(
            Step(
                step_id=f"org-adoption-step-{i}",
                requester_share=share,
                tiers=(1, 2),
                window=Window(
                    window_id=f"org-adoption-window-{i}",
                    start=start,
                    end=start + timedelta(days=14),
                    closed_at=start + timedelta(days=16),
                ),
                required=refs,
                governed_basis_points=9900,
                around_limit=0,
                signed_at=START - timedelta(days=32),
                owners=tuple(
                    Owner(
                        role=role,
                        person="alex" if role == "reviewer" else "morgan",
                        work=work[role],
                    )
                    for role in ROLES
                ),
                manual_route="authenticated review and audit; payment through refund.execute",
            )
        )
    return RolloutPlan(
        plan_id="org-adoption-constructed-plan",
        canary_id="org-adoption-fixed-hash",
        candidate_fingerprint=candidate_fingerprint,
        signed_by="morgan",
        signed_at=START - timedelta(days=32),
        steps=tuple(steps),
        widen_rule="zero paid around and governed share at least 99 percent",
        hold_rule="hold paid workaround, low share or absent evidence; next window requires named change",
        reverse_rule="forbidden payment, budget spent, tier2 payment without proposal-bound approval",
    )


def accounting_fixture(n, *, console):
    window = constructed_plan("accounting-exercise-only").steps[0].window
    intake, decisions = [], []
    for i in range(n):
        request = c.Request(
            request_id=f"org-adoption-ex-r-{i:03d}",
            requester_id=f"customer-{i}",
            order_reference=f"org-adoption-ex-o-{i:03d}",
            at=START + timedelta(hours=1),
            amount_cents=4000,
            reason="not_delivered",
        )
        intake.append(
            Intake(
                request=request,
                order_id=request.order_reference,
                received_at=request.at,
                decision_deadline=request.at + timedelta(hours=24),
            )
        )
        if i == 0:  # Planted: logged phone request with no decision record.
            continue
        chosen = c.Decision(
            request_id=request.request_id,
            outcome=c.Outcome.DENY,
            amount_cents=4000,
            tier=c.Tier.AGENT,
            reason="already_delivered",
            policy_citation="refund_cap_v4@2026-09-01#eligibility",
        )
        audit = c.AuditRecord(
            audit_id="audit-" + request.request_id,
            trace_id="trace-" + request.request_id,
            at=request.at + timedelta(hours=2),
            decision=chosen,
            freshness={},
            risk_score=None,
            approval_ids=[],
        )
        decisions.append(
            DecisionEntry(
                audit=audit,
                actor_id="alex",
                path="governed",
                review_id="review-" + request.request_id,
                authorization_id="constructed-verified-review-alex",
            )
        )
    provider, ledger = [], []
    if console:
        request = intake[1].request
        payment = ProviderPayment(
            payment_id="org-adoption-ex-console",
            order_id=intake[1].order_id,
            amount_cents=4000,
            at=request.at + timedelta(hours=1),
            request_id=request.request_id,
            channel="console",
        )
        provider.append(payment)
        # Imported history alone does not match a request made through refund.execute.
        ledger.append(
            c.Receipt(
                receipt_id=payment.payment_id,
                order_id=payment.order_id,
                amount_cents=payment.amount_cents,
                at=payment.at,
                idempotency_key="prior-" + payment.payment_id,
            )
        )
    return {
        "window": window,
        "intake": tuple(intake),
        "decisions": tuple(decisions),
        "ledger": tuple(ledger),
        "provider": tuple(provider),
    }


def serialize_fixture(fixture):
    return {
        key: value.model_dump(mode="json")
        if isinstance(value, c.Record)
        else [row.model_dump(mode="json") for row in value]
        for key, value in fixture.items()
    }


def read_fixture(value):
    from refund_agent.rollout.org_adoption_records import (
        DecisionEntry,
        Intake,
        ProviderPayment,
        Window,
    )

    models = {
        "window": Window,
        "intake": Intake,
        "decisions": DecisionEntry,
        "ledger": c.Receipt,
        "provider": ProviderPayment,
    }
    return {
        key: models[key].model_validate_json(json.dumps(row))
        if key == "window"
        else tuple(
            models[key].model_validate_json(json.dumps(item)) for item in row
        )
        for key, row in value.items()
        if key in models
    }
