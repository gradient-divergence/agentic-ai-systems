"""Constructed records are reproducible, truth-separated, and region-input-bound."""

from datetime import timedelta

import pytest

from refund_agent.release.candidate import build_candidate
from refund_agent.release.suite import build_suite

# Match the spine's fixed-core initialization before cold component imports.
CORE = build_candidate(build_suite(trials=1))

# isort: split
from refund_agent import contracts as c  # noqa: E402
from refund_agent.audit.audits_explanations_month import constructed_month  # noqa: E402
from refund_agent.audit.audits_explanations_rates import (  # noqa: E402
    confusion_by_segment,
    criteria,
)
from refund_agent.audit.audits_explanations_records import SegmentAssignments  # noqa: E402


def assignments(n=2000):
    return SegmentAssignments(
        by_order={
            f"audits-explanations-o-{i:04d}": ("A", "B", "C")[i // 3 % 3]
            for i in range(n)
        },
        version="constructed-test-regions",
    )


def test_full_size_generator_is_fixed_seed_and_truth_never_reaches_tools():
    month = constructed_month(assignments())
    assert month == constructed_month(assignments())
    assert len(month.data.requests) == 2000 and month.seed == 7
    assert len(month.truth) == 2000 and month.tool_data().correct == []
    assert "truth" not in month.tool_data().model_dump()
    assert all(1 <= r.amount_cents <= 4999 for r in month.data.requests)
    for r, o, s, p in zip(
        month.data.requests,
        month.data.orders,
        month.data.shipments,
        month.data.retrievals,
    ):
        at = month.evaluation_times[r.request_id]
        assert at - r.at == timedelta(hours=36)
        assert (
            at - o.observed_at == at - s.observed_at == timedelta(seconds=120)
        )
        assert at - p.observed_at == timedelta(hours=1)
    assert month != constructed_month(assignments(), seed=11)


def test_absent_or_unknown_partner_region_refused():
    with pytest.raises(ValueError, match="assignment is required"):
        constructed_month(SegmentAssignments(by_order={}, version="test"), n=1)
    with pytest.raises(ValueError, match="names must"):
        constructed_month(
            SegmentAssignments(
                by_order={"audits-explanations-o-0000": "D"}, version="test"
            ),
            n=1,
        )


def test_12_case_confusion_counts_and_criteria_by_hand():
    month = constructed_month(assignments(12), n=12)
    # Four decided-or-referred cases per region, independent of generated truth.
    truths = [
        t.model_copy(update={"owed": i % 2 == 0})
        for i, t in enumerate(month.truth)
    ]
    month = month.model_copy(update={"truth": tuple(truths)})
    decisions = []
    for i, r in enumerate(month.data.requests):
        outcome = (
            c.Outcome.DENY
            if i % 3 == 0
            else c.Outcome.APPROVE
            if i % 3 == 1
            else c.Outcome.REFER
        )
        decisions.append(
            c.Decision(
                request_id=r.request_id,
                outcome=outcome,
                amount_cents=r.amount_cents,
                tier=c.Tier.AGENT,
                reason="constructed",
                policy_citation="refund_cap_v4@2026-04-01#eligibility",
            )
        )
    counts = confusion_by_segment(month, tuple(decisions))
    assert counts["A"].model_dump() == {
        "tp": 1,
        "fp": 1,
        "fn": 1,
        "tn": 1,
        "referred": 2,
    }
    assert counts["B"].model_dump() == {
        "tp": 1,
        "fp": 0,
        "fn": 0,
        "tn": 1,
        "referred": 1,
    }
    assert counts["C"].model_dump() == {
        "tp": 0,
        "fp": 1,
        "fn": 1,
        "tn": 0,
        "referred": 1,
    }
    assert criteria(counts["A"]) == {
        "denial_share": 0.5,
        "tpr": 0.5,
        "fpr": 0.5,
        "ppv": 0.5,
        "base_rate": 0.5,
    }
    with pytest.raises(ValueError, match="one oracle"):
        confusion_by_segment(month, tuple(decisions[:-1]))


def test_constructed_month_runs_selected_canary_and_retains_real_explanations():
    from refund_agent.release.candidate import build_candidate
    from refund_agent.release.suite import build_suite

    build_candidate(build_suite(trials=1))
    from refund_agent.audit.audits_explanations_component import STORE_PATH, selector
    from refund_agent.audit.audits_explanations_holds import AuditHolds
    from refund_agent.audit.audits_explanations_month import START, run_month
    from refund_agent.audit.audits_explanations_rates import AuditRule
    from refund_agent.audit.audits_explanations_records import ReviewRoute
    from refund_agent.audit.audits_explanations_stage6 import AuditInputs
    from refund_agent.canary.cloud_native_deployment_component import SELECTOR
    from refund_agent.memory.register import PolicyRegister

    month = constructed_month(assignments(12), n=12)
    rule = AuditRule(signed_at=START - timedelta(days=1))
    register = PolicyRegister()
    for policy in month.data.policies:
        register.announce(policy, policy.effective_at)
    snapshot = register.snapshot(month.period.end)
    candidate = build_candidate(build_suite(trials=1, components=(SELECTOR, selector(rule))))

    def context_for(local, request):
        assert local.correct == []
        return {"audits_explanations": AuditInputs(
            rule, month.segments, snapshot, month.history,
            {request.request_id: ReviewRoute(
                request_id=request.request_id, ticket_id="review-" + request.request_id,
                deadline=month.evaluation_times[request.request_id] + timedelta(days=2),
            )}, lambda *_: True, AuditHolds(STORE_PATH),
        )}

    result = run_month(month, candidate, context_for)
    assert len(result.traces) == len(result.observations) == 12
    assert result.paid_cents > 0
    explanations = {e.request_id: e for e in result.explanations}
    denied = [t for t in result.traces if t.decision.outcome == c.Outcome.DENY]
    assert denied
    for trace in denied:
        evidence = explanations[trace.decision.request_id]
        assert len(evidence.checks) == 6 and all(evidence.checks.values())
        assert evidence.failure is None
