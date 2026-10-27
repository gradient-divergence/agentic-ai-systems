"""Real partner contributions obey the five-part bundle contract."""

from datetime import datetime, timedelta, timezone

import pytest

from refund_agent.release.candidate import build_candidate
from refund_agent.release.suite import build_suite

CORE = build_candidate(build_suite(trials=1))

# isort: split
from refund_agent.audit.audits_explanations_component import selector as audits  # noqa: E402
from refund_agent.audit.audits_explanations_rates import AuditRule  # noqa: E402
from refund_agent.batch_stream.component import selector as events  # noqa: E402
from refund_agent.canary.cloud_native_deployment_component import SELECTOR as CANARY  # noqa: E402
from refund_agent.data.generate import _policies  # noqa: E402
from refund_agent.memory.register import PolicyRegister  # noqa: E402
from refund_agent.observability_reliability.component import selector as observe  # noqa: E402
from refund_agent.promotion.mlops_mle_bundle import from_candidate  # noqa: E402
from refund_agent.promotion.mlops_mle_component import selector as promotion  # noqa: E402
from refund_agent.release import extensions  # noqa: E402
from refund_agent.security.security_privacy_component import SELECTOR as SCOPE  # noqa: E402
from refund_agent.sources.data_systems_release import SELECTOR as CONTRACTS  # noqa: E402


@pytest.fixture(scope="module")
def all_partner_candidate():
    at = datetime(2026, 10, 1, tzinfo=timezone.utc)
    register = PolicyRegister()
    policies = _policies()
    for policy in policies:
        register.announce(policy, policy.effective_at)
    selections = (
        CONTRACTS, events(), CANARY,
        promotion(register.snapshot(at), [p.version for p in policies]),
        observe(), SCOPE, audits(AuditRule(signed_at=at - timedelta(days=1))),
    )
    return build_candidate(build_suite(trials=1, components=selections))


def test_all_partner_bundle_accepts_scope_and_audit_items(all_partner_candidate):
    candidate = all_partner_candidate
    bundle = from_candidate(candidate)
    contributions = extensions.contributions(candidate, "bundle_items")
    for name in (SCOPE["name"], "audits-explanations.stage6"):
        assert set(contributions[name]) == {"tools"}
        for item, value in contributions[name]["tools"].items():
            assert bundle.tools.items[item] == value
    requirements = extensions.contributions(candidate, "promotion_requirements")
    assert requirements["audits-explanations.stage6"][0]["changes"] == (
        "audits-explanations.stage6",
        "audits-explanations.generator",
        "audits-explanations.rule",
    )


@pytest.mark.parametrize("producer", [SCOPE["name"], "audits-explanations.stage6"])
def test_flat_partner_contribution_is_refused(all_partner_candidate, monkeypatch, producer):
    original = extensions.contributions

    def flat(candidate, hook, **context):
        contributions = original(candidate, hook, **context)
        if hook == "bundle_items":
            items = contributions[producer]
            contributions[producer] = items.get("tools", items)
        return contributions

    monkeypatch.setattr(extensions, "contributions", flat)
    with pytest.raises(ValueError, match="^bundle contribution conflicts with an owned item$"):
        from_candidate(all_partner_candidate)


def test_all_partner_constructed_month_first_request_uses_scoped_view():
    from refund_agent.audit.audits_explanations_component import STORE_PATH
    from refund_agent.audit.audits_explanations_holds import AuditHolds
    from refund_agent.audit.audits_explanations_month import START, constructed_month, run_month
    from refund_agent.audit.audits_explanations_records import ReviewRoute, SegmentAssignments
    from refund_agent.audit.audits_explanations_stage6 import AuditInputs
    from refund_agent.batch_stream.generate import snapshot_log
    from refund_agent.observability_reliability.silence import region_of
    from refund_agent.security.security_privacy_experiment import runtime

    oid = "audits-explanations-o-0000"
    segments = SegmentAssignments(
        by_order={oid: "ABC"[region_of(oid)]}, version="chapter34-region_of"
    )
    month = constructed_month(segments, seed=7, n=1)
    before = month.model_dump()
    rule = AuditRule(signed_at=START - timedelta(days=1))
    register = PolicyRegister()
    for policy in month.data.policies:
        register.announce(policy, policy.effective_at)
    snapshot = register.snapshot(month.period.end + timedelta(days=30))
    candidate = build_candidate(build_suite(trials=1, components=(
        CONTRACTS, events(), CANARY,
        promotion(snapshot, [p.version for p in month.data.policies]),
        observe(), SCOPE, audits(rule),
    )))

    def context(local, request):
        at = month.evaluation_times[request.request_id]
        issuer, security = runtime(local)
        return {
            "event_log": snapshot_log(local),
            "security_privacy": security,
            "security_credential": issuer.issue_request(request, security.accounts, at),
            "audits_explanations": AuditInputs(
                rule, segments, snapshot, month.history,
                {request.request_id: ReviewRoute(
                    request_id=request.request_id, ticket_id="first-request-review",
                    deadline=at + timedelta(days=2),
                )},
                security.accounts.may_act, AuditHolds(STORE_PATH),
            ),
        }

    result = run_month(month, candidate, context)
    assert len(result.traces) == len(result.observations) == 1
    assert result.traces[0].decision.request_id == "audits-explanations-r-7-0000"
    assert result.traces[0].decision.reason == "eligible"
    assert result.traces[0].decision.executed
    assert month.model_dump() == before
