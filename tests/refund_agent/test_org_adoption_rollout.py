"""Chapter 38 contracts, joins, signed authority and the actual balance gate."""

from datetime import datetime, timedelta, timezone
from itertools import product

import pytest

from refund_agent import contracts as c
from refund_agent.agent import Agent
from refund_agent.canary.cloud_native_deployment_plan import in_cohort
from refund_agent.model import ScriptedModel
from refund_agent.release.candidate import build_candidate
from refund_agent.release.suite import build_suite
from refund_agent.rollout.org_adoption_authority import (
    AuthorityTools,
    BudgetExhaustion,
    StageAuthority,
)
from refund_agent.rollout.org_adoption_desk import (
    EXERCISE_BALANCE,
    TEXT_BALANCE,
    balance_case,
    balance_data,
    decision_entries,
)
from refund_agent.rollout.org_adoption_matching import (
    governed_payment,
    index_window,
)
from refund_agent.rollout.org_adoption_plan import (
    EVIDENCE_TYPES,
    ROLES,
    EvidenceRef,
    Owner,
    RolloutPlan,
    Step,
    validate,
)
from refund_agent.rollout.org_adoption_provider import Provider
from refund_agent.rollout.org_adoption_reconcile import reconcile
from refund_agent.rollout.org_adoption_records import (
    Cause,
    DecisionEntry,
    EvidenceRecord,
    Intake,
    ProviderPayment,
    Window,
)
from refund_agent.rollout.org_adoption_rule import Change, decide, next_window
from refund_agent.tools import Tools

START = datetime(2026, 11, 29, tzinfo=timezone.utc)


def plan_fixture(candidate="constructed-test-candidate"):
    steps, evidence = [], []
    for i, share in enumerate((0.25, 1.0)):
        start = START + timedelta(days=45 * i)
        period_end = start - timedelta(days=31)
        refs = tuple(
            EvidenceRef(
                record_type=kind,
                record_id=f"test-{kind}-{i}",
                period_start=period_end - timedelta(days=28 if not i else 14)
                if kind == "audit"
                else None,
                period_end=period_end if kind == "audit" else None,
            )
            for kind in EVIDENCE_TYPES
        )
        step = Step(
            step_id=f"org-adoption-step-{i + 1}",
            requester_share=share,
            tiers=(1, 2),
            window=Window(
                window_id=f"org-adoption-window-{i + 1}",
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
                    person="morgan" if role != "reviewer" else "alex",
                    work="review, reconcile or decide as assigned",
                )
                for role in ROLES
            ),
            manual_route="authenticated review and audit; payment through refund.execute",
        )
        steps.append(step)
        for ref in refs:
            evidence.append(
                EvidenceRecord(
                    record_type=ref.record_type,
                    record_id=ref.record_id,
                    candidate_fingerprint=candidate,
                    signed_by="morgan",
                    signed_at=start - timedelta(hours=1),
                    status="pass",
                    period_start=ref.period_start,
                    period_end=ref.period_end,
                    labels_mature_at=start - timedelta(days=1)
                    if ref.record_type == "audit"
                    else None,
                    denials={"A": 7, "B": 0, "C": 0} if ref.record_type == "audit" else {},
                    denials_read={"A": 7, "B": 0, "C": 0} if ref.record_type == "audit" else {},
                )
            )
    plan = RolloutPlan(
        plan_id="org-adoption-test-plan",
        canary_id="org-adoption-test-canary",
        candidate_fingerprint=candidate,
        signed_by="morgan",
        signed_at=START - timedelta(days=32),
        steps=tuple(steps),
        widen_rule="zero paid around; governed share at least 99%",
        hold_rule="hold and record causes; next window names a change",
        reverse_rule="forbidden payment, budget spent, tier2 unbound approval",
    )
    return plan, tuple(evidence)


def entry(request, at=None, *, person=False, outcome=c.Outcome.DENY):
    chosen = c.Decision(
        request_id=request.request_id,
        outcome=outcome,
        amount_cents=request.amount_cents,
        tier=c.Tier.AGENT,
        reason="already_delivered"
        if outcome == c.Outcome.DENY
        else "approval_required",
        policy_citation="refund_cap_v4@2026-09-01#eligibility",
    )
    audit = c.AuditRecord(
        audit_id="audit-" + request.request_id,
        trace_id="trace-" + request.request_id,
        at=at or request.at,
        decision=chosen,
        freshness={},
        risk_score=None,
        approval_ids=[],
    )
    return DecisionEntry(
        audit=audit,
        path="governed",
        actor_id="alex" if person else "refund-agent",
        review_id="review-1" if person else None,
        authorization_id="verified-credential-alex" if person else None,
    )


def records(n, missing=0):
    window = plan_fixture()[0].steps[0].window
    intake, decisions = [], []
    for i in range(n):
        request = c.Request(
            request_id=f"org-adoption-r-{i}",
            requester_id=f"person-{i}",
            order_reference=f"org-adoption-o-{i}",
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
        if i >= missing:
            decisions.append(entry(request, person=True))
    return intake, decisions, window


def causes_for(count):
    ids = [p.payment_id for p in count.around] + list(count.missing)
    return tuple(
        Cause(
            exception_id=key,
            cause="constructed test cause",
            owner="morgan",
            found_at=count.window.closed_at,
            recorded_at=count.window.closed_at,
            disposition="accepted_open",
        )
        for key in ids
    )


def test_plan_accepts_complete_timely_resolved_records():
    plan, evidence = plan_fixture()
    assert validate(plan, evidence) == plan


@pytest.mark.parametrize(
    "change",
    ["late", "shrink", "owner", "empty", "threshold", "duration", "types"],
)
def test_plan_refuses_invalid_settings(change):
    plan, evidence = plan_fixture()
    index = 1 if change == "shrink" else 0
    step = plan.steps[index]
    updates = {
        "late": {"signed_at": step.window.start},
        "shrink": {"requester_share": 0.10},
        "owner": {"owners": step.owners[:-1]},
        "empty": {"manual_route": ""},
        "threshold": {"governed_basis_points": 9800},
        "duration": {
            "window": step.window.model_copy(
                update={"end": step.window.end - timedelta(days=1)}
            )
        },
        "types": {"required": step.required[:-1]},
    }[change]
    bad = plan.model_copy(
        update={
            "steps": tuple(
                row.model_copy(update=updates) if i == index else row
                for i, row in enumerate(plan.steps)
            )
        }
    )
    with pytest.raises(ValueError):
        validate(bad, evidence)


def test_same_hash_expansion_includes_canary_and_previous_step():
    for i in range(1000):
        requester = f"requester-{i}"
        if in_cohort("fixed", requester):
            assert in_cohort("fixed", requester, 0.25)


@pytest.mark.parametrize(
    "change",
    [
        "missing",
        "late",
        "candidate",
        "failed",
        "wrong_period",
        "immature",
        "sample",
    ],
)
def test_plan_refuses_bad_evidence(change):
    plan, evidence = plan_fixture()
    if change == "missing":
        evidence = evidence[1:]
    else:
        index = 2 if change in ("wrong_period", "immature", "sample") else 0
        row = evidence[index]
        update = {
            "late": {"signed_at": START},
            "candidate": {"candidate_fingerprint": "different"},
            "failed": {"status": "hold"},
            "wrong_period": {"period_start": START - timedelta(days=70)},
            "immature": {"labels_mature_at": START + timedelta(days=1)},
            "sample": {"denials_read": {"A": 6, "B": 0, "C": 0}},
        }[change]
        evidence = (
            evidence[:index]
            + (row.model_copy(update=update),)
            + evidence[index + 1 :]
        )
    with pytest.raises(ValueError):
        validate(plan, evidence)


def test_audit_cannot_claim_maturity_before_thirty_day_payment_labels():
    plan, evidence = plan_fixture()
    audit = evidence[2]
    bad = audit.model_copy(
        update={"labels_mature_at": audit.period_end + timedelta(days=1)}
    )
    with pytest.raises(ValueError, match="30-day"):
        validate(plan, evidence[:2] + (bad,) + evidence[3:])


def test_imported_prior_receipt_remains_paid_around_path():
    intake, entries, window = records(1)
    payment = ProviderPayment(
        payment_id="console-1",
        order_id=intake[0].order_id,
        amount_cents=4000,
        at=START + timedelta(hours=2),
        request_id=intake[0].request.request_id,
        channel="console",
    )
    prior = c.Receipt(
        receipt_id=payment.payment_id,
        order_id=payment.order_id,
        amount_cents=payment.amount_cents,
        at=payment.at,
        idempotency_key="prior-console-1",
    )
    count = reconcile(intake, entries, [prior], [payment], window)
    assert count.around == (payment,) and count.governed == 0


def test_phone_request_is_in_intake_denominator_and_person_denial_counts():
    intake, entries, window = records(80, missing=1)
    count = reconcile(intake, entries, [], [], window)
    assert (
        count.received,
        count.governed,
        len(count.missing),
        count.governed_percent,
    ) == (80, 79, 1, 98.75)


def test_person_entry_requires_authenticated_review_record():
    intake, entries, _ = records(1)
    with pytest.raises(ValueError, match="authenticated"):
        DecisionEntry(audit=entries[0].audit, path="governed", actor_id="alex")


def test_human_payment_through_guarded_tool_is_governed():
    data = balance_data(TEXT_BALANCE)
    request = data.requests[0].model_copy(update={"preparer_id": "alex"})
    tools, provider = (
        Tools(data.model_copy(update={"requests": [request]})),
        Provider(),
    )
    trace = Agent(tools).run(request)
    assert trace.receipt is not None
    provider.record_tools(tools)
    at = request.at
    count = reconcile(
        [
            Intake(
                request=request,
                order_id=data.orders[0].order_id,
                received_at=at,
                decision_deadline=at + timedelta(hours=4),
            )
        ],
        decision_entries(
            tools,
            actor_id="alex",
            review_id="review-person",
            authorization_id="verified-alex",
        ),
        list(tools.ledger.values()),
        provider.payments,
        Window(
            window_id="person",
            start=at - timedelta(hours=1),
            end=at + timedelta(hours=1),
            closed_at=at + timedelta(hours=1),
        ),
    )
    assert count.governed == 1 and not count.around


def test_referred_pending_deadline_and_later_payment_are_retained():
    intake, entries, window = records(2)
    referred = entry(intake[0].request, outcome=c.Outcome.REFER)
    late = entry(intake[1].request, at=window.end + timedelta(hours=1))
    payment = ProviderPayment(
        payment_id="late-console",
        order_id=intake[1].order_id,
        amount_cents=4000,
        at=window.end + timedelta(hours=1),
        request_id=intake[1].request.request_id,
        channel="console",
    )
    count = reconcile(intake, [referred, late], [], [payment], window)
    assert (
        count.pending[0].overdue
        and count.pending[0].decision_deadline == intake[0].decision_deadline
    )
    assert count.referred == (intake[0].request.request_id,)
    assert count.late_decisions == (intake[1].request.request_id,)
    assert (
        count.payments_after_window == ("late-console",)
        and len(count.around) == 1
    )


@pytest.mark.parametrize("path", ["governed", "shadow"])
def test_unmatched_provider_and_decision_are_exceptions_before_causes(path):
    intake, entries, window = records(1)
    stranger = intake[0].request.model_copy(update={"request_id": "orphan"})
    payment = ProviderPayment(
        payment_id="orphan-money",
        order_id="unknown",
        amount_cents=4000,
        at=START + timedelta(hours=2),
        request_id=None,
        channel="console",
    )
    count = reconcile(
        intake,
        entries + [entry(stranger).model_copy(update={"path": path})],
        [],
        [payment],
        window,
    )
    assert set(count.exceptions) == {
        "decision_without_intake:orphan",
        "payment_without_unique_intake:orphan-money",
    }


def test_conflicting_record_identity_refuses_count_and_duplicate_is_idempotent():
    intake, entries, window = records(1)
    assert reconcile(intake * 2, entries * 2, [], [], window).received == 1
    with pytest.raises(ValueError, match="conflicting"):
        reconcile(
            intake + [intake[0].model_copy(update={"order_id": "other"})],
            entries,
            [],
            [],
            window,
        )


def test_shadow_prediction_is_not_a_persons_decision_record():
    intake, entries, window = records(1)
    shadow = entries[0].model_copy(
        update={"path": "shadow", "actor_id": "refund-agent"}
    )
    count = reconcile(intake, [shadow], [], [], window)
    assert count.governed == 0
    assert count.missing == (intake[0].request.request_id,)


def test_not_yet_due_request_remains_pending_at_closing_snapshot():
    intake, _, window = records(1)
    waiting = intake[0].model_copy(
        update={"decision_deadline": window.closed_at + timedelta(hours=1)}
    )
    count = reconcile([waiting], [], [], [], window)
    assert len(count.pending) == 1 and not count.pending[0].overdue
    assert count.pending[0].decision_deadline == waiting.decision_deadline
    assert not count.missing


@pytest.mark.parametrize(
    "n,missing,expected",
    [(100, 0, "widen"), (100, 1, "widen"), (80, 1, "hold")],
)
def test_unpaid_gap_uses_signed_threshold(n, missing, expected):
    plan, evidence = plan_fixture()
    intake, entries, window = records(n, missing)
    count = reconcile(intake, entries, [], [], window)
    verdict = decide(
        plan, plan.steps[0], count, evidence, causes=causes_for(count)
    )
    assert verdict.outcome == expected


def test_paid_workaround_holds_even_above_threshold():
    plan, evidence = plan_fixture()
    intake, entries, window = records(200)
    payment = ProviderPayment(
        payment_id="around",
        order_id=intake[0].order_id,
        amount_cents=4000,
        at=START + timedelta(hours=2),
        request_id=intake[0].request.request_id,
        channel="console",
    )
    count = reconcile(intake, entries, [], [payment], window)
    verdict = decide(
        plan, plan.steps[0], count, evidence, causes=causes_for(count)
    )
    assert count.governed_percent == 99.5 and verdict.outcome == "hold"
    assert verdict.reasons == ("paid_around_path",)


@pytest.mark.parametrize("identified", [True, False])
def test_older_request_paid_in_window_holds_without_changing_intake(
    identified,
):
    plan, evidence = plan_fixture()
    intake, entries, window = records(100)
    old_request = intake[0].request.model_copy(
        update={
            "request_id": "older-request",
            "order_reference": "older-order",
            "at": window.start - timedelta(days=2),
        }
    )
    intake.append(
        Intake(
            request=old_request,
            order_id=old_request.order_reference,
            received_at=old_request.at,
            decision_deadline=old_request.at + timedelta(hours=24),
        )
    )
    entries.append(entry(old_request))
    payment = ProviderPayment(
        payment_id="older-console-payment",
        order_id=old_request.order_reference,
        amount_cents=4000,
        at=window.start + timedelta(hours=2),
        request_id=old_request.request_id if identified else None,
        channel="console",
    )
    count = reconcile(intake, entries, [], [payment], window)
    assert (count.received, count.governed) == (100, 100)
    assert count.around == (payment,) and not count.exceptions
    verdict = decide(
        plan, plan.steps[0], count, evidence, causes=causes_for(count)
    )
    assert verdict.outcome == "hold"
    assert verdict.reasons == ("paid_around_path",)


@pytest.mark.parametrize("other_current", [False, True])
@pytest.mark.parametrize("when", ["before", "during", "after"])
def test_shared_order_payment_is_retained_as_exception_and_holds(
    other_current, when
):
    plan, evidence = plan_fixture()
    n = 1 if other_current else 100
    intake, entries, window = records(n)
    shared_at = (
        window.start + timedelta(hours=2)
        if other_current
        else (window.start - timedelta(days=2))
    )
    other = intake[0].request.model_copy(
        update={"request_id": "shared-order-request", "at": shared_at}
    )
    shared = Intake(
        request=other,
        order_id=intake[0].order_id,
        received_at=shared_at,
        decision_deadline=shared_at + timedelta(hours=24),
    )
    payment = ProviderPayment(
        payment_id="shared-order-console",
        order_id=shared.order_id,
        amount_cents=4000,
        at={
            "before": window.start - timedelta(hours=1),
            "during": window.start + timedelta(hours=3),
            "after": window.end + timedelta(hours=1),
        }[when],
        request_id=None,
        channel="console",
    )
    unambiguous = reconcile(intake, entries, [], [payment], window)
    assert (unambiguous.received, unambiguous.governed) == (n, n - 1)
    assert unambiguous.around == (payment,) and not unambiguous.exceptions
    count = reconcile(
        intake + [shared], entries + [entry(other)], [], [payment], window
    )
    assert (count.received, count.governed) == (
        2 if other_current else 100,
        2 if other_current else 100,
    )
    assert count.around == (payment,)
    assert count.exceptions == (
        "payment_without_unique_intake:shared-order-console",
    )
    assert count.payments_after_window == (
        (payment.payment_id,) if when == "after" else ()
    )
    verdict = decide(
        plan, plan.steps[0], count, evidence, causes=causes_for(count)
    )
    assert verdict.outcome == "hold"
    assert verdict.reasons == (
        "paid_around_path",
        "unmatched_record_exceptions",
    )


@pytest.mark.parametrize("attribution", ["known", "unknown", "mismatched"])
@pytest.mark.parametrize("late", [False, True])
def test_current_order_does_not_reassign_valid_older_request(
    attribution, late
):
    plan, evidence = plan_fixture()
    intake, entries, window = records(100)
    other = intake[0].request.model_copy(
        update={
            "request_id": "older-request",
            "order_reference": "other-order"
            if attribution == "mismatched"
            else (intake[0].order_id),
            "at": window.start - timedelta(days=2),
        }
    )
    intake.append(
        Intake(
            request=other,
            order_id=other.order_reference,
            received_at=other.at,
            decision_deadline=other.at + timedelta(hours=24),
        )
    )
    entries.append(entry(other))
    payment = ProviderPayment(
        payment_id="attributed-console",
        order_id=intake[0].order_id,
        amount_cents=4000,
        at=window.end + timedelta(hours=1)
        if late
        else (window.start - timedelta(hours=1)),
        request_id="unknown-request"
        if attribution == "unknown"
        else (other.request_id),
        channel="console",
    )
    count = reconcile(intake, entries, [], [payment], window)
    assert (count.received, count.governed) == (100, 100)
    assert count.around == (() if attribution == "known" else (payment,))
    assert (
        count.exceptions
        == {
            "known": (),
            "unknown": ("payment_without_unique_intake:attributed-console",),
            "mismatched": ("payment_order_mismatch:attributed-console",),
        }[attribution]
    )
    assert decide(
        plan, plan.steps[0], count, evidence, causes=causes_for(count)
    ).outcome == ("widen" if attribution == "known" else "hold")


@pytest.mark.parametrize("identified", [False, True])
def test_payment_before_window_for_current_order_holds(identified):
    plan, evidence = plan_fixture()
    intake, entries, window = records(100)
    payment = ProviderPayment(
        payment_id="before-window-console",
        order_id=intake[0].order_id,
        amount_cents=4000,
        at=window.start - timedelta(hours=1),
        request_id=intake[0].request.request_id if identified else None,
        channel="console",
    )
    count = reconcile(intake, entries, [], [payment], window)
    assert (count.received, count.governed) == (100, 99)
    assert count.around == (payment,) and not count.exceptions
    assert not count.payments_after_window
    assert (
        decide(
            plan, plan.steps[0], count, evidence, causes=causes_for(count)
        ).outcome
        == "hold"
    )


def test_duplicate_payment_reports_count_once_but_distinct_payments_remain():
    intake, entries, window = records(100)
    payment = ProviderPayment(
        payment_id="duplicate-console",
        order_id=intake[0].order_id,
        amount_cents=4000,
        at=window.start + timedelta(hours=2),
        request_id=intake[0].request.request_id,
        channel="console",
    )
    count = reconcile(intake, entries, [], [payment, payment], window)
    assert (count.received, count.governed) == (100, 99)
    assert count.around == (payment,) and not count.exceptions
    second = payment.model_copy(update={"payment_id": "second-console"})
    distinct = reconcile(intake, entries, [], [payment, second], window)
    assert distinct.around == (payment, second) and distinct.governed == 99
    with pytest.raises(ValueError, match="conflicting record identity"):
        reconcile(
            intake,
            entries,
            [],
            [payment, payment.model_copy(update={"amount_cents": 1})],
            window,
        )


@pytest.mark.parametrize("shared", ["none", "older", "current"])
@pytest.mark.parametrize("when", ["before", "start", "during", "end", "close"])
def test_generated_small_cases_conserve_unique_payments(shared, when):
    data = balance_data(TEXT_BALANCE)
    request = data.requests[0]
    tools, provider = Tools(data), Provider()
    assert Agent(tools).run(request).receipt is not None
    provider.record_tools(tools)
    intake = [
        Intake(
            request=request,
            order_id=request.order_reference,
            received_at=request.at,
            decision_deadline=request.at + timedelta(hours=24),
        )
    ]
    entries = list(decision_entries(tools))
    window = plan_fixture()[0].steps[0].window
    if shared != "none":
        other_at = (
            window.start + timedelta(hours=2)
            if shared == "current"
            else (window.start - timedelta(days=2))
        )
        other = request.model_copy(
            update={"request_id": "another-request", "at": other_at}
        )
        intake.append(
            Intake(
                request=other,
                order_id=request.order_reference,
                received_at=other_at,
                decision_deadline=other_at + timedelta(hours=24),
            )
        )
        entries.append(entry(other))
    at = {
        "before": window.start - timedelta(hours=1),
        "start": window.start,
        "during": window.start + timedelta(hours=3),
        "end": window.end,
        "close": window.closed_at,
    }[when]
    console = ProviderPayment(
        payment_id="known-console",
        order_id=request.order_reference,
        amount_cents=4000,
        at=at,
        request_id=request.request_id,
        channel="console",
    )
    order_only = console.model_copy(
        update={"payment_id": "order-only-console", "request_id": None}
    )
    orphan = console.model_copy(
        update={
            "payment_id": "orphan-console",
            "order_id": "unknown-order",
            "request_id": None,
            "at": window.start + timedelta(hours=3),
        }
    )
    # Labels come from the constructed records, independently of the matcher.
    labels = (
        "governed",
        "workaround",
        "exception" if shared != "none" else ("workaround"),
        "exception",
    )
    ledger = list(tools.ledger.values()) * 2
    for identified, repetitions in product(
        (False, True), product(range(3), repeat=4)
    ):
        governed = provider.payments[0].model_copy(
            update={"request_id": request.request_id if identified else None}
        )
        atoms = (governed, console, order_only, orphan)
        payments = [p for p, n in zip(atoms, repetitions) for _ in range(n)]
        expected = {
            label: {
                p.payment_id
                for p, kind, n in zip(atoms, labels, repetitions)
                if kind == label and n
            }
            for label in ("governed", "workaround", "exception")
        }
        args = (intake * 2, entries * 2, ledger, payments[::-1], window)
        count = reconcile(*args)
        _, indexed_entries, receipts, paid, _ = index_window(*args)
        payment_ids = {p.payment_id for p in payments}
        assert {p.payment_id for p, _ in paid} == payment_ids
        assert len(paid) == len(payment_ids)
        matched = {
            p.payment_id
            for p, key in paid
            if governed_payment(p, key, receipts, indexed_entries)
        }
        around = {p.payment_id for p in count.around}
        assert len(count.around) == len(around)
        exceptions = {reason.split(":", 1)[1] for reason in count.exceptions}
        assert matched == expected["governed"]
        assert exceptions == expected["exception"]
        # around reports all money without governed proof; exceptions tag uncertainty.
        assert around == expected["workaround"] | expected["exception"]
        categories = (matched, around - exceptions, exceptions)
        assert set.union(*categories) == payment_ids
        assert sum(map(len, categories)) == len(payment_ids)
        money = {p.payment_id: p.amount_cents for p in payments}
        assert sum(p.amount_cents for p in count.around) + sum(
            money[key] for key in matched
        ) == sum(money.values())
        received = 2 if shared == "current" else 1
        affected = bool(
            repetitions[1] or (shared == "none" and repetitions[2])
        )
        assert (count.received, count.governed) == (
            received,
            received - affected,
        )


@pytest.mark.parametrize(
    "condition",
    ["forbidden_payment", "budget_spent", "tier2_unbound_approval"],
)
def test_reversal_conditions_move_back(condition):
    plan, evidence = plan_fixture()
    intake, entries, window = records(1)
    count = reconcile(intake, entries, [], [], window)
    assert (
        decide(
            plan, plan.steps[0], count, evidence, reversals=[condition]
        ).outcome
        == "move_back"
    )


def test_bound_spent_budget_record_itself_moves_back():
    plan, evidence = plan_fixture()
    intake, entries, window = records(1)
    spent = evidence[0].model_copy(update={"status": "spent"})
    result = decide(
        plan,
        plan.steps[0],
        reconcile(intake, entries, [], [], window),
        (spent,) + evidence[1:],
    )
    assert result.outcome == "move_back" and "budget_spent" in result.reasons


def test_missing_record_holds_and_next_window_needs_change_after_hold():
    plan, _ = plan_fixture()
    intake, entries, window = records(1)
    held = decide(
        plan, plan.steps[0], reconcile(intake, entries, [], [], window), []
    )
    assert held.outcome == "hold" and any(
        "missing required record" in r for r in held.reasons
    )
    later = window.model_copy(
        update={
            "window_id": "next",
            "start": held.at + timedelta(days=1),
            "end": held.at + timedelta(days=15),
            "closed_at": held.at + timedelta(days=16),
        }
    )
    with pytest.raises(ValueError, match="change"):
        next_window(held, None, later)
    change = Change(
        change_id="cover",
        description="name absent reviewer's cover",
        owner="morgan",
        at=held.at + timedelta(hours=1),
        held_window_id=held.window_id,
    )
    assert next_window(held, change, later) == later
    with pytest.raises(ValueError, match="follow hold"):
        next_window(held, change.model_copy(update={"at": held.at}), later)


@pytest.mark.parametrize(
    "fixture,delayed,excess,early_balance",
    [
        (TEXT_BALANCE, 8000, 4000, 0),
        (EXERCISE_BALANCE, 10500, 1500, 3000),
    ],
)
def test_real_gate_delayed_and_early_settlement(
    fixture, delayed, excess, early_balance
):
    late, early = (
        balance_case(fixture, early=False),
        balance_case(fixture, early=True),
    )
    assert (
        late["balance_read_cents"],
        late["agent_paid_cents"],
        late["total_refunded_cents"],
        late["excess_cents"],
    ) == (fixture.paid_cents, fixture.request_cents, delayed, excess)
    assert late["agent_payments"] == 1 and len(late["count"].around) == 1
    assert (
        early["balance_read_cents"] == early_balance
        and early["agent_paid_cents"] == 0
    )
    assert (
        early["reason"] == "amount_over_policy"
        and early["imported_receipts"] == 1
    )
    assert len(early["count"].around) == 1 and early["count"].governed == 0


def authority_fixture(tmp_path):
    plan, evidence = plan_fixture()
    authority = StageAuthority(plan, tmp_path / "org-adoption-authority.jsonl")
    return authority, plan, evidence


def requester_at_share(plan, *, inside):
    return next(
        f"person-{i}"
        for i in range(10000)
        if in_cohort(plan.canary_id, f"person-{i}", 0.25) == inside
        and not in_cohort(plan.canary_id, f"person-{i}")
    )


def test_scope_requires_signed_record_and_reloads(tmp_path):
    authority, plan, evidence = authority_fixture(tmp_path)
    request = (
        balance_data(TEXT_BALANCE)
        .requests[0]
        .model_copy(
            update={"requester_id": requester_at_share(plan, inside=True)}
        )
    )
    assert authority.route(request, request.at) == "person"
    authority.sign_step(
        plan.steps[0].step_id,
        by="morgan",
        at=START - timedelta(minutes=30),
        evidence=evidence,
    )
    assert authority.route(request, request.at) == "agent"
    reloaded = StageAuthority(plan, authority.path)
    assert reloaded.route(request, request.at) == "agent"
    outside = request.model_copy(
        update={"requester_id": requester_at_share(plan, inside=False)}
    )
    assert reloaded.route(outside, request.at) == "person"


@pytest.mark.parametrize("scope", [None, "tier1"])
def test_forbidden_payment_withdraws_only_tier1(tmp_path, scope):
    authority, plan, evidence = authority_fixture(tmp_path)
    authority.sign_step(
        plan.steps[0].step_id,
        by="morgan",
        at=START - timedelta(minutes=30),
        evidence=evidence,
    )
    request = (
        balance_data(TEXT_BALANCE)
        .requests[0]
        .model_copy(
            update={"requester_id": requester_at_share(plan, inside=True)}
        )
    )
    tier2 = request.model_copy(update={"amount_cents": 5000})
    assert authority.route(request, request.at) == "agent"
    assert authority.route(tier2, tier2.at) == "agent"
    withdrawal = authority.withdraw(
        "forbidden_payment",
        by="morgan",
        at=START,
        record_id="forbidden-tier1",
        scope=scope,
    )
    assert withdrawal.scope == "tier1"
    reloaded = StageAuthority(plan, authority.path)
    assert reloaded.events()[-1] == withdrawal
    assert reloaded.reason(request, request.at) == "org_adoption_withdrawn"
    assert reloaded.route(tier2, tier2.at) == "agent"


@pytest.mark.parametrize("scope", ["tier2", "all"])
def test_forbidden_payment_refuses_other_withdrawal_scopes(tmp_path, scope):
    authority, _, _ = authority_fixture(tmp_path)
    before = authority.events()
    with pytest.raises(ValueError, match="scope differs"):
        authority.withdraw(
            "forbidden_payment",
            by="morgan",
            at=START,
            record_id="wrong-forbidden-scope",
            scope=scope,
        )
    assert authority.events() == before


def test_condition4_withdrawal_sends_every_tier_to_person(tmp_path):
    authority, plan, evidence = authority_fixture(tmp_path)
    authority.sign_step(
        plan.steps[0].step_id,
        by="morgan",
        at=START - timedelta(minutes=30),
        evidence=evidence,
    )
    request = (
        balance_data(TEXT_BALANCE)
        .requests[0]
        .model_copy(
            update={"requester_id": requester_at_share(plan, inside=True)}
        )
    )
    requests = tuple(
        request.model_copy(update={"amount_cents": amount})
        for amount in (4000, 5000, 50001)
    )
    assert [authority.reason(r, r.at) for r in requests] == [
        None,
        None,
        "org_adoption_person_required",
    ]
    withdrawal = authority.withdraw(
        "owner",
        by="morgan",
        at=START,
        record_id="tier3-without-named-approver",
        scope="all",
    )
    assert withdrawal.scope == "all"
    reloaded = StageAuthority(plan, authority.path)
    for item in requests:
        assert reloaded.reason(item, item.at) == "org_adoption_withdrawn"
        assert reloaded.route(item, item.at) == "person"


@pytest.mark.parametrize("condition", ["budget_spent", "forbidden_payment"])
@pytest.mark.parametrize("reason", ["damaged", "not_delivered"])
def test_withdrawal_stops_payment_and_denial_beyond_canary_without_refunding(
    tmp_path, reason, condition
):
    authority, plan, evidence = authority_fixture(tmp_path)
    authority.sign_step(
        plan.steps[0].step_id,
        by="morgan",
        at=START - timedelta(minutes=30),
        evidence=evidence,
    )
    data = balance_data(TEXT_BALANCE)
    request = data.requests[0].model_copy(
        update={
            "requester_id": requester_at_share(plan, inside=True),
            "reason": reason,
            "at": data.requests[0].at + timedelta(minutes=2),
        }
    )
    kind = type("OrgAdoptionTestTools", (AuthorityTools, Tools), {})
    tools = kind(
        data.model_copy(update={"requests": [request]}),
        stage_authority=authority,
        org_adoption_candidate=build_candidate(build_suite(trials=1)),
        org_adoption_context={},
    )
    # Pay a different request before withdrawing; moving back refunds nothing.
    seed = request.model_copy(
        update={
            "request_id": "paid-before-withdrawal",
            "order_reference": "paid-before-order",
            "reason": "damaged",
            "at": request.at - timedelta(minutes=2),
        }
    )
    tools.data = tools.data.model_copy(
        update={
            "requests": [seed, request],
            "orders": [
                *data.orders,
                data.orders[0].model_copy(
                    update={
                        "order_id": seed.order_reference,
                        "reference": seed.order_reference,
                        "payment_id": "paid-before-purchase",
                    }
                ),
            ],
            "shipments": [
                *data.shipments,
                data.shipments[0].model_copy(
                    update={"order_id": seed.order_reference}
                ),
            ],
            "retrievals": [
                *data.retrievals,
                data.retrievals[0].model_copy(
                    update={"request_id": seed.request_id}
                ),
            ],
        }
    )
    seeded = Agent(tools).run(seed)
    assert seeded.receipt is not None and len(tools.payments) == 1
    before = tools.ledger
    authority.withdraw(
        condition,
        by="morgan",
        at=request.at - timedelta(minutes=1),
        record_id="spent-1",
    )
    assert tools.ledger == before
    trace = Agent(tools).run(request)
    assert trace.decision.outcome == c.Outcome.REFER
    assert (
        trace.decision.reason == "org_adoption_withdrawn"
        and tools.ledger == before
    )
    assert len(tools.shadow_traces) == 1
    assert tools.shadow_traces[0].decision.outcome == (
        c.Outcome.APPROVE if reason == "damaged" else c.Outcome.DENY
    )
    assert not tools.shadow_recorder.payments


def test_tier2_withdrawal_and_tier3_never_gain_agent_authority(tmp_path):
    authority, plan, evidence = authority_fixture(tmp_path)
    authority.sign_step(
        plan.steps[0].step_id,
        by="morgan",
        at=START - timedelta(minutes=30),
        evidence=evidence,
    )
    request = (
        balance_data(TEXT_BALANCE)
        .requests[0]
        .model_copy(
            update={
                "requester_id": requester_at_share(plan, inside=True),
                "amount_cents": 5000,
            }
        )
    )
    assert authority.route(request, request.at) == "agent"
    authority.withdraw(
        "tier2_unbound_approval", by="morgan", at=START, record_id="unbound"
    )
    assert authority.route(request, request.at) == "person"
    tier3 = request.model_copy(update={"amount_cents": 50001})
    assert authority.route(tier3, tier3.at) == "person"


@pytest.mark.parametrize(
    "wrong_identity", ["plan", "budget", "window", "cross_step"]
)
def test_budget_callback_binds_plan_window_and_remains_closed_after_reload(
    tmp_path, wrong_identity
):
    authority, plan, evidence = authority_fixture(tmp_path)
    authority.sign_step(
        plan.steps[0].step_id,
        by="morgan",
        at=START - timedelta(minutes=30),
        evidence=evidence,
    )
    status = BudgetExhaustion(
        budget_id="test-budget-0",
        plan_id=plan.plan_id,
        window_id=plan.steps[0].window.window_id,
        at=START,
        spent=True,
    )
    updates = {
        "plan": {"plan_id": "other"},
        "budget": {"budget_id": "other-budget"},
        "window": {"window_id": "other-window"},
        "cross_step": {"window_id": plan.steps[1].window.window_id},
    }[wrong_identity]
    message = (
        "another rollout"
        if wrong_identity == "plan"
        else "another budget/window"
    )
    with pytest.raises(ValueError, match=message):
        authority.on_budget_spent(status.model_copy(update=updates))
    assert len(authority.events()) == 1
    authority.on_budget_spent(status)
    authority.on_budget_spent(status)
    assert len(authority.events()) == 2
    reloaded = StageAuthority(plan, authority.path)
    request = (
        balance_data(TEXT_BALANCE)
        .requests[0]
        .model_copy(
            update={"requester_id": requester_at_share(plan, inside=True)}
        )
    )
    assert reloaded.route(request, request.at) == "person"


def test_latest_signed_step_expands_scope_only_after_prior_widening(tmp_path):
    authority, plan, evidence = authority_fixture(tmp_path)
    first, second = plan.steps
    authority.sign_step(
        first.step_id,
        by="morgan",
        at=START - timedelta(minutes=30),
        evidence=evidence,
    )
    at = second.window.start - timedelta(minutes=30)
    with pytest.raises(ValueError, match="previous window"):
        authority.sign_step(
            second.step_id, by="morgan", at=at, evidence=evidence
        )
    intake, entries, window = records(1)
    previous = decide(
        plan, first, reconcile(intake, entries, [], [], window), evidence
    )
    held = decide(plan, first, reconcile(intake, entries, [], [], window), ())
    assert held.outcome == "hold"
    with pytest.raises(ValueError, match="previous window"):
        authority.sign_step(
            second.step_id,
            by="morgan",
            at=at,
            evidence=evidence,
            previous=held,
        )
    assert len(authority.events()) == 1
    authority.sign_step(
        second.step_id,
        by="morgan",
        at=at,
        evidence=evidence,
        previous=previous,
    )
    request = (
        balance_data(TEXT_BALANCE)
        .requests[0]
        .model_copy(
            update={"requester_id": requester_at_share(plan, inside=False)}
        )
    )
    assert (
        authority.route(request, first.window.start + timedelta(hours=1))
        == "person"
    )
    assert (
        authority.route(request, second.window.start + timedelta(hours=1))
        == "agent"
    )


def test_hold_change_recorded_retry_widen_preserves_original_hold(tmp_path):
    authority, plan, evidence = authority_fixture(tmp_path)
    first, second = plan.steps
    authority.sign_step(
        first.step_id,
        by="morgan",
        at=START - timedelta(minutes=30),
        evidence=evidence,
    )
    intake, entries, window = records(200)
    payment = ProviderPayment(
        payment_id="held-console",
        order_id=intake[0].order_id,
        amount_cents=4000,
        at=intake[0].received_at,
        request_id=intake[0].request.request_id,
        channel="console",
    )
    count = reconcile(intake, entries, [], [payment], window)
    held = decide(plan, first, count, evidence, causes=causes_for(count))
    assert held.outcome == "hold" and held.reasons == ("paid_around_path",)
    change = Change(
        change_id="named-absence-cover",
        description="name Morgan as cover on Alex's absent days",
        owner="morgan",
        at=held.at + timedelta(hours=1),
        held_window_id=held.window_id,
    )
    replacement = window.model_copy(
        update={
            "window_id": "org-adoption-retry-window",
            "start": held.at + timedelta(days=1),
            "end": held.at + timedelta(days=15),
            "closed_at": held.at + timedelta(days=16),
        }
    )
    for bad_hold in (
        held.model_copy(update={"plan_id": "other-plan"}),
        held.model_copy(update={"window_id": "unrecorded-window"}),
        held.model_copy(update={"outcome": "widen"}),
    ):
        with pytest.raises(ValueError):
            authority.retry_window(
                bad_hold,
                change,
                replacement,
                by="morgan",
                at=change.at + timedelta(minutes=1),
            )
        assert len(authority.events()) == 1
    retry = authority.retry_window(
        held,
        change,
        next_window(held, change, replacement),
        by="morgan",
        at=change.at + timedelta(minutes=1),
    )
    reloaded = StageAuthority(plan, authority.path)
    assert reloaded.events()[-1] == retry
    assert retry.held == held and retry.change == change
    retry_at = replacement.start + timedelta(hours=1)
    shifted_intake = [
        Intake(
            request=row.request.model_copy(
                update={
                    "request_id": "retry-" + row.request.request_id,
                    "at": retry_at,
                }
            ),
            order_id=row.order_id,
            received_at=retry_at,
            decision_deadline=retry_at + timedelta(hours=24),
        )
        for row in intake
    ]
    shifted_entries = [
        entry(row.request, at=row.received_at) for row in shifted_intake
    ]
    clean = reconcile(
        intake + shifted_intake,
        entries + shifted_entries,
        [],
        [payment],
        replacement,
    )
    assert (clean.received, clean.governed) == (200, 200)
    assert not clean.around and retry.held == held
    with pytest.raises(ValueError, match="exact step window"):
        decide(plan, first, clean, evidence)
    widened = decide(plan, first, clean, evidence, retry=retry)
    assert widened.outcome == "widen" and not widened.reasons
    with pytest.raises(ValueError, match="held plan"):
        decide(
            plan,
            first,
            clean,
            evidence,
            retry=retry.model_copy(update={"plan_fingerprint": "other"}),
        )
    reloaded.sign_step(
        second.step_id,
        by="morgan",
        at=second.window.start - timedelta(minutes=30),
        evidence=evidence,
        previous=widened,
    )
    restarted = StageAuthority(plan, authority.path)
    assert restarted.events()[1].held == held
    assert restarted.events()[1].change.change_id == change.change_id
    request = intake[0].request.model_copy(
        update={"requester_id": requester_at_share(plan, inside=False)}
    )
    assert restarted.route(request, second.window.start) == "agent"
    restarted.on_budget_spent(
        BudgetExhaustion(
            budget_id="test-budget-0",
            plan_id=plan.plan_id,
            window_id=retry.window.window_id,
            at=second.window.start,
            spent=True,
        )
    )
    assert (
        StageAuthority(plan, authority.path).route(
            request, second.window.start
        )
        == "person"
    )


def test_withdrawal_between_assessment_and_execute_is_rechecked(tmp_path):
    authority, plan, evidence = authority_fixture(tmp_path)
    authority.sign_step(
        plan.steps[0].step_id,
        by="morgan",
        at=START - timedelta(minutes=30),
        evidence=evidence,
    )
    data = balance_data(TEXT_BALANCE)
    request = data.requests[0].model_copy(
        update={"requester_id": requester_at_share(plan, inside=True)}
    )
    kind = type("OrgAdoptionLastMomentTools", (AuthorityTools, Tools), {})
    tools = kind(
        data.model_copy(update={"requests": [request]}),
        stage_authority=authority,
        org_adoption_candidate=build_candidate(build_suite(trials=1)),
        org_adoption_context={},
    )

    class WithdrawAtExecute(ScriptedModel):
        def reply(self, state):
            if state.phase == c.Phase.EXECUTE:
                authority.withdraw(
                    "budget_spent",
                    by="morgan",
                    at=state.execution_at,
                    record_id="during-run",
                )
            return super().reply(state)

    trace = Agent(tools, WithdrawAtExecute()).run(request)
    assert trace.decision.outcome == c.Outcome.REFER and not tools.payments
    assert trace.decision.reason == "org_adoption_withdrawn"


def test_withdrawn_tier2_drafts_only_in_shadow_for_person(tmp_path):
    authority, plan, evidence = authority_fixture(tmp_path)
    authority.sign_step(
        plan.steps[0].step_id,
        by="morgan",
        at=START - timedelta(minutes=30),
        evidence=evidence,
    )
    data = balance_data(EXERCISE_BALANCE)
    request = data.requests[0].model_copy(
        update={
            "requester_id": requester_at_share(plan, inside=True),
            "amount_cents": 5000,
        }
    )
    authority.withdraw(
        "tier2_unbound_approval", by="morgan", at=START, record_id="tier2-stop"
    )
    kind = type("OrgAdoptionTier2Tools", (AuthorityTools, Tools), {})
    tools = kind(
        data.model_copy(update={"requests": [request]}),
        stage_authority=authority,
        org_adoption_candidate=build_candidate(build_suite(trials=1)),
        org_adoption_context={},
    )
    trace = Agent(tools).run(request)
    assert trace.decision.outcome == c.Outcome.REFER
    assert len(tools.shadow_recorder.shadow_reviews) == 1
    assert not tools.reviews and not tools.payments



def test_audit_accepts_census_of_thirty_denials_in_each_region():
    _,evidence=plan_fixture()
    row=next(r for r in evidence if r.record_type=="audit")
    complete={"A":30,"B":30,"C":30}
    checked=EvidenceRecord.model_validate({**row.model_dump(),"denials":complete,"denials_read":complete})
    assert sum(checked.denials_read.values())==90


@pytest.mark.parametrize("read", [{"A":30,"B":30,"C":20},{"A":80,"B":0,"C":0}])
def test_audit_refuses_pooled_eighty_readings_without_regional_coverage(read):
    _,evidence=plan_fixture()
    row=next(r for r in evidence if r.record_type=="audit")
    with pytest.raises(ValueError):
        EvidenceRecord.model_validate({**row.model_dump(),"denials":{"A":30,"B":30,"C":30},"denials_read":read})
