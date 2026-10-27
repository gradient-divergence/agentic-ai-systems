import json

import pytest
from capstone_blueprint_fixtures import AT, packet_fixture
from test_capstone_blueprint_decision import ConstructedAuthority

from refund_agent.launch.capstone_blueprint_budget import (
    BudgetExhaustion,
    budget_spent,
)
from refund_agent.launch.capstone_blueprint_outcomes import (
    PaymentOutcome,
    outcomes,
)
from refund_agent.launch.capstone_blueprint_timeline import Demand, operate


def constructed_driver():
    from capstone_blueprint_fixtures import OWNERS, ConstructedRegistry

    from refund_agent.canary.cloud_native_deployment_rehearsal import fixture
    from refund_agent.launch.capstone_blueprint_requirement import (
        refund_requirement,
    )
    from refund_agent.launch.capstone_blueprint_scan import LaunchLogs
    from refund_agent.launch.capstone_blueprint_schemas import (
        validate_owned_record,
    )
    from refund_agent.launch.capstone_blueprint_timeline import StageResult

    class Registry(ConstructedRegistry):
        def validate_record(self, kind, payload, claims):
            if kind in {"comparison", "tier3"}:
                return validate_owned_record(kind, payload, claims)
            return super().validate_record(kind, payload, claims)

    class ConstructedDriver:
        # Test-only callbacks establish ordering, not a substitute deployment.
        registry = Registry(refund_requirement("B6"))
        owners = OWNERS
        authority = ConstructedAuthority()
        refresh_producers = {}
        controls = ()
        open_items = ()

        def __init__(self):
            self.calls = []

        def generate_stage6(self, seed, start, days, demand):
            self.calls.append("generate")
            return fixture("capstone-blueprint", "clock", start)[0]

        def stage4(self, stream, out):
            self.calls.append("stage4")
            return StageResult(end_day=6, records=())

        def stage5(self, stream, out, after):
            self.calls.append("stage5")
            return StageResult(end_day=34, records=())

        def audit_period(self, stream, out, period):
            self.calls.append("audit")
            return StageResult(end_day=period.end + 30.0001, records=())

        def stage6_bundle(self, stream, out, after):
            self.calls.append("stage6_bundle")
            return "B6"

        def rollout_step(self, stream, out, step, period):
            self.calls.append("step" + str(step))
            self.clock = period.end
            return StageResult(end_day=period.end, records=())

        def sign_requirement(self, requirement, out):
            self.calls.append("sign")
            assert requirement.signed_day < requirement.comparison_period.start
            self.registry.requirement = requirement
            return requirement

        def month_controls(self, stream, out, period, requirement):
            self.calls.append("month_controls")
            (out / "capstone-blueprint-logs.json").write_text(
                LaunchLogs().model_dump_json()
            )
            return StageResult(end_day=max(period.end + 30, self.clock), records=())

        def launch_logs(self):
            return LaunchLogs()

    return ConstructedDriver()


def test_driver_orders_audits_and_signs_before_second_step(tmp_path):
    driver = constructed_driver()
    run = operate(7, demand=Demand(), out=tmp_path / "run", driver=driver)
    assert driver.calls == [
        "generate",
        "stage4",
        "stage5",
        "audit",
        "stage6_bundle",
        "step1",
        "audit",
        "sign",
        "step2",
        "month_controls",
    ]
    assert run.status == "operated" and run.missing_kinds == (
        "comparison",
        "tier3",
    )
    gate = json.loads(
        (run.path / "capstone-blueprint-gate-02.json").read_text()
    )
    assert (
        gate["verdict"] == "hold"
    )  # The constructed callbacks leave producer records missing.


def test_no_missing_partner_is_replaced_by_a_stub(tmp_path):
    out = tmp_path / "run"
    run = operate(7, demand=Demand(), out=out)
    assert run.status == "integration_missing" and run.requirement is None
    assert "stream" in run.missing_kinds and "comparison" in run.missing_kinds
    assert not (out / "capstone-blueprint-stream.json").exists()
    records = json.loads(
        (out / "missing/capstone-blueprint-stream.json").read_text()
    )
    assert "no tier_mix" in records["reason"]
    before = {p.name: p.read_bytes() for p in out.iterdir() if p.is_file()}
    assert operate(7, demand=Demand(), out=out) == run
    assert {
        p.name: p.read_bytes() for p in out.iterdir() if p.is_file()
    } == before
    with pytest.raises(ValueError, match="another seed"):
        operate(8, demand=Demand(), out=out)


def test_tier3_holds_without_a_signed_stage6_predecessor(tmp_path):
    run = operate(
        7, demand=Demand(), out=tmp_path / "run", driver=constructed_driver()
    )
    assert "tier3" in run.missing_kinds
    assert not (run.path / "capstone-blueprint-tier3.json").exists()
    missing = json.loads(
        (run.path / "missing/capstone-blueprint-tier3.json").read_text()
    )
    assert "signed stage 6 routing record" in missing["reason"]


@pytest.mark.parametrize("fault", [None, "late", "wrong_role", "earlier_bundle", "false"])
def test_tier3_reads_signature_and_provenance_from_stage6(tmp_path, fault):
    from capstone_blueprint_fixtures import evidence

    from refund_agent.launch.capstone_blueprint_packet import (
        assemble,
        write_record,
    )
    from refund_agent.launch.capstone_blueprint_requirement import (
        refund_requirement,
    )

    driver = constructed_driver()
    need = next(
        n
        for row in refund_requirement("B6").rows
        for n in row.records
        if n.kind == "routing"
    )

    def stage6_bundle(stream, out, after):
        changes = {
            "record_id": "signed-stage6-routing",
            "day": after + 0.0005,
            "signed_day": 33,
            "run_started_day": after + 0.0002,
        }
        if fault == "late":
            changes["signed_day"] = after + 0.002
        elif fault == "wrong_role":
            changes["signed_by"] = "alex"
        elif fault == "earlier_bundle":
            changes["bundle"] = "B5"
        elif fault == "false":
            driver.registry.false.add(changes["record_id"])
        write_record(out, evidence(need, **changes))
        return "B6"

    driver.stage6_bundle = stage6_bundle
    run = operate(7, demand=Demand(), out=tmp_path / "run", driver=driver)
    path = run.path / "capstone-blueprint-tier3.json"
    if fault is not None:
        assert "tier3" in run.missing_kinds and not path.exists()
        return
    assert "tier3" not in run.missing_kinds and path.exists()
    packet = assemble(run.path, driver.registry, run.requirement)
    source = next(r for r in packet.records if r.claims.kind == "routing")
    tier3 = next(r for r in packet.records if r.claims.kind == "tier3")
    assert (
        tier3.claims.signed_by,
        tier3.claims.signer_role,
        tier3.claims.signed_day,
    ) == (
        "morgan",
        "decision_owner",
        33,
    )
    provenance = json.loads(
        (
            run.path / "capstone-blueprint-tier3-governing-record.json"
        ).read_text()
    )
    assert provenance == source.model_dump(mode="json")
    assert tier3.claims.record_id.endswith("-under-" + source.sha256)


def test_operating_driver_propagates_scripted_model_disagreement(tmp_path):
    from test_capstone_blueprint_compare import comparison_fixture

    driver = constructed_driver()
    original = driver.month_controls

    def month_controls(stream, out, period, requirement):
        result = original(stream, out, period, requirement)
        _, inputs = comparison_fixture()
        offset = requirement.comparison_period.start - inputs.period.start
        requests = tuple(
            row.model_copy(
                update={
                    "requested_day": row.requested_day + offset,
                    "label_day": row.label_day + offset,
                }
            )
            for row in inputs.requests
        )

        def shifted(arm):
            return arm.model_copy(
                update={
                    "decisions": tuple(
                        row.model_copy(
                            update={
                                "answer_day": row.answer_day + offset,
                            }
                        )
                        for row in arm.decisions
                    )
                }
            )

        agent, rules = shifted(inputs.agent), shifted(inputs.rules)
        choices = tuple(
            row.model_copy(update={"outcome": "deny", "paid_cents": 0})
            if row.request_id == "r2"
            else row
            for row in rules.decisions
        )
        inputs = inputs.model_copy(
            update={
                "period": requirement.comparison_period,
                "requests": requests,
                "agent": agent,
                "manual_answer_days": {
                    rid: day + offset for rid, day in inputs.manual_answer_days.items()
                },
                "rules": rules.model_copy(update={"decisions": choices}),
            }
        )
        (out / "capstone-blueprint-comparison-inputs.json").write_text(
            inputs.model_dump_json()
        )
        return result

    driver.month_controls = month_controls
    out = tmp_path / "run"
    with pytest.raises(ValueError, match="scripted model departs from rules"):
        operate(7, demand=Demand(), out=out, driver=driver)
    assert not (out / "missing/capstone-blueprint-comparison.json").exists()
    assert not (out / "capstone-blueprint-gate-01.json").exists()
    assert not (out / "capstone-blueprint-run.json").exists()


def test_human_shadow_payments_acquire_outcomes_and_canary_adds_agent_payments():
    human = PaymentOutcome(
        receipt_id="human",
        request_id="r1",
        payment_day=2,
        amount_cents=4999,
        actor="human",
        mode="shadow",
        found_false_30d=False,
        generated_label=False,
    )
    agent = human.model_copy(
        update={
            "receipt_id": "agent",
            "request_id": "r2",
            "payment_day": 10,
            "actor": "agent",
            "mode": "executed",
            "found_false_30d": True,
        }
    )
    assert outcomes((human, agent), 32).final == (human,)
    assert outcomes((human, agent), 32).pending == ("agent",)
    assert len(outcomes((human, agent), 40).final) == 2
    assert (
        len(
            outcomes(
                (human.model_copy(update={"generated_label": True}),), 2
            ).final
        )
        == 1
    )
    with pytest.raises(ValueError, match="hypothetical shadow"):
        PaymentOutcome.model_validate({**agent.model_dump(), "mode": "shadow"})
    with pytest.raises(ValueError, match="one outcome"):
        outcomes((human, human), 40)


def test_budget_callback_is_bound_to_plan_window_and_budget(tmp_path):
    req, _, registry = packet_fixture(tmp_path)
    from test_org_adoption_rollout import authority_fixture

    authority, plan, _ = authority_fixture(tmp_path)
    status = BudgetExhaustion(
        budget_id="test-budget-0", plan_id=plan.plan_id,
        window_id=plan.steps[0].window.window_id, at=AT, spent=True,
    )
    budget_spent(
        status, authority=authority, registry=registry, requirement=req, day=65
    )
    assert authority.events()[-1].scope == "tier1"
    for field in ("budget_id", "plan_id", "window_id"):
        with pytest.raises(ValueError, match="another"):
            budget_spent(
                status.model_copy(update={field: "another"}),
                authority=authority,
                registry=registry,
                requirement=req,
                day=65,
            )


@pytest.mark.parametrize("retry", [False, True])
def test_budget_callback_shares_retry_identity_and_withdraws_once(tmp_path, retry):
    from datetime import timedelta

    from test_org_adoption_rollout import authority_fixture, causes_for, records

    from refund_agent.rollout.org_adoption_authority import StageAuthority, Withdrawal
    from refund_agent.rollout.org_adoption_rule import Change, decide, next_window

    req, _, registry = packet_fixture(tmp_path / "packet")
    authority, plan, evidence = authority_fixture(tmp_path)
    step = plan.steps[0]
    authority.sign_step(step.step_id, by="morgan", at=step.window.start - timedelta(minutes=30),
                        evidence=evidence)
    window = step.window
    if retry:
        from refund_agent.rollout.org_adoption_reconcile import reconcile
        from refund_agent.rollout.org_adoption_records import ProviderPayment

        intake, entries, original = records(200)
        payment = ProviderPayment(
            payment_id="console-around", order_id=intake[0].order_id, amount_cents=4000,
            at=intake[0].received_at, request_id=intake[0].request.request_id, channel="console",
        )
        count = reconcile(intake, entries, [], [payment], original)
        held = decide(plan, step, count, evidence, causes=causes_for(count))
        change = Change(change_id="repair", description="cover the absence", owner="morgan",
                        at=held.at + timedelta(hours=1), held_window_id=held.window_id)
        window = original.model_copy(update={
            "window_id": "retry-window", "start": held.at + timedelta(days=1),
            "end": held.at + timedelta(days=15), "closed_at": held.at + timedelta(days=16),
        })
        authority.retry_window(held, change, next_window(held, change, window), by="morgan",
                               at=change.at + timedelta(minutes=1))
    status = BudgetExhaustion(budget_id="test-budget-0", plan_id=plan.plan_id,
                             window_id=window.window_id, at=window.start, spent=True)
    first = authority.on_budget_spent(status)
    second = budget_spent(status, authority=authority, registry=registry, requirement=req, day=65)
    assert second == first
    withdrawals = [row for row in authority.events() if isinstance(row, Withdrawal)]
    assert len(withdrawals) == 1 and withdrawals[0].record_id == "org-adoption-budget-test-budget-0"
    assert StageAuthority(plan, authority.path).events()[-1] == first
