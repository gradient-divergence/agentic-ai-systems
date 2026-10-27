"""Chapter 41 contracts, numerical decisions and fail-closed joins."""

from dataclasses import replace
from datetime import timedelta
from types import SimpleNamespace

import numpy as np
import pytest
from pydantic import BaseModel, JsonValue

from refund_agent.canary.cloud_native_deployment_rehearsal import RehearsalRecord
from refund_agent.data.history import DEFAULT_ROWS, END_AT, _false_probability
from refund_agent.oversight.human_oversight_risk_demand import Stage6Demand
from refund_agent.oversight.human_oversight_risk_design import check_design as check_stage6_design
from refund_agent.oversight.human_oversight_risk_design import default_design
from refund_agent.oversight.human_oversight_risk_simulate import QueueArrival, simulate
from refund_agent.renewal.future_directions_adopt import RECORDS, AdoptionPorts, adopt
from refund_agent.renewal.future_directions_candidate import recalibrate, selected_score
from refund_agent.renewal.future_directions_decide import decide
from refund_agent.renewal.future_directions_evidence import Check, Evidence, evaluate
from refund_agent.renewal.future_directions_requirement import ReplacementRequirement, validate
from refund_agent.renewal.future_directions_windows import Window, generate_forward, validate_window


@pytest.fixture(scope="module")
def score():
    return selected_score()


@pytest.fixture(scope="module")
def window():
    return Window.generate("future-directions-window-01", 7, END_AT, 2000)


@pytest.fixture
def req(score):
    return ReplacementRequirement(
        incumbent_bundle_fingerprint="incumbent-bundle",
        incumbent_candidate_fingerprint="incumbent",
        candidate_bundle_fingerprint="candidate-bundle",
        candidate_fingerprint="candidate",
        policy_version="2026-09-01",
        selection_fingerprint=score.selection_fingerprint,
        sealed_receipt=score.sealed_receipt,
        score_fingerprint=score.component.fingerprint,
        reversal_target="incumbent-bundle",
        first_request_day=0.02,
    )


def evidence(req, **changes):
    checks = tuple(
        Check(name=name, passed=True, detail="constructed test evidence")
        for name in (
            "calibration",
            "cost",
            "wrongful_denials",
            "capacity",
            "incumbent_admission",
            "release",
        )
    )
    values = dict(
        requirement_fingerprint=req.fingerprint,
        candidate_bundle_fingerprint=req.candidate_bundle_fingerprint,
        candidate_fingerprint="candidate",
        score_fingerprint=req.score_fingerprint,
        window_fingerprint="window",
        window_id="window-01",
        paid=2000,
        unique=2000,
        all_matured=True,
        selected=True,
        wrong=130,
        prediction=0.065,
        interval=(0.055, 0.077),
        difference=(-0.2, -0.3, -0.1),
        referrals=200,
        added_minutes=1000,
        checks=checks,
        first_day=1,
        last_day=60,
        closed_day=90,
        day=90,
    )
    return Evidence.model_validate({**values, **changes})


def test_requirement_owner_date_fields_and_target(req):
    admitted = (
        {
            "bundle_fingerprint": "incumbent-bundle",
            "candidate_fingerprint": "incumbent",
            "stage": 6,
            "status": "admitted",
        },
    )
    assert validate(req, admitted) == req
    with pytest.raises(ValueError, match="admission"):
        validate(req, ())
    for field, value in (
        ("signed_by", "Agent"),
        ("signer_role", "reviewer"),
        ("signed_day", 1),
        ("attempts", 3),
        ("rebuild_on", "cost"),
        ("reversal_mechanism", ""),
        ("reversal_owner", "Agent"),
        ("reversal_target", "unadmitted"),
        ("experiment_assumption", ""),
        ("release_required", False),
        ("incumbent_candidate_fingerprint", "another-candidate"),
        ("candidate_bundle_fingerprint", None),
    ):
        with pytest.raises(ValueError):
            validate(req.model_copy(update={field: value}), admitted)


def test_fresh_draws_and_maturity(window):
    assert list(window.rows) == generate_forward(7, END_AT, 2000)
    assert len(window.paid_ids) == 2000
    assert all(int(r.request_id.rsplit("-", 1)[1]) >= DEFAULT_ROWS for r in window.rows)
    for r in window.rows:
        if r.paid:
            assert r.amount_cents < 5000
            assert r.label_final_at == r.at + timedelta(days=30)
            earlier = r.model_copy(update={"at": END_AT - timedelta(days=31)})
            assert _false_probability(r) == 2 * _false_probability(earlier)
        else:
            assert r.found_false_30d is None


def test_window_refuses_partial_immature_duplicates_and_reselection(window):
    rows = [r for r in window.rows if r.paid]
    for partial in (rows[:1], rows[:-1], [rows[0]] * 2000):
        with pytest.raises(ValueError):
            validate_window(
                window,
                partial,
                paid=2000,
                evaluated_at=window.mature_at,
                expected_start=END_AT,
                seed=7,
            )
    with pytest.raises(ValueError, match="immature"):
        validate_window(
            window,
            rows,
            paid=2000,
            evaluated_at=window.mature_at - timedelta(seconds=1),
            expected_start=END_AT,
            seed=7,
        )
    changed = replace(window, rows=tuple(reversed(window.rows)))
    with pytest.raises(ValueError):
        validate_window(
            changed, rows, paid=2000, evaluated_at=window.mature_at, expected_start=END_AT, seed=7
        )


def test_selected_score_is_the_sealed_rebuild(score):
    assert (
        score.sealed_receipt == "f9b14846b5ee8b9c554e45eae06355a0e0b4bbc1f8b52c564f38a8cd44156276"
    )
    assert score.rebuild_fingerprint == score.component.fingerprint


def test_raw_metrics_and_missing_capacity_release_fail_closed(req, score, window):
    rows = [r for r in window.rows if r.paid]
    ev = evaluate(
        rows,
        score.component,
        "incumbent",
        req,
        window.mature_at,
        window=window,
        expected_start=END_AT,
    )
    checks = {c.name: c for c in ev.checks}
    assert not checks["capacity"].passed
    assert not checks["release"].passed
    assert not checks["incumbent_admission"].passed
    assert ev.wrong == sum(r.found_false_30d for r in rows)
    p = score.component.predict(rows)
    cost = np.where(
        p < 0.080, np.asarray([r.found_false_30d * r.amount_cents / 100 for r in rows]), 4
    )
    assert ev.difference[0] == pytest.approx(
        float((cost - [r.found_false_30d * r.amount_cents / 100 for r in rows]).mean())
    )
    assert ev.added_minutes == ev.referrals * 5
    assert decide(req, ev, 1).verdict == "keep"


def test_every_stage6_class_passes_both_consumers(req, score, window):
    from refund_agent.launch.capstone_blueprint_compare import ReviewWork
    from refund_agent.oversight.human_oversight_risk_demand import Stage6Demand
    from refund_agent.oversight.human_oversight_risk_design import check_design, default_design

    demand = Stage6Demand()
    monday = END_AT + timedelta(days=(0 - END_AT.weekday()) % 7, hours=3)
    arrivals = []
    for work in demand.classes():
        review = ReviewWork(
            review_id="producer-" + work.name, request_id=None,
            work_class=work.name, minutes=work.minutes,
        )
        arrivals.append({"request_id": review.review_id, "at": monday,
                         "class": review.work_class, "review_minutes": review.minutes,
                         "blocking": work.blocking})
    ev = evaluate(
        [r for r in window.rows if r.paid], score.component, "incumbent", req,
        window.mature_at, window=window, expected_start=END_AT,
        check_design=check_design, capacity_design=default_design("D3", END_AT),
        capacity_demand=demand, capacity_arrivals=tuple(arrivals),
    )
    assert next(c for c in ev.checks if c.name == "capacity").passed
    assert ev.referrals == 261 and ev.added_minutes == 1305


def test_added_referral_rate_alone_crosses_utilization_ceiling(req, score, window):
    demand = Stage6Demand(tier2_daily=22)
    before = check_stage6_design(D3, demand)
    assert before["utilization"] < D3.utilization_target == 0.8
    assert not before["failures"]
    ev = evaluate(
        [r for r in window.rows if r.paid], score.component, "incumbent", req,
        window.mature_at, window=window, expected_start=END_AT,
        check_design=check_stage6_design, capacity_design=D3,
        capacity_demand=demand, capacity_arrivals=(),
    )
    assert ev.referrals == 261 and ev.added_minutes == 1305
    capacity = next(c for c in ev.checks if c.name == "capacity")
    assert not capacity.passed
    assert capacity.detail == "necessary counts; weekly utilization exceeds 0.8"
    decision = decide(req, ev, 1)
    assert decision.verdict == "keep" and "capacity" in decision.reasons


@pytest.mark.parametrize("error_type", [AttributeError, RuntimeError, OSError])
def test_capacity_callback_exception_fails_closed(req, score, window, error_type):
    def broken_check(*args, **kwargs):
        raise error_type("capacity producer failed")

    ev = evaluate(
        [r for r in window.rows if r.paid], score.component, "incumbent", req,
        window.mature_at, window=window, expected_start=END_AT,
        check_design=broken_check, capacity_design=D3, capacity_demand=Stage6Demand(),
        capacity_arrivals=(),
    )
    capacity = next(c for c in ev.checks if c.name == "capacity")
    assert not capacity.passed and "capacity producer failed" in capacity.detail
    assert "capacity" in decide(req, ev, 1).reasons


@pytest.mark.parametrize("fault", ["bundle", "launch_with_failures"])
def test_launch_verdict_is_validated_and_bound_after_durable_promotion(req, fault):
    decision = decide(req, evidence(req), 1)
    records = {name: {"requirement_fingerprint": req.fingerprint, "passed": True}
               for name in RECORDS}
    gate = launch_result(decision.candidate_bundle_fingerprint)
    gate = gate.model_copy(update=(
        {"launched_bundle": "WRONG-BUNDLE"} if fault == "bundle" else {"verdict": "launch"}
    ))
    registry = []
    clients = replace(ports(req, decision, registry), on_promotion=lambda **k: gate)
    result = adopt(decision, req, registry, "morgan", 91, records=records, ports=clients)
    assert result.verdict == "adopted-launch-pending"
    assert len(registry) == 1 and result.promotion_record == registry[0]
    assert result.watch_id == "future-directions-watch"
    assert result.launch_verdict is None and result.launch_record is None


def test_added_referrals_are_blocking_arrivals_and_capacity_cannot_take_them(req, score, window):
    friday = END_AT + timedelta(days=(4 - END_AT.weekday()) % 7, hours=17)
    demand = tuple({
        "request_id": f"existing-tier2-{n}", "at": friday, "class": "tier2",
        "review_minutes": 15, "blocking": True,
    } for n in range(96))
    before = simulate(7, END_AT, 30, demand=Stage6Demand(), people=D3.people,
                      service_model="fixed", arrivals=tuple(
                          QueueArrival(r["request_id"], r["class"], r["at"]) for r in demand))
    assert not any(r["failures"] for r in check_stage6_design(
        D3, Stage6Demand(), weekends=before.weekends)["observed_weekends"])
    ev = evaluate(
        [r for r in window.rows if r.paid], score.component, "incumbent", req,
        window.mature_at, window=window, expected_start=END_AT,
        check_design=check_stage6_design, capacity_design=D3,
        capacity_demand=Stage6Demand(), capacity_arrivals=demand,
    )
    capacity = next(c for c in ev.checks if c.name == "capacity")
    assert not capacity.passed and "pooled_capacity" in capacity.detail
    assert ev.referrals == 261 and ev.added_minutes == 1305
    decision = decide(req, ev, 1)
    assert decision.verdict == "keep" and "capacity" in decision.reasons


@pytest.mark.parametrize(
    "failed",
    ["calibration", "cost", "wrongful_denials", "capacity", "release", "incumbent_admission"],
)
def test_decisions_keep_every_failure_except_first_calibration_alone(req, failed):
    ev = evidence(req)
    ev = ev.model_copy(
        update={
            "checks": tuple(c.model_copy(update={"passed": c.name != failed}) for c in ev.checks)
        }
    )
    first = decide(req, ev, 1)
    assert first.verdict == ("rebuild" if failed == "calibration" else "keep")
    if failed == "calibration":
        second = ev.model_copy(
            update={
                "first_decision_fingerprint": first.fingerprint,
                "fit_window_fingerprint": "first-window",
            }
        )
        assert decide(req, second, 2, first).verdict == "keep"


def test_all_pass_and_inconclusive_cost(req, window):
    ev = evidence(req)
    assert decide(req, ev, 1).verdict == "replace"
    rows = [r for r in window.rows if r.paid]
    wrong = max(
        (i for i, r in enumerate(rows) if r.found_false_30d), key=lambda i: rows[i].amount_cents
    )
    review_count = (rows[wrong].amount_cents - 1) // 400
    sound = [i for i, r in enumerate(rows) if not r.found_false_30d][: review_count - 1]
    predictions = np.full(2000, 0.04)
    predictions[[wrong, *sound]] = 0.09
    candidate = SimpleNamespace(fingerprint=req.score_fingerprint, predict=lambda _: predictions)
    failed = evaluate(
        rows,
        candidate,
        "incumbent",
        req,
        window.mature_at,
        window=window,
        expected_start=END_AT,
    )
    assert failed.difference[0] == pytest.approx(
        (review_count * 4 - rows[wrong].amount_cents / 100) / 2000
    )
    assert failed.difference[0] < 0 < failed.difference[2]
    assert not next(c for c in failed.checks if c.name == "cost").passed
    decision = decide(req, failed, 1)
    assert decision.verdict == "keep" and "cost" in decision.reasons


def test_unbound_candidate_cannot_replace(req):
    unbound = req.model_copy(update={"candidate_fingerprint": None})
    with pytest.raises(ValueError, match="candidate identity"):
        decide(unbound, evidence(unbound, candidate_fingerprint=None), 1)


def test_unbound_or_mismatched_bundle_cannot_replace(req):
    with pytest.raises(ValueError, match="signed requirement"):
        decide(req, evidence(req, candidate_bundle_fingerprint="another-bundle"), 1)
    unbound = req.model_copy(update={"candidate_bundle_fingerprint": None})
    with pytest.raises(ValueError, match="bundle identity"):
        decide(unbound, evidence(unbound, candidate_bundle_fingerprint=None), 1)


@pytest.mark.parametrize("count,mature", [(1, True), (1999, True), (2000, False)])
def test_decide_refuses_invalid_window(req, count, mature):
    with pytest.raises(ValueError, match="window"):
        decide(req, evidence(req, paid=count, unique=count, all_matured=mature), 1)


def test_rebuild_uses_only_first_matured_window(score, window, tmp_path, monkeypatch):
    from demos.future_directions_demo import run
    from refund_agent.renewal import future_directions_windows as windows

    rows = [r for r in window.rows if r.paid]
    rebuilt = recalibrate(score.component, rows, evaluated_at=window.mature_at)
    assert rebuilt.fingerprint != score.component.fingerprint
    assert rebuilt.model is score.component.model
    original = run(tmp_path / "original", inputs=partner_inputs(tmp_path / "original"))
    generate = windows.generate_forward

    def changed_future(seed, start, paid):
        generated = generate(seed, start, paid)
        if start == END_AT:
            return generated
        changed = []
        for r in generated:
            if r.paid:
                assert r.found_false_30d is not None
                r = r.model_copy(update={"found_false_30d": 1 - r.found_false_30d})
            changed.append(r)
        return changed

    monkeypatch.setattr(windows, "generate_forward", changed_future)
    changed = run(tmp_path / "changed", inputs=partner_inputs(tmp_path / "changed"))
    assert changed["evidence2"]["wrong"] == 2000 - original["evidence2"]["wrong"]
    assert (
        changed["rebuilt_score_fingerprint"]
        == original["rebuilt_score_fingerprint"]
        == rebuilt.fingerprint
    )
    assert changed["decision"] == original["decision"]
    with pytest.raises(ValueError, match="matured"):
        recalibrate(score.component, rows, evaluated_at=window.mature_at - timedelta(seconds=1))


class ConstructedLaunchVerdict(BaseModel):
    """Chapter 39's full payload, with a permissive verdict for fault injection.

    Contract: launch/capstone_blueprint_gate.py::LaunchVerdict.
    """

    verdict: JsonValue
    launched_bundle: str
    packet_fingerprint: str
    requirement_fingerprint: str
    day: float
    rows: tuple[dict, ...]
    failures: tuple[dict, ...]


def launch_result(bundle="candidate-bundle", verdict="hold"):
    failures = (
        (
            {
                "row": "1",
                "kind": "comparison",
                "check": "complete",
                "reason": "scripted model still equals rules",
            },
            {
                "row": "4",
                "kind": "release",
                "check": "consistent",
                "reason": "strict records reopened",
            },
        )
        if verdict == "hold"
        else ()
    )
    return ConstructedLaunchVerdict(
        verdict=verdict,
        launched_bundle=bundle,
        packet_fingerprint="launch-packet",
        requirement_fingerprint="launch-requirement",
        day=91,
        rows=tuple(
            {
                "row": str(i),
                "checks": {
                    name: not any(f["row"] == str(i) and f["check"] == name for f in failures)
                    for name in ("complete", "consistent", "signed_in_order", "reversible")
                },
                "counted": (),
            }
            for i in range(1, 8)
        ),
        failures=failures,
    )


def ports(req, decision, registry):
    def promote(**kwargs):
        record = {
            "stage": 6,
            "verdict": "admitted",
            "rollback_target_fingerprint": req.incumbent_bundle_fingerprint,
            "rollback_target_candidate_fingerprint": req.incumbent_candidate_fingerprint,
            "candidate_bundle_fingerprint": decision.candidate_bundle_fingerprint,
            "candidate_fingerprint": decision.candidate_fingerprint,
            "decision_fingerprint": decision.fingerprint,
        }
        registry.append(record)
        return record

    return AdoptionPorts(
        check_requirements=lambda **k: {"passed": True, "signed_decline_accepted": True},
        release=lambda **k: {
            "decision": "pass",
            "signature": {"owner": "morgan"},
            "software_fingerprint": decision.candidate_fingerprint,
        },
        rehearse=lambda **k: rehearsal_record(**k),
        promote=promote,
        on_promotion=lambda **k: launch_result(decision.candidate_bundle_fingerprint),
        register_watch=lambda **k: {
            "budget_withdrawal_registered": True,
            "confirmation_persisted": True,
            "watch_id": "future-directions-watch",
            "plan_fingerprint": "plan",
            "window_fingerprint": "window",
        },
    )


def test_adoption_records_missing_owner_decline_and_launch_hold(req):
    decision = decide(req, evidence(req), 1)
    records = {
        name: {"requirement_fingerprint": req.fingerprint, "passed": True} for name in RECORDS
    }
    registry = []
    clients = ports(req, decision, registry)
    for name in RECORDS:
        absent = {k: v for k, v in records.items() if k != name}
        assert (
            adopt(decision, req, registry, "morgan", 91, records=absent, ports=clients).verdict
            == "held"
        )
    assert not registry
    assert (
        adopt(decision, req, registry, "Agent", 91, records=records, ports=clients).verdict
        == "held"
    )
    refusal = replace(
        clients, check_requirements=lambda **k: {"passed": True, "signed_decline_accepted": False}
    )
    assert (
        adopt(decision, req, registry, "morgan", 91, records=records, ports=refusal).verdict
        == "held"
    )
    assert not registry
    result = adopt(decision, req, registry, "morgan", 91, records=records, ports=clients)
    assert result.verdict == "adopted" and result.stage == 6 and result.launch_verdict == "hold"
    assert len(registry) == 1 and registry[0]["rollback_target_fingerprint"] == "incumbent-bundle"


@pytest.mark.parametrize("verdict", ["hold", "launch"])
def test_launch_model_is_serialized_at_adoption(req, verdict):
    import json

    decision = decide(req, evidence(req), 1)
    records = {
        name: {"requirement_fingerprint": req.fingerprint, "passed": True} for name in RECORDS
    }
    registry = []
    gate = launch_result(decision.candidate_bundle_fingerprint, verdict)
    clients = replace(ports(req, decision, registry), on_promotion=lambda **k: gate)
    result = adopt(decision, req, registry, "morgan", 91, records=records, ports=clients)
    assert result.verdict == "adopted" and result.stage == 6
    assert len(registry) == 1 and result.launch_verdict == verdict
    saved = json.loads(result.model_dump_json())["launch_record"]
    assert saved["verdict"] == verdict
    assert saved["launched_bundle"] == "candidate-bundle"
    assert saved["packet_fingerprint"] == "launch-packet"
    assert saved["requirement_fingerprint"] == "launch-requirement"
    assert saved["day"] == 91
    assert len(saved["rows"]) == 7 and saved["rows"][0]["counted"] == []
    assert result.launch_record["rows"] == saved["rows"]
    assert result.launch_record["failures"] == saved["failures"]
    if verdict == "hold":
        assert saved["failures"][0] == {
            "row": "1",
            "kind": "comparison",
            "check": "complete",
            "reason": "scripted model still equals rules",
        }
    else:
        assert saved["failures"] == []
        assert all(all(row["checks"].values()) for row in saved["rows"])


@pytest.mark.parametrize("verdict", ["pass", "admitted", "HOLD", "", None, True, 7, [], {}])
def test_unsupported_launch_verdict_is_named_and_keeps_monitoring(req, verdict):
    decision = decide(req, evidence(req), 1)
    records = {
        name: {"requirement_fingerprint": req.fingerprint, "passed": True} for name in RECORDS
    }
    registry = []
    clients = replace(
        ports(req, decision, registry),
        on_promotion=lambda **k: launch_result(decision.candidate_bundle_fingerprint, verdict),
    )
    result = adopt(decision, req, registry, "morgan", 91, records=records, ports=clients)
    assert result.verdict == "adopted-launch-pending"
    assert "InvalidLaunchVerdict" in result.reason
    assert result.launch_verdict is None and result.launch_record is None
    assert len(registry) == 1 and result.promotion_record == registry[0]
    assert result.promotion_fingerprint is not None
    assert (result.watch_id, result.plan_fingerprint, result.window_fingerprint) == (
        "future-directions-watch",
        "plan",
        "window",
    )


@pytest.mark.parametrize(
    "field",
    [
        "candidate_bundle_fingerprint",
        "candidate_fingerprint",
        "rollback_target_fingerprint",
        "rollback_target_candidate_fingerprint",
    ],
)
def test_promotion_must_bind_both_candidate_and_target_identities(req, tmp_path, field):
    from refund_agent.renewal.future_directions_watch import WatchStore, watch

    decision = decide(req, evidence(req), 1)
    records = {
        name: {"requirement_fingerprint": req.fingerprint, "passed": True} for name in RECORDS
    }
    registry = []
    clients = ports(req, decision, registry)
    promote = clients.promote
    clients = replace(clients, promote=lambda **k: {**promote(**k), field: "another-identity"})
    result = adopt(decision, req, registry, "morgan", 91, records=records, ports=clients)
    assert result.verdict == "adopted-launch-pending" and "promotion" in result.reason
    assert len(registry) == 1 and registry[0]["verdict"] == "admitted"
    assert result.promotion_record == {**registry[0], field: "another-identity"}
    assert result.promotion_fingerprint is not None
    assert result.launch_record is None and result.launch_verdict is None
    assert (result.watch_id, result.plan_fingerprint, result.window_fingerprint) == (
        "future-directions-watch",
        "plan",
        "window",
    )
    adoption = {**result.model_dump(mode="json"), "fingerprint": result.fingerprint}
    store = WatchStore(tmp_path / "watch.jsonl", result.fingerprint, req.fingerprint)
    assert watch([], adoption, req, 92, store, period="month-01").verdict == "pending"


@pytest.mark.parametrize(
    "field", ["candidate_bundle_fingerprint", "rollback_target_candidate_fingerprint"]
)
def test_mismatched_promotion_keeps_budget_withdrawal_active(req, tmp_path, field):
    import json

    from refund_agent.renewal.future_directions_watch import BudgetStatus, WatchStore, watch

    decision = decide(req, evidence(req), 1)
    records = {
        name: {"requirement_fingerprint": req.fingerprint, "passed": True} for name in RECORDS
    }
    registry = []
    clients = ports(req, decision, registry)
    promote = clients.promote
    clients = replace(clients, promote=lambda **k: {**promote(**k), field: "another-identity"})
    result = adopt(decision, req, registry, "morgan", 91, records=records, ports=clients)
    adoption = {**json.loads(result.model_dump_json()), "fingerprint": result.fingerprint}
    store = WatchStore(tmp_path / "watch.jsonl", result.fingerprint, req.fingerprint)
    assert watch([], adoption, req, 92, store, period="month-01").verdict == "pending"
    calls = []

    def withdraw(**kwargs):
        calls.append("withdraw")
        return {"tier1_authority": "shadow", "decider": "person"}

    spent = BudgetStatus(
        plan_fingerprint="plan",
        window_fingerprint="window",
        spent=True,
        status_fingerprint="spent-budget",
    )
    restarted = WatchStore(store.path, result.fingerprint, req.fingerprint)
    stopped = watch(
        [], adoption, req, 93, restarted, period="budget", budget=spent, withdraw=withdraw
    )
    assert stopped.verdict == "withdrawn" and stopped.state.grant_withdrawn
    assert calls == ["withdraw"] and len(registry) == 1
    assert watch([], adoption, req, 94, restarted, period="month-01").verdict == "withdrawn"


@pytest.mark.parametrize("verdict", ["hold", "launch"])
def test_adoption_output_reports_launch_failures(req, verdict, capsys):
    import json
    from pathlib import Path

    from demos import future_directions_demo as demo

    decision = decide(req, evidence(req), 1)
    records = {
        name: {"requirement_fingerprint": req.fingerprint, "passed": True} for name in RECORDS
    }
    gate = launch_result(verdict=verdict)
    clients = replace(ports(req, decision, []), on_promotion=lambda **k: gate)
    adoption = adopt(decision, req, [], "morgan", 91, records=records, ports=clients)
    source = Path(demo.__file__)
    region = source.read_text().split(
        "# region book:fnd.future-directions.adopt-01 output\n", 1
    )[1].split("# endregion book:fnd.future-directions.adopt-01", 1)[0]
    exec(
        compile(region, str(source), "exec"),
        {
            "__name__": "__main__",
            "json": json,
            "result": {
                "adoption": adoption.model_dump(mode="json"),
                "requirement": req.model_dump(mode="json"),
                "missing": {},
            },
        },
    )
    output = capsys.readouterr().out
    assert "Launch gate: " + verdict in output
    if verdict == "hold":
        assert 'Held/reopened rows: ["1", "4"]' in output
        assert "Row 1: comparison/complete; scripted model still equals rules" in output
        assert "Row 4: release/consistent; strict records reopened" in output
    else:
        assert "Held/reopened rows: []" in output
        assert "Row 1:" not in output and "Row 4:" not in output


def test_incomplete_rehearsal_record_cannot_admit_a_replacement(req):
    decision = decide(req, evidence(req), 1)
    records = {
        name: {"requirement_fingerprint": req.fingerprint, "passed": True} for name in RECORDS
    }
    registry = []
    clients = replace(ports(req, decision, registry), rehearse=lambda **k: {"verdict": "pass", **k})
    result = adopt(decision, req, registry, "morgan", 91, records=records, ports=clients)
    assert result.verdict == "held" and "RehearsalRecord" in result.reason
    assert not registry


def rehearsal_record(*, candidate_fingerprint, rollback_target_fingerprint):
    """Constructed evidence in chapter 32's real, complete serialized contract."""
    return RehearsalRecord(
        rehearsal_id="future-directions-rehearsal",
        candidate_fingerprint=candidate_fingerprint,
        rollback_target_fingerprint=rollback_target_fingerprint,
        cadence="4h",
        first_payment_cents=2499,
        triggering_payment_cents=4999,
        payments_after_trigger=0,
        cents_after_trigger=0,
        expected_further="5/9",
        exposure_stop_trigger="recheck_overdue",
        exposure_stop_at=END_AT + timedelta(days=2, hours=4),
        detected_at=END_AT + timedelta(days=2, hours=4),
        paid_twice=0,
        unreadable_records=0,
        checks={"ledger_preserved": True, "components_restored": True},
        components={
            name: True
            for name in (
                "agent",
                "model",
                "runtime",
                "adapter",
                "tools_composition",
                "offer_policy",
            )
        },
        policy_version="2026-09-01",
        verdict="pass",
        restart_fresh_process=True,
        corrections=0,
        recheck_overdue_pauses=1,
        final_outcome="ended",
        blockers=(),
    )


@pytest.mark.parametrize("error_type", [ValueError, RuntimeError, OSError])
def test_launch_callback_error_cannot_hide_an_existing_promotion(req, tmp_path, error_type):
    from refund_agent.renewal.future_directions_watch import WatchStore, watch

    decision = decide(req, evidence(req), 1)
    records = {
        name: {"requirement_fingerprint": req.fingerprint, "passed": True} for name in RECORDS
    }
    registry = []

    def unavailable(**kwargs):
        raise error_type("launch evidence unavailable")

    clients = replace(ports(req, decision, registry), on_promotion=unavailable)
    result = adopt(decision, req, registry, "morgan", 91, records=records, ports=clients)
    assert len(registry) == 1 and result.verdict == "adopted-launch-pending"
    assert result.promotion_fingerprint is not None and result.launch_verdict is None
    assert (result.watch_id, result.plan_fingerprint, result.window_fingerprint) == (
        "future-directions-watch",
        "plan",
        "window",
    )
    adoption = {**result.model_dump(), "fingerprint": result.fingerprint}
    store = WatchStore(tmp_path / "watch.jsonl", result.fingerprint, req.fingerprint)
    assert watch([], adoption, req, 92, store, period="month-01").verdict == "pending"


D3 = default_design("D3", END_AT)


def stage6_demand(window):
    """Seeded observed arrivals use the real producer's classes and rates."""
    rng = np.random.default_rng(window.seed)
    days = (window.rows[-1].at - window.start).total_seconds() / 86400
    classes = {work.name: work for work in Stage6Demand().classes()}
    result = []
    for kind in ("referral", "tier2", "tier3", "recheck", "blind", "planted",
                 "held_denial", "modification", "refusal_followup", "adjudication"):
        work = classes[kind]
        count = rng.poisson(work.rate_per_day * days)
        for n, offset in enumerate(sorted(rng.uniform(0, days, count))):
            result.append({
                "request_id": f"existing-{kind}-{n}",
                "at": window.start + timedelta(days=float(offset)),
                "class": kind, "review_minutes": work.minutes, "blocking": work.blocking,
            })
    return tuple(sorted(result, key=lambda r: (r["at"], r["request_id"])))


def test_capacity_fixture_uses_complete_brief_demand(window):
    demand = stage6_demand(window)
    assert {r["class"] for r in demand} == {"referral", "tier2", "tier3", "recheck", "blind", "planted"}
    assert any(r["at"].weekday() == 4 and r["at"].hour >= 17 for r in demand)
    assert stage6_demand(window) == demand
    result = check_stage6_design(D3, Stage6Demand())
    assert result["total_hours"] == pytest.approx(56.4, abs=0.02)
    assert (result["capacity_hours"], result["monday_capacity"], result["tier3_monday_capacity"]) == (88, 24, 8)


@pytest.mark.parametrize("kind,count,reason", [
    ("tier3", 17, "tier3_authorized_capacity"),
    ("tier2", 97, "pooled_capacity"),
    ("recheck", 0, "weekly utilization exceeds 0.8"),
])
def test_capacity_refuses_each_brief_inequality(req, score, window, kind, count, reason):
    friday = END_AT + timedelta(days=(4 - END_AT.weekday()) % 7, hours=17)
    demand = Stage6Demand(tier1_monthly=3000, recheck_share=1) if kind == "recheck" else Stage6Demand()
    work = next(work for work in demand.classes() if work.name == kind)
    arrivals = tuple({
        "request_id": f"overload-{n}", "at": friday, "class": kind,
        "review_minutes": work.minutes, "blocking": work.blocking,
    } for n in range(count))
    ev = evaluate(
        [r for r in window.rows if r.paid], score.component, "incumbent", req,
        window.mature_at, window=window, expected_start=END_AT,
        check_design=check_stage6_design, capacity_design=D3,
        capacity_demand=demand, capacity_arrivals=arrivals,
    )
    capacity = next(c for c in ev.checks if c.name == "capacity")
    assert not capacity.passed and reason in capacity.detail


def version_bundle_identity(candidate):
    """Test-only chapter 33 components; refs are outside its five-part hash."""
    from refund_agent.adaptation.prompt import StructuredPrompt
    from refund_agent.selection import fingerprint

    components = {
        "model": {
            **candidate.model,
            "risk_score": next(
                (
                    s["settings"]
                    for s in candidate.tools_composition.get("components", [])
                    if s["name"] == "future-directions.risk"
                ),
                {"name": "text-rule", "threshold": 0.2},
            ),
        },
        "prompt": {
            "template": "refund-draft-2026-10-10",
            "fingerprint": StructuredPrompt().fingerprint,
        },
        "tools": {
            "agent": candidate.agent,
            "runtime": candidate.runtime,
            "adapter": candidate.adapter,
            "composition": candidate.tools_composition,
        },
        "policy": {
            "versions": ["2026-09-01"],
            "offer_rule": candidate.offer_policy["policy"],
            "register_snapshot": fingerprint({"constructed": "policy-register"}),
        },
        "index": {
            "encoder": "constructed-policy-encoder",
            "sha256": fingerprint({"constructed": "policy-index"}),
        },
    }
    return {
        "bundle_fingerprint": fingerprint(
            [fingerprint(components[k]) for k in ("model", "prompt", "tools", "policy", "index")]
        ),
        "candidate_fingerprint": candidate.fingerprint,
    }


def partner_inputs(directory):
    """Independent chapter 29/33/34/37/38/39 records, confined to tests."""
    from refund_agent.canary.cloud_native_deployment_store import AppendJournal
    from refund_agent.release.candidate import build_candidate
    from refund_agent.release.suite import build_suite
    from refund_agent.renewal.future_directions_watch import BudgetStatus
    from refund_agent.shadow.__main__ import validate_release

    directory.mkdir(parents=True, exist_ok=True)
    suite = build_suite()
    incumbent = build_candidate(suite).fingerprint
    identity = version_bundle_identity(build_candidate(suite))
    admitted = {**identity, "stage": 6, "status": "admitted"}
    launch_state = {**identity, "verdict": "hold", "signature": None}
    registry, releases, calls = [admitted], {}, []

    def release(*, suite, component, day, directory):
        candidate = build_candidate(suite)
        assert any(s["settings"]["fingerprint"] == component.fingerprint for s in suite.components)
        record_id = "release-" + candidate.fingerprint[:16]
        record = {
            "record_id": record_id,
            "decision": "pass",
            "candidate_fingerprint": candidate.fingerprint,
            "software_fingerprint": candidate.fingerprint,
            "suite_fingerprint": suite.fingerprint,
            "signature": {"owner": "morgan", "at": day.isoformat()},
            "unresolved_earlier_failures": [],
            "failure_dispositions": [],
        }
        journal = AppendJournal(directory / (record_id + ".jsonl"))
        journal.append({**record, "signature": None})
        journal.append(
            {"event": "sign", "record_id": record["record_id"], "signature": record["signature"]}
        )
        validate_release(record, candidate, suite, journal)
        releases[candidate.fingerprint] = (record, journal, suite)
        calls.append("release")
        return record, journal

    def check_requirements(**kwargs):
        req, decision, records = (kwargs[k] for k in ("req", "decision", "records"))
        assert records["selection"]["fingerprint"] == req.selection_fingerprint
        assert records["rebuild_check"]["sealed_receipt"] == req.sealed_receipt
        assert records["calibration"]["evidence_fingerprint"] == decision.evidence_fingerprint
        assert records["signed_cost"]["fingerprint"] == req.cost_fingerprint
        assert records["signed_decline"]["signer"] == req.signed_by
        calls.append("requirements")
        return {"passed": True, "signed_decline_accepted": True}

    def released(*, candidate_fingerprint):
        record, journal, suite = releases[candidate_fingerprint]
        validate_release(record, build_candidate(suite), suite, journal)
        return record

    def run_rehearsal(*, candidate_fingerprint, rollback_target_fingerprint):
        calls.append("run_rehearsal")
        assert rollback_target_fingerprint == incumbent
        return rehearsal_record(
            candidate_fingerprint=candidate_fingerprint,
            rollback_target_fingerprint=rollback_target_fingerprint,
        )

    def promote(**kwargs):
        calls.append("promote")
        decision = kwargs["decision"]
        record = {
            "stage": kwargs["stage"],
            "verdict": "admitted",
            "rollback_target_fingerprint": kwargs["rollback_target_fingerprint"],
            "rollback_target_candidate_fingerprint": kwargs[
                "rollback_target_candidate_fingerprint"
            ],
            "candidate_bundle_fingerprint": kwargs["candidate_bundle_fingerprint"],
            "candidate_fingerprint": decision.candidate_fingerprint,
            "decision_fingerprint": decision.fingerprint,
            "rehearsal_fingerprint": kwargs["rehearsal_fingerprint"],
        }
        registry.append(record)
        return record

    def on_promotion(state, promotion):
        calls.append("on_promotion")
        assert state == launch_state and promotion == registry[-1]
        return launch_result(promotion["candidate_bundle_fingerprint"])

    wiring = {
        "watch_id": "future-directions-watch",
        "plan_fingerprint": "plan",
        "window_fingerprint": "window",
        "budget_withdrawal_registered": True,
        "confirmation_persisted": True,
    }
    ports = AdoptionPorts(
        check_requirements, released, run_rehearsal, promote, on_promotion, lambda **kwargs: wiring
    )
    return SimpleNamespace(
        incumbent_suite=suite,
        bundle_identity=version_bundle_identity,
        admitted_targets=(admitted,),
        launch_state=launch_state,
        check_design=check_stage6_design,
        capacity_design=D3,
        capacity_demand=Stage6Demand(),
        demand=stage6_demand,
        release=release,
        adoption_ports=ports,
        registry=registry,
        budget_status=lambda adoption: BudgetStatus(
            plan_fingerprint="plan",
            window_fingerprint="window",
            spent=False,
            status_fingerprint="budget",
        ),
        withdraw=lambda **kwargs: {"tier1_authority": "shadow", "decider": "person"},
        rollback=lambda **kwargs: {
            "target_fingerprint": identity["bundle_fingerprint"],
            "target_candidate_fingerprint": incumbent,
            "stage": 6,
            "verdict": "restored",
        },
        calls=calls,
    )


def test_watched_month_keeps_higher_tiers_out_of_the_score_and_cohort(
    req, window, tmp_path, monkeypatch
):
    from demos import future_directions_demo as demo
    from refund_agent.renewal.future_directions_adopt import Adoption
    from refund_agent.renewal.future_directions_watch import BudgetStatus

    paid = [r for r in window.rows if r.paid][:40]
    higher = next(r for r in window.rows if r.amount_cents >= 5000)
    monkeypatch.setattr(demo, "generate_forward", lambda *args: [*paid, higher])

    def predict(rows):
        assert all(r.amount_cents < 5000 for r in rows), "risk score received a higher-tier request"
        return np.full(len(rows), 0.05)

    component = SimpleNamespace(predict=predict, threshold=0.080)
    adoption = Adoption(
        verdict="adopted",
        day=0,
        reason="constructed test admission",
        decision_fingerprint="decision",
        requirement_fingerprint=req.fingerprint,
        candidate_bundle_fingerprint=req.candidate_bundle_fingerprint,
        candidate_fingerprint=req.candidate_fingerprint,
        incumbent_bundle_fingerprint=req.incumbent_bundle_fingerprint,
        incumbent_candidate_fingerprint=req.incumbent_candidate_fingerprint,
        watch_id="watch",
        plan_fingerprint="plan",
        window_fingerprint="window",
    )
    inputs = SimpleNamespace(
        budget_status=lambda _: BudgetStatus(
            plan_fingerprint="plan",
            window_fingerprint="window",
            spent=False,
            status_fingerprint="budget",
        ),
        withdraw=None,
        rollback=None,
    )
    _, month = demo.watched_month(tmp_path, component, adoption, req, inputs)
    assert month["scored"] == month["paid"] == 40
    assert higher.request_id not in {r["request_id"] for r in month["requests"]}
    assert [r["ordinal"] for r in month["requests"]] == list(range(1, 41))


def test_whole_continuation_consumes_partner_records_before_evidence(tmp_path, monkeypatch, capsys):
    import json

    from demos.future_directions_demo import load, run

    directory = tmp_path / "integration"
    inputs = partner_inputs(directory)
    generate = Window.generate

    def after_signature(*args, **kwargs):
        requirement = json.loads((directory / "future-directions-requirement.json").read_text())
        assert requirement["candidate_fingerprint"] is not None
        assert requirement["candidate_bundle_fingerprint"] is not None
        assert (
            len({requirement["candidate_fingerprint"], requirement["candidate_bundle_fingerprint"]})
            == 2
        )
        assert (
            requirement["incumbent_bundle_fingerprint"] == inputs.launch_state["bundle_fingerprint"]
        )
        assert (
            requirement["incumbent_candidate_fingerprint"]
            == inputs.launch_state["candidate_fingerprint"]
        )
        assert requirement["signed_day"] < requirement["first_request_day"]
        return generate(*args, **kwargs)

    monkeypatch.setattr(Window, "generate", after_signature)
    result = run(directory, inputs=inputs)
    assert result["decision"]["verdict"] == "rebuild"
    assert result["decision"]["reasons"] == ["calibration"]
    first = result["evidence"]
    assert first["paid"] == 2000 and first["wrong"] == 136 and first["referrals"] == 261
    assert first["prediction"] == pytest.approx(0.047724, abs=0.000001)
    assert first["difference"][0] == pytest.approx(-0.462240, abs=0.000001)
    assert result["decision2"]["verdict"] == "replace"
    assert (
        result["decision2"]["first_decision_fingerprint"]
        == decide(
            ReplacementRequirement.model_validate(result["requirement"]),
            Evidence.model_validate(first),
            1,
        ).fingerprint
    )
    second = result["evidence2"]
    assert second["paid"] == 2000 and all(c["passed"] for c in second["checks"])
    assert second["wrong"] == 124 and second["referrals"] == 309
    assert second["prediction"] == pytest.approx(0.067171111, abs=1e-9)
    assert second["difference"] == pytest.approx((-0.484395, -0.7506695, -0.231276875))
    assert second["difference"][0] < second["difference"][2] < 0
    assert result["adoption"]["verdict"] == "adopted" and result["adoption"]["stage"] == 6
    assert result["adoption"]["launch_verdict"] == "hold"
    assert [f["check"] for f in result["adoption"]["launch_record"]["failures"]] == [
        "complete",
        "consistent",
    ]
    req = ReplacementRequirement.model_validate(result["requirement"])
    adoption = result["adoption"]
    promotion = adoption["promotion_record"]
    assert adoption["incumbent_bundle_fingerprint"] == req.incumbent_bundle_fingerprint
    assert adoption["incumbent_candidate_fingerprint"] == req.incumbent_candidate_fingerprint
    assert promotion["candidate_bundle_fingerprint"] == second["candidate_bundle_fingerprint"]
    assert promotion["candidate_fingerprint"] == second["candidate_fingerprint"]
    assert promotion["rollback_target_fingerprint"] == req.incumbent_bundle_fingerprint
    assert promotion["rollback_target_candidate_fingerprint"] == req.incumbent_candidate_fingerprint
    rehearsal = RehearsalRecord.model_validate(adoption["rehearsal_record"])
    assert set(adoption["rehearsal_record"]) == set(RehearsalRecord.model_fields)
    assert rehearsal.candidate_fingerprint == adoption["candidate_fingerprint"]
    assert rehearsal.rollback_target_fingerprint == req.incumbent_candidate_fingerprint
    assert inputs.calls == [
        "release",
        "release",
        "requirements",
        "run_rehearsal",
        "promote",
        "on_promotion",
    ]
    assert len(inputs.registry) == 2 and inputs.registry[0]["status"] == "admitted"
    assert result["watch"]["verdict"] == "clear"
    assert (result["watched_month"]["paid"], result["watched_month"]["wrong"]) == (797, 19)
    assert not result["watch"]["state"]["grant_withdrawn"]
    assert result["timing"]["adoption_day"] == pytest.approx(183.983333333)
    assert result["missing"] == {} and load(directory) == result
    from refund_agent.renewal.__main__ import main
    from refund_agent.renewal.future_directions_watch import check_month

    assert main(["decide", "--run-dir", str(directory)]) == 0
    assert json.loads(capsys.readouterr().out)["verdict"] == "replace"
    assert main(["watch", "--run-dir", str(directory)]) == 0
    assert json.loads(capsys.readouterr().out)["verdict"] == "clear"
    month = result["watched_month"]
    assert (
        check_month(
            [SimpleNamespace(**r) for r in month["requests"]],
            {"status": "clear"},
            period=month["period"],
            day=month["day"],
        )["status"]
        == result["watch"]["verdict"]
    )
    with pytest.raises(ValueError, match="fresh pre-evidence"):
        run(directory, inputs=inputs)


@pytest.mark.parametrize("field", ["bundle_fingerprint", "candidate_fingerprint"])
def test_held_launch_must_bind_each_incumbent_identity(tmp_path, field):
    from demos.future_directions_demo import run

    inputs = partner_inputs(tmp_path)
    inputs.launch_state[field] = "another-identity"
    result = run(tmp_path, inputs=inputs)
    assert result["decision"]["verdict"] == "keep"
    assert "incumbent_admission" in result["decision"]["reasons"]
    assert result["decision2"] is None and result["adoption"]["verdict"] == "held"
    assert result["missing"]["launch-record"] == "held launch names another incumbent"


def test_version_bundle_cannot_reference_another_candidate(tmp_path):
    from demos.future_directions_demo import run

    inputs = partner_inputs(tmp_path)
    inputs.bundle_identity = lambda candidate: {
        **version_bundle_identity(candidate),
        "candidate_fingerprint": "another-candidate",
    }
    with pytest.raises(ValueError, match="references another chapter 29 candidate"):
        run(tmp_path, inputs=inputs)
