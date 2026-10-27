import json
import runpy
from pathlib import Path

import pytest

from demos import regions as _regions
from demos.human_oversight_risk_capture import ROOT, SOURCES, capture_chapter

NAMES = ("mm1-01", "arrivals-01", "queue-01", "timeout-01", "absence-01",
         "catch-01", "sample-01", "screen-01", "keys-01", "wait-data-01", "review-records-01")
PREFIX = "fnd.human-oversight-risk."


@pytest.fixture(scope="module")
def captured(tmp_path_factory):
    return capture_chapter(tmp_path_factory.mktemp("human-oversight-risk-capture"))


@pytest.mark.parametrize("name", NAMES)
def test_outputs_reproduce_through_the_repository_capture(name, captured):
    path = ROOT / "book_outputs" / f"_{PREFIX}{name}.txt"
    assert path.read_bytes() == captured[PREFIX + name].encode()
    assert "companion/" not in captured[PREFIX + name]
    assert max(map(len, captured[PREFIX + name].splitlines())) <= 79


def test_printed_region_width_limit():
    extractor = vars(_regions)
    for source in SOURCES:
        for region in extractor["extract_snippets_from_text"](source.read_text(), source):
            assert max(map(len, region.content.splitlines())) <= 79
            compile(region.content, str(source), "exec")


def test_plot_data_is_constructed_and_uses_matching_staffed_traffic(captured):
    data = json.loads(captured[PREFIX + "wait-data-01"])
    assert data["constructed"]
    for point in data["points"]:
        if point["desk"]:
            assert point["calendar_lambda"] == pytest.approx(point["u"] * 2 * 40 / 168)


def test_pending_partner_integrations_are_reported(captured):
    assert "stage-6 join PENDING" in captured[PREFIX + "arrivals-01"]
    assert "integration pending" in captured[PREFIX + "timeout-01"]
    assert "PENDING" in captured[PREFIX + "screen-01"]
    assert "one 10% month" in captured[PREFIX + "queue-01"]


def test_missing_partner_code_is_reported_and_never_substituted(monkeypatch):
    from refund_agent.oversight.human_oversight_risk_component import OversightTools

    fixtures = runpy.run_path(str(ROOT / "tests/refund_agent/test_human_oversight_risk_lifecycle.py"))
    data, tools, desk, ticket = fixtures["fixture"]()
    context = dict(reviewer_authenticate=fixtures["authenticate"],
                   denial_explainer=fixtures["explain_fixture"], decision_owner="decision-owner",
                   assigned_reviewer="alex", oversight_mode="execute")
    for missing in ("reviewer_authenticate", "denial_explainer"):
        with pytest.raises(ValueError):
            OversightTools(data, **{**context, missing: None})

    # The available core method cannot accept chapter 35's credential signature.
    # A retry through that core method would pay and is forbidden.
    from refund_agent.tools import Tools

    calls = []

    def unauthenticated_only(self, approval):
        calls.append(approval)
        return Tools.record_approval(self, approval)

    monkeypatch.setattr(type(tools), "record_approval", unauthenticated_only)
    at = fixtures["REVIEW_AT"]
    fixtures["freshen"](tools, at)
    before = (tools.audits, tools.ledger, dict(tools._approvals), dict(desk.tickets))
    with pytest.raises(TypeError, match="credential"):
        desk.approve(ticket.review_id, "alex", fixtures["issue"]("alex", at), at)
    assert calls == []
    assert (tools.audits, tools.ledger, dict(tools._approvals), desk.tickets) == before


def test_waiting_observations_remain_visible_outside_decided_denominators(monkeypatch, capsys):
    from datetime import timedelta

    from demos import human_oversight_risk_demo as demo
    from refund_agent.oversight.human_oversight_risk_monitor import ReviewObservation

    rows = (ReviewObservation("waiting", demo.START, "alex", 20, True, "approved",
                              demo.START + timedelta(days=8)),
            ReviewObservation("decided", demo.START, "alex", 20, True, "refused",
                              demo.START + timedelta(days=1)))
    monkeypatch.setattr(demo, "attention_rows", lambda: rows)
    monkeypatch.setattr(demo, "queue_experiment", lambda: {("D3", "exponential"): {"first": None}})
    monkeypatch.setattr(demo, "planted_records", lambda _: {})
    monkeypatch.setattr(demo, "alarm_replications", lambda: dict(
        replications=1, median=None, p10=None, p90=None, alarmed_by_week2=0, undetected=1))
    demo.print_catches()
    week1 = capsys.readouterr().out.split("week 1", 1)[1].split("week 2", 1)[0]
    assert "overloaded catch 1/1 unresolved 1" in week1


def test_catch_output_reports_the_replication_distribution_and_original_seed(captured):
    import re

    text = captured[PREFIX + "catch-01"]
    assert "Seed 7 first alarm (week, decided busy planted cases): undetected" in text
    assert "Replications: 1000" in text and "seeds 7..1006" in text
    quantiles = re.search(r"percentile: (None|[\d.]+) / (None|[\d.]+) / (None|[\d.]+)", text)
    assert quantiles
    median, p10, p90 = quantiles.groups()
    assert median == p90 == "None" and 1 <= float(p10) <= 5
    assert "Share alarming by week 2:" in text and "Undetected by day 30:" in text


def test_alarm_replications_change_seed_and_hold_the_simulated_design_fixed(monkeypatch):
    from datetime import timedelta

    from demos import human_oversight_risk_demo as demo
    from refund_agent import contracts as c
    from refund_agent.oversight.human_oversight_risk_demand import Stage6Demand
    from refund_agent.oversight.human_oversight_risk_simulate import staffing

    fixtures = runpy.run_path(str(ROOT / "tests/refund_agent/test_human_oversight_risk_lifecycle.py"))
    _, _, desk, ticket = fixtures["fixture"]()
    prototype = desk.packets[ticket.review_id][0]
    prototype = prototype.model_copy(update={"evidence": prototype.evidence.model_copy(update={
        "order": prototype.evidence.order.model_copy(update={"paid_cents": 30000}),
        "shipment_events": [c.ShipmentEvent(at=demo.START - timedelta(days=1), status="delivered")],
    })})
    demand = Stage6Demand()
    monkeypatch.setattr(demo, "ALARM_SEEDS", range(7, 17))
    monkeypatch.setattr(demo, "available_base_measurement", lambda: {"demand": demand})
    monkeypatch.setattr(demo, "planted_prototype", lambda: prototype)
    original, calls = demo.simulate, []

    def observe_real_simulation(seed, start, **kwargs):
        calls.append((seed, start, kwargs))
        result = original(seed, start, **kwargs)
        assert result.end - result.start == timedelta(days=30)
        return result

    monkeypatch.setattr(demo, "simulate", observe_real_simulation)
    demo.alarm_replications.cache_clear()
    try:
        summary = demo.alarm_replications()
    finally:
        demo.alarm_replications.cache_clear()
    assert summary["replications"] == 10
    assert [seed for seed, _, _ in calls] == list(range(7, 17))
    assert all(start == demo.START and kwargs == dict(
        demand=demand, people=staffing("D3"), service_model="exponential") for _, start, kwargs in calls)


def test_screen_and_review_evidence_print_actual_records_with_pending_interfaces(captured):
    screen = captured[PREFIX + "screen-01"]
    assert "1 Request" in screen and "2 Proposed denial:" in screen
    assert "5 Explanation (proposed):" in screen and "10 Reviewer riley; credential" in screen
    assert "ReviewerContext" in screen and "DenialExplanation" in screen
    records = captured[PREFIX + "review-records-01"]
    assert "TEST-CONSTRUCTED" in records and "assembled execution PENDING" in records
    assert "execution_refused:outside_return_window" in records
    assert "execution_refused:stale_shipment" in records
    assert "Waiting eligibility expiry: 1" in records and "expired=True" in records
    assert "owner: decision-owner" in records and "Notice:" in records
    assert "Old/new policy: 2026-09-01 2026-10-05" in records
    assert "Deputy during absence: payment" in records and "Deputy during absence: denial" in records
    assert "request state: denial_confirmed" in records and "request state: paid" in records


def test_sample_numbers_are_labelled_stage4_and_stage6_is_kept_pending(captured):
    text = captured[PREFIX + "sample-01"]
    assert "STAGE-4 SENSITIVITY RUN" in text and "ASSEMBLED STAGE-6 NUMBERS PENDING" in text
    assert "about 731 payments; 183/73 rechecks" in text and "48 blind" in text


def test_revised_staffing_passes_every_month_including_the_rework_stress(captured):
    import re

    text = captured[PREFIX + "queue-01"]
    for model in ("fixed", "exponential"):
        for name in ("D4", "D4-rework"):
            block = text.split(f"{name} {model} capacity", 1)[1].split("Boundary:", 1)[0]
            rows = re.findall(r"^\s+(\d{1,3})\s+(\d+)/(\d+)\s+([\d.]+)%\s+(\d+)\s*$", block, re.M)
            assert len(rows) == 100
            assert all(int(late) * 100 <= int(total) and int(pending) == 0
                       for _, late, total, _, pending in rows)
            assert "breach months=0/100" in block
    d3 = text.split("D3 exponential capacity", 1)[1].split("Boundary:", 1)[0]
    assert "breach months=1/100" in d3 and "accepted=False" in d3


@pytest.mark.parametrize("changed_run", [False, True])
def test_held_denial_stress_tracks_the_audited_breach_month(
    monkeypatch, capsys, tmp_path, changed_run,
):
    import re

    from demos import human_oversight_risk_demo as demo

    audit = (ROOT / "book_outputs/_fnd.audits-explanations.audit-01.txt").read_text()
    held = sum(map(int, re.findall(r"Send (\d+) \S+ denials to a person\.", audit)))
    assert held > 0
    if changed_run:
        # A new audit must change the stress input without another code edit.
        record = json.loads((demo.ROOT / "demos/data/audits_explanations_run.json").read_text())
        row = next(o for o in record["run"]["observations"]
                   if o["segment"] == "C" and o["reason"] == "already_delivered"
                   and o["outcome"] == "deny")
        row["outcome"] = "pay"
        path = tmp_path / "demos/data/audits_explanations_run.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(record))
        monkeypatch.setattr(demo, "ROOT", tmp_path)
        held -= 1
    monkeypatch.setattr(demo, "available_base_measurement", lambda: {"demand": demo.Stage6Demand()})
    monkeypatch.setattr(demo, "queue_experiment", lambda: {})

    demo.print_queue()

    block = capsys.readouterr().out.split("held-denial breach month\n", 1)[1].split("tier1 reversal", 1)[0]
    assert f"held_denial offered hours/week={held / 30 * 7 * 5 / 60:.6f}" in block


@pytest.mark.skipif(
    not (ROOT.parent / "_quarto.yml").is_file(),
    reason="requires the book repository",
)
def test_legacy_retirement_does_not_change_another_chapters_region():
    import subprocess

    path = Path("demos/part7_8_demo.py")
    before = subprocess.check_output(
        ["git", "--no-optional-locks", "show", f"5dee1f07:companion/{path}"],
        cwd=ROOT, text=True,
    )
    after = (ROOT / path).read_text()
    # Chapters 38–41 retire their own regions after this chapter joins the spine.
    key = "# region book:fnd.appendix-math-foundations.core-01"
    assert before[before.index(key):] == after[after.index(key):]
