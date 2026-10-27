"""Summary statements follow recorded arrivals, admission fields and review effects."""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace


def test_arrival_counts_use_distinct_scans_in_the_declared_fourteen_days():
    from demos.batch_stream_demo import arrival_counts

    assert arrival_counts() == {"within minutes": 5307, "depot sync": 224, "daily file": 69}


def test_arrival_summary_reads_the_regions_during_the_silent_interval():
    from demos.observability_reliability_demo import arrival_summary

    start = datetime(2026, 10, 1, tzinfo=timezone.utc)
    row = SimpleNamespace(
        last_arrival=start,
        batch_at=start + timedelta(hours=3),
        arrivals_by_region_hour=tuple(
            {
                "hour": (start + timedelta(hours=h)).isoformat(),
                "region": r,
                "scans": 0 if r == 1 else 4,
            }
            for h in (1, 2)
            for r in range(3)
        ),
    )
    assert (
        arrival_summary(row)
        == "Hourly regional arrivals: two regions keep sending while one falls silent"
    )
    row.arrivals_by_region_hour = tuple({**v, "scans": 4} for v in row.arrivals_by_region_hour)
    assert (
        arrival_summary(row)
        != "Hourly regional arrivals: two regions keep sending while one falls silent"
    )


def test_budget_summary_checks_recorded_admission_fields_without_start():
    from demos.observability_reliability_demo import START, budget_summary

    plan = SimpleNamespace(error_budget_ref="signed-budget")
    record = {
        "signed_by": "morgan",
        "signed_at": (START - timedelta(seconds=1)).isoformat(),
        "objectives": [
            {"definition": "timely", "window": "28 days", "allowance": 1, "population": "tier1"}
        ]
        * 2,
    }
    runs = ({"rehearsal": {"plan": plan, "budget_admission": record}},)
    assert (
        budget_summary(runs)
        == "Budget admission fields pass chapter 32's start reader; no start issued"
    )
    record["objectives"][0]["allowance"] = -1
    assert (
        budget_summary(runs)
        == "Budget admission fields fail chapter 32's start reader; no start issued"
    )


def test_deadline_counts_come_from_ticket_events_and_the_receipt_ledger():
    from refund_agent.oversight.human_oversight_risk_deadline import deadline_counts

    ticket = SimpleNamespace(
        request_id="r-1",
        events=(
            SimpleNamespace(kind="approval"),
            SimpleNamespace(kind="denial_confirmation"),
            SimpleNamespace(kind="escalation"),
        ),
    )
    tools = SimpleNamespace(
        ledger={
            "a": SimpleNamespace(request=SimpleNamespace(request_id="r-1")),
            "b": SimpleNamespace(request=SimpleNamespace(request_id="r-2")),
        }
    )
    assert deadline_counts(ticket, tools) == (1, 1, 1)


def test_timeout_prints_the_actual_empty_ledger_counts(capsys):
    from refund_agent.oversight.human_oversight_risk_deadline import print_timeout

    print_timeout()
    assert (
        "Deadline approvals=0 denials=0 receipts=0; item remains open." in capsys.readouterr().out
    )
