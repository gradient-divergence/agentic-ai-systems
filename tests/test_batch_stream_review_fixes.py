"""Chapter 31's blind-review regressions, independent of release migration."""

import json
import re
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from demos import batch_stream_demo as demo
from demos.regions import extract_snippets_from_text
from refund_agent import contracts as c
from refund_agent.agent import Agent
from refund_agent.batch_stream import experiment
from refund_agent.batch_stream.generate import SETTINGS, START, generate, opening, scan
from refund_agent.batch_stream.log import EventLog
from refund_agent.batch_stream.replay import Views
from refund_agent.batch_stream.tools import EventTimeTools
from refund_agent.batch_stream.view import (
    ALLOWED,
    MARGIN,
    late_events,
    shipment_as_of,
    watermark,
)
from refund_agent.batch_stream.windows import day_counts

ROOT = Path(__file__).resolve().parents[1]
GUIDE = ROOT / "docs/chapters/31-batch-stream.md"


def test_absence_fixture_probe_keeps_agent_call_observations_unchanged():
    from refund_agent.batch_stream.suite import extend
    from refund_agent.release.candidate import build_candidate
    from refund_agent.release.suite import build_suite, run_case

    suite = extend(build_suite(trials=1))
    candidate = build_candidate(suite)
    case = next(case for case in suite.cases if case.handler == "batch-stream-absence")
    result = run_case(case, candidate)
    assert result["failed"] == []
    assert all(
        call["arguments"]["idempotency_key"]
        != "batch-stream-boundary-batch-stream-opening"
        for call in result["tool_calls"]
    )


def test_delay_shares_are_selection_probabilities_not_realized_proportions():
    stream = generate()
    originals = {}
    for event in stream.log.entries:
        if START <= event.at < START + timedelta(days=SETTINGS["days"]):
            originals.setdefault(event.event_id, event)
    counts = [0, 0, 0]
    for event in originals.values():
        seconds = (event.arrived_at - event.at).total_seconds()
        counts[0 if seconds < 3600 else 1 if seconds <= 9 * 3600 else 2] += 1
    assert SETTINGS["delay_shares"] == (0.95, 0.04, 0.01)
    assert counts == [5307, 224, 69] and len(originals) == 5600
    assert counts[2] / len(originals) != SETTINGS["delay_shares"][2]
    assert "95%, 4% and 1% are selection probabilities" in GUIDE.read_text()


def test_replay_cutoff_is_two_days_after_the_request():
    result = experiment.replay_run()
    assert result["cutoff"] == START + timedelta(hours=18, minutes=5)
    corrected_at = result["corrected"].observed_at
    assert corrected_at == START + timedelta(days=2, hours=6, minutes=8)
    assert (corrected_at.date() - result["cutoff"].date()).days == 2
    assert "06:08 two days after the request" in GUIDE.read_text()


def test_runtime_journal_is_registered_and_capture_snapshots_are_not():
    from refund_agent.batch_stream.component import RUN_JOURNAL, selector
    from refund_agent.release import extensions

    record = extensions.records((selector(),))[0]
    assert record["stores"] == [vars(RUN_JOURNAL)]
    assert RUN_JOURNAL.owner == "morgan" and RUN_JOURNAL.mode == "append"
    assert not any("book_outputs" in store["path"] or "exercises" in store["path"] for store in record["stores"])


@pytest.mark.parametrize("field,value", [("owner", "unlisted"), ("mode", "write")])
def test_runtime_journal_owner_and_mode_are_validated(monkeypatch, field, value):
    from dataclasses import replace

    from refund_agent.batch_stream.component import selector
    from refund_agent.release import extensions

    plugin = extensions._REGISTRY["batch-stream"]
    store = replace(plugin.stores[0], **{field: value})
    monkeypatch.setitem(extensions._REGISTRY, plugin.name, replace(plugin, stores=(store,)))
    with pytest.raises(ValueError, match="runtime store needs a path, listed owner and valid mode"):
        extensions.records((selector(),))


@pytest.mark.parametrize("operation", ["rewrite", "truncate", "remove", "snapshot"])
def test_runtime_journal_enforces_append_and_keeps_history(tmp_path, monkeypatch, operation):
    from dataclasses import replace

    from refund_agent.batch_stream import component
    from refund_agent.release import extensions
    from refund_agent.release.journal import Journal

    path = tmp_path / "journal.jsonl"
    snapshot = tmp_path / "records.json"
    snapshot.write_text("capture snapshot\n")
    store = replace(component.RUN_JOURNAL, path=str(path))
    plugin = extensions._REGISTRY["batch-stream"]
    monkeypatch.setitem(extensions._REGISTRY, plugin.name, replace(plugin, stores=(store,)))
    extensions.records((component.selector(),))
    namespace = {**vars(component), "journal_path": path, "snapshot_path": snapshot}
    exec(compile(
        "def append():\n"
        "    RunJournal(journal_path).append({'event': 'run'})\n"
        "def forbidden(operation):\n"
        "    if operation == 'rewrite':\n"
        "        journal_path.write_text('replacement')\n"
        "    elif operation == 'truncate':\n"
        "        with journal_path.open('a+') as stream:\n"
        "            stream.truncate(0)\n"
        "    elif operation == 'remove':\n"
        "        journal_path.unlink()\n"
        "    else:\n"
        "        snapshot_path.write_text('component replacement')\n",
        component.__file__, "exec"
    ), namespace)
    namespace["append"]()
    before = path.read_bytes()
    namespace["append"]()
    assert path.read_bytes().startswith(before)
    assert [row["entry"] for row in Journal(path).read()] == [1, 2]
    history = path.read_bytes()
    with pytest.raises(ValueError, match="runtime store|store mode"):
        namespace["forbidden"](operation)
    assert path.read_bytes() == history
    assert snapshot.read_text() == "capture snapshot\n"


@pytest.mark.parametrize("old_id,fresh_id", [("a", "z"), ("z", "a")])
@pytest.mark.parametrize("at_hours,expected_late", [(0, True), (1, False)])
def test_tied_arrival_lateness_uses_the_complete_group(
    old_id, fresh_id, at_hours, expected_late
):
    arrival = START + timedelta(hours=10)
    old = scan("old", "delivered", START + timedelta(hours=at_hours), arrival)
    fresh = scan("fresh", "shipped", arrival, arrival)
    log = EventLog(
        [
            old.model_copy(update={"event_id": old_id}),
            fresh.model_copy(update={"event_id": fresh_id}),
        ]
    )
    assert watermark(log, arrival) == START + timedelta(hours=1)
    assert [e.order_id for e in late_events(log)] == (
        ["old"] if expected_late else []
    )


@pytest.mark.parametrize("old_id,fresh_id", [("a", "z"), ("z", "a")])
@pytest.mark.parametrize("offset_us", [-1, 0, 1])
def test_tied_arrivals_respect_the_final_cutoff(old_id, fresh_id, offset_us):
    arrival = START + timedelta(days=1) + ALLOWED + MARGIN
    arrival += timedelta(microseconds=offset_us)
    log = EventLog(
        [
            scan(
                "old", "delivered", START + timedelta(hours=10), arrival
            ).model_copy(update={"event_id": old_id}),
            scan("fresh", "shipped", arrival, arrival).model_copy(
                update={"event_id": fresh_id}
            ),
        ]
    )
    row = day_counts(log, START, arrival)
    assert row["settled"] == int(offset_us < 0)
    assert row["n_final"] == (None if offset_us < 0 else 0)
    assert row["corrections"] == ([] if offset_us < 0 else [old_id])
    assert row["fast"] == row["n_day"] + row["copies"] - row["misses"]


@pytest.mark.parametrize("reason", ["damaged", "not_delivered"])
def test_delivery_before_shipping_rejects_payment_and_denial_evidence(reason):
    data, original, request = opening()
    made = START + timedelta(hours=11, minutes=40)
    log = EventLog(
        [e for e in original.entries if e.order_id != request.order_reference]
        + [
            scan(request.order_reference, "delivered", made, made),
            scan(
                request.order_reference,
                "shipped",
                made + timedelta(minutes=1),
                made + timedelta(minutes=1),
            ),
        ]
    )
    with pytest.raises(ValueError, match="delivery_before_shipment"):
        shipment_as_of(log, request.order_reference, request.at)
    tools = EventTimeTools(data, event_log=log)
    trace = Agent(tools).run(request.model_copy(update={"reason": reason}))
    assert trace.decision.outcome == c.Outcome.REFER
    assert not trace.decision.executed and not tools.payments
    assert trace.steps[-1].result.ok


def test_delivered_only_and_simultaneous_shipping_histories_remain_valid():
    made = START + timedelta(hours=11, minutes=40)
    delivery = scan("parcel", "delivered", made, made)
    for events in (
        [delivery],
        [delivery, scan("parcel", "shipped", made, made)],
    ):
        shipment = shipment_as_of(EventLog(events), "parcel", made)
        assert any(e.status == "delivered" for e in shipment.events)


def test_replay_decisions_consume_the_rebuilt_shipment(monkeypatch):
    build = Views.shipment

    def without_delivery(self, order_id, at):
        shipment = build(self, order_id, at)
        return shipment.model_copy(
            update={
                "events": [
                    e for e in shipment.events if e.status != "delivered"
                ]
            }
        )

    monkeypatch.setattr(Views, "shipment", without_delivery)
    result = experiment.replay_run()
    assert [e.status for e in result["corrected"].events] == ["shipped"]
    assert result["corrected_decision"]["reason"] == "eligible"
    assert not result["decisions_equal"]
    assert result["tool_calls"] == 0 and result["ledger_unchanged"]


def test_rebuilt_replay_decisions_match_both_recorded_decisions():
    result = experiment.replay_run()
    assert result["original_decision"]["reason"] == "shipment_incomplete"
    assert result["corrected_decision"]["reason"] == "already_delivered"
    assert result["decisions_equal"]
    assert result["tool_calls"] == 0 and result["ledger_unchanged"]


def test_payment_exercise_has_a_request_time_watermark():
    payment = experiment.payment_exercise()
    request_at = datetime.fromisoformat(payment["request_at"])
    progress = datetime.fromisoformat(payment["watermark"])
    assert progress <= request_at - MARGIN
    log = EventLog(
        experiment.ScanEvent.model_validate_json(json.dumps(e))
        for e in payment["events"]
    )
    assert progress == watermark(log, request_at)
    observed_at = max(e.arrived_at for _, e in log.arrived_by(request_at))
    assert (request_at - observed_at).total_seconds() == payment[
        "record_age_seconds"
    ]
    stored = json.loads(
        (ROOT / "exercises/batch-stream/payment-05.json").read_text()
    )
    assert stored == payment


@pytest.mark.skipif(
    not (ROOT.parent / "_quarto.yml").is_file(),
    reason="requires the book repository",
)
def test_flink_reference_pins_the_reviewed_release_without_a_year():
    bibliography = (ROOT.parent / "references.bib").read_text()
    entry = bibliography.split("@online{flink2026guarantees,", 1)[1].split(
        "\n}", 1
    )[0]
    assert re.search(r"\byear\s*=", entry) is None
    assert (
        "flink-docs-release-2.1/docs/connectors/datastream/guarantees/"
        in entry
    )
    assert "2.1.1" in entry and "2026-10-12" in entry


def test_claims_prints_only_discussed_lines_and_retains_the_full_output(
    monkeypatch, capsys
):
    records = json.loads(
        (ROOT / "demos/data/batch-stream-records.json").read_text()
    )["claims"]
    monkeypatch.setattr(demo, "claim_run", lambda: records)
    demo.print_claims()
    excerpt = capsys.readouterr().out
    assert len(excerpt.splitlines()) <= 24
    assert (
        "Settings:" not in excerpt
        and "Constructed stream sha256:" not in excerpt
    )
    for expected in (
        "552",
        "fast: 700 claims, 148 paid, 8 contradicted",
        "watermark: 700 claims, 144 paid, 4 contradicted",
        "final: 700 claims, 140 paid, 0 contradicted",
        "9.06",
        "36.04",
        "36.06",
        "36.24",
        "35.75",
    ):
        assert expected in excerpt
    demo.print_claims_full()
    full = capsys.readouterr().out
    assert len(full.splitlines()) > len(excerpt.splitlines())
    assert "Settings:" in full and "Constructed stream sha256:" in full
    assert "0.054054" in full and "0.054054" in excerpt
    source = ROOT / "demos/batch_stream_demo.py"
    outputs = {
        s.snippet_id
        for s in extract_snippets_from_text(source.read_text(), source)
        if s.output
    }
    assert "fnd.batch-stream.claims-full-01" in outputs
