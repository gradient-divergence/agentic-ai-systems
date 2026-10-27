# ruff: noqa: E402 -- initialize the fixed core before component imports.
"""Check captured records against independently derived chapter 34 quantities."""

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Literal

from demos import regions as _regions
from refund_agent.release.candidate import build_candidate
from refund_agent.release.suite import build_suite

CORE = build_candidate(build_suite())

# isort: split
from refund_agent import contracts as c
from refund_agent.batch_stream.component import selector as event_selector
from refund_agent.batch_stream.view import (
    ALLOWED,
    first_complete,
    shipment_as_of,
)
from refund_agent.observability_reliability.component import selector
from refund_agent.observability_reliability.fortnight import make_stream
from refund_agent.observability_reliability.silence import region_of
from refund_agent.sources.data_systems_release import SELECTOR
from refund_agent.tools import Tools

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "book_outputs"


def figure():
    return json.loads((OUTPUTS / "_fnd.observability-reliability.figure-01.txt").read_text())


def test_demo_import_has_no_stdout_or_writes(tmp_path):
    result = subprocess.run(
        [sys.executable, "-c", "import demos.observability_reliability_demo"],
        cwd=tmp_path,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout == "" and result.stderr == ""
    assert tuple(tmp_path.iterdir()) == ()


def test_budget_bytes_recapture_through_repository_function():
    extractor = vars(_regions)
    source = ROOT / "demos/observability_reliability_demo.py"
    key = "fnd.observability-reliability.budget-01"
    regions = extractor["extract_snippets_from_text"](source.read_text(), source)
    # The first output region needs no fortnight, unlike later incident outputs.
    outputs = extractor["capture_outputs"](
        {key: next(s for s in regions if s.snippet_id == key)}, sys.executable
    )
    assert outputs[key].encode() == (OUTPUTS / f"_{key}.txt").read_bytes()


def test_prebatch_bytes_recapture_from_event_records():
    extractor = vars(_regions)
    source = ROOT / "demos/observability_reliability_exercises.py"
    regions = {
        s.snippet_id: s
        for s in extractor["extract_snippets_from_text"](source.read_text(), source)
        if s.output
    }
    outputs = extractor["capture_outputs"](regions, sys.executable)
    key = "fnd.observability-reliability.prebatch-01"
    assert outputs[key].encode() == (OUTPUTS / f"_{key}.txt").read_bytes()


def test_printed_span_region_runs_and_meets_limits():
    extractor = vars(_regions)
    source = ROOT / "refund_agent/observability_reliability/spans.py"
    region = next(
        s
        for s in extractor["extract_snippets_from_text"](source.read_text(), source)
        if s.snippet_id == "fnd.observability-reliability.spans-01"
    )
    assert len(region.content.splitlines()) <= 25
    assert max(map(len, region.content.splitlines())) <= 79
    assert "attempt: Attempt" in region.content
    from refund_agent.observability_reliability.spans import Attempt

    namespace = {
        "c": c,
        "Attempt": Attempt,
        "datetime": datetime,
        "Literal": Literal,
        "Tools": Tools,
    }
    exec(region.content, namespace)
    assert issubclass(namespace["TracedTools"], Tools)


def test_affected_requests_match_independent_event_time_population():
    row = figure()["incident"]
    stream = make_stream()
    start, end = (
        datetime.fromisoformat(row["last_arrival"]),
        datetime.fromisoformat(row["batch_at"]),
    )
    expected = set()
    for request in stream.requests:
        if region_of(request.order_reference) != row["region"]:
            continue
        reads = [request.at]
        view = shipment_as_of(stream.log, request.order_reference, request.at)
        delivered = any(e.status == "delivered" and e.at <= request.at for e in view.events)
        if request.reason == "not_delivered" and not delivered:
            reads.append(first_complete(stream.log, request.at + ALLOWED))
        if any(start <= at < end for at in reads):
            expected.add(request.request_id)
    assert set(row["affected_decisions"]) == expected
    assert row["region"] == stream.silence.region
    assert datetime.fromisoformat(row["batch_at"]) == stream.silence.end


def test_capture_keeps_attempts_causal_links_and_shadow_cost_separate():
    data = figure()
    spans = data["spans"]
    roots = [s for s in spans if s["name"] == "run"]
    assert [s["attempt"] for s in roots] == [1, 2]
    assert roots[0]["status"] == "held" and roots[1]["status"] == "ok"
    indexed = {s["span_id"]: s for s in spans}
    advance, relation = roots[1]["links"][0]
    assert relation == "released_by" and indexed[advance]["name"] == "watermark_advance"
    correction = next(s for s in spans if s["name"] == "correction")
    payment, relation = correction["links"][0]
    assert relation == "corrects" and indexed[payment]["name"] == "store:ledger"
    row = data["incident"]
    assert row["corrected_modes"] == {"shadow": row["corrections"]}
    assert row["cash_loss_cents"] == 4999
    assert 0 < row["shadow_exposure_cents"] < 10000
    assert row["illustrative_review_cost_cents"] == 400 * row["policy_referrals"]
    assert row["cohort_exposure"] == 1
    assert row["cohort_exposure_cents"] == 4999
    assert row["cohort_corrections"] == 1 and row["releases"] == 0
    assert len(row["bundles"]) == 2 and row["unstamped_lineage"] == 0
    assert all(s["bundle"] in row["bundles"] for s in spans)
    assert not row["budget_status"]["d6"]["available"]
    assert row["budget_status"]["d6"]["pass"] is None
    assert "mature person-payment labels" in row["budget_status"]["d6"]["reason"]
    stop = row["cohort_stops"][0]
    assert stop["trigger"] == "correction" and stop["outcome"] == "paused"
    assert stop["payments_after_trigger"] == 0
    assert data["rehearsals"]["as-built"]["verdict"] == "observed"
    assert data["rehearsals"]["fixed"]["corrections"] == 0
    assert data["rehearsals"]["fixed"]["final_outcome"] == "running"
    assert len(row["arrivals_by_region_hour"]) == 14 * 24 * 3
    assert data["fixed_status"]["objectives"]["corrections"]["bad"] == 0
    assert row["timeline"] == sorted(row["timeline"], key=lambda s: (s["at"], s["span_id"]))


def test_captured_candidate_matches_current_selected_implementation():
    from demos.observability_reliability_demo import bundle_selector

    suite = build_suite(
        trials=1,
        components=[
            SELECTOR,
            event_selector(),
            bundle_selector(),
            selector(region_finality=True),
        ],
    )
    assert build_candidate(suite).fingerprint == figure()["fixed_status"]["candidate_fingerprint"]


def test_incident_output_uses_reader_descriptions_and_reports_d6(monkeypatch, capsys):
    from demos import observability_reliability_demo as demo
    from refund_agent.observability_reliability.incident import IncidentRecord

    row = IncidentRecord.model_validate_json(json.dumps(figure()["incident"]))
    monkeypatch.setattr(demo, "experiment", lambda: (None, None, row))
    demo.print_incident()
    output = capsys.readouterr().out
    assert "figure-01 incident.arrivals_by_region_hour" not in output
    assert "Hourly regional arrivals: two regions keep sending while one falls silent" in output
    assert f"bundle count {len(row.bundles)}; unstamped lineage {row.unstamped_lineage}" in output
    assert f"registry promotions/rollbacks during the fortnight: {row.releases}" in output
    assert "D6 unavailable:" in output and "mature person-payment labels" in output
