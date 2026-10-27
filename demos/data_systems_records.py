"""Chapter 30's deterministic observations, separate from their printed views."""

import contextlib
import io
import runpy
from collections import Counter
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path
from tempfile import TemporaryDirectory

from refund_agent import contracts as c
from refund_agent.agent import Agent
from refund_agent.data.generate import generate
from refund_agent.release.candidate import build_candidate, build_tools, run_request
from refund_agent.release.suite import build_suite
from refund_agent.sources.data_systems_lineage import rebuild
from refund_agent.sources.data_systems_release import SELECTOR
from refund_agent.sources.data_systems_releases import add_status, rename_delivery
from refund_agent.tools import Tools

COMPANION = Path(__file__).resolve().parents[1]
EXERCISES = COMPANION / "exercises/data-systems"


@lru_cache(maxsize=1)
def experiment():
    data = generate(7, orders=40, requests=40).model_copy(update={"correct": []})
    candidate = build_candidate(build_suite(components=[SELECTOR]))
    versions = (
        ("generated-bare", data, False),
        ("generated-contracts", data, True),
        ("rename-bare", rename_delivery(data), False),
        ("rename-contracts", rename_delivery(data), True),
        ("harmless-bare", add_status(data), False),
        ("harmless-contracts", add_status(data), True),
    )
    runs = {}
    for name, version, contracted in versions:
        tools = build_tools(version, candidate, "execute") if contracted else Tools(version)
        traces = [
            run_request(candidate, tools, r) if contracted else Agent(tools).run(r)
            for r in data.requests
        ]
        runs[name] = {
            "tools": tools,
            "traces": traces,
            "counts": Counter((t.decision.outcome.value, t.decision.reason) for t in traces),
            "paid_cents": sum(p.amount_cents for p in tools.payments),
        }
    return data, candidate, runs


def generator_probe():
    # Inputs only; no labels inspected, no golden-window decision run here.
    data = generate().model_copy(update={"correct": []})
    bad = [s for s in data.shipments if any(e.at > s.observed_at for e in s.events)]
    ids = {s.order_id for s in bad}
    requests = [r for r in data.requests if r.order_reference in ids]
    return {
        "records": len(data.shipments),
        "requests": len(data.requests),
        "bad_records": len(bad),
        "readers": [r.request_id for r in requests],
        "reader_dates": [r.at.isoformat() for r in requests],
        "public_readers": len(set(r.request_id for r in requests) & set(data.public_scenarios)),
        "offsets_seconds": sorted(
            {
                (e.at - s.observed_at).total_seconds()
                for s in bad
                for e in s.events
                if e.at > s.observed_at
            }
        ),
    }


def lineage_example():
    _, _, runs = experiment()
    tools = runs["generated-contracts"]["tools"]
    row = next(r for r in tools.lineage if r.request.request_id == "r-7-0007")
    return row, tools.record_store, rebuild(row, tools.record_store)


def exercise_05_fixture():
    row, store, _ = lineage_example()
    raw = row.model_dump(mode="json")
    records = {ref.fingerprint: store.get(ref.fingerprint) for ref in row.inputs}
    ref = next(r for r in raw["inputs"] if r["source"] == "shipment")
    shipment = records.pop(ref["fingerprint"])
    observed = datetime.fromisoformat(raw["decision_at"]) + timedelta(minutes=40)
    shipment["observed_at"] = observed.isoformat()
    from refund_agent.selection import fingerprint

    ref.update(
        fingerprint=fingerprint(shipment), observed_at=observed.isoformat(), age_seconds=-2400
    )
    records[ref["fingerprint"]] = shipment
    return {
        "constructed": True,
        "lineage": raw,
        "records": records,
    }


def reference_answers():
    reference = runpy.run_path(str(EXERCISES / "data_systems_reference.py"))
    card = runpy.run_path(str(EXERCISES / "data_systems_reference_06.py"))
    return {
        "hospital": reference["exercise_3"](),
        "quiet_and_day": reference["exercise_4"](),
        "diagnosis": reference["exercise_5"](exercise_05_fixture()),
        "card": card,
    }


def chapter_2_denial():
    # Reuse the actual chapter 2 construction, without printing its other captures.
    with contextlib.redirect_stdout(io.StringIO()):
        ns = runpy.run_path(str(COMPANION / "demos/modern_ai_stack_demo.py"))
    data = ns["load"](ns["releases"]["in dollars"])
    request = next(r for r in data.requests if r.request_id == ns["REQUEST"])
    _, candidate, _ = experiment()
    decisions = []
    for tools in (Tools(data), build_tools(data, candidate, "execute")):
        state = c.State(request=request, trace_id=f"trace-{request.request_id}")
        for name in ("order.lookup", "shipment.status", "policy.retrieve"):
            tools.call(c.ToolCall(name=name), state)
        decisions.append(tools.assess(state))
    return decisions


@lru_cache(maxsize=1)
def release_evidence():
    from refund_agent.release.journal import Journal
    from refund_agent.release.rule import evaluate_release

    suite = build_suite(components=[SELECTOR], trials=5)
    candidate = build_candidate(suite)
    with TemporaryDirectory(prefix="data-systems-release-") as directory:
        record = evaluate_release(
            candidate,
            suite,
            Journal(Path(directory) / "journal.jsonl"),
            5,
            at="2026-10-12T12:00:00+00:00",
        )
    positive = [r for r in record["case_results"] if r["part"] == "positive"]
    payments = [
        sum(
            r["trials"][trial]["decision"]["amount_cents"]
            for r in positive
            if r["trials"][trial]["decision"]["executed"]
        )
        for trial in range(5)
    ]
    return {
        key: record[key]
        for key in (
            "at",
            "candidate",
            "candidate_fingerprint",
            "suite_fingerprint",
            "decision",
            "failed_cases",
            "floor",
            "cases_run",
            "trials_run",
            "record_id",
            "signature",
        )
    } | {
        "must_pay_cases": len(positive),
        "must_pay_cents_per_trial": payments,
        "scope": "chapter 30 unsigned release evidence; no canary authority",
    }
