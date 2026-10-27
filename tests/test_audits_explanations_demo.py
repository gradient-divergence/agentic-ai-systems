"""Explicit-source capture, snippet execution, and pending integration refusal."""

import ast
import runpy
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from demos import regions as _regions
from refund_agent.release.candidate import build_candidate
from refund_agent.release.suite import build_suite

CORE = build_candidate(build_suite(trials=1))

# isort: split
from demos.audits_explanations_capture import capture_chapter  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def test_six_independent_outputs_reproduce_and_have_printable_width(tmp_path):
    outputs = capture_chapter(tmp_path)
    assert len(outputs) == 6
    for name, text in outputs.items():
        stored = ROOT / "book_outputs" / f"_{name}.txt"
        assert stored.read_bytes() == text.encode()
        assert text.endswith("\n") and "companion/" not in text
        assert max(map(len, text.splitlines())) <= 79


def test_printed_regions_are_unique_and_run_with_their_source_dependencies():
    extractor = vars(_regions)
    paths = [
        ROOT / "refund_agent/audit" / f"audits_explanations_{n}.py"
        for n in ("rates", "explain")
    ]
    snippets = []
    for path in paths:
        source = path.read_text()
        snippets.extend(extractor["extract_snippets_from_text"](source, path))
        namespace = {
            "__name__": "audits_explanations_listing",
            "__file__": str(path),
        }
        exec(compile(source, str(path), "exec"), namespace)
        for snippet in extractor["extract_snippets_from_text"](source, path):
            assert max(map(len, snippet.content.splitlines())) <= 79
            exec(compile(snippet.content, str(path), "exec"), namespace)
        if path.stem.endswith("rates"):
            test = namespace["ceiling_test"]
            convert = namespace["wrongful_denial_rate"]
            assert callable(test) and callable(convert)
            assert test(5, 5, 990, 5, 0.01) == (5, 1)
            assert convert(0.4, 3 / 80, 12 / 360) == pytest.approx(9 / 357)
        else:
            from demos.audits_explanations_examples import (
                context_from_records,
                example_records,
            )
            from refund_agent.rules import draft

            context = context_from_records(
                example_records(), may_act=lambda *_: True
            )
            state = context.current.state()
            generate = namespace["explain"]
            check = namespace["replay"]
            assert callable(generate) and callable(check)
            explanation = generate(
                state, draft(state), state.refunded_cents, context=context
            )
            assert all(check(explanation, state, context=context).values())
    assert {s.snippet_id for s in snippets} == {
        "fnd.audits-explanations.audit-core-01",
        "fnd.audits-explanations.explain-core-01",
    }
    for name in (
        "audits_explanations_demo.py",
        "audits_explanations_integrated_demo.py",
        "audits_explanations_integrated_capture.py",
    ):
        path = ROOT / "demos" / name
        snippets.extend(
            extractor["extract_snippets_from_text"](path.read_text(), path)
        )
    assert len({s.snippet_id for s in snippets}) == len(snippets)


def test_independent_expected_results_and_exhausted_case():
    out = ROOT / "book_outputs"
    keys = (out / "_fnd.audits-explanations.keys-01.txt").read_text()
    assert "false declines 140" in keys and "false declines 350" in keys
    assert "0.818181818" in keys and "Breach False" in keys
    release = (out / "_fnd.audits-explanations.release-01.txt").read_text()
    assert "Clean readings 80" in release and "Release: False" in release
    assert "Minimum clean sample for release: 126" in release
    checks = (out / "_fnd.audits-explanations.checks-01.txt").read_text()
    assert checks.count("Replay 6/6") == 4
    assert (
        "no_admissible_change" in checks and "named_approver:morgan" in checks
    )


def test_checks_days_follow_submission_execution_and_counterfactual_records(
    capsys, monkeypatch
):
    from demos import audits_explanations_demo as demo
    from demos.audits_explanations_examples import AT, example_records
    from refund_agent.memory.register import clause_text

    capsys.readouterr()

    def changed_records(kind):
        records = example_records(kind)
        if kind == "two-clause":
            for row, days in (
                (records["current"], 1), (records["history"][0], -1)
            ):
                delta = timedelta(days=days)
                for field in ("at", "policy_observed_at"):
                    row[field] = (
                        datetime.fromisoformat(row[field]) + delta
                    ).isoformat()
                for source in ("order", "shipment"):
                    row[source]["observed_at"] = (
                        datetime.fromisoformat(row[source]["observed_at"])
                        + delta
                    ).isoformat()
                row["request"]["at"] = (AT + timedelta(days=28)).isoformat()
                row["policy"]["return_days"] = 29
                row["policy"]["text"] = row["policy"]["text"].replace(
                    clause_text("eligibility", 30),
                    clause_text("eligibility", 29),
                )
        return records

    monkeypatch.setattr(demo, "example_records", changed_records)
    demo.print_checks()
    text = capsys.readouterr().out
    assert (
        "Submission day 28, execution day 32, "
        "counterfactual evaluation day 29."
    ) in text
    assert text.count("Replay 6/6") == 4


def test_producers_import_without_canonical_artifact(tmp_path, monkeypatch):
    monkeypatch.setenv("AUDITS_EXPLANATIONS_RUN_PATH", str(tmp_path / "absent.json"))
    namespace = runpy.run_path(str(ROOT / "demos/audits_explanations_integrated_demo.py"))
    assert callable(namespace["write_integrated_run"])
    assert callable(namespace["write_audit_report_05"])


def test_canonical_capture_refuses_missing_partner_artifact(tmp_path, monkeypatch):
    monkeypatch.setenv("AUDITS_EXPLANATIONS_RUN_PATH", str(tmp_path / "absent.json"))
    with pytest.raises(ValueError, match="integrate chapters 33-35 first"):
        capture_chapter(tmp_path, integrated=True)
    assert not (tmp_path / "book_outputs").exists()


def test_saved_canonical_outputs_reload_in_a_fresh_process(tmp_path):
    outputs = capture_chapter(tmp_path, integrated=True)
    assert len(outputs) == 5
    for name, text in outputs.items():
        stored = ROOT / "book_outputs" / f"_{name}.txt"
        assert stored.read_bytes() == text.encode()


def test_saved_canonical_loader_refuses_stale_candidate():
    from demos.audits_explanations_integrated_demo import _validate_run, integrated_run

    record = integrated_run()
    candidate = record.run.model_copy(deep=True).candidate
    candidate["agent"]["fingerprint"] = "0" * 64
    stale = record.model_copy(
        update={"run": record.run.model_copy(update={"candidate": candidate})}
    )
    with pytest.raises(ValueError, match="selected implementation is stale"):
        _validate_run(stale)


def test_producer_writes_constructed_records_and_printed_regions_capture_them(
    tmp_path,
    monkeypatch,
):
    from demos import audits_explanations_integrated_demo as producer
    from demos.audits_explanations_examples import (
        AT,
        context_from_records,
        example_records,
    )
    from refund_agent import contracts as c
    from refund_agent.agent import Agent
    from refund_agent.audit.audits_explanations_explain import explain, replay
    from refund_agent.audit.audits_explanations_month import (
        SETTINGS,
        Month,
        MonthRun,
        Truth,
    )
    from refund_agent.audit.audits_explanations_rates import (
        AuditObservation,
        AuditRule,
        Period,
        audit_period,
    )
    from refund_agent.audit.audits_explanations_records import SegmentAssignments
    from refund_agent.audit.audits_explanations_stage6 import ExplanationAttachment
    from refund_agent.data.generate import Dataset
    from refund_agent.release.candidate import build_candidate
    from refund_agent.release.suite import build_suite
    from refund_agent.tools import Tools

    records = example_records("tier2")
    records["current"]["request"]["amount_cents"] = 2499
    ctx = context_from_records(records, may_act=lambda *_: True)
    current = ctx.current
    data = Dataset(
        requests=[current.request],
        orders=[current.order],
        shipments=[current.shipment],
        policies=[current.policy],
        correct=[],
        retrievals=[
            c.PolicySnapshot(
                request_id=current.request.request_id,
                version=current.policy.version,
                observed_at=current.policy_observed_at,
            )
        ],
    )
    trace = Agent(Tools(data)).run(current.request, at=current.at)
    state = trace.steps[-1].state
    explanation = explain(state, trace.decision, 0, context=ctx)
    attachment = ExplanationAttachment(
        request_id=current.request.request_id,
        proposed_decision=trace.decision,
        explanation=explanation,
        checks=replay(explanation, state, context=ctx),
        failure=None,
        evaluation=current,
    )
    month = Month(
        seed=7,
        period=Period(start=AT, end=AT + timedelta(days=30)),
        settings=SETTINGS,
        segments=SegmentAssignments(
            by_order={current.order.order_id: "C"}, version="constructed-test"
        ),
        data=data,
        truth=(
            Truth(
                request_id=current.request.request_id,
                received=False,
                genuine=True,
                owed=True,
                found_false_if_paid=False,
            ),
        ),
        history={},
        evaluation_times={current.request.request_id: current.at},
    )
    run = MonthRun(
        candidate=build_candidate(build_suite(1)).record(),
        traces=(trace,),
        explanations=(attachment,),
        paid_cents=0,
        observations=(
            AuditObservation(
                request_id=current.request.request_id,
                order_id=current.order.order_id,
                segment="C",
                at=current.at,
                outcome="deny",
                reason="already_delivered",
                owed=True,
                found_false_if_paid=None,
                label_mature_at=current.at + timedelta(days=30),
            ),
        ),
    )
    rule = AuditRule(signed_at=AT - timedelta(days=1))
    audit = audit_period(
        run.observations,
        month.period,
        rule,
        7,
        at=month.period.end + timedelta(days=30),
    )
    path = tmp_path / "constructed-run.json"
    arguments = dict(
        month=month,
        run=run,
        audit=audit,
        public_traces=(trace,),
        public_explanations=(attachment,),
        partner_components={},
    )
    with pytest.raises(ValueError, match="all six Part VI partners"):
        producer.write_integrated_run(path, **arguments)
    assert not path.exists()
    # Test serialization/capture plumbing independently of absent partners.
    # No predecessor implementation or canonical integration is substituted.
    monkeypatch.setattr(producer, "_validate_run", lambda record: record)
    producer.write_integrated_run(path, **arguments)
    assert producer.IntegratedRun.model_validate_json(path.read_text()).run == run
    source = ROOT / "demos/audits_explanations_integrated_capture.py"
    test_source = tmp_path / "audits_explanations_constructed_capture.py"
    test_source.write_text(
        "from pathlib import Path\n"
        "from demos.audits_explanations_integrated_demo import IntegratedRun\n"
        + source.read_text().replace(
            "record = integrated_run()",
            f"record = IntegratedRun.model_validate_json(Path({str(path)!r}).read_text())",
        )
    )
    extractor = vars(_regions)
    snippets = {
        s.snippet_id: s
        for s in extractor["extract_snippets_from_text"](
            test_source.read_text(),
            test_source,
        )
    }
    outputs = extractor["capture_outputs"](snippets, sys.executable)
    assert len(outputs) == 5
    assert "Constructed seed-7 month" in outputs["fnd.audits-explanations.month-01"]
    assert "shipment.deliveries" in outputs["fnd.audits-explanations.explain-01"]
    assert "passing denial explanations 1/1" in outputs["fnd.audits-explanations.explain-02"]


def test_all_new_sources_parse():
    for path in (ROOT / "refund_agent/audit").iterdir():
        if path.suffix == ".py":
            ast.parse(path.read_text(), filename=str(path))
