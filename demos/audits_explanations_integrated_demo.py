"""Capture canonical month evidence after the partner components have joined.

The input is written by write_integrated_run after real candidate runs. Absence
or stale code refuses capture. There is no fixture or absent-partner substitute.
"""

import json
import os
import re
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from textwrap import fill

# Core initialization precedes selected component helpers, as on the spine.
from refund_agent.release.candidate import Candidate, build_candidate, source_fingerprint
from refund_agent.release.extensions import active
from refund_agent.release.suite import IMPLEMENTATION_FILES, build_suite
from refund_agent.runtimes.adapter import AdapterSettings

CORE = build_candidate(build_suite(trials=1))

# isort: split
from refund_agent import contracts as c
from refund_agent.audit.audits_explanations_explain import DENIAL_REASONS
from refund_agent.audit.audits_explanations_month import Month, MonthRun
from refund_agent.audit.audits_explanations_rates import (
    AuditRecord,
    audit_period,
    ceiling_test,
    confusion_by_segment,
    criteria,
    power,
)
from refund_agent.audit.audits_explanations_stage6 import ExplanationAttachment
from refund_agent.data.generate import number

ROOT = Path(__file__).resolve().parents[1]
RUN_PATH = ROOT / "demos/data/audits_explanations_run.json"


class IntegratedRun(c.Record):
    month: Month
    run: MonthRun
    audit: AuditRecord
    public_traces: tuple[c.Trace, ...]
    public_explanations: tuple[ExplanationAttachment, ...]
    partner_components: dict[str, str]


def _validate_run(record):
    candidate = Candidate(**record.run.candidate)
    suite = build_suite(
        trials=1, components=candidate.tools_composition.get("components", ())
    )
    current = build_candidate(
        suite,
        candidate.name,
        candidate.runtime["identity"],
        candidate.tools_composition["verifier"],
        settings=AdapterSettings(**candidate.adapter["settings"]),
    )
    if current.record() != candidate.record():
        raise ValueError("canonical run's selected implementation is stale")
    components = {p.name: p for p, settings in active(candidate)}
    required = {
        "data",
        "event_time",
        "canary",
        "promotion",
        "observe",
        "scope",
    }
    if (
        set(record.partner_components) != required
        or any(
            name not in components
            for name in record.partner_components.values()
        )
        or not components[record.partner_components["scope"]].outermost
    ):
        raise ValueError(
            "canonical capture requires all six Part VI partners and outermost scope"
        )
    if "audits-explanations.stage6" not in components:
        raise ValueError("canonical capture requires the stage-6 component")
    if candidate.agent["fingerprint"] != source_fingerprint(
        "agent/__init__.py", "tools.py", "contracts.py", "rules.py"
    ) or candidate.tools_composition["fingerprint"] != source_fingerprint(
        *IMPLEMENTATION_FILES
    ):
        raise ValueError("canonical run's core implementation is stale")
    if record.month.seed != 7 or len(record.run.traces) != 2000:
        raise ValueError(
            "canonical month requires one seeded run for all 2000 requests"
        )
    if record.audit != audit_period(
        record.run.observations,
        record.month.period,
        record.audit.rule,
        record.month.seed,
        at=record.audit.at,
    ):
        raise ValueError("audit differs from captured decision observations")
    if (
        len(record.public_traces) != 40
        or len({t.decision.request_id for t in record.public_traces}) != 40
    ):
        raise ValueError(
            "canonical public replay requires all forty distinct requests"
        )
    for traces, attachments in (
        (record.run.traces, record.run.explanations),
        (record.public_traces, record.public_explanations),
    ):
        explanations = {a.request_id: a for a in attachments}
        for trace in traces:
            d = trace.decision
            if (
                d.outcome == c.Outcome.DENY
                or d.outcome == c.Outcome.REFER
                and d.reason in DENIAL_REASONS
            ):
                a = explanations.get(d.request_id)
                if (
                    a is None
                    or a.explanation is None
                    or a.failure
                    or len(a.checks) != 6
                    or not all(a.checks.values())
                ):
                    raise ValueError(
                        "canonical capture has an unexplained denial"
                    )
    return record


def write_integrated_run(
    path,
    *,
    month,
    run,
    audit,
    public_traces,
    public_explanations,
    partner_components,
):
    record = _validate_run(
        IntegratedRun(
            month=month,
            run=run,
            audit=audit,
            public_traces=public_traces,
            public_explanations=public_explanations,
            partner_components=partner_components,
        )
    )
    c.write_json(Path(path), record)
    sidecar = Path(path).with_name("audits_explanations_figures.json")
    sidecar.write_text(
        json.dumps(
            {
                "audit": audit.model_dump(mode="json"),
                "oracle": {
                    g: criteria(v)
                    for g, v in confusion_by_segment(
                        month, tuple(t.decision for t in run.traces)
                    ).items()
                },
                "identity": [
                    {"base_rate": p, "fpr": 0.8 * p / (1 - p)}
                    for p in (0.1, 0.3)
                ],
            },
            sort_keys=True,
            indent=2,
        )
        + "\n"
    )
    return record


def write_audit_report_05(path, *, month, run, rule, at):
    if month.seed != 11 or len(run.traces) != 2000:
        raise ValueError(
            "exercise 5 requires the real seed-11 constructed month"
        )
    candidate = Candidate(**run.candidate)
    selected = active(candidate)
    if not any(p.outermost for p, settings in selected):
        raise ValueError("exercise 5 requires the integrated scope boundary")
    audit = audit_period(run.observations, month.period, rule, 11, at=at)
    sample = {rid for s in audit.segments for rid in s.sample_ids}
    truth = {t.request_id: t for t in month.truth}
    positions = {r.request_id: i for i, r in enumerate(month.data.requests)}
    appeals = {g: 0 for g in rule.segments}
    denials = {g: 0 for g in rule.segments}
    observations = []
    for o in run.observations:
        if o.outcome == "deny":
            denials[o.segment] += 1
            probability = (
                0.5
                if truth[o.request_id].owed
                else 0.10
                if o.segment == "B"
                else 0.02
            )
            appeals[o.segment] += number(
                11,
                positions[o.request_id],
                "audits-explanations-audit-appeal",
                10000,
            ) < int(probability * 10000)
        observations.append(
            o.model_copy(
                update={"owed": o.owed if o.request_id in sample else None}
            )
        )
    rates = {
        g: appeals[g] / denials[g] if denials[g] else None
        for g in rule.segments
    }
    leader = max(
        (g for g in rates if rates[g] is not None), key=lambda g: rates[g]
    )
    record = {
        "seed": 11,
        "period": month.period.model_dump(mode="json"),
        "rule": rule.model_dump(mode="json"),
        "at": at.isoformat(),
        "observations": [o.model_dump(mode="json") for o in observations],
        "appeals": appeals,
        "denials": denials,
        "agent_report": {
            "measure": "appeals_per_denial",
            "rates": rates,
            "held": leader,
            "planted_error": "appeals_substituted_for_blind_readings",
        },
    }
    Path(path).write_text(json.dumps(record, sort_keys=True, indent=2) + "\n")
    return record


def integrated_run():
    path = Path(os.environ.get("AUDITS_EXPLANATIONS_RUN_PATH", RUN_PATH))
    return _read_run(path)


@lru_cache(maxsize=1)
def _read_run(path):
    if not path.is_file():
        raise ValueError(
            "canonical audits-explanations run is absent; integrate chapters 33-35 first"
        )
    return _validate_run(IntegratedRun.model_validate_json(path.read_text()))


def _rate(value):
    return "unassessable" if value is None else f"{value:.9f}"


def _relative_times(text, origin):
    def replace(match):
        at = datetime.fromisoformat(match[0])
        return f"Day {(at - origin).days + 1}, {at:%H:%M:%S} UTC"

    return re.sub(
        r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?[+]00:00", replace, text
    )


def print_month(record):
    month, run = record.month, record.run
    print("Constructed seed-7 month. Truth is never read by tools or rules.")
    counts = confusion_by_segment(month, tuple(t.decision for t in run.traces))
    for g, row in counts.items():
        rates = criteria(row)
        print(
            f"{g}: TP {row.tp}, FP {row.fp}, FN {row.fn}, TN {row.tn}, referred {row.referred}."
        )
        print(
            f"Decided {row.tp + row.fp + row.fn + row.tn}, denied {row.tp + row.fp}, paid {row.fn + row.tn}."
        )
        for name, value in rates.items():
            print(f"{name}: {_rate(value)}")
        print(
            f"Denials wrong: {_rate(1 - rates['ppv'] if rates['ppv'] is not None else None)}"
        )
        print(
            f"Illustrative wrongful-denial cost, 20 dollars each: {row.fp * 20} dollars."
        )
    requests = {
        r.request_id
        for r in month.data.requests
        if r.reason == "not_delivered"
    }
    restricted = month.model_copy(
        update={
            "truth": tuple(t for t in month.truth if t.request_id in requests)
        }
    )
    for g, row in confusion_by_segment(
        restricted,
        tuple(
            t.decision for t in run.traces if t.decision.request_id in requests
        ),
    ).items():
        rates = criteria(row)
        print(
            f"Not-delivered {g}: TPR {_rate(rates['tpr'])}, FPR {_rate(rates['fpr'])}."
        )
        print(
            f"Not-delivered {g}: PPV {_rate(rates['ppv'])}, base {_rate(rates['base_rate'])}."
        )


def print_audit(record):
    rule = record.audit.rule
    print("Constructed seed-7 audit. All payment labels have matured.")
    for name in (
        "version",
        "segments",
        "read_per_segment",
        "ceiling",
        "alpha_family",
        "test",
        "correction",
        "owner",
        "signed_by",
        "source_owner",
        "response",
    ):
        print(fill(f"{name}: {rule.model_dump()[name]}", width=79))
    print("Signature checked before the period's first day.")
    for r in record.audit.segments:
        print(
            f"{r.segment}: decided {r.decided}, denied {r.denials}, paid {r.paid}, referred {r.referred}."
        )
        print(
            f"Denial share {_rate(r.denial_share)}, read {r.read}, wrongful {r.wrongful}."
        )
        print(f"Wrongful sample share {_rate(r.wrongful_share)}.")
        print(f"Wilson interval {r.wilson}.")
        print(
            f"Payments found false {_rate(r.found_false_share)}, owed payments {r.owed_payments}."
        )
        print(f"Owed-denial estimate {_rate(r.estimate)}.")
        print(f"Conditional range {r.estimate_range}.")
        print(
            f"Kceil {r.ceiling_count}, exact tail {_rate(r.chance)}, breach {r.breach}."
        )
        print(
            f"Status {r.status}, upper owed-denial share {_rate(r.upper_owed_denial_share)}."
        )
        print(
            f"Release {r.release_allowed}, minimum clean sample {r.minimum_clean_release_sample}."
        )
        print(
            f"Held proposals sampled: {r.held_proposals}. Reasons held: {r.held_reasons}."
        )
        if r.read >= 3:
            print(
                f"If sample held 3: exact tail {float(ceiling_test(3, r.denials, r.owed_payments, r.read, rule.ceiling)[1]):.9f}."
            )
        if r.breach:
            for reason in r.held_reasons:
                held = sum(
                    o.segment == r.segment
                    and o.reason == reason
                    and o.outcome == "deny"
                    for o in record.run.observations
                )
                print(f"Send {held} {r.segment}/{reason} denials to a person.")
            print("Notify the carrier integration team.")


def print_power(record):
    counts = confusion_by_segment(
        record.month, tuple(t.decision for t in record.run.traces)
    )
    for row in record.audit.segments:
        print(
            f"{row.segment}: exact power {float(power(record.audit.rule, row.denials, counts[row.segment].fp, row.owed_payments, row.read)):.9f}."
        )
        if row.ceiling_count < row.denials:
            print(
                f"One wrongful above ceiling: {float(power(record.audit.rule, row.denials, row.ceiling_count + 1, row.owed_payments, row.read)):.9f}."
            )


def print_explanation(record):
    truth = {t.request_id: t for t in record.month.truth}
    selected = {
        t.decision.request_id
        for t in record.run.traces
        if t.decision.reason == "already_delivered"
        and t.decision.outcome == c.Outcome.DENY
    }
    request = next(
        r
        for r in record.month.data.requests
        if r.request_id in selected
        and record.month.segments.region_of(r.order_reference) == "C"
        and not truth[r.request_id].received
    )
    attachment = next(
        a
        for a in record.run.explanations
        if a.request_id == request.request_id
    )
    e = attachment.explanation
    print(
        f"{request.request_id}: {request.amount_cents} cents, tier 1, region C."
    )
    for clause in e.clauses:
        print(fill(f"{clause.citation}: {clause.text}", width=79))
    for evidence in e.evidence:
        print(
            fill(
                _relative_times(
                    f"{evidence.field}: {evidence.value}",
                    record.month.period.start,
                ),
                width=79,
            )
        )
        print(
            fill(
                f"source {evidence.source}, age {evidence.age_seconds:g} seconds",
                width=79,
            )
        )
        print(fill(f"record: {evidence.record_id}", width=79))
        print(
            _relative_times(
                f"Observed at: {evidence.observed_at.isoformat()}",
                record.month.period.start,
            )
        )
    print(
        fill(
            _relative_times(e.rendered, record.month.period.start),
            width=79,
            replace_whitespace=False,
        )
    )
    for name, passed in attachment.checks.items():
        print(f"Replay {name}: {'pass' if passed else 'fail'}")
    print(
        "Constructed truth, never read by the agent: the parcel did not arrive."
    )


def print_replays(record):
    for name, traces, attachments in (
        ("Month", record.run.traces, record.run.explanations),
        (
            "Forty public scenarios",
            record.public_traces,
            record.public_explanations,
        ),
    ):
        ids = {
            t.decision.request_id
            for t in traces
            if t.decision.outcome == c.Outcome.DENY
        }
        passed = sum(
            a.request_id in ids
            and a.failure is None
            and all(a.checks.values())
            for a in attachments
        )
        print(f"{name}: passing denial explanations {passed}/{len(ids)}.")
    print("Two-clause and exhausted-balance record checks: see checks-01.")
