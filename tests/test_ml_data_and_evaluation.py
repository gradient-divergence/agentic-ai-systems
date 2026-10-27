"""Tests for the listings and outputs of the chapter on ML data and evaluation."""

import importlib
import io
import json
import math
import statistics
import subprocess
import sys
from contextlib import redirect_stdout
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from refund_agent.baseline import score_decisions
from refund_agent.data.generate import generate
from refund_agent.labels import LABEL_POLICY, relabel
from refund_agent.split import CUTS, WINDOWS, fingerprint, freeze, time_split

COMPANION = Path(__file__).resolve().parents[1]
DEMO = COMPANION / "demos" / "ml_data_evaluation_demo.py"
FIXTURE = COMPANION / "demos" / "data" / "ml_data_evaluation_split_05.json"


def demo():
    return importlib.import_module("demos.ml_data_evaluation_demo")


@pytest.fixture(scope="module")
def data():
    return generate(7)


def test_import_prints_nothing_and_writes_nothing(tmp_path: Path) -> None:
    code = (
        "import contextlib, io, os, sys\n"
        f"sys.path.insert(0, {str(COMPANION)!r})\n"
        "buffer = io.StringIO()\n"
        "with contextlib.redirect_stdout(buffer):\n"
        "    from demos.ml_data_evaluation_demo import wilson_interval, measures\n"
        "assert buffer.getvalue() == '', buffer.getvalue()\n"
        "assert os.listdir('.') == [], os.listdir('.')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=tmp_path, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


def test_relabeling_under_the_latest_version_changes_seventy_denials(data) -> None:
    labels = {d.request_id: d for d in data.correct}
    latest = {d.request_id: d for d in relabel(data, "2026-09-01")}
    changed = [
        r for r in data.requests if latest[r.request_id].outcome != labels[r.request_id].outcome
    ]
    assert len(changed) == 70
    assert {labels[r.request_id].outcome.value for r in changed} == {"deny"}
    assert all(data.policy_at(r.at).version != "2026-09-01" for r in changed)
    assert LABEL_POLICY.window_days == 0
    # The label in force is the generator's own: relabeling under each request's version is a no-op.
    for r in data.requests[:50]:
        same = relabel(data, data.policy_at(r.at).version)
        assert same[data.requests.index(r)] == labels[r.request_id]


def test_time_split_counts_and_the_final_label_condition(data) -> None:
    windows = time_split(data.requests, CUTS, timedelta(0))
    assert [len(windows[name]) for name in WINDOWS] == [398, 81, 121]
    assert (
        max(r.at for r in windows["training"]) < CUTS[0] <= min(r.at for r in windows["validation"])
    )
    assert (
        max(r.at for r in windows["validation"]) < CUTS[1] <= min(r.at for r in windows["golden"])
    )
    # A 30-day label window drops every request whose label would be open at the next cut.
    gap = timedelta(days=30)
    held = time_split(data.requests, CUTS, gap)
    assert all(r.at + gap <= CUTS[0] for r in held["training"])
    assert all(r.at + gap <= CUTS[1] for r in held["validation"])
    dropped = sum(1 for r in data.requests if r.at < CUTS[0] and r.at + gap > CUTS[0])
    assert len(held["training"]) == 398 - dropped and dropped > 0
    assert len(held["golden"]) == 121


def test_measures_base_rate_and_intervals() -> None:
    module = demo()
    bank = module.measures(tp=18, fp=6, fn=2, tn=74)
    assert bank["precision"] == pytest.approx(0.75)
    assert bank["recall"] == pytest.approx(0.9)
    assert bank["F1"] == pytest.approx(36 / 44)
    assert bank["accuracy"] == pytest.approx(0.92)
    assert bank["false-positive rate"] == pytest.approx(0.075)
    # F1 is the harmonic mean of precision and recall.
    p, r = bank["precision"], bank["recall"]
    assert bank["F1"] == pytest.approx(2 * p * r / (p + r))
    assert module.precision_at(0.2, 0.9, 0.075) == pytest.approx(0.75)
    assert module.precision_at(0.01, 0.9, 0.075) == pytest.approx(0.009 / 0.08325)
    with pytest.raises(ZeroDivisionError):  # a model that never flags has no precision
        module.measures(tp=0, fp=0, fn=1, tn=99)
    assert module.wald_interval(0, 15) == (0.0, 0.0)
    low, high = module.wilson_interval(0, 15)
    assert low == 0.0 and high == pytest.approx(1.96**2 / (15 + 1.96**2))
    assert module.wald_interval(10, 100) == pytest.approx((0.0412, 0.1588), abs=1e-4)
    assert module.wilson_interval(10, 100) == pytest.approx((0.0552, 0.1744), abs=1e-4)
    assert module.cases_needed(0.1, 0.05) == 139
    assert module.cases_needed(0.2, 0.05) == 246
    # The bank's recall: 2 misses among its 20 fraud cases, not among its 100 payments.
    assert module.wilson_interval(2, 20) == pytest.approx((0.0279, 0.3010), abs=1e-4)


def test_a_nominal_95_percent_interval_does_not_cover_95_percent_of_the_time() -> None:
    module = demo()
    # At 15 cases and a rate of 0.01, only the sample with no failure gives a Wilson
    # interval that holds the rate, so the actual coverage is the chance of that sample.
    assert module.wilson_interval(1, 15)[0] > 0.01
    assert module.coverage(15, 0.01) == pytest.approx(0.99**15)
    assert module.coverage(15, 0.01) < 0.95 < module.coverage(15, 0.2)
    # The Wald interval at 100 cases, at the two rates the chapter quotes from its source.
    assert module.coverage(100, 0.106, module.wald_interval) == pytest.approx(0.952, abs=5e-4)
    assert module.coverage(100, 0.107, module.wald_interval) == pytest.approx(0.911, abs=5e-4)
    # The exact one-sided bound of the chapter before keeps its 5%: no failure in 15
    # cases has a chance of exactly 0.05 at the bound, and of less at Wilson's upper end.
    exact = 1 - 0.05 ** (1 / 15)
    assert (1 - exact) ** 15 == pytest.approx(0.05)
    assert (1 - module.wilson_interval(0, 15)[1]) ** 15 < 0.05


@pytest.mark.parametrize("k,n", [(0, 15), (3, 50), (10, 100), (20, 38), (50, 50)])
def test_wilson_ends_solve_the_score_quadratic(k: int, n: int) -> None:
    module = demo()
    z, rate = 1.96, k / n
    low, high = module.wilson_interval(k, n)
    for end in (low, high):
        assert (rate - end) ** 2 == pytest.approx(z * z * end * (1 - end) / n, abs=1e-12)
    # The interval for successes is the mirror of the interval for failures.
    mirror = module.wilson_interval(n - k, n)
    assert (1 - mirror[1], 1 - mirror[0]) == pytest.approx((low, high))


def test_golden_set_is_fixed_and_every_read_is_logged(data) -> None:
    windows = time_split(data.requests, CUTS, timedelta(0))
    golden = freeze(data, windows["golden"], LABEL_POLICY)
    assert golden.manifest["requests"] == 121
    assert golden.manifest["by_tier"] == {1: 38, 2: 34, 3: 49}
    assert golden.manifest["by_policy_version"] == {"2026-07-01": 61, "2026-09-01": 60}
    assert golden.intact()
    assert golden.log == []
    rows = golden.consult("test", "one read")
    assert len(rows) == 121
    assert golden.log == [{"entry": 1, "set": "golden-1", "reader": "test", "purpose": "one read"}]
    altered = list(golden.labels)
    altered[0] = altered[0].model_copy(update={"reason": "changed"})
    assert fingerprint(list(golden.requests), altered) != golden.manifest["sha256"]


def test_a_slice_keeps_its_parents_labels_name_and_access_log(data) -> None:
    windows = time_split(data.requests, CUTS, timedelta(0))
    golden = freeze(data, windows["golden"], LABEL_POLICY)
    golden.consult("report", "first read")
    september = [r for r in data.requests if r.at >= data.policies[-1].effective_at]
    piece = freeze(data, september, LABEL_POLICY, name="golden-1-september", parent=golden)
    assert piece.manifest["parent"] == "golden-1"
    assert piece.manifest["parent_sha256"] == golden.manifest["sha256"]
    assert piece.manifest["requests"] == 60 and piece.intact()
    assert piece.log is golden.log and len(piece.log) == 1  # the parent's history, kept
    rows = piece.consult("exercise", "second read")
    assert len(rows) == 60 and golden.intact()
    assert [(e["entry"], e["set"]) for e in golden.log] == [
        (1, "golden-1"),
        (2, "golden-1-september"),
    ]
    held = dict(zip((r.request_id for r in golden.requests), golden.labels))
    assert all(held[r.request_id] == label for r, label in rows)
    with pytest.raises(ValueError):  # a July request is no part of the golden set
        freeze(data, windows["validation"][:1], LABEL_POLICY, name="not-a-slice", parent=golden)


def test_a_slice_rejects_a_changed_parent_request(data) -> None:
    windows = time_split(data.requests, CUTS, timedelta(0))
    golden = freeze(data, windows["golden"], LABEL_POLICY)
    request = next(r for r in golden.requests if r.amount_cents == 4999)
    changed = request.model_copy(update={"amount_cents": 5000})
    with pytest.raises(ValueError, match="a slice must keep its parent's requests unchanged"):
        freeze(data, [changed], LABEL_POLICY, parent=golden)
    assert request.amount_cents == 4999 and golden.intact()


def test_value_check_misses_the_audit_field_and_the_time_check_does_not(data) -> None:
    module = demo()
    labels = {d.request_id: module.payment_forbidden(d) for d in data.correct}
    outcomes = {d.request_id: d.outcome.value for d in data.correct}
    some = time_split(data.requests, CUTS, timedelta(0))["validation"][:40]  # July
    audits = module.audit_fields(data, some)
    assert {after for _, after in audits.values()} == {0.0}  # stamped at the request's time
    paid = [audits[r.request_id][0] for r in some]
    truth = [labels[r.request_id] for r in some]
    assert module.value_check(truth, truth) == "flags"
    assert module.value_check(paid, truth) == "passes"
    assert module.time_check("paid_amount_cents", 0.0) == "flags: decision writes it"
    assert module.time_check("days_since_delivery", 0.0) == "passes"
    # A paid amount appears exactly when the label approves.
    assert [p > 0 for p in paid] == [outcomes[r.request_id] == "approve" for r in some]


def test_baseline_matches_every_golden_label_by_the_stage_0_scorer(data) -> None:
    module = demo()
    golden = freeze(data, time_split(data.requests, CUTS, timedelta(0))["golden"], LABEL_POLICY)
    read = golden.consult("test", "the baseline's errors")
    drafts = module.baseline_decisions(data, [r for r, _ in read])
    card = score_decisions(drafts, [label for _, label in read])
    assert (card.correct, card.total) == (121, 121)
    flagged = {r.request_id: module.payment_forbidden(label) for r, label in read}
    counts = module.tier_counts(read, drafts, flagged)
    assert [(tier, n, errors) for tier, n, errors, *_ in counts] == [
        (1, 38, 0),
        (2, 34, 0),
        (3, 49, 0),
        ("All", 121, 0),
    ]
    assert [fp + fn for *_, fp, fn in counts] == [0, 0, 0, 0]  # perfect flags, as given
    # One cent more on an approved amount keeps the outcome and is still an error.
    i = next(i for i, d in enumerate(drafts) if d.outcome.value == "approve")
    altered = list(drafts)
    altered[i] = drafts[i].model_copy(update={"amount_cents": drafts[i].amount_cents + 1})
    assert altered[i].outcome == read[i][1].outcome
    assert [errors for _, _, errors, *_ in module.tier_counts(read, altered, flagged)] == [
        1,
        0,
        0,
        1,
    ]


def test_tier_three_trace_uses_the_existing_golden_read_and_keeps_named_authority(data):
    from demos.regions import extract_snippets_from_text

    marked = {s.snippet_id: s for s in extract_snippets_from_text(DEMO.read_text(), DEMO)}
    assert "fnd.ml-data-and-evaluation.trace-01" in marked
    golden = freeze(data, time_split(data.requests, CUTS, timedelta(0))["golden"], LABEL_POLICY)
    read = golden.consult("test", "the existing chapter report")
    log = list(golden.log)
    namespace = {"__name__": "__main__", "data": data, "read": read}
    printed = io.StringIO()
    with redirect_stdout(printed):
        exec(marked["fnd.ml-data-and-evaluation.trace-01"].content, namespace)
    request, label = namespace["trace_request"], namespace["trace_label"]
    eligible = [r for r, label in read if r.amount_cents > 50000 and label.reason == "approval_required"]
    assert request == min(eligible, key=lambda r: (r.at, r.request_id))
    assert label.outcome.value == "refer" and int(label.tier) == 3
    assert data.policy_at(request.at).named_approver in printed.getvalue()
    assert "agent prepares the packet only; named approver decides, denials included" in printed.getvalue()
    assert golden.log == log and golden.intact()
    assert printed.getvalue().encode() == (
        COMPANION / "book_outputs/_fnd.ml-data-and-evaluation.trace-01.txt"
    ).read_bytes()


@pytest.mark.parametrize("policy_outcome", ["refer", "deny"])
def test_tier_three_agent_leaves_both_policy_outcomes_to_the_named_approver(data, policy_outcome):
    module = demo()
    labels = {label.request_id: label for label in data.correct}
    request = next(
        r for r in time_split(data.requests, CUTS, timedelta(0))["golden"]
        if r.amount_cents > 50000 and labels[r.request_id].outcome.value == policy_outcome
    )
    tools = module.Tools(module.one_request_data(data, request))
    trace = module.Agent(tools).run(request)
    assert trace.decision.outcome.value == "refer" and int(trace.decision.tier) == 3
    assert trace.receipt is None


def test_demo_reaches_golden_rows_only_through_the_logged_consultation() -> None:
    source = DEMO.read_text(encoding="utf-8")
    frozen = source.index('golden = freeze(data, windows.pop("golden"), LABEL_POLICY)')
    first_read = source.index("golden.consult(")
    assert frozen < source.index("# region book:fnd.ml-data-and-evaluation.leak-01") < first_read
    # From the freeze to the first consultation, the leak checks and the model
    # comparisons name no golden row: no window of that name, no frozen row, no raw
    # label, and the script's own labels are cut down to the development cases.
    between = source[frozen:first_read]
    assert "labels = {r.request_id: labels[r.request_id] for r in development}" in between
    for name in ('windows["golden"]', "golden.requests", "golden.labels", "data.correct"):
        assert name not in between, name
    assert between.count("data.requests") == 0
    # After it, golden rows come from a consultation and nowhere else.
    after = source[first_read:]
    for name in ('windows["golden"]', "golden.requests", "golden.labels", "data.correct"):
        assert name not in after, name


def test_decision_time_inputs_read_nothing_from_after_the_request(data) -> None:
    module = demo()
    rows = module.decision_time_inputs(data)
    assert set(rows[0]) == set(module.INPUTS)
    ambiguous = next(i for i, r in enumerate(data.requests) if r.failure == "ambiguous_order")
    assert math.isnan(rows[ambiguous]["balance_cents"])  # two orders match; none is read
    for row, request in zip(rows, data.requests):
        assert row["return_days"] == data.policy_at(request.at).return_days
        assert math.isnan(row["days_since_delivery"]) or row["days_since_delivery"] >= 0


def test_exercise_5_manifest_has_one_planted_error(data) -> None:
    manifest = json.loads(FIXTURE.read_text(encoding="utf-8"))
    windows = time_split(data.requests, CUTS, timedelta(0))
    split = manifest["split"]
    assert [split[name]["requests"] for name in WINDOWS] == [398, 81, 121]
    assert split["training"]["to"] < split["validation"]["from"] <= split["golden"]["from"]
    assert split["gap_days"] == LABEL_POLICY.window_days
    assert manifest["labels"]["policy_version_used"] == "in force at each request's time"
    records = {item["record"] for item in manifest["inputs"]}
    assert "audit record" not in records
    golden = freeze(data, windows["golden"], LABEL_POLICY)
    assert manifest["golden_set"]["sha256"] == golden.manifest["sha256"][:16]
    assert manifest["golden_set"]["access_log_entries"] == 0
    # The planted error: a statistic fitted on every request, not on the training rows.
    fitted = manifest["fitted_statistics"]
    assert len(fitted) == 1 and fitted[0]["fitted_on"] == "all 600 requests"
    rows = demo().decision_time_inputs(data)
    days = [row["days_since_delivery"] for row in rows]
    position = {r.request_id: i for i, r in enumerate(data.requests)}
    training = [days[position[r.request_id]] for r in windows["training"]]
    assert fitted[0]["value"] == statistics.median(d for d in days if not math.isnan(d))
    assert fitted[0]["value"] != statistics.median(d for d in training if not math.isnan(d))


def test_split_dates_are_utc_days() -> None:
    assert CUTS == (
        datetime(2026, 7, 1, tzinfo=timezone.utc),
        datetime(2026, 8, 1, tzinfo=timezone.utc),
    )
