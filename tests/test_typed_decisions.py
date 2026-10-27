"""Typed answers, proper scores, sealed retests and runnable Chapter 20 listings.

The seed-20 source audit traverses the entire companion tree, including canonical
sealed directories. It requires FND_ALLOW_SEALED=1, set only for a separately
authorized sealed-execution run.
"""

import ast
import contextlib
import importlib
import io
import json
import math
import runpy
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from pydantic import ValidationError

from demos import regions as _regions

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "demos/typed_decisions_demo.py"
PREFIX = "fnd.typed-decisions."
NAMES = ("typed-01", "router-01", "scores-01", "retest-01", "bands-01",
         "conformal-01", "figures-01", "keys-01")
TOOLS = vars(_regions)


def demo():
    return importlib.import_module("demos.typed_decisions_demo")


def typed():
    return importlib.import_module("refund_agent.typed")


def regions():
    assert DEMO.exists(), "Chapter 20's runnable demo has not been implemented"
    return {s.snippet_id: s for s in TOOLS["extract_snippets_from_text"](DEMO.read_text(), DEMO)}


def test_printed_listings_run_in_chapter_order_without_hidden_setup(tmp_path, monkeypatch):
    monkeypatch.setenv("BOOK_GOLDEN_DIRECTORY", str(tmp_path))
    namespace = {"__name__": "__main__"}
    snippets = regions()
    for name in ("typed-01", "scores-01", "conformal-01"):
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            exec(compile(snippets[PREFIX + name].content, name, "exec"), namespace)
        assert stdout.getvalue().encode() == (
            ROOT / "book_outputs" / f"_{PREFIX}{name}.txt"
        ).read_bytes()


@pytest.fixture(scope="module")
def experiment(tmp_path_factory):
    return demo().build_experiment(tmp_path_factory.mktemp("typed-journal"))


@pytest.fixture(scope="module")
def captured():
    return TOOLS["capture_outputs"](regions(), sys.executable)


@pytest.mark.parametrize("name", NAMES)
def test_every_stored_output_regenerates_byte_for_byte(name, captured):
    stored = ROOT / "book_outputs" / f"_{PREFIX}{name}.txt"
    assert captured[PREFIX + name].encode() == stored.read_bytes()


def test_seed_and_fresh_process_capture_are_deterministic(captured):
    assert captured == TOOLS["capture_outputs"](regions(), sys.executable)


def test_import_fits_nothing_prints_nothing_and_writes_nothing(tmp_path):
    code = """from refund_agent.data import history
def forbidden(*args, **kwargs):
    raise AssertionError('import generated history')
history.generate_history = forbidden
import demos.typed_decisions_demo
"""
    result = subprocess.run([sys.executable, "-c", code], cwd=tmp_path,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout == ""
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("mu", [0.0, 0.04, 0.3, 0.7, 1.0])
def test_proper_scores_minimize_at_truth_and_absolute_error_at_an_endpoint(mu):
    m = demo()
    grid = sorted(set(np.linspace(0, 1, 101)) | {mu})
    for rule in ("brier", "log"):
        best = m.expected_score(mu, mu, rule)
        assert all(m.expected_score(mu, p, rule) >= best - 1e-14 for p in grid)
        assert all(m.expected_score(mu, p, rule) > best for p in grid if abs(p - mu) > 1e-12)
    endpoint = 0 if mu < 0.5 else 1
    assert min(m.expected_score(mu, p, "absolute") for p in grid) == pytest.approx(
        m.expected_score(mu, endpoint, "absolute")
    )


def test_worked_scores_and_smoothing_do_not_clamp_the_unbounded_log_score():
    m = demo()
    assert [m.expected_score(.7, p, "brier") for p in (.5, .7, .9, 1)] == pytest.approx(
        [.25, .21, .25, .30]
    )
    assert [m.expected_score(.7, p, "log") for p in (.5, .7, .9)] == pytest.approx(
        [.69314718056, .61086430205, .76452788886]
    )
    assert math.isinf(m.expected_score(.7, 1, "log"))
    assert m.smoothed_minimizer(.04, .05) == pytest.approx(.086)
    assert m.smoothed_minimizer(.04, .05) > .080


def test_decomposition_is_exact_for_few_distinct_forecasts_and_reports_binning_residual():
    m = demo()
    rng = np.random.default_rng(20)
    for _ in range(50):
        p = rng.choice([.05, .2, .7], 100)
        y = rng.binomial(1, p)
        parts = m.decomposition(y, p, bins=None)
        assert parts["brier"] == pytest.approx(
            parts["reliability"] - parts["resolution"] + parts["uncertainty"], abs=1e-14
        )
    y = np.array([1, 0, 0, 0, 0, 1, 1, 1, 0, 0])
    parts = m.decomposition(y, [.1] * 5 + [.7] * 5, bins=None)
    assert [parts[k] for k in ("brier", "reliability", "resolution", "uncertainty")] == pytest.approx(
        [.21, .01, .04, .24]
    )
    continuous = m.decomposition([0, 1, 0, 1], [.1, .2, .3, .8], bins=2)
    assert abs(continuous["residual"]) > .001
    assert continuous["brier"] == pytest.approx(
        continuous["reliability"] - continuous["resolution"]
        + continuous["uncertainty"] + continuous["residual"]
    )


def test_rank_and_triage_set_and_rank_beyond_sample():
    m = demo()
    rank, threshold = m.conformal_threshold([.05, .10, .10, .15, .20, .25, .30, .40, .60], .2)
    assert (rank, threshold) == (8, .4)
    assert m.prediction_set([.7, .2, .1], threshold) == (0,)
    rank, threshold = m.conformal_threshold([.1, .2], .1)
    assert rank == 3 and math.isinf(threshold)
    assert m.prediction_set([.7, .2, .1], threshold) == (0, 1, 2)
    assert m.conformal_threshold([.1], .5) == (1, .1)


def test_constant_forecast_ties_cover_sound_cases_in_both_calibrations(tmp_path, monkeypatch):
    m = demo()
    source = m.generate_history(seed=20, rows=200)
    months = {name: [row.model_copy(update={
        "request_id": f"{name}-{i}", "at": m.JULY.replace(month=month),
        "paid": True, "amount_cents": 4000, "found_false_30d": i >= (190 if month == 7 else 95),
        "label_final_at": m.JULY.replace(month=month + 1),
    }) for i, row in enumerate(source if month == 7 else source[:100])]
        for name, month in (("july", 7), ("august", 8), ("september", 9))}
    scores = SimpleNamespace(predict=lambda rows: {"proposed": np.full(len(rows), .05)},
                             sha256="constant-forecast", validation={})
    monkeypatch.setattr(m, "rebuild_scores", lambda: scores)
    monkeypatch.setattr(m, "redrawn_windows", lambda: months.copy())
    monkeypatch.setattr(m.ScriptedVendorModel, "predict", lambda self, rows, negated=False:
                        np.full(len(rows), .95 if negated else .05).tolist())
    experiment = m.build_experiment(tmp_path)
    for month in experiment["months"].values():
        for name in ("marginal", "scripted"):
            summary = month[name]
            assert summary["sets"]["empty"] == 0
            assert summary["sets"]["sound"] == 100
            assert summary["covered"] == 95
            assert summary["rate"] >= 1 - m.ALPHA
        assert month["class-conditional"]["rate"] == 1
    # The 100-case calibration also keeps exact ties at the marginal threshold.
    p, y = np.full(100, .05), np.array([0] * 95 + [1] * 5)
    zeta = m.nonconformity_scores(np.column_stack((1 - p, p)))[np.arange(100), y]
    _, threshold = m.conformal_threshold(zeta, m.ALPHA)
    summary = m.coverage_summary(p, y, threshold)
    assert summary["sets"]["empty"] == 0
    assert summary["rate"] >= 1 - m.ALPHA


@pytest.mark.parametrize("probabilities,loss", [([0, 1], 0.), ([1., 0.], math.inf)])
def test_month_log_loss_handles_correct_and_impossible_endpoints(probabilities, loss):
    m = demo()
    rows = [row.model_copy(update={"found_false_30d": bool(i)})
            for i, row in enumerate(m.generate_history(seed=20, rows=2))]
    scores = SimpleNamespace(predict=lambda rows: {"proposed": np.array(probabilities)})
    with np.errstate(divide="ignore", invalid="raise"):
        report, _ = m.summarize_month(rows, scores, dict(marginal=1., sound=1., wrong=1., scripted=1.))
    assert report["scores"]["proposed"]["log"] == loss


@pytest.mark.parametrize("absent", ["sound", "wrong"])
def test_conformal_rendering_marks_absent_class_and_keeps_marginal_result(absent, capsys):
    m = demo()
    record = json.loads((ROOT / "demos/data/typed_decisions_record.json").read_text())
    y = np.full(100, 1 if absent == "sound" else 0)
    summary = m.coverage_summary(np.full(100, .05), y, .1)
    experiment = dict(record=record, months={"one-class": dict(
        requests=100, marginal=summary, **{"class-conditional": summary})})
    m.print_conformal(experiment)
    output = capsys.readouterr().out
    assert output.count(f"{absent} coverage: undefined, no observations") == 2
    assert f"coverage: {summary['covered']}/100 = {summary['rate']:.6f}" in output
    assert "one-class, marginal sets:" in output
    assert "one-class, class-conditional sets:" in output


@pytest.mark.parametrize("scores,alpha", [([], .1), ([.1], 0), ([.1], 1), ([float("nan")], .1)])
def test_bad_conformal_calibration_is_rejected(scores, alpha):
    with pytest.raises(ValueError):
        demo().conformal_threshold(scores, alpha)


def test_exchangeable_marginal_and_each_class_cover_in_two_thousand_seeded_repetitions():
    m = demo()
    rng = np.random.default_rng(20)
    marginal, conditional = [], {0: [], 1: []}
    for _ in range(2000):
        mu = rng.uniform(.02, .98, 201)
        y = rng.binomial(1, mu)
        p = 1 / (1 + np.exp(-2 * np.log(mu / (1 - mu))))
        scores = m.nonconformity_scores(np.column_stack((1 - p, p)))
        zeta = scores[np.arange(200), y[:-1]]
        _, threshold = m.conformal_threshold(zeta, .1)
        target = int(y[-1])
        marginal.append(target in m.prediction_set([1 - p[-1], p[-1]], threshold))
        _, threshold = m.conformal_threshold(zeta[y[:-1] == target], .1)
        conditional[target].append(target in m.prediction_set([1 - p[-1], p[-1]], threshold))
    for covered in (marginal, *conditional.values()):
        assert np.mean(covered) >= .9 - 3 * math.sqrt(.9 * .1 / len(covered))


@pytest.mark.parametrize("p,want", [(0, "pay"), (.03, "pay"), (.080, "review"),
                                    (.12, "review"), (.80, "review"), (.85, "third"), (1, "third")])
def test_bands_keep_cost_ties_in_review_and_never_authorize_denial(p, want):
    m = typed()
    assert m.band(p) == want
    assert m.ACTION_MAP[want] in ("pay_if_gate_allows", "refer")
    assert "deny" not in m.ACTION_MAP.values()
    assert m.ACTION_MAP["cannot_tell"] == "refer"


@pytest.mark.parametrize("payload", [
    {"probabilities": {"sound": .8, "unknown": .2}},
    {"probabilities": {"sound": 1.1, "wrong": -.1}},
    {"probabilities": {"sound": .8, "wrong": .3}},
    {"probabilities": {"sound": float("nan"), "wrong": .2}},
    {"probabilities": {"sound": .8}},
    {"probabilities": {"sound": .8, "wrong": .2}, "cannot_tell": True},
    {},
])
def test_typed_answer_refuses_undeclared_incomplete_or_invalid_probabilities(payload):
    m = typed()
    with pytest.raises(ValidationError):
        m.TypedAnswer(question=m.RISK_QUESTION, model_version="test", **payload)


def test_declared_refusal_and_amount_support_are_enforced():
    m = typed()
    question = m.TypedQuestion(name="binary", kind="true_false", answers=("no", "yes"))
    with pytest.raises(ValidationError, match="declared"):
        m.TypedAnswer(question=question, cannot_tell=True, model_version="test")
    assert m.risk_answer(.03, 4999, "test").cannot_tell is False
    assert m.risk_answer(.03, 5000, "test").cannot_tell is True
    assert m.risk_answer(.03, 7500, "test").probabilities is None
    with pytest.raises(ValidationError):
        m.TypedQuestion(name="bad", kind="choice", answers=("x", "x"))


def test_probability_sum_tolerance_accepts_roundoff_and_rejects_a_real_gap():
    m = typed()
    answer = m.TypedAnswer(question=m.RISK_QUESTION, model_version="test",
                           probabilities={"sound": .8, "wrong": .2 + 1e-10})
    assert answer.probabilities["wrong"] > .2
    with pytest.raises(ValidationError, match="sum"):
        m.TypedAnswer(question=m.RISK_QUESTION, model_version="test",
                      probabilities={"sound": .8, "wrong": .2 + 1e-8})


@pytest.mark.parametrize("covered,n,want", [(15, 15, "inconclusive"), (8, 10, "inconclusive"),
                                           (900, 1000, "coverage shown"),
                                           (850, 1000, "undercoverage shown")])
def test_predeclared_coverage_verdicts_use_exact_bounds_for_small_counts(covered, n, want):
    low, high = demo().coverage_interval(covered, n)
    assert demo().coverage_verdict(covered, n) == want
    if (covered, n) == (15, 15):
        assert low == pytest.approx(.78198064, abs=1e-7)
        assert high == 1
    if (covered, n) == (8, 10):
        assert low == pytest.approx(.44390454, abs=1e-7)


def test_amount_weighted_band_calibration_prices_the_actual_loss():
    row = demo().band_statistics([.1, .1], [.05, .15], [10, 40])
    assert row["prediction"] == pytest.approx(.1)
    assert row["observed"] == pytest.approx(.1)
    assert row["weighted_prediction"] == pytest.approx(.1)
    assert row["weighted_observed"] == pytest.approx(.13)
    assert row["predicted_loss"] == pytest.approx(2.5)
    assert row["realized_loss"] == pytest.approx(3.25)


def test_referral_shares_count_overlapping_band_and_conformal_referrals_once():
    shares = demo().referral_summary([.02, .09, .85], .01, np.array([1, 1]))
    assert shares["bands"]["referred"] == 2
    assert shares["bands"]["review"] == shares["bands"]["third"] == 1
    assert shares["bands"]["cannot_tell"] == 0
    assert shares["marginal"]["referred"] == shares["class-conditional"]["referred"] == 3
    assert shares["marginal"]["share"] == 1


def test_router_is_top_label_calibrated_but_misprices_escalation():
    report = demo().router_report()
    assert report["requests"] == 20
    assert report["top_probability"] == pytest.approx(.8)
    assert report["top_correct"] == 16
    assert report["escalation_prediction"] == pytest.approx(.05)
    assert report["escalation_count"] == 3
    assert report["stated_route"] == "standard"
    assert report["corrected_route"] == "priority"


def test_rebuild_reproduces_chapter_fives_validation_and_reads_no_seed_seven_holdout(monkeypatch):
    m = demo()
    original = m.generate_history

    class GuardedRow:
        def __init__(self, row):
            self.row = row

        def __getattr__(self, name):
            if self.row.at >= m.GO_LIVE and name in ("found_false_30d", "label_final_at"):
                raise AssertionError("seed-7 frozen outcome was read")
            return getattr(self.row, name)

    monkeypatch.setattr(m, "generate_history", lambda **kw: [GuardedRow(r) for r in original(**kw)])
    bundle = m.rebuild_scores()
    assert bundle.validation["requests"] == 988
    assert bundle.validation["wrong"] == 44
    assert bundle.validation["boosted_mean"] == pytest.approx(.0331, abs=.00005)
    assert bundle.validation["boosted_error"] == pytest.approx(.0132, abs=.00005)
    assert bundle.validation["weighted_mean"] == pytest.approx(.3646, abs=.00005)
    assert bundle.validation["weighted_error"] == pytest.approx(.3201, abs=.00005)


def test_both_months_are_frozen_before_thresholds_and_consulted_once_afterwards(tmp_path, monkeypatch):
    m = demo()
    original_freeze, original_threshold = m.frozen, m.conformal_threshold
    events = []

    class GuardedSet:
        def __init__(self, held):
            self.held = held
            self.manifest = held.manifest
            self.log = held.log

        def persist(self, directory):
            self.held.persist(directory)

        def consult(self, reader, purpose):
            assert sum(e == "threshold" for e in events) == 4
            events.append("consult:" + self.manifest["name"])
            return self.held.consult(reader, purpose)

    def freeze(rows, name):
        events.append("freeze:" + name)
        return GuardedSet(original_freeze(rows, name))

    def threshold(scores, alpha):
        assert sum(e.startswith("freeze:") for e in events) == 2
        assert not any(e.startswith("consult:") for e in events)
        events.append("threshold")
        return original_threshold(scores, alpha)

    monkeypatch.setattr(m, "frozen", freeze)
    monkeypatch.setattr(m, "conformal_threshold", threshold)
    report = m.build_experiment(tmp_path)
    assert len(events) == 8  # three score thresholds and the exercise fixture's threshold
    assert events[:2] == ["freeze:typed-decisions-seed20-august", "freeze:typed-decisions-seed20-september"]
    for month in ("august", "september"):
        journal = report["record"]["journals"][month]
        assert len(journal["access"]) == 1
        path = tmp_path / (journal["manifest"]["sha256"] + ".access.jsonl")
        assert len(path.read_text().splitlines()) == 1
        assert report["record"]["threshold_sha256"] in journal["access"][0]["purpose"]


def test_journal_resumes_without_erasing_the_previous_consultation(tmp_path):
    first, second = demo().build_experiment(tmp_path), demo().build_experiment(tmp_path)
    for month in ("august", "september"):
        a, b = (r["record"]["journals"][month] for r in (first, second))
        assert a["manifest"] == b["manifest"]
        assert [e["entry"] for e in b["access"]] == [1, 2]
        assert b["access"][0] == a["access"][0]
    assert first["months"] == second["months"]


def test_exercise_six_uses_only_july_and_catches_the_scripted_models_two_faults(monkeypatch):
    m = demo()
    original = m.generate_history

    class GuardedRow:
        def __init__(self, row):
            self.row = row

        def __getattr__(self, name):
            if self.row.at.month in (8, 9) and self.row.at.year == 2026 and name == "found_false_30d":
                raise AssertionError("exercise read a held-out month")
            return getattr(self.row, name)

    monkeypatch.setattr(m, "generate_history", lambda **kw: [GuardedRow(r) for r in original(**kw)])
    report = m.scripted_diagnostics()
    assert report["seed"] == 20
    assert report["decomposition"]["reliability"] > report["reliability_limit"]
    assert report["negation_violations"] == report["requests"]
    assert "deny" not in typed().ACTION_MAP.values()


def test_fixtures_figures_and_record_regenerate_from_the_one_consultation(experiment, captured):
    m = demo()
    for name, value in (("typed_decisions_acceptance_05.json", m.acceptance_fixture(experiment)),
                        ("typed_decisions_figures.json", experiment["figures"]),
                        ("typed_decisions_record.json", experiment["record"])):
        stored = ROOT / "demos/data" / name
        assert stored.read_bytes() == (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    assert json.loads(captured[PREFIX + "figures-01"]) == experiment["figures"]
    fixture = m.acceptance_fixture(experiment)
    assert fixture["coverage"]["wrong"]["covered"] / fixture["coverage"]["wrong"]["requests"] < .3
    assert "nine in ten wrong payments" in fixture["agent_conclusion"]


@pytest.mark.sealed
def test_seed_twenty_is_unused_by_any_other_history_draw():
    for path in (ROOT).rglob("*.py"):
        if path == DEMO or "tests" in path.parts or ".venv" in path.parts:
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Call):
                name = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
                if name == "generate_history":
                    assert not any(k.arg == "seed" and isinstance(k.value, ast.Constant)
                                   and k.value.value == 20 for k in node.keywords), path


def test_plotter_reads_generated_data_and_exports_pdf_png_and_grayscale(tmp_path):
    plotter = runpy.run_path(str(ROOT.parent / "figures/plots/fig_typed_decisions.py"))
    plotter["main"](out=tmp_path)
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(
        f"fig-typed-decisions-{name}{suffix}" for name in ("sets", "scores")
        for suffix in (".pdf", ".png", "-gray.png")
    )


def test_unfinished_starter_remains_an_explicit_reader_task():
    starter = runpy.run_path(str(ROOT / "exercises/typed-decisions/starter.py"))
    with pytest.raises(NotImplementedError):
        starter["accept"]([], typed().ScriptedVendorModel())
