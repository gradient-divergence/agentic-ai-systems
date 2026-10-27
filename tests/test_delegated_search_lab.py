"""Teaching experiments stay outside the selection record and frozen windows."""

import itertools
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from demos import delegated_model_demo as delegated
from demos import delegated_search_lab as lab

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "demos/delegated_search_lab.py"
OUTPUTS = ROOT / "book_outputs"
PREFIX = "fnd.automl-and-selection."


@pytest.fixture(scope="module")
def window():
    return delegated.prepare()[0]


@pytest.fixture(scope="module")
def searches(window):
    calls = []
    original = lab.fit_delivery

    def checked(name, builder, settings, received, commission, trials=None):
        assert received is window and commission.intact()
        calls.append((builder, settings))
        return original(name, builder, settings, received, commission, trials)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(lab, "fit_delivery", checked)
        records = lab.compare(window)
    return records, calls


def test_random_arm_is_the_chapters_twenty_trials_in_order(window, searches):
    records, _ = searches
    random = records["random"]
    # The old catalog permutation fails here, before comparing any scores.
    assert [t["settings"] for t in random] == delegated.configurations(20, 7)
    deliveries, trials, winner = delegated.random_search(
        window, delegated.commission_for(window), []
    )
    assert [t["loss"] for t in random] == [t["loss"] for t in trials]
    best = min(random, key=lambda t: t["loss"])
    assert best["settings"] == winner.settings
    assert best["loss"] == winner.reported_loss == min(d.reported_loss for d in deliveries)
    assert f"{best['loss']:.4f}" == "0.1666"
    truth = json.loads((OUTPUTS / f"_{PREFIX}truth-01.txt").read_text())
    expected = next(r for r in truth["models"] if r["name"] == "search winner")
    assert best["true_lead"] == expected["true_lead"]


def test_twenty_distinct_full_fits_per_arm_and_shared_start(searches):
    records, calls = searches
    assert set(records) == {"grid", "random", "model-based"}
    for method, trials in records.items():
        fitted = [settings for builder, settings in calls if builder == method]
        assert len(fitted) == len(trials) == 20
        assert fitted == [t["settings"] for t in trials]
        assert len({json.dumps(settings, sort_keys=True) for settings in fitted}) == 20
        assert all(delegated.allowed(s, method, delegated.SPACE) for s in fitted)
        assert [t["entry"] for t in trials] == list(range(1, 21))
        np.testing.assert_array_equal(
            [t["best_so_far"] for t in trials],
            np.minimum.accumulate([t["loss"] for t in trials]),
        )
    assert [t["settings"] for t in records["model-based"][:3]] == [
        t["settings"] for t in records["random"][:3]
    ]


def test_grid_is_four_logistic_points_plus_a_two_to_the_four_product(searches):
    expected = [{"family": "logistic", "C": C} for C in (0.01, 0.1, 1.0, 10.0)]
    for iterations, rate, leaves, minimum in itertools.product(
        (50, 200), (0.025, 0.1), (3, 15), (50, 200)
    ):
        expected.append(
            {
                "family": "boosted",
                "max_iter": iterations,
                "learning_rate": rate,
                "max_leaf_nodes": leaves,
                "min_samples_leaf": minimum,
                "l2_regularization": 1.0,
            }
        )
    assert [t["settings"] for t in searches[0]["grid"]] == expected


def test_expected_improvement_prefers_uncertain_b_over_lower_mean_a():
    values = lab.expected_improvement([0.1660, 0.1672], [0.0002, 0.0030], 0.1666)
    assert [f"{value:.5f}" for value in values] == ["0.00060", "0.00092"]
    assert values.argmax() == 1
    np.testing.assert_allclose(
        lab.expected_improvement([0.1, 0.2], [0, 0], 0.15), [0.05, 0], rtol=0, atol=1e-15
    )


def test_model_based_step_maximizes_ei_instead_of_minimizing_mean(monkeypatch, window):
    space = lab.catalog()
    start = delegated.configurations(3, 7)
    a, b = [i for i, s in enumerate(space) if s not in start][:2]

    class Surrogate:
        def __init__(self, **kwargs):
            pass

        def fit(self, inputs, losses):
            self.best = min(losses)

        def predict(self, inputs, return_std):
            mean, std = np.full(len(inputs), 0.9), np.zeros(len(inputs))
            mean[a], std[a] = self.best - 0.0006, 0.0002
            mean[b], std[b] = self.best + 0.0006, 0.0030
            return mean, std

    monkeypatch.setattr(lab, "GaussianProcessRegressor", Surrogate)
    fourth = lab.compare(window, budget=4)["model-based"][3]
    assert fourth["settings"] == space[b]


def test_stacking_folds_and_both_combiners_are_reported(monkeypatch, window):
    rows = window["training"]
    X, y, _ = lab.arrays(rows)
    row_indices = {tuple(row): i for i, row in enumerate(X)}
    assert len(row_indices) == len(rows)  # Inputs identify the actual training rows.
    validation_ids = {r.request_id for r in window["validation"]}
    original_arrays, make_model = lab.arrays, lab.make_model
    member_fits, combiner_fits = [], []

    def marked_arrays(received):
        inputs, labels, amount = original_arrays(received)
        # Binary training labels cannot be confused with a validation label.
        labels = np.array([
            2 if row.request_id in validation_ids else label
            for row, label in zip(received, labels)
        ])
        return inputs, labels, amount

    class RecordingMember:
        def __init__(self, settings, seed):
            self.model = make_model(settings, seed)
            self.predictions = []

        def fit(self, inputs, labels):
            self.training = np.array([row_indices[tuple(row)] for row in inputs])
            np.testing.assert_array_equal(labels, y[self.training])
            self.model.fit(inputs, labels)
            member_fits.append(self)
            return self

        def predict_proba(self, inputs):
            probabilities = self.model.predict_proba(inputs)
            self.predictions.append((inputs.copy(), probabilities[:, 1].copy()))
            return probabilities

    class RecordingCombiner(lab.LogisticRegression):
        def fit(self, inputs, labels):
            inputs, labels = np.array(inputs), np.array(labels)
            assert np.isin(labels, [0, 1]).all(), "combiner received validation labels"
            combiner_fits.append((inputs.copy(), labels.copy()))
            return super().fit(inputs, labels)

    monkeypatch.setattr(lab, "arrays", marked_arrays)
    monkeypatch.setattr(lab, "make_model", RecordingMember)
    monkeypatch.setattr(lab, "LogisticRegression", RecordingCombiner)
    result = lab.stacking(window)
    assert len(member_fits) == 10 and len(combiner_fits) == 2
    held_size = len(rows) // 5
    oof, held_labels = [], []
    for fold in range(4):
        start = len(rows) - (4 - fold) * held_size
        held = np.arange(start, start + held_size)
        expected_training = np.array([
            i for i in range(start) if rows[i].label_final_at <= rows[start].at
        ])
        columns = []
        for member in member_fits[2 * fold:2 * fold + 2]:
            np.testing.assert_array_equal(member.training, expected_training)
            assert not np.intersect1d(member.training, held).size
            assert all(rows[i].label_final_at <= rows[start].at for i in member.training)
            assert len(member.predictions) == 1
            inputs, probabilities = member.predictions[0]
            np.testing.assert_array_equal(inputs, X[held])
            columns.append(probabilities)
        oof.append(np.column_stack(columns))
        held_labels.extend(y[held])
    np.testing.assert_array_equal(combiner_fits[0][0], np.vstack(oof))
    np.testing.assert_array_equal(combiner_fits[0][1], held_labels)
    own = []
    for member in member_fits[-2:]:
        np.testing.assert_array_equal(member.training, np.arange(len(rows)))
        assert len(member.predictions) == 2
        inputs, probabilities = member.predictions[1]
        np.testing.assert_array_equal(inputs, X)
        own.append(probabilities)
    np.testing.assert_array_equal(combiner_fits[1][0], np.column_stack(own))
    np.testing.assert_array_equal(combiner_fits[1][1], y)
    assert len(result["folds"]) == 4
    assert result["meta_rows"] == sum(f["held"] for f in result["folds"])
    assert all(f["latest_label"] <= f["first_held"] for f in result["folds"])
    assert set(result["member_losses"]) == {"logistic", "boosted"}
    assert set(result["weights"]) == {"intercept", "logistic", "boosted"}
    assert set(result["in_sample_weights"]) == set(result["weights"])
    assert result["validation_loss"] > 0 and result["in_sample_loss"] > 0


def test_right_way_selector_reads_training_labels_only(monkeypatch, window):
    original = lab.arrays
    training_ids = {r.request_id for r in window["training"]}

    def guarded(rows):
        assert all(r.request_id in training_ids for r in rows), "validation label read"
        return original(rows)

    noise = np.random.default_rng(7).standard_normal((len(window["training"]), 200))
    monkeypatch.setattr(lab, "arrays", guarded)
    selected = lab.select_inside_training(window["training"], noise)
    assert len(set(selected["columns"])) == 5
    assert len(selected["losses"]) == 200
    assert selected["latest_label"] <= selected["first_held"]
    assert set(selected["columns"]) == set(np.argsort(selected["losses"], kind="stable")[:5])


def test_each_selection_refits_the_same_logistic_with_five_noise_inputs(window, capsys):
    result = lab.selection(window)
    assert result["noise_columns"] == 200
    assert set(result["models"]) == {"baseline", "validation-chosen", "training-chosen"}
    for name, model in result["models"].items():
        assert model["noise_selected"] == (0 if name == "baseline" else 5)
        assert model["validation_loss"] > 0 and model["expected_loss"] > 0
    assert result["fresh_requests"] == 2026
    lab.print_selection(result)
    printed = capsys.readouterr().out
    assert all(len(line) <= 79 for line in printed.splitlines())
    assert " ".join(printed.split()).endswith("Validation-chosen inputs lower validation loss in this run.")
    compact = " ".join(printed.split())
    assert "baseline" not in compact and "logistic reference" in compact
    assert "lead over logistic reference +0.002669" in compact
    assert "lead over logistic reference -0.002333" in compact
    assert "generator's expectation on a fresh June draw" in compact
    assert "validation-chosen=-0.000336; training-chosen=-0.000713" in compact


def test_new_regions_are_standalone_and_keep_listing_limits():
    from demos.regions import extract_snippets_from_text

    regions = extract_snippets_from_text(SOURCE.read_text(), SOURCE)
    assert {r.snippet_id.removeprefix(PREFIX) for r in regions} == {
        "lab-01", "lab-trace-01", "ei-01", "select-01", "stack-01"
    }
    for region in regions:
        assert region.output
        limit = {"lab-01": 15, "stack-01": 10}.get(region.snippet_id.removeprefix(PREFIX))
        if limit:
            assert len(region.content.splitlines()) <= limit
        assert all(len(line) <= 79 for line in region.content.splitlines())


@pytest.mark.parametrize("name", ["lab-01", "ei-01", "select-01", "stack-01"])
def test_new_printable_region_runs_on_its_own(name, tmp_path):
    from demos.regions import extract_snippets_from_text

    regions = {r.snippet_id: r for r in extract_snippets_from_text(SOURCE.read_text(), SOURCE)}
    region = regions[PREFIX + name]
    script = tmp_path / "listing.py"
    script.write_text(region.content)
    result = subprocess.run(
        [sys.executable, str(script)], cwd=tmp_path,
        env={**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
             "VECLIB_MAXIMUM_THREADS": "1", "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.encode() == (OUTPUTS / f"_{PREFIX}{name}.txt").read_bytes()
    assert list(tmp_path.iterdir()) == [script]


def test_fresh_generator_pool_is_reused_per_process(monkeypatch):
    original = delegated.generate_history
    calls = []

    def counted(*args, **kwargs):
        calls.append(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(delegated, "generate_history", counted)
    delegated.fresh_truth.cache_clear()
    first = delegated.fresh_truth()
    second = delegated.fresh_truth()
    assert first is second and len(first[0]) == 2026
    assert calls == [{"seed": 71, "rows": 72000}]


@pytest.mark.parametrize("function", ["compare", "stacking", "selection", "feature_construction"])
def test_lab_rejects_open_and_frozen_rows_before_fitting(function, window):
    from datetime import timedelta
    from types import SimpleNamespace

    # A stand-in is enough: no real frozen request is handed to a lab function.
    frozen = SimpleNamespace(at=delegated.GO_LIVE, label_final_at=delegated.GO_LIVE + timedelta(days=30))
    with pytest.raises(ValueError, match="development rows only"):
        getattr(lab, function)({**window, "validation": [frozen]})
    july = SimpleNamespace(at=delegated.GO_LIVE - timedelta(days=10), label_final_at=delegated.GO_LIVE + timedelta(days=20))
    with pytest.raises(ValueError, match="development rows only"):
        getattr(lab, function)({**window, "validation": [july]})


def test_lab_never_receives_frozen_rows_or_adds_an_august_consultation(monkeypatch, window, tmp_path):
    import copy

    _, august = delegated.prepare()
    log = json.loads((OUTPUTS / f"_{PREFIX}record-full-01.txt").read_text())["log"]
    august.persist(tmp_path, prior_log=log)
    before_log = copy.deepcopy(august.log)
    before_files = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    fresh_rows = delegated.fresh_truth()[0]
    permitted = {r.request_id for rows in window.values() for r in rows} | {
        r.request_id for r in fresh_rows
    }
    original_arrays, original_predict = delegated.arrays, delegated.Delivery.predict

    def checked_rows(rows):
        assert all(r.request_id in permitted for r in rows), "a frozen row reached the lab"

    def arrays(rows):
        checked_rows(rows)
        return original_arrays(rows)

    def predict(self, rows, mapped=True):
        checked_rows(rows)
        return original_predict(self, rows, mapped)

    def forbidden(*args, **kwargs):
        pytest.fail("a teaching run touched a consultation or the selection record")

    monkeypatch.setattr(lab, "prepare", lambda: (window, august))
    monkeypatch.setattr(lab, "arrays", arrays)
    monkeypatch.setattr(delegated, "arrays", arrays)
    monkeypatch.setattr(delegated.Delivery, "predict", predict)
    monkeypatch.setattr(type(august), "consult", forbidden)
    monkeypatch.setattr(delegated, "evaluate", forbidden)
    monkeypatch.setattr(delegated, "SelectionRecord", forbidden)
    lab.search_run.cache_clear()
    run = lab.search_run()
    assert len(run["trials"]["random"]) == 20
    assert lab.selection(window)["models"]["baseline"]["noise_selected"] == 0
    assert lab.stacking(window)["meta_rows"] > 0
    assert lab.feature_construction(window)["own_label_left_out"]
    assert august.log == before_log and len(august.log) == 3
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == before_files
    lab.search_run.cache_clear()


def test_trace_plot_refuses_changed_quoted_numbers(monkeypatch):
    import copy
    import runpy

    monkeypatch.syspath_prepend(str(ROOT.parent / "figures/plots"))
    plot = runpy.run_path(str(ROOT.parent / "figures/plots/fig_delegated_search.py"))
    data = json.loads((OUTPUTS / f"_{PREFIX}lab-trace-01.txt").read_text())
    plot["validate_trace"](data)
    changed = copy.deepcopy(data)
    best = min(changed["trials"]["random"], key=lambda t: t["loss"])
    best["true_lead"] = 0.01
    with pytest.raises(AssertionError):
        plot["validate_trace"](changed)
