"""Numerical contracts, runnable listings and captured output for training control."""

import ast
import contextlib
import importlib.util
import io
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from demos import regions as EXTRACTOR

ROOT = Path(__file__).resolve().parents[1]

capture_outputs = EXTRACTOR.capture_outputs
extract_snippets_from_text = EXTRACTOR.extract_snippets_from_text
DEMO = ROOT / "demos/training_control_demo.py"
PREFIX = "fnd.optimization-regularization."
NAMES = (
    "overshoot-01",
    "descent-01",
    "valley-01",
    "adam-01",
    "rotate-01",
    "noise-01",
    "decay-01",
    "stop-01",
    "report-01",
    "keys-01",
)
PRINTED = ("descent-01", "valley-01", "adam-01", "decay-01")


def demo():
    from demos import training_control_demo

    return training_control_demo


def regions():
    return extract_snippets_from_text(DEMO.read_text(), source_path=DEMO)


def test_printed_listings_run_in_chapter_order_with_no_hidden_setup():
    namespace = {"__name__": "__main__"}
    snippets = {s.snippet_id: s for s in regions()}
    for name in PRINTED:
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            exec(compile(snippets[PREFIX + name].content, name, "exec"), namespace)
        stored = ROOT / "book_outputs" / f"_{PREFIX}{name}.txt"
        assert stdout.getvalue().encode() == stored.read_bytes()


@pytest.fixture(scope="module")
def captured():
    snippets = {s.snippet_id: s for s in regions()}
    return capture_outputs(snippets, sys.executable)


@pytest.mark.parametrize("name", NAMES)
def test_every_stored_output_regenerates_byte_for_byte(name, captured):
    stored = ROOT / "book_outputs" / f"_{PREFIX}{name}.txt"
    assert captured[PREFIX + name].encode() == stored.read_bytes()


def test_output_regions_are_in_outline_order():
    assert [s.snippet_id for s in regions()] == [PREFIX + n for n in NAMES]
    assert all(s.output for s in regions())


def test_capture_is_deterministic_in_a_fresh_process(captured):
    assert captured == capture_outputs({s.snippet_id: s for s in regions()}, sys.executable)


def test_import_has_no_output_or_file_effects(tmp_path):
    result = subprocess.run(
        [sys.executable, "-c", "import demos.training_control_demo"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == ""
    assert list(tmp_path.iterdir()) == []


def test_one_weight_follows_the_exact_error_recurrence():
    m = demo()
    X, y = np.array([[2.0]]), np.array([1.0])
    for eta in (0.1, 0.25, 0.45, 0.5, 0.6):
        path = m.descend(X, y, eta, 5, w0=[0.25])[:, 0]
        expected = 0.5 - 0.25 * (1 - 4 * eta) ** np.arange(6)
        np.testing.assert_allclose(path, expected, atol=1e-14)
    assert m.step_limit([4]) == 0.5


def test_valley_limit_and_training_only_standardization():
    m = demo()
    X, y = m.valley()
    np.testing.assert_allclose(m.curvatures(X), [0.04, 4])
    assert m.steps_to_loss(X, y, m.descend(X, y, 0.49, 1000)) == 332
    assert m.loss(X, y, m.descend(X, y, 0.51, 1000)[-1]) > 1e15
    Z = X / np.sqrt(np.mean(X**2, axis=0))
    np.testing.assert_allclose(m.curvatures(Z), [1, 1])
    np.testing.assert_allclose(m.descend(Z, y, 1, 1)[-1], [1, 1])


def test_valley_excerpt_keeps_computations_without_report_formatting():
    snippet = next(s for s in regions() if s.snippet_id == PREFIX + "valley-01")
    tree = ast.parse(snippet.content)
    assert isinstance(tree.body[0], ast.FunctionDef)
    assert tree.body[0].name == "valley"
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    names = {node.func.id for node in calls if isinstance(node.func, ast.Name)}
    names |= {node.func.attr for node in calls if isinstance(node.func, ast.Attribute)}
    assert "print" not in names
    assert {"valley", "curvatures", "step_limit", "steps_to_loss", "report_valley"} <= names
    m = demo()
    X, y = m.valley()
    assert m.steps_to_loss(X, y, m.descend(X, y, 0.25, 2000)) == 653
    assert m.steps_to_loss(X, y, m.descend(X, y, 0.5, 2000)) is None


def test_adam_first_step_and_bias_correction():
    m = demo()
    X, y = m.valley()
    corrected = m.adam(X, y, 0.1, 1)[-1]
    uncorrected = m.adam(X, y, 0.1, 1, correct=False)[-1]
    np.testing.assert_allclose(corrected, [0.1, 0.1], atol=1e-8)
    np.testing.assert_allclose(uncorrected / corrected, math.sqrt(10), rtol=2e-6)
    assert 1 - 0.9**10 == pytest.approx(0.6513215599)


def test_rotation_preserves_gradient_descent_and_momentum_but_not_adam():
    m = demo()
    X, y = m.valley()
    rotation = m.rotation()
    for run in (lambda a: m.descend(a, y, 0.25, 100), lambda a: m.adam(a, y, 1, 100, beta2=None)):
        np.testing.assert_allclose(run(X @ rotation) @ rotation.T, run(X), atol=1e-12)
    assert m.steps_to_loss(X @ rotation, y, m.adam(X @ rotation, y, 0.1, 1000)) > m.steps_to_loss(
        X, y, m.adam(X, y, 0.1, 1000)
    )


@pytest.mark.parametrize("eta,batch", [(0.1, 1), (0.01, 1), (0.1, 16)])
def test_seeded_noise_matches_the_stationary_formula_within_five_percent(eta, batch):
    path = demo().sgd(4, eta, batch, 50000, seed=7)
    observed = np.std(path[2000:])
    predicted = math.sqrt(eta / (batch * 4 * (2 - eta * 4)))
    assert observed == pytest.approx(predicted, rel=0.05)
    np.testing.assert_array_equal(path, demo().sgd(4, eta, batch, 50000, seed=7))


def test_schedule_and_clipping():
    m = demo()
    assert [m.warmup_cosine(t, 0.4, 10, 100) for t in (0, 5, 10, 55, 100)] == pytest.approx(
        [0, 0.2, 0.4, 0.2, 0]
    )
    np.testing.assert_allclose(m.clip([3, 4], 2), [1.2, 1.6])
    np.testing.assert_array_equal(m.clip([0, 0], 2), [0, 0])


def test_penalty_and_early_stop_shrink_differently():
    m = demo()
    X, y = m.valley()
    penalized = m.penalized_minimizer(X, y, 0.04)
    stopped = m.descend(X, y, 0.25, 69)[-1]
    np.testing.assert_allclose(penalized, [0.495049504950495, 2.5], atol=1e-12)
    np.testing.assert_allclose(stopped, [0.5, 2.5], atol=0.01)
    assert penalized[0] != stopped[0]  # only the flat direction approximately agrees


def test_penalty_equals_decay_for_plain_descent_to_twelve_decimals():
    m = demo()
    rng = np.random.default_rng(7)
    for _ in range(20):
        w, g = rng.normal(size=(2, 5))
        np.testing.assert_allclose(
            m.decay_step(w, g, 0.1, 0.2), w - 0.1 * (g + 0.2 * w), rtol=0, atol=1e-12
        )
    assert m.decay_step(2, 3, 0.1, 0.2, denominator=2) == pytest.approx(1.81)
    assert 2 - 0.1 * (3 + 0.2 * 2) / 2 == pytest.approx(1.83)
    np.testing.assert_allclose(
        m.adam(np.array([[1.0]]), np.array([-1.0]), 0.1, 1, w0=[2], decay=0.2)[-1],
        [1.86],
        atol=1e-8,
    )


def test_training_fits_do_not_read_validation_or_frozen_data():
    m = demo()
    problem = m.noisy_problem(7)
    altered = {
        **problem,
        "validation_X": problem["validation_X"] * 1000,
        "validation_y": problem["validation_y"] + 1000,
    }
    first, second = m.regularization_experiment(problem), m.regularization_experiment(altered)
    for key in ("mean", "scale", "path", "ridge_weights"):
        np.testing.assert_array_equal(first[key], second[key])
    assert not np.array_equal(first["validation_loss"], second["validation_loss"])
    # No test/golden window exists; validation is intentionally spent on selection.
    assert set(problem) == {"training_X", "training_y", "validation_X", "validation_y"}
    source = DEMO.read_text()
    assert "refund_agent" not in source and "read_text(" not in source


def test_seed_seven_reproduces_data_and_fits():
    m = demo()
    a, b = m.noisy_problem(7), m.noisy_problem(7)
    for key in a:
        np.testing.assert_array_equal(a[key], b[key])
    assert not np.array_equal(a["training_X"], m.noisy_problem(8)["training_X"])


def test_exercise_record_regenerates_and_exposes_the_single_bad_conclusion():
    m = demo()
    path = ROOT / "exercises/optimization-regularization/run-05.json"
    assert path.read_bytes() == (json.dumps(m.exercise_record(), indent=2) + "\n").encode()
    record = json.loads(path.read_text())
    rows, eta = record["steps"], record["learning_rate"]
    assert len(rows) == 61
    g0, g1 = np.array(rows[0]["gradients"]), np.array(rows[1]["gradients"])
    np.testing.assert_allclose((1 - g1 / g0) / eta, [9, 0.25], atol=1e-12)
    assert abs(g1[0] / g0[0]) == pytest.approx(0.9995)
    assert all(a["gradients"][0] * b["gradients"][0] < 0 for a, b in zip(rows, rows[1:]))
    assert abs(rows[-1]["weights"][0] - 1) > 0.97
    assert "converged" in record["agent_conclusion"]
    for row in rows:
        w = np.array(row["weights"])
        np.testing.assert_allclose(row["gradients"], np.array([9, 0.25]) * (w - [1, 4]))
        assert row["loss"] == pytest.approx(0.5 * np.sum(np.array([9, 0.25]) * (w - [1, 4]) ** 2))


def test_agent_report_stability_is_not_convergence_and_rerun_uses_training_scale():
    m = demo()
    run = m.energy_experiment(7)
    assert np.all(np.diff(run["raw_loss"]) <= 1e-10)
    assert run["covered"][0] < 0.011 and run["covered"][2] < 0.00001
    assert run["standardized_training_loss"] < run["raw_loss"][-1] / 1000
    np.testing.assert_allclose(run["scale"] ** 2, [144, 1440000, 0.04])


def test_figure_data_is_generated_byte_for_byte():
    path = ROOT / "demos/data/training_control_figures.json"
    assert (
        path.read_bytes()
        == (json.dumps(demo().figure_data(), indent=2, allow_nan=False) + "\n").encode()
    )


def test_plotting_code_writes_all_colour_and_grayscale_artifacts(tmp_path):
    pytest.importorskip(
        "matplotlib", reason="plotting needs Matplotlib; absent in the supplied companion Python"
    )
    plot = ROOT.parent / "figures/plots/fig_optimization_regularization.py"
    spec = importlib.util.spec_from_file_location("training_control_plot", plot)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.OUT = tmp_path
    module.main()
    assert len(list(tmp_path.glob("*.pdf"))) == 8
    assert len(list(tmp_path.glob("*.png"))) == 8


def test_exercise_six_acceptance_against_the_demo_solution():
    tests = ROOT / "exercises/optimization-regularization/test_acceptance.py"
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", str(tests)],
        env={**os.environ, "TRAINING_CONTROL_SOLUTION": str(DEMO)},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    "eta,beta2,expected", [(1.0, None, 130), (5.0, None, 105), (0.1, 0.999, 135), (0.3, 0.999, 112)]
)
def test_first_threshold_crossings_match_the_optimizer_records(eta, beta2, expected):
    m = demo()
    X, y = m.valley()
    assert m.steps_to_loss(X, y, m.adam(X, y, eta, 2000, beta2=beta2)) == expected


def test_exercise_keys_follow_loss_factors_and_distance_factors():
    product = 1 + math.sqrt(2)
    assert (1 - product) ** 2 == pytest.approx(2)
    assert (1 - (1 - math.sqrt(0.99))) ** 2 == pytest.approx(0.99)
    for eta, count in ((1 / 9, 82), (2 / 9.25, 42)):
        factor = 1 - eta * 0.25
        assert factor**count <= 0.1 < factor ** (count - 1)


def test_early_stopping_scores_all_checkpoints_but_does_not_refit_on_validation():
    result = demo().regularization_experiment()
    best = np.argmin(result["validation_loss"])
    assert 0 < best < len(result["path"]) - 1
    assert result["training_loss"][-1] < result["training_loss"][best]
    assert result["validation_loss"][best] < result["validation_loss"][-1]
    chosen = np.argmin(result["ridge_validation"])
    assert 0 < chosen < len(result["lambdas"]) - 1
    assert result["ridge_validation"][chosen] < result["ridge_validation"][0]
