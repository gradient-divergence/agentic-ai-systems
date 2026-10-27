"""Behavior, arithmetic and independent capture checks for chapter 9."""

import contextlib
import importlib
import io
import json
import runpy
import sys
from pathlib import Path

import numpy as np
import pytest

from demos.regions import capture_outputs, extract_snippets_from_text

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "demos/backprop_demo.py"
PREFIX = "fnd.neural-networks-backprop."
PRINTED = ("forward-01", "backward-01", "check-01", "train-01")
OUTPUTS = (*PRINTED, "scalar-01", "kink-01", "epsilon-01", "curves-01", "delegated-01", "depth-01", "keys-01")


def demo():
    return importlib.import_module("demos.backprop_demo")


def regions():
    return {s.snippet_id: s for s in extract_snippets_from_text(DEMO.read_text(), DEMO)}


def test_f_printed_listings_run_alone_in_chapter_order():
    snippets = regions()
    namespace = {"__name__": "__main__"}
    for name in PRINTED:
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            exec(compile(snippets[PREFIX + name].content, str(DEMO), "exec"), namespace)
        path = ROOT / "book_outputs" / f"_{PREFIX}{name}.txt"
        assert stream.getvalue().encode() == path.read_bytes()


def test_a_worked_pass_and_simultaneous_steps():
    m = demo()
    x, y, theta = m.count_words("My parcel never arrived.", m.WORDS), np.array([2]), m.hand_weights()
    f, g = m.forward(x, theta), m.backward(x, y, theta)
    np.testing.assert_array_equal(x, [1, 0, 0, 1])
    np.testing.assert_array_equal(f["u"], [[1, -1]])
    np.testing.assert_array_equal(f["h"], [[1, 0]])
    np.testing.assert_array_equal(f["z"], [[2, 1, 0]])
    np.testing.assert_allclose(f["p"], [[0.6652409558, 0.2447284711, 0.0900305732]])
    assert m.loss(x, y, theta) == pytest.approx(2.4076059644)
    np.testing.assert_allclose(g["h"], [[1.5752103826, -0.4205124847]])
    np.testing.assert_allclose(g["W1"][0], [1.5752103826, 0, 0, 1.5752103826])
    assert sum(np.count_nonzero(g[k] == 0) for k in theta) == 10
    assert sum(np.sum(g[k] ** 2) for k in theta) == pytest.approx(10.1048270715)
    moved = {k: theta[k] - 0.1 * g[k] for k in theta}
    assert m.loss(x, y, moved) == pytest.approx(1.542893359, abs=1e-9)
    moved = {k: theta[k] - 0.3 * g[k] for k in theta}
    assert m.forward(x, moved)["u"][0, 0] == pytest.approx(-0.4176893443)
    assert m.loss(x, y, moved) == pytest.approx(0.846136592, abs=1e-9)


def test_b_every_parameter_and_batch_mean_agree_with_finite_differences():
    m = demo()
    rng = np.random.default_rng(19)
    theta = {k: v + rng.normal(scale=0.07, size=v.shape) for k, v in m.hand_weights().items()}
    X = np.array([[1, 0, 0, 1], [0, 1, 0, 0], [1, 0, 1, 0]], dtype=float)
    y = np.array([2, 0, 1])
    assert np.min(np.abs(m.forward(X, theta)["u"])) > 0.01
    analytical = m.backward(X, y, theta)
    numerical, switched = m.central_difference(X, y, theta)
    assert switched == []
    for k in theta:
        assert m.relative_gap(analytical[k], numerical[k]) < 1e-7
        singles = [m.backward(x, [target], theta)[k] for x, target in zip(X, y)]
        np.testing.assert_allclose(analytical[k], np.mean(singles, axis=0), atol=1e-15)
    original = m.hand_weights()
    np.testing.assert_array_equal(original["W1"], [[1, 0, 0, 1], [1, 1, 0, -2]])


def test_c_relu_corner_is_a_false_alarm():
    m = demo()
    X = m.count_words("I changed my mind.", m.WORDS)
    theta = m.hand_weights()
    assert m.forward(X, theta)["u"].tolist() == [[-1, 0]]
    assert m.backward(X, [1], theta)["b1"][1] == 0
    numerical, switched = m.central_difference(X, [1], theta)
    assert numerical["b1"][1] == pytest.approx(-0.5, abs=1e-5)
    assert switched == [("W1", (1, 2)), ("b1", (1,))]


def test_c_a_nudge_can_cross_a_corner_from_a_sum_that_is_not_zero():
    m = demo()
    X = m.count_words("I changed my mind.", m.WORDS)
    theta = m.hand_weights()
    theta["b1"][1] = 4e-6
    assert m.forward(X, theta)["u"][0, 1] == pytest.approx(4e-6) and m.forward(X, theta)["u"][0, 1] != 0
    backward = m.backward(X, [1], theta)["b1"][1]
    assert backward == pytest.approx(-1, abs=1e-5)
    crossing, switched = m.central_difference(X, [1], theta, 1e-5)
    assert crossing["b1"][1] == pytest.approx(-0.7, abs=1e-5)
    assert switched == [("W1", (1, 2)), ("b1", (1,))]
    smaller, switched = m.central_difference(X, [1], theta, 1e-6)
    assert smaller["b1"][1] == pytest.approx(backward, abs=1e-8) and switched == []


def test_c_the_check_runs_two_nudged_forward_passes_for_each_parameter_and_one_base_pass(monkeypatch):
    m = demo()
    X = m.count_words("My parcel never arrived.", m.WORDS)
    theta = m.hand_weights()
    passes, forward = [], m.forward
    monkeypatch.setattr(m, "forward", lambda X, theta: passes.append(1) or forward(X, theta))
    m.central_difference(X, [2], theta)
    parameters = sum(array.size for array in theta.values())
    assert (parameters, len(passes)) == (19, 2 * 19 + 1)
    printed = (ROOT / "book_outputs" / f"_{PREFIX}check-01.txt").read_text()
    assert "all 19 parameters, 38 nudged forward passes and 1 base pass; float64, epsilon 0.00001" in " ".join(printed.split())


def test_c_check_arrays_sets_aside_the_entries_whose_nudge_switched_a_unit():
    m = demo()
    _, X, y = m.dataset()
    gaps, switched = m.check_arrays(X, y, m.hand_weights())
    assert {name for name, _ in switched} == {"W1", "b1"} and len(switched) == 10
    assert np.isnan(gaps["W1"]) and np.isnan(gaps["b1"])
    assert gaps["W2"] < 1e-7 and gaps["b2"] < 1e-7
    X = m.count_words("I changed my mind.", m.WORDS)
    gaps, switched = m.check_arrays(X, [1], m.hand_weights())
    assert len(switched) == 2 and max(gaps.values()) < 1e-7


def test_d_shapes_and_falling_loss_do_not_validate_the_missing_gate():
    m = demo()
    _, X, y = m.dataset()
    run = m.train(X, y, gate=False)
    assert run["losses"][-1] < run["losses"][0]
    theta = run["parameters"][10]
    assert np.min(np.abs(m.forward(X, theta)["u"])) > 1e-5
    good, bad = m.backward(X, y, theta), m.backward(X, y, theta, gate=False)
    numerical, switched = m.central_difference(X, y, theta)
    assert switched == []
    for k in theta:
        assert good[k].shape == bad[k].shape == theta[k].shape
        assert m.relative_gap(good[k], numerical[k]) < 1e-7
        gap = m.relative_gap(bad[k], numerical[k])
        assert gap > 0.1 if k in ("W1", "b1") else gap < 1e-7


@pytest.fixture(scope="module")
def depth():
    return demo().depth_scales()


def test_e_initialization_scales_and_sampled_gradient_checks(depth):
    by_scale = {row["scale"]: row for row in depth}
    assert by_scale[0.5]["gradient_ratio"] < 0.1
    assert 0.5 < by_scale[1.0]["gradient_ratio"] < 2
    assert by_scale[2.0]["gradient_ratio"] > 10
    for row in depth:
        assert row["sampled_weights"] == 75
        assert row["check_gap"] < 1e-4
        assert len(row["hidden_rms"]) == len(row["gradient_rms"]) == 11


def test_scalar_join_and_exercise_keys():
    m = demo()
    value = m.scalar_pass()
    assert value["hidden_input"] == 1 and value["prediction"] == 2 and value["loss"] == 0.5
    assert value["gradient"] == (4, 2, 1, 1)
    assert m.scalar_pass(target=3)["gradient"] == (-4, -2, -1, -1)
    # Each loss is quadratic in one parameter alone, so the central difference is exact.
    assert m.scalar_check() == pytest.approx([4, 2, 1, 1], abs=1e-12)
    assert m.scalar_step(0.05)["loss"] == pytest.approx(0.0028125)
    assert m.scalar_step(0.1)["loss"] == pytest.approx(0.605)
    assert m.scalar_step(0.1)["hidden_input"] == pytest.approx(0, abs=1e-15)
    assert m.exercise_six()["central"] == pytest.approx(-1)
    assert m.exercise_six()["plus"] == pytest.approx(0.1152)
    assert m.exercise_six()["minus"] == pytest.approx(0.1352)
    key8 = m.exercise_eight()
    assert key8["minimum_hidden"] > 1e-5
    assert max(key8["gaps"]) < 1e-6
    assert key8["delivered"] > 0 and key8["relative_drop_error"] < 0.05
    assert key8["corner_backward"] == 0 and abs(key8["corner_central"]) > 0.1
    assert key8["switched"] == 0 and key8["corner_switched"] is True
    gaps, switched = m.clinic_check()
    assert switched == 0 and gaps["b1"] == pytest.approx(99 / 101) and gaps["W1"] < 1e-7


def test_g_clinic_fixture_omits_biases_with_one_planted_error():
    fixture = ROOT / "exercises/neural-networks-backprop/check-report-07.json"
    source = fixture.with_name("check_report_07.py")
    report = json.loads(fixture.read_text())
    ns = runpy.run_path(str(source))
    assert ns["make_report"]() == report
    assert set(report["checked_arrays"]) == {"W1", "W2"}
    assert report["largest_relative_gap"] < 1e-7
    assert report["entries_whose_nudge_switched_a_unit"] == 0
    # The learner's copy must not say where the error is, in words or by its structure.
    text = source.read_text().lower()
    for giveaway in ("planted", "faulty", "bug", "wrong", "incomplete", "misses", "again", "correct"):
        assert giveaway not in text
    assert "backward" not in "".join(line for line in text.splitlines() if "import" in line)
    m = demo()
    _, X, y = m.dataset("clinic", n=100)
    theta = m.smooth_weights()
    bad = ns["backward"](X, y, theta)
    good = m.backward(X, y, theta)
    numerical, switched = m.central_difference(X, y, theta)
    assert switched == []
    for k in theta:
        expected = good[k] / len(y) if k in ("b1", "b2") else good[k]
        np.testing.assert_array_equal(bad[k], expected)
        assert m.relative_gap(bad[k], numerical[k]) > 0.9 if k in ("b1", "b2") else m.relative_gap(bad[k], numerical[k]) < 1e-7


def test_seeded_tables_and_deterministic_training():
    m = demo()
    data = importlib.import_module("demos.backprop_data")
    for name, table in data.PATTERNS.items():
        assert len(table) == 3 and all(len(patterns) == 5 for patterns in table.values())
        texts, X, y = m.dataset(name)
        repeated = m.dataset(name)
        assert texts == repeated[0]
        np.testing.assert_array_equal(X, repeated[1])
        np.testing.assert_array_equal(y, repeated[2])
        assert len(texts) == 300 and len(set(y)) == 3
        assert texts != m.dataset(name, seed=8)[0]
    _, X, y = m.dataset()
    np.testing.assert_array_equal(m.train(X, y)["losses"], m.train(X, y)["losses"])



def test_checked_gradients_can_stall_or_raise_loss_and_counts_limit_accuracy():
    m = demo()
    _, X, y = m.dataset()
    good, bad = m.train(X, y), m.train(X, y, gate=False)
    assert not np.all(np.diff(good["losses"]) < 0)
    assert np.all(np.diff(bad["losses"]) < 0)
    ceiling = sum(max(np.bincount(y[np.all(X == row, axis=1)], minlength=3)) for row in np.unique(X, axis=0))
    assert max(good["correct"]) == ceiling == 287
    assert good["correct"][-1] > bad["correct"][-1]
    stalled = m.train(X, y, eta=2)
    assert stalled["active"][1] == stalled["active"][-1] == [0, 0]
    shares = np.bincount(y, minlength=3) / len(y)
    assert stalled["losses"][-1] == pytest.approx(-np.sum(shares * np.log(shares)))
    np.testing.assert_allclose(m.forward(X, stalled["theta"])["p"], np.tile(shares, (len(y), 1)))
    raised = m.train(X, y, eta=16)
    assert raised["losses"][1] > raised["losses"][0] and raised["losses"][-1] > raised["losses"][0]
    for run in (good, stalled, raised):
        gaps, switched = m.check_arrays(X, y, run["parameters"][10])
        assert switched == [] and max(gaps.values()) < 1e-7


def test_units_that_start_alike_stay_alike_and_outgoing_weights_part_them():
    m = demo()
    x = m.count_words("My parcel never arrived.", m.WORDS)
    theta = m.hand_weights()
    theta["W1"][1], theta["b1"][1] = theta["W1"][0], theta["b1"][0]
    f, g = m.forward(x, theta), m.backward(x, [2], theta)
    # Equal incoming weights give equal hidden values, and still unequal gradients.
    assert f["h"][0, 0] == f["h"][0, 1]
    assert not np.allclose(g["W1"][0], g["W1"][1]) and not np.isclose(g["b1"][0], g["b1"][1])
    theta["W2"][:, 1] = theta["W2"][:, 0]
    g = m.backward(x, [2], theta)
    np.testing.assert_allclose(g["W1"][0], g["W1"][1], rtol=0, atol=1e-15)
    np.testing.assert_allclose(g["W2"][:, 0], g["W2"][:, 1], rtol=0, atol=1e-15)
    key = m.symmetry_check()
    assert key["hidden"] == [1, 1] and key["still_equal"] is True and key["moved"] > 1
    assert key["outgoing_differ"][0] != pytest.approx(key["outgoing_differ"][1], abs=0.1)
    assert key["outgoing_same"][0] == pytest.approx(key["outgoing_same"][1], abs=1e-15)
    # Weights that differ without any chance in them also break the tie: the hand-set start is one.
    _, X, y = m.dataset()
    assert m.train(X, y)["correct"][-1] == 287

def test_no_external_or_frozen_data_is_read_and_import_has_no_effect(monkeypatch):
    m = demo()
    data = importlib.import_module("demos.backprop_data")
    def forbidden(*args, **kwargs):
        raise AssertionError("a model operation read a file")
    monkeypatch.setattr("builtins.open", forbidden)
    monkeypatch.setattr(Path, "open", forbidden)
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        importlib.reload(data)
        importlib.reload(m)
        _, X, y = m.dataset()
        m.train(X, y, steps=10)
        m.delegated_check()
        m.depth_scales(width=8, inputs=8, seeds=(7,), samples=2)
        m.symmetry_check(steps=5)
    assert stream.getvalue() == ""


def test_every_output_regenerates_byte_for_byte_twice():
    snippets = regions()
    assert {k[len(PREFIX):] for k, v in snippets.items() if v.output} == set(OUTPUTS)
    first = capture_outputs(snippets, sys.executable)
    assert first == capture_outputs(snippets, sys.executable)
    for name, output in first.items():
        assert output.encode() == (ROOT / "book_outputs" / f"_{name}.txt").read_bytes()


def test_figure_data_and_keys_regenerate():
    m = demo()
    path = ROOT / "demos/data/backprop_figures.json"
    assert json.loads(path.read_text()) == m.figure_data()
    # Output-only keys deliberately use the tested module functions.
    out = capture_outputs(regions(), sys.executable)[PREFIX + "keys-01"]
    assert "37 parameters" in out and "reject" in out and "unit 2" in out
    assert "0.0028125" in out and "0.6050000" in out
