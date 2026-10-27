"""Exercise 8 checks every array, a ReLU corner and the predicted loss decrease."""

import os
import runpy
from pathlib import Path

import numpy as np
import pytest

from demos import backprop_demo as demo

SOLUTION = runpy.run_path(
    str(Path(os.environ.get("BACKPROP_SOLUTION", Path(__file__).with_name("starter.py"))))
)


@pytest.fixture(scope="module")
def case():
    _, X, y = demo.dataset()
    original = demo.smooth_weights()
    theta = {
        "W1": original["W1"],
        "b1": original["b1"],
        "W2": np.array([[0.8, 0.2], [-0.1, 1.1]], dtype=np.float64),
        "b2": np.array([0.17, 0.13]),
        "W3": original["W2"],
        "b3": original["b2"],
    }
    return X, y, theta


@pytest.mark.parametrize("name", ("W1", "b1", "W2", "b2", "W3", "b3"))
def test_every_array_has_no_switched_nudge_and_gap_below_one_millionth(case, name):
    X, y, theta = case
    base = SOLUTION["forward"](X, y, theta)
    analytical = SOLUTION["backward"](X, y, theta)[name]
    numerical = np.zeros_like(theta[name])
    for index in np.ndindex(theta[name].shape):
        losses = []
        for sign in (1, -1):
            changed = {k: v.copy() for k, v in theta.items()}
            changed[name][index] += sign * 1e-5
            value = SOLUTION["forward"](X, y, changed)
            assert all(np.array_equal(base[k] > 0, value[k] > 0) for k in ("u1", "u2"))
            losses.append(value["loss"])
        numerical[index] = (losses[0] - losses[1]) / 2e-5
    gap = np.linalg.norm(analytical - numerical) / max(
        1e-12, np.linalg.norm(analytical) + np.linalg.norm(numerical)
    )
    assert gap < 1e-6, f"{name}: gap {gap}"


def test_forward_matches_independent_two_layer_softmax(case):
    X, y, t = case
    h1 = np.maximum(X @ t["W1"].T + t["b1"], 0)
    h2 = np.maximum(h1 @ t["W2"].T + t["b2"], 0)
    z = h2 @ t["W3"].T + t["b3"]
    p = np.exp(z - z.max(axis=1, keepdims=True))
    p /= p.sum(axis=1, keepdims=True)
    value = SOLUTION["forward"](X, y, t)
    np.testing.assert_allclose(value["p"], p, rtol=0, atol=1e-14)
    assert value["loss"] == pytest.approx(-np.log(p[np.arange(len(y)), y]).mean())


def test_corner_names_first_hidden_layer_unit_two(case):
    _, _, theta = case
    old = demo.hand_weights()
    theta = {**theta, "W1": old["W1"], "b1": old["b1"], "W3": old["W2"], "b3": old["b2"]}
    X = demo.count_words("I changed my mind.", demo.WORDS)[None, :]
    y = np.array([1])
    base = SOLUTION["forward"](X, y, theta)
    assert base["u1"][0, 1] == 0
    values = []
    masks = []
    for sign in (1, -1):
        t = {k: v.copy() for k, v in theta.items()}
        t["b1"][1] += sign * 1e-5
        value = SOLUTION["forward"](X, y, t)
        values.append(value["loss"])
        masks.append(value["u1"] > 0)
    assert not np.array_equal(*masks)
    central = (values[0] - values[1]) / 2e-5
    analytical = SOLUTION["backward"](X, y, theta)["b1"][1]
    assert analytical == 0 and abs(central - analytical) > 1e-3


def test_small_update_matches_first_order_prediction_within_five_percent(case):
    X, y, theta = case
    gradients = SOLUTION["backward"](X, y, theta)
    eta = 0.001
    moved = {k: v - eta * gradients[k] for k, v in theta.items()}
    prediction = eta * sum(np.sum(g * g) for g in gradients.values())
    decrease = SOLUTION["forward"](X, y, theta)["loss"] - SOLUTION["forward"](X, y, moved)["loss"]
    assert decrease > 0 and abs(decrease - prediction) / prediction < 0.05
