"""Exercise 6: run the three decays and judge the validation results.

The adaptive gradient and the decay both use the weight before the update. Keep
validation out of fitting. The provided experiment fits scales on training data,
then compares all three completed runs on validation, spending that window.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from demos.training_control_demo import exercise_six_results  # noqa: E402


def decay_step(w, gradient, eta, lam, denominator=1.0):
    """Return one decoupled update; denominator may be a scalar or a vector."""
    return w - eta * gradient / denominator - eta * lam * w


def adam(X, y, eta, steps, beta1=0.9, beta2=0.999, delta=1e-8, w0=None, correct=True, decay=0.0):
    w = np.zeros(X.shape[1]) if w0 is None else np.array(w0, dtype=float)
    m, v = np.zeros_like(w), np.zeros_like(w)
    hessian, target = X.T @ X / len(y), X.T @ y / len(y)
    path = [w.copy()]
    for t in range(1, steps + 1):
        g = hessian @ w - target
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g**2
        mh = m / (1 - beta1**t) if correct else m
        vh = v / (1 - beta2**t) if correct else v
        w = decay_step(w, mh / (np.sqrt(vh) + delta), eta, decay)
        path.append(w.copy())
    return np.array(path)


if __name__ == "__main__":
    print("seed 7; eta 0.01; 2000 steps; lambda: training loss, validation loss")
    for lam, train, validation in exercise_six_results(adam):
        print(f"{lam:g}: {train:.6f}, {validation:.6f}")
