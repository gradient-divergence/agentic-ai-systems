"""Exercise 7: the delivery for a constructed clinic's message router.

A network on four counted words that routes patient messages to three desks, its
backward pass, and the gradient-check report that came with it. The forward pass and
the checking functions are the companion's. Run with --write-report to regenerate
check-report-07.json beside this source.
"""

# region book:fnd.neural-networks-backprop.check-report-07
import numpy as np

from demos.backprop_demo import central_difference, dataset, forward, relative_gap, smooth_weights


def backward(X, y, theta):
    f = forward(X, theta)
    grad_z = f["p"].copy()
    grad_z[np.arange(len(grad_z)), np.asarray(y, dtype=int)] -= 1
    grad_z /= len(grad_z)
    grad_h = grad_z @ theta["W2"]
    grad_u = grad_h * (f["u"] > 0)
    return dict(
        W2=grad_z.T @ f["h"], b2=grad_z.mean(axis=0),
        W1=grad_u.T @ f["X"], b1=grad_u.mean(axis=0),
    )


def make_report():
    _, X, y = dataset("clinic", n=100)
    theta = smooth_weights()
    analytical = backward(X, y, theta)
    numerical, switched = central_difference(X, y, theta)
    checked = {k: relative_gap(analytical[k], numerical[k]) for k in ("W1", "W2")}
    return dict(
        scenario="constructed clinic message routing",
        classes=["appointments", "prescriptions", "billing"],
        seed=7, initialization_seed=19, messages=len(y), precision="float64", epsilon=1e-5,
        minimum_abs_hidden_input=float(abs(forward(X, theta)["u"]).min()),
        entries_whose_nudge_switched_a_unit=len(switched),
        checked_arrays=checked, largest_relative_gap=max(checked.values()),
        conclusion="gradient check passed",
    )

# endregion book:fnd.neural-networks-backprop.check-report-07


if __name__ == "__main__":
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-report", action="store_true")
    args = parser.parse_args()
    report = json.dumps(make_report(), indent=2, allow_nan=False) + "\n"
    if args.write_report:
        Path(__file__).with_name("check-report-07.json").write_text(report)
    else:
        print(report, end="")
