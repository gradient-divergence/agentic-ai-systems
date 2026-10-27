"""Seeded quadratic experiments for Training Optimization and Regularization.

The four printed regions run in chapter order with no intervening setup. Every
region's stdout is captured by scripts/snippets/extract_snippets.py. Importing this
module does no work. Loss means half the mean squared prediction error throughout;
a penalty is lambda/2 times the squared norm. Counts report the first iterate below
a threshold, not a guarantee that subsequent momentum or Adam iterates stay there.

Validation chooses a penalty or checkpoint in the thirty-input experiment. It is
spent on that selection, not a held-out estimate. No frozen or test window is read.
Use --write-data to regenerate the plot data; exercise fixtures have their own CLI.
"""


# region book:fnd.optimization-regularization.overshoot-01 output
def network_step(eta):
    """Chapter 9's two-layer ReLU, updated along its checked initial gradient."""
    a, b, u, v = [w - eta * g for w, g in zip((0.5, 0, 2, 0), (4, 2, 1, 1))]
    hidden_input = a * 2 + b
    prediction = u * max(0, hidden_input) + v
    return hidden_input, 0.5 * (prediction - 1) ** 2


if __name__ == "__main__":
    print("x 2, y 1; initial weights (0.5, 0, 2, 0), gradient (4, 2, 1, 1)")
    print(f"initial loss: {network_step(0)[1]:.7f}")
    for eta in (0.05, 0.08, 0.09, 0.1):
        hidden, value = network_step(eta)
        print(
            f"eta {eta:.2f}: hidden input {0.0 if abs(hidden) < 1e-15 else hidden:.4f}, loss {value:.7f}"
        )
# endregion book:fnd.optimization-regularization.overshoot-01


# region book:fnd.optimization-regularization.descent-01 output
import numpy as np


def descend(X, y, eta, steps, w0=None, penalty=0.0):
    w = np.zeros(X.shape[1]) if w0 is None else np.array(w0, dtype=float)
    hessian, target = X.T @ X / len(y), X.T @ y / len(y)
    path = [w.copy()]
    for _ in range(steps):
        w -= eta * (hessian @ w - target + penalty * w)
        path.append(w.copy())
    return np.array(path)


def loss(X, y, w):
    return float(np.mean((X @ w - y) ** 2) / 2)


if __name__ == "__main__":
    one_X, one_y = np.array([[2.0]]), np.array([1.0])
    print("x 2, y 1, w0 0.25; curvature 4, minimum 0.5, initial loss 0.125")
    for eta in (0.1, 0.25, 0.45, 0.5, 0.6):
        path = descend(one_X, one_y, eta, 5, w0=[0.25])
        print(
            f"eta {eta:.2f}, error factor {1 - 4 * eta:.2f}, loss factor "
            f"{(1 - 4 * eta) ** 2:.2f}"
        )
        print("  weights: " + ", ".join(f"{w[0]:.6f}" for w in path[1:]))
        print(
            "  losses:  "
            + ", ".join(f"{loss(one_X, one_y, w):.6f}" for w in path[1:])
        )
# endregion book:fnd.optimization-regularization.descent-01


def report_valley(X, y, c, limit, condition, counts, rms, scaled_c, scaled_count, h, normalized):
    """Format the valley computations without filling the printed excerpt with reports."""
    print("examples: (2, 0.2, 2), (2, -0.2, 0), (-2, 0.2, 0), (-2, -0.2, -2)")
    print(f"curvatures: {c[1]:.2f}, {c[0]:.2f}; limit {limit:.2f}; condition {condition:.0f}")
    print("minimum: " + ", ".join(f"{v:.6f}" for v in np.linalg.lstsq(X, y, rcond=None)[0]))
    print(f"initial loss {loss(X, y, np.zeros(2)):.6f}; threshold 0.000001")
    for eta, count in counts.items():
        status = str(count) if count is not None else "never (boundary)" if eta == limit else "diverges"
        print(
            f"eta {eta:.2f}: steps {status}; "
            f"direction factors {1 - eta * c[1]:.4f}, {1 - eta * c[0]:.4f}"
        )
    print(f"training RMS: {rms[0]:.1f}, {rms[1]:.1f}")
    print(
        f"standardized curvatures: {scaled_c[0]:.1f}, {scaled_c[1]:.1f}; "
        f"limit {step_limit(scaled_c):.1f}"
    )
    print(f"standardized eta 1.0: steps {scaled_count}; minimum (1.0, 1.0)")
    print(f"batch values (2, 4, 6, 8): mean {h.mean():.1f}, variance {h.var():.1f}")
    print("normalized: " + ", ".join(f"{v:.3f}" for v in normalized))


# region book:fnd.optimization-regularization.valley-01 output
def valley():
    return np.array([[2, 0.2], [2, -0.2], [-2, 0.2], [-2, -0.2]]), np.array(
        [2, 0, 0, -2]
    )


def curvatures(X):
    return np.linalg.eigvalsh(X.T @ X / len(X))


def step_limit(values):
    return 2 / np.max(values)


def steps_to_loss(X, y, path, threshold=1e-6):
    losses = np.mean((X @ path.T - y[:, None]) ** 2, axis=0) / 2
    hits = np.flatnonzero(losses < threshold)
    return int(hits[0]) if len(hits) else None


if __name__ == "__main__":
    X, y = valley()
    c = curvatures(X)
    limit, condition = step_limit(c), c[1] / c[0]
    counts = {
        eta: steps_to_loss(X, y, descend(X, y, eta, 2000))
        if eta < limit
        else None
        for eta in (0.25, 0.49, 0.5, 0.51)
    }
    rms = np.sqrt(np.mean(X**2, axis=0))
    Z = X / rms
    scaled_c = curvatures(Z)
    scaled_count = steps_to_loss(Z, y, descend(Z, y, 1, 1))
    h = np.array([2, 4, 6, 8])
    normalized = (h - h.mean()) / h.std()
    # The helper only formats the computed values for display.
    from demos.training_control_demo import report_valley

    report_valley(
        X,
        y,
        c,
        limit,
        condition,
        counts,
        rms,
        scaled_c,
        scaled_count,
        h,
        normalized,
    )
# endregion book:fnd.optimization-regularization.valley-01


# region book:fnd.optimization-regularization.adam-01 output
def adam(
    X,
    y,
    eta,
    steps,
    beta1=0.9,
    beta2=0.999,
    delta=1e-8,
    w0=None,
    correct=True,
    decay=0.0,
):
    """beta2=None gives momentum; decay weights outside the adaptive step."""
    w = np.zeros(X.shape[1]) if w0 is None else np.array(w0, dtype=float)
    m, v = np.zeros_like(w), np.zeros_like(w)
    hessian, target = X.T @ X / len(y), X.T @ y / len(y)
    path = [w.copy()]
    for t in range(1, steps + 1):
        g = hessian @ w - target
        m = beta1 * m + (1 - beta1) * g
        if beta2 is None:
            update = m  # momentum uses the uncorrected average
        else:
            v = beta2 * v + (1 - beta2) * g**2
            mh = m / (1 - beta1**t) if correct else m
            vh = v / (1 - beta2**t) if correct else v
            update = mh / (np.sqrt(vh) + delta)
        w = w - eta * update - eta * decay * w
        path.append(w.copy())
    return np.array(path)


if __name__ == "__main__":
    print(
        "gradient at zero: " + ", ".join(f"{v:.1f}" for v in -X.T @ y / len(y))
    )
    for method, path in (
        ("descent eta 0.25", descend(X, y, 0.25, 1)),
        ("momentum eta 1", adam(X, y, 1, 1, beta2=None)),
        ("momentum eta 5", adam(X, y, 5, 1, beta2=None)),
        ("Adam eta 0.1", adam(X, y, 0.1, 1)),
        ("Adam uncorrected eta 0.1", adam(X, y, 0.1, 1, correct=False)),
    ):
        print(
            f"first step, {method}: " + ", ".join(f"{v:.6f}" for v in path[-1])
        )
    print(
        f"uncorrected multiplier (delta -> 0): "
        f"{(1 - 0.9) / np.sqrt(1 - 0.999):.6f}"
    )
    print(
        f"constant-gradient first-moment fraction: t 1 {1 - 0.9:.6f}, t 10 "
        f"{1 - 0.9**10:.6f}"
    )
    print("beta1 0.9, beta2 0.999, delta 1e-8; first loss below 0.000001")
    for eta in (1.0, 5.0):
        count = steps_to_loss(X, y, adam(X, y, eta, 2000, beta2=None))
        print(f"momentum eta {eta:.1f}: {count} steps")
    for eta in (0.1, 0.3):
        count = steps_to_loss(X, y, adam(X, y, eta, 2000))
        print(f"Adam eta {eta:.1f}: {count} steps")
# endregion book:fnd.optimization-regularization.adam-01


def rotation():
    angle = np.pi / 4
    return np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])


# region book:fnd.optimization-regularization.rotate-01 output
if __name__ == "__main__":
    rotated = X @ rotation()
    print("valley rotated 45 degrees; first loss below 0.000001")
    for name, eta, run in (
        ("descent", 0.25, descend),
        ("momentum", 1.0, lambda a, b, c, d: adam(a, b, c, d, beta2=None)),
        ("momentum", 5.0, lambda a, b, c, d: adam(a, b, c, d, beta2=None)),
        ("Adam", 0.1, adam),
        ("Adam", 0.3, adam),
    ):
        raw_count = steps_to_loss(X, y, run(X, y, eta, 2000))
        turn_count = steps_to_loss(rotated, y, run(rotated, y, eta, 2000))
        print(f"{name} eta {eta:.2f}: raw {raw_count}, rotated {turn_count} steps")
# endregion book:fnd.optimization-regularization.rotate-01


def warmup_cosine(t, peak=0.4, warm=10, total=100):
    if not 0 < warm < total or not 0 <= t <= total:
        raise ValueError("require 0 < warm < total and 0 <= t <= total")
    return (
        peak * t / warm
        if t <= warm
        else peak * (1 + np.cos(np.pi * (t - warm) / (total - warm))) / 2
    )


def clip(g, maximum):
    g = np.asarray(g, dtype=float)
    norm = np.linalg.norm(g)
    return g * min(1, maximum / norm) if norm else g.copy()


def sgd(c, eta, batch, steps, seed=7, method="sgd", schedule=None):
    """Independent additive Gaussian gradient noise, variance 1/batch, w*=0."""
    noise = np.random.default_rng(seed).normal(size=steps) / np.sqrt(batch)
    w, m, v = 0.0, 0.0, 0.0
    path = [w]
    for t, epsilon in enumerate(noise, 1):
        rate = eta if schedule is None else schedule(t)
        g = c * w + epsilon
        if method == "adam":
            m, v = 0.9 * m + 0.1 * g, 0.999 * v + 0.001 * g * g
            g = (m / (1 - 0.9**t)) / (np.sqrt(v / (1 - 0.999**t)) + 1e-8)
        elif method != "sgd":
            raise ValueError("method must be sgd or adam")
        w -= rate * g
        path.append(w)
    return np.array(path)


# region book:fnd.optimization-regularization.noise-01 output
if __name__ == "__main__":
    print("seed 7; curvature 4; one-example gradient noise variance 1")
    print("50000 updates, first 2000 discarded; empirical weight standard deviation")
    for eta, batch in ((0.1, 1), (0.01, 1), (0.1, 16)):
        observed = sgd(4, eta, batch, 50000)[2000:].std()
        predicted = np.sqrt(eta / (batch * 4 * (2 - eta * 4)))
        print(f"SGD eta {eta:.2f}, batch {batch}: observed {observed:.6f}, formula {predicted:.6f}")
    for eta in (0.1, 0.01):
        observed = sgd(4, eta, 1, 50000, method="adam")[2000:].std()
        print(f"Adam eta {eta:.2f}, batch 1: observed {observed:.6f}")
    for t in (0, 5, 10, 55, 100):
        print(f"schedule peak 0.4, warm 10, total 100: step {t}, eta {warmup_cosine(t):.6f}")
    scheduled = sgd(4, 0.4, 1, 100, schedule=warmup_cosine)
    print(
        f"scheduled SGD at step 100: weight {scheduled[-1]:.6f}, loss {2 * scheduled[-1] ** 2:.6f}"
    )
    print("clipping (3, 4) at norm 2: " + ", ".join(f"{v:.6f}" for v in clip([3, 4], 2)))
# endregion book:fnd.optimization-regularization.noise-01


# region book:fnd.optimization-regularization.decay-01 output
def penalized_minimizer(X, y, lam):
    return np.linalg.solve(
        X.T @ X / len(y) + lam * np.eye(X.shape[1]), X.T @ y / len(y)
    )


def decay_step(w, gradient, eta, lam, denominator=1.0):
    return w - eta * gradient / denominator - eta * lam * w


if __name__ == "__main__":
    penalty, eta = 0.04, 0.25
    regularized = penalized_minimizer(X, y, penalty)
    stopped = descend(X, y, eta, 69)[-1]
    print("penalty lambda/2 * ||w||^2; lambda 0.04")
    print("penalized minimum: " + ", ".join(f"{v:.6f}" for v in regularized))
    c = curvatures(X)
    print(
        f"penalized curvatures {c[1] + penalty:.2f}, {c[0] + penalty:.2f}; "
        f"condition {(c[1] + penalty) / (c[0] + penalty):.1f}"
    )
    print(
        "retained fractions: "
        + ", ".join(f"{v:.6f}" for v in regularized / [0.5, 5])
    )
    print("eta 0.25, stopped at 69: " + ", ".join(f"{v:.6f}" for v in stopped))
    w, g, eta, lam = 2.0, 3.0, 0.1, 0.2
    print("one step: w 2, gradient 3, eta 0.1, lambda 0.2")
    print(
        f"plain descent, penalty {w - eta * (g + lam * w):.12f}, "
        f"decoupled {decay_step(w, g, eta, lam):.12f}"
    )
    print(
        f"fixed sqrt(vhat) 2: penalty {w - eta * (g + lam * w) / 2:.2f}, "
        f"decoupled {decay_step(w, g, eta, lam, 2):.2f}"
    )
# endregion book:fnd.optimization-regularization.decay-01


def fit_scale(training_X):
    mean = training_X.mean(axis=0)
    scale = np.sqrt(np.mean((training_X - mean) ** 2, axis=0))
    return mean, scale


def noisy_problem(seed=7):
    """Thirty differently scaled inputs, forty training and 4000 validation cases."""
    rng = np.random.default_rng(seed)
    scales = np.geomspace(0.1, 10, 30)
    truth = rng.normal(size=30) / (scales * np.sqrt(30))
    training_X = rng.normal(size=(40, 30)) * scales
    training_y = training_X @ truth + rng.normal(size=40)
    validation_X = rng.normal(size=(4000, 30)) * scales
    validation_y = validation_X @ truth + rng.normal(size=4000)
    return dict(
        training_X=training_X,
        training_y=training_y,
        validation_X=validation_X,
        validation_y=validation_y,
    )


def regularization_experiment(problem=None):
    problem = noisy_problem() if problem is None else problem
    train, train_y = problem["training_X"], problem["training_y"]
    mean, scale = fit_scale(train)
    Z = (train - mean) / scale
    intercept = train_y.mean()
    target = train_y - intercept
    eta = 1 / curvatures(Z)[-1]
    path = descend(Z, target, eta, 2000)
    lambdas = np.array([0, 0.001, 0.01, 0.1, 1, 10])
    ridge_weights = np.array([penalized_minimizer(Z, target, lam) for lam in lambdas])
    # All fitting above uses training data. Validation below selects, and is spent.
    validation = (problem["validation_X"] - mean) / scale
    validation_y = problem["validation_y"] - intercept
    training_loss = np.mean((Z @ path.T - target[:, None]) ** 2, axis=0) / 2
    # In blocks to avoid a large validation-by-step matrix.
    validation_loss = np.array([loss(validation, validation_y, w) for w in path])
    ridge_training = np.array([loss(Z, target, w) for w in ridge_weights])
    ridge_validation = np.array([loss(validation, validation_y, w) for w in ridge_weights])
    return dict(
        mean=mean,
        scale=scale,
        path=path,
        eta=eta,
        lambdas=lambdas,
        training_loss=training_loss,
        validation_loss=validation_loss,
        ridge_weights=ridge_weights,
        ridge_training=ridge_training,
        ridge_validation=ridge_validation,
    )


# region book:fnd.optimization-regularization.stop-01 output
if __name__ == "__main__":
    stop_run = regularization_experiment()
    best_step = int(np.argmin(stop_run["validation_loss"]))
    best_ridge = int(np.argmin(stop_run["ridge_validation"]))
    print(f"seed 7; 30 inputs, 40 training, 4000 validation; eta {stop_run['eta']:.6f}")
    print("scales and intercept fitted on training only; validation spent on selection")
    print("lambda: training loss, validation loss (prediction loss, without penalty)")
    for lam, train, val in zip(
        stop_run["lambdas"], stop_run["ridge_training"], stop_run["ridge_validation"]
    ):
        print(f"{lam:g}: {train:.6f}, {val:.6f}")
    print(
        f"selected lambda {stop_run['lambdas'][best_ridge]:g}: validation loss {stop_run['ridge_validation'][best_ridge]:.6f}"
    )
    print(
        f"selected step {best_step}: training loss {stop_run['training_loss'][best_step]:.6f}, validation loss {stop_run['validation_loss'][best_step]:.6f}"
    )
    print(
        f"end step 2000: training loss {stop_run['training_loss'][-1]:.6f}, validation loss {stop_run['validation_loss'][-1]:.6f}"
    )
# endregion book:fnd.optimization-regularization.stop-01


def energy_experiment(seed=7):
    signs = np.array([[a, b, c] for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)])
    train = np.tile(signs, (10, 1)) * [12, 1200, 0.2] + [10, 2000, 0.5]
    validation = np.tile(signs, (20, 1)) * [12, 1200, 0.2] + [10, 2000, 0.5]
    rng = np.random.default_rng(seed)
    truth = np.array([-2, 0.04, 50])
    train_y = 40 + train @ truth + rng.normal(scale=0.5, size=len(train))
    val_y = 40 + validation @ truth + rng.normal(scale=0.5, size=len(validation))
    mean, scale = fit_scale(train)
    centered, target = train - mean, train_y - train_y.mean()
    values = curvatures(centered)
    eta = 1 / values[-1]
    path = descend(centered, target, eta, 100)
    minimum = np.linalg.lstsq(centered, target, rcond=None)[0]
    standardized = descend(centered / scale, target, 1, 100)
    result = dict(
        mean=mean,
        scale=scale,
        rms=np.sqrt(np.mean(train**2, axis=0)),
        curvatures=values,
        eta=eta,
        covered=path[-1] / minimum,
        raw_loss=np.array([loss(centered, target, w) for w in path]),
        divergent_loss=loss(centered, target, descend(centered, target, 3 / values[-1], 10)[-1]),
        standardized_training_loss=loss(centered / scale, target, standardized[-1]),
        standardized_validation_loss=loss(
            (validation - mean) / scale, val_y - train_y.mean(), standardized[-1]
        ),
    )
    result["sweep"] = [
        (
            rate,
            loss(centered / scale, target, descend(centered / scale, target, rate, 100)[-1]),
            loss(
                (validation - mean) / scale,
                val_y - train_y.mean(),
                descend(centered / scale, target, rate, 100)[-1],
            ),
        )
        for rate in (0.25, 1, 1.9)
    ]
    return result


# region book:fnd.optimization-regularization.report-01 output
from demos.output_wrap import print_wrapped

if __name__ == "__main__":
    report = energy_experiment()
    print_wrapped("constructed heating-energy run; seed 7; 80 training, 160 validation; 100 updates")
    print_wrapped("input: raw RMS, training mean, training centered RMS, curvature, distance covered")
    for i, name in enumerate(("outdoor temperature", "floor area", "occupancy share")):
        print_wrapped(
            f"{name}: {report['rms'][i]:.6f}, {report['mean'][i]:.6f}, {report['scale'][i]:.6f}, {report['scale'][i] ** 2:.6f}, {100 * report['covered'][i]:.6f}%"
        )
    c = report["curvatures"]
    print_wrapped(
        f"limit {step_limit(c):.12f}, condition {c[-1] / c[0]:.0f}, agent eta {report['eta']:.12f}"
    )
    print_wrapped(f"diverging eta {3 / c[-1]:.12f}, loss after 10 steps {report['divergent_loss']:.6f}")
    print_wrapped(
        f"stable raw training loss: start {report['raw_loss'][0]:.6f}, end {report['raw_loss'][-1]:.6f}"
    )
    print_wrapped("rerun, training-fitted standardization; eta: training loss, validation loss at step 100")
    for eta, train, val in report["sweep"]:
        print_wrapped(f"{eta:.2f}: {train:.6f}, {val:.6f}")
    print_wrapped("decision: refuse the raw run; stability concealed two nearly unlearned directions")
# endregion book:fnd.optimization-regularization.report-01


def exercise_record():
    eta = 2 / 9 * (1 - 0.00025)
    steps = []
    w = np.zeros(2)
    c, minimum = np.array([9, 0.25]), np.array([1, 4])
    for t in range(61):
        gradient = c * (w - minimum)
        steps.append(
            dict(
                step=t,
                weights=w.tolist(),
                gradients=gradient.tolist(),
                loss=float(np.sum(c * (w - minimum) ** 2) / 2),
            )
        )
        w -= eta * gradient
    return dict(
        scenario="constructed next-day hospital bed occupancy",
        objective="half mean squared error; no penalty; exact full-batch gradients",
        inputs=[[3, 0.5], [3, -0.5], [-3, 0.5], [-3, -0.5]],
        targets=[5, 1, -1, -5],
        initial_weights=[0, 0],
        learning_rate=eta,
        updates=60,
        steps=steps,
        agent_conclusion="The loss has leveled off, so the model has converged.",
    )


def exercise_six_results(candidate=adam, seed=7):
    problem = noisy_problem(seed)
    train, target = problem["training_X"], problem["training_y"]
    mean, scale = fit_scale(train)
    Z, target_mean = (train - mean) / scale, target.mean()
    fitted = [
        (lam, candidate(Z, target - target_mean, 0.01, 2000, decay=lam)[-1])
        for lam in (0, 0.01, 0.1)
    ]
    validation = (problem["validation_X"] - mean) / scale
    return [
        (
            lam,
            loss(Z, target - target_mean, w),
            loss(validation, problem["validation_y"] - target_mean, w),
        )
        for lam, w in fitted
    ]


# region book:fnd.optimization-regularization.keys-01 output
if __name__ == "__main__":
    print(
        "exercise 1: loss doubles, eta*c "
        f"{1 + np.sqrt(2):.6f} (eta>0); level alternating {2:.6f}; "
        f"loss falls 1%, same side {1 - np.sqrt(0.99):.6f}"
    )
    print("exercise 2: standardizing and penalty change curvature; Adam and warmup change the step")
    print(
        "Adam first step approaches eta for a nonzero gradient as delta -> 0; a zero gradient does not move"
    )
    print(f"exercise 3: limit {2 / 9:.6f}, condition {9 / 0.25:.0f}")
    for eta in (1 / 9, 2 / (9 + 0.25)):
        factor = 1 - eta * 0.25
        count = int(np.ceil(np.log(0.1) / np.log(factor)))
        print(
            f"eta {eta:.6f}: flat-direction factor {factor:.6f}, {count} steps for tenfold distance reduction"
        )
    print("standardized: curvatures 1, 1; eta 1, one step")
    g, w, eta, lam = 3, 2, 0.1, 0.2
    first = w - eta * g / (abs(g) + 1e-8)
    coupled = w - eta * (g + lam * w) / (abs(g + lam * w) + 1e-8)
    decoupled = decay_step(w, g, eta, lam, abs(g) + 1e-8)
    print(f"exercise 4: Adam {first:.6f}, penalty {coupled:.6f}, decoupled {decoupled:.6f}")
    print(
        f"uncorrected first step {eta * (1 - 0.9) * g / (np.sqrt((1 - 0.999) * g * g) + 1e-8):.6f}"
    )
    record = exercise_record()
    steep_factor = 1 - record["learning_rate"] * 9
    print(
        f"exercise 5: curvatures 9, 0.25; steep factor {steep_factor:.6f}; distance lost per update {100 * (1 - abs(steep_factor)):.6f}%"
    )
    exercise_X, exercise_y = np.array(record["inputs"]), np.array(record["targets"])
    rerun = descend(exercise_X, exercise_y, 1 / 9, 60)
    print(f"rerun eta {1 / 9:.6f}, 60 steps: loss {loss(exercise_X, exercise_y, rerun[-1]):.6f}")
    print("exercise 6, Adam with decoupled decay; seed 7, eta 0.01, 2000 steps")
    print("lambda: training loss, validation loss (validation spent on comparison)")
    for lam, train, val in exercise_six_results():
        print(f"{lam:g}: {train:.6f}, {val:.6f}")
    print("exercise 6 judgment: validation chooses lambda 0.1, the lowest of these three losses")
    print("selection spent validation; estimate new-data loss on an untouched test window")
# endregion book:fnd.optimization-regularization.keys-01


def figure_data():
    """The arrays consumed by the plotter, generated by the same functions as stdout."""
    X, y = valley()
    Z = X / np.sqrt(np.mean(X**2, axis=0))
    rates = np.unique(np.r_[np.geomspace(0.001, 2.2, 180), 0.25, 0.45, 0.49, 0.5, 1, 2])
    paths = {
        "descent_025": descend(X, y, 0.25, 30),
        "descent_045": descend(X, y, 0.45, 30),
        "momentum": adam(X, y, 1, 30, beta2=None),
        "adam": adam(X, y, 0.1, 30),
    }
    stop = regularization_experiment()
    return dict(
        seed=7,
        sweep=dict(
            rates=rates.tolist(),
            raw=[loss(X, y, descend(X, y, eta, 50)[-1]) for eta in rates],
            standardized=[loss(Z, y, descend(Z, y, eta, 50)[-1]) for eta in rates],
            raw_limit=float(step_limit(curvatures(X))),
            standardized_limit=float(step_limit(curvatures(Z))),
        ),
        paths={name: path.tolist() for name, path in paths.items()},
        contour=dict(curvatures=[4, 0.04], minimum=[0.5, 5]),
        schedule=dict(steps=list(range(101)), rates=[float(warmup_cosine(t)) for t in range(101)]),
        stopping=dict(
            steps=list(range(2001)),
            training=stop["training_loss"].tolist(),
            validation=stop["validation_loss"].tolist(),
            selected=int(np.argmin(stop["validation_loss"])),
        ),
        ridge=dict(
            lambdas=stop["lambdas"].tolist(),
            training=stop["ridge_training"].tolist(),
            validation=stop["ridge_validation"].tolist(),
        ),
    )


if __name__ == "__main__":
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-data", action="store_true")
    args = parser.parse_args()
    if args.write_data:
        path = Path(__file__).with_name("data") / "training_control_figures.json"
        path.write_text(json.dumps(figure_data(), indent=2, allow_nan=False) + "\n")
