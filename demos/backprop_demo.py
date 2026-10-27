"""Deterministic NumPy passes, checks and training experiments for chapter 9.

The four printed regions run together in chapter order without intervening setup.
Output-only regions add the scalar join to chapter 10, corners, precision, delegated
code, initialization and exercise keys. All data are generated for this chapter;
training losses are not held-out estimates. Importing the module does no work.
Use --write-data for the plot arrays. The clinic fixture has its own generator.
"""

# region book:fnd.neural-networks-backprop.forward-01 output
import re

import numpy as np

WORDS = ("arrived", "broken", "mind", "never")
CLASSES = ("damaged", "changed_mind", "not_delivered")


def count_words(text, words):
    tokens = re.findall(r"[a-z]+", text.lower())
    return np.array([tokens.count(word) for word in words], dtype=float)


def hand_weights(dtype=np.float64):
    return {
        "W1": np.array([[1, 0, 0, 1], [1, 1, 0, -2]], dtype=dtype),
        "b1": np.array([-1, 0], dtype=dtype),
        "W2": np.array([[2, -1], [1, 1], [0, 0]], dtype=dtype),
        "b2": np.zeros(3, dtype=dtype),
    }


def forward(X, theta):
    X = np.atleast_2d(np.asarray(X, dtype=theta["W1"].dtype))
    u = X @ theta["W1"].T + theta["b1"]
    h = np.maximum(0, u)
    z = h @ theta["W2"].T + theta["b2"]
    shifted = z - z.max(axis=1, keepdims=True)
    logp = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    return dict(X=X, u=u, h=h, z=z, p=np.exp(logp), logp=logp)


def loss_of(f, y):
    logp = f["logp"]
    return float(-logp[np.arange(len(logp)), np.asarray(y, dtype=int)].mean())


def loss(X, y, theta):
    return loss_of(forward(X, theta), y)


def show(name, value, digits=6):
    text = np.array2string(
        np.asarray(value), precision=digits, floatmode="fixed"
    )
    print(f"{name}: {text}")


if __name__ == "__main__":
    x = count_words("My parcel never arrived.", WORDS)
    y, theta = np.array([2]), hand_weights()
    values = forward(x, theta)
    for name in ("X", "u", "h", "z", "p"):
        show(name, values[name][0])
    print(f"loss {loss(x, y, theta):.9f}; uniform loss {np.log(3):.9f}")
    print(
        "parameter shapes: "
        + ", ".join(f"{k} {v.shape}" for k, v in theta.items())
    )
    print(f"parameters: {sum(v.size for v in theta.values())}")
# endregion book:fnd.neural-networks-backprop.forward-01


# region book:fnd.neural-networks-backprop.backward-01 output
def backward(X, y, theta, gate=True):
    f = forward(X, theta)
    grad_z = f["p"].copy()
    grad_z[np.arange(len(grad_z)), np.asarray(y, dtype=int)] -= 1
    grad_z /= len(grad_z)  # gradient of the mean loss
    grad_h = grad_z @ theta["W2"]
    grad_u = grad_h * (f["u"] > 0) if gate else grad_h
    return dict(
        z=grad_z, h=grad_h, u=grad_u,
        W2=grad_z.T @ f["h"], b2=grad_z.sum(axis=0),
        W1=grad_u.T @ f["X"], b1=grad_u.sum(axis=0),
    )


if __name__ == "__main__":
    gradient = backward(x, y, theta)
    grad_p = np.zeros_like(values["p"])
    grad_p[0, y[0]] = -1 / values["p"][0, y[0]]
    show("gradient p", grad_p)
    show("gradient X", gradient["u"] @ theta["W1"])
    for name in ("z", "h", "u", "W2", "b2", "W1", "b1"):
        show("gradient " + name, gradient[name])
    zero_count = sum(np.count_nonzero(gradient[k] == 0) for k in theta)
    total = sum(v.size for v in theta.values())
    print(f"zero parameter gradients: {zero_count} of {total}")
# endregion book:fnd.neural-networks-backprop.backward-01


# region book:fnd.neural-networks-backprop.scalar-01 output
def scalar_pass(weights=(0.5, 0, 2, 0), x=2, target=1):
    w1, b1, w2, b2 = weights
    u = w1 * x + b1
    h = max(0, u)
    prediction = w2 * h + b2
    error = prediction - target
    grad_u = error * w2 * (u > 0)
    return dict(hidden_input=u, hidden=h, prediction=prediction,
                loss=0.5 * error**2, gradient=(grad_u * x, grad_u, error * h, error))


def scalar_step(eta, target=1):
    weights = (0.5, 0, 2, 0)
    gradient = scalar_pass(target=target)["gradient"]
    moved = tuple(w - eta * g for w, g in zip(weights, gradient))
    return dict(weights=moved, **scalar_pass(moved, target=target))


def scalar_check(epsilon=0.01, weights=(0.5, 0, 2, 0)):
    """A central difference for each of the four parameters, the others held."""
    estimates = []
    for k in range(len(weights)):
        plus = tuple(w + epsilon * (i == k) for i, w in enumerate(weights))
        minus = tuple(w - epsilon * (i == k) for i, w in enumerate(weights))
        difference = scalar_pass(plus)["loss"] - scalar_pass(minus)["loss"]
        estimates.append(difference / (2 * epsilon))
    return estimates


if __name__ == "__main__":
    initial = scalar_pass()
    print("x 2, target 1; weights (w1, b1, w2, b2) = (0.5, 0, 2, 0)")
    print(f"forward: hidden input {initial['hidden_input']:.4f}, hidden {initial['hidden']:.4f}, prediction {initial['prediction']:.4f}, loss {initial['loss']:.7f}")
    show("backward (w1, b1, w2, b2)", initial["gradient"])
    show("central differences, epsilon 0.01", scalar_check())
    print("one simultaneous update from the original weights at each step size")
    for eta in (0.05, 0.08, 0.09, 0.1):
        step = scalar_step(eta)
        show(f"eta {eta:.2f}, weights", step["weights"])
        print(f"eta {eta:.2f}: hidden input {0.0 if abs(step['hidden_input']) < 1e-15 else step['hidden_input']:.4f}, prediction {step['prediction']:.4f}, loss {step['loss']:.7f}")
    one_sided = ((2.01)**2 - 2**2) / 0.01
    central = ((2.01)**2 - (1.99)**2) / 0.02
    print(f"x squared at 2, epsilon 0.01: one-sided {one_sided:.6f}, central {central:.6f}")
# endregion book:fnd.neural-networks-backprop.scalar-01


# region book:fnd.neural-networks-backprop.check-01 output
from demos.output_wrap import print_wrapped


def nudged(theta, name, index, step):
    moved = {k: v.copy() for k, v in theta.items()}
    moved[name][index] += step
    return moved


def central_difference(X, y, theta, epsilon=1e-5):
    """Estimates, and the entries whose nudge switched a unit on or off."""
    estimates, switched = {}, []
    on = forward(X, theta)["u"] > 0
    for name, array in theta.items():
        estimates[name] = np.empty_like(array)
        for index in np.ndindex(array.shape):
            plus = forward(X, nudged(theta, name, index, epsilon))
            minus = forward(X, nudged(theta, name, index, -epsilon))
            difference = loss_of(plus, y) - loss_of(minus, y)
            estimates[name][index] = difference / (2 * epsilon)
            if np.any((plus["u"] > 0) != on) or np.any((minus["u"] > 0) != on):
                switched.append((name, index))
    return estimates, switched


def relative_gap(a, b):
    return float(
        np.max(np.abs(a - b) / np.maximum(1e-8, np.abs(a) + np.abs(b)))
    )


def hand_check(X, y, theta, name, index, epsilon=0.01):
    high = loss(X, y, nudged(theta, name, index, epsilon))
    low = loss(X, y, nudged(theta, name, index, -epsilon))
    return (
        high,
        low,
        (high - low) / (2 * epsilon),
        (high - loss(X, y, theta)) / epsilon,
    )


if __name__ == "__main__":
    print_wrapped(
        "hand check, epsilon 0.01: "
        "plus loss, minus loss, central, one-sided, backward"
    )
    for name, index in (("W1", (0, 3)), ("b2", (2,)), ("W1", (1, 3))):
        values = (*hand_check(x, y, theta, name, index), gradient[name][index])
        print_wrapped(
            f"{name}{index}: " + ", ".join(f"{v:.9f}" for v in values)
        )
    numerical, switched = central_difference(x, y, theta)
    print_wrapped(
        f"all {total} parameters, {2 * total} nudged forward passes "
        "and 1 base pass; float64, epsilon 0.00001"
    )
    print_wrapped(f"entries whose nudge switched a unit: {len(switched)}")
    for name in theta:
        gap = relative_gap(gradient[name], numerical[name])
        print_wrapped(f"{name}: largest relative gap {gap:.3e}")
# endregion book:fnd.neural-networks-backprop.check-01


# region book:fnd.neural-networks-backprop.train-01 output
from demos.backprop_data import WORDS as DOMAIN_WORDS
from demos.backprop_data import generate


def dataset(domain="refund", n=300, seed=7):
    texts, labels = generate(domain, n, seed)
    X = np.array([count_words(text, DOMAIN_WORDS[domain]) for text in texts])
    return texts, X, labels


def train(X, y, eta=0.2, steps=300, gate=True, initial=None):
    theta = hand_weights()
    if initial is not None:
        theta = {k: v.copy() for k, v in initial.items()}
    losses, correct, active, parameters = [], [], [], []
    for step in range(steps + 1):
        f = forward(X, theta)
        losses.append(loss(X, y, theta))
        correct.append(int(np.sum(f["p"].argmax(axis=1) == y)))
        active.append((f["u"] > 0).sum(axis=0).tolist())
        parameters.append({k: v.copy() for k, v in theta.items()})
        if step < steps:
            g = backward(X, y, theta, gate)
            theta = {k: v - eta * g[k] for k, v in theta.items()}
    return dict(theta=theta, losses=losses, correct=correct, active=active,
                parameters=parameters)


if __name__ == "__main__":
    texts, X, labels = dataset()
    print("generated requests; seed 7; 300 updates; eta 0.2; training loss only")
    tally = ", ".join(f"{name} {sum(labels == i)}" for i, name in enumerate(CLASSES))
    print("class counts: " + tally)
    runs = {gate: train(X, labels, gate=gate) for gate in (True, False)}
    for gate, run in runs.items():
        print("gate " + ("present" if gate else "removed"))
        for step in (0, 10, 100, 300):
            print(f"step {step}: loss {run['losses'][step]:.9f}, "
                  f"correct {run['correct'][step]} of {len(labels)}")
        changes = np.diff(run["losses"])
        print(f"strictly falling at every update: {bool(np.all(changes < 0))}")
        for step in np.flatnonzero(changes > 0) + 1:
            before, now = run["losses"][step - 1], run["losses"][step]
            print(f"increase at step {step}: {before:.9f} to {now:.9f}, "
                  f"rise {now - before:.9f}")
# endregion book:fnd.neural-networks-backprop.train-01


def smooth_weights(seed=19):
    rng = np.random.default_rng(seed)
    return {k: v + rng.normal(scale=0.07, size=v.shape) for k, v in hand_weights().items()}


def check_arrays(X, y, theta, gate=True, epsilon=1e-5):
    """Each array's largest gap over the entries whose nudges switched no unit, and the rest."""
    analytical = backward(X, y, theta, gate)
    numerical, switched = central_difference(X, y, theta, epsilon)
    gaps = {}
    for name in theta:
        keep = np.ones(theta[name].shape, dtype=bool)
        for other, index in switched:
            if other == name:
                keep[index] = False
        kept = analytical[name][keep], numerical[name][keep]
        gaps[name] = relative_gap(*kept) if keep.any() else float("nan")
    return gaps, switched


def epsilon_errors():
    records = []
    X, y = count_words("My parcel never arrived.", WORDS), [2]
    exact = backward(X, y, hand_weights())["W1"][0, 3]
    for epsilon in np.logspace(-1, -13, 13):
        _, _, central, one_sided = hand_check(X, y, hand_weights(), "W1", (0, 3), epsilon)
        _, _, single, _ = hand_check(X, y, hand_weights(np.float32), "W1", (0, 3), epsilon)
        records.append(dict(epsilon=float(epsilon), one_sided=abs(one_sided - exact),
                            central=abs(central - exact), single=abs(single - exact)))
    return records


# region book:fnd.neural-networks-backprop.kink-01 output
if __name__ == "__main__":
    corner = count_words("I changed my mind.", WORDS)
    f = forward(corner, hand_weights())
    show("corner counts", corner)
    show("corner hidden input", f["u"][0])
    show("corner probabilities", f["p"][0])
    numerical, switched = central_difference(corner, [1], hand_weights())
    print(f"unit 2 bias: backward {backward(corner, [1], hand_weights())['b1'][1]:.6f}, central {numerical['b1'][1]:.6f}")
    high, low, _, _ = hand_check(corner, [1], hand_weights(), "b1", (1,), 1e-5)
    base = loss(corner, [1], hand_weights())
    print(f"left slope {(base - low) / 1e-5:.6f}, right slope {(high - base) / 1e-5:.6f}")
    print("entries whose nudge switched a unit: " + ", ".join(f"{name}{index}" for name, index in switched))
    near = hand_weights()
    near["b1"][1] = 4e-6
    print(f"unit 2 bias moved to 0.000004: hidden input {forward(corner, near)['u'][0, 1]:.6f}, backward {backward(corner, [1], near)['b1'][1]:.6f}")
    for epsilon in (1e-5, 1e-6):
        numerical, switched = central_difference(corner, [1], near, epsilon)
        print(f"epsilon {epsilon:.6f}: central {numerical['b1'][1]:.6f}, entries whose nudge switched a unit: {len(switched)}")
    _, X, labels = dataset()
    at_zero = np.any(forward(X, hand_weights())["u"] == 0, axis=1)
    print(f"requests with a unit exactly at zero: {sum(at_zero)} of {len(X)}")
    _, switched = central_difference(X, labels, hand_weights())
    print(f"all 300 at the hand-set weights: entries whose nudge switched a unit: {len(switched)} of 19")
# endregion book:fnd.neural-networks-backprop.kink-01


# region book:fnd.neural-networks-backprop.epsilon-01 output
if __name__ == "__main__":
    print("W1[0, 3]; absolute error against float64 backward gradient")
    print("epsilon: one-sided float64, central float64, central float32")
    for row in epsilon_errors():
        print(f"{row['epsilon']:.0e}: {row['one_sided']:.9e}, {row['central']:.9e}, {row['single']:.9e}")
# endregion book:fnd.neural-networks-backprop.epsilon-01


# region book:fnd.neural-networks-backprop.curves-01 output
if __name__ == "__main__":
    _, X, labels = dataset()
    counts = np.bincount(labels, minlength=3)
    shares = counts / len(labels)
    print("initial weights: corners present; checking all 19 entries at each run's step 10")
    for eta in (0.2, 2.0, 16.0):
        run = train(X, labels, eta)
        checked = run["parameters"][10]
        gaps, switched = check_arrays(X, labels, checked)
        print(f"eta {eta:g}, step 10: minimum abs(hidden input) {np.min(np.abs(forward(X, checked)['u'])):.9f}; entries whose nudge switched a unit: {len(switched)}")
        print("gaps W1, b1, W2, b2: " + ", ".join(f"{v:.3e}" for v in gaps.values()))
        for step in (0, 1, 10, 300):
            print(f"eta {eta:g}, step {step}: loss {run['losses'][step]:.9f}, active units {run['active'][step]}")
        if eta == 2:
            show("final first request probabilities", forward(X, run["theta"])["p"][0])
            g = backward(X, labels, run["theta"])
            print(f"final largest abs(parameter gradient): {max(np.max(np.abs(g[k])) for k in run['theta']):.3e}")
    print("class shares: " + ", ".join(f"{v:.9f}" for v in shares))
    print(f"class-share entropy: {-np.sum(shares * np.log(shares)):.9f}")
    ceiling = sum(max(np.bincount(labels[np.all(X == row, axis=1)], minlength=3)) for row in np.unique(X, axis=0))
    print(f"best possible correct from these counts: {ceiling} of {len(labels)}")
    print("same counts, different reasons: It arrived fine, I do not want it. / The order has not arrived.")
    for gate in (True, False):
        checked = train(X, labels, gate=gate)["parameters"][10]
        gaps, switched = check_arrays(X, labels, checked, gate)
        print("gate " + ("present" if gate else "removed") + f": gaps W1, b1, W2, b2; entries whose nudge switched a unit: {len(switched)}")
        print(", ".join(f"{v:.3e}" for v in gaps.values()))
# endregion book:fnd.neural-networks-backprop.curves-01


def delegated_check():
    _, X, labels = dataset("ticket")
    theta = smooth_weights()
    gaps, switched = check_arrays(X, labels, theta, gate=False)
    corrected, _ = check_arrays(X, labels, theta)
    return dict(gaps=gaps, corrected=corrected, switched=len(switched),
                minimum_hidden=float(np.min(np.abs(forward(X, theta)["u"]))),
                start_loss=loss(X, labels, theta),
                end_loss=train(X, labels, gate=False, initial=theta)["losses"][-1])


# region book:fnd.neural-networks-backprop.delegated-01 output
from demos.output_wrap import print_wrapped

if __name__ == "__main__":
    ticket = delegated_check()
    print_wrapped("constructed IT service desk; seed 7; 300 tickets, three queues; float64, epsilon 0.00001")
    print_wrapped(f"minimum abs(hidden input): {ticket['minimum_hidden']:.9f}; entries whose nudge switched a unit: {ticket['switched']}")
    print_wrapped(f"missing-gate training loss: {ticket['start_loss']:.9f} to {ticket['end_loss']:.9f}")
    for name in ticket["gaps"]:
        print_wrapped(f"{name}: delegated gap {ticket['gaps'][name]:.3e}, corrected gap {ticket['corrected'][name]:.3e}")
    print_wrapped("decision: reject; corrected line: grad_u = grad_h * (f['u'] > 0)")
# endregion book:fnd.neural-networks-backprop.delegated-01


def stack_pass(X, y, weights, biases):
    """Any number of ReLU layers, followed by a linear softmax head."""
    hidden, inputs = [np.asarray(X, dtype=float)], []
    for W, b in zip(weights[:-1], biases[:-1]):
        u = hidden[-1] @ W.T + b
        inputs.append(u)
        hidden.append(np.maximum(0, u))
    scores = hidden[-1] @ weights[-1].T + biases[-1]
    shifted = scores - scores.max(axis=1, keepdims=True)
    logp = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    value = float(-logp[np.arange(len(X)), y].mean())
    g = np.exp(logp)
    g[np.arange(len(X)), y] -= 1
    g /= len(X)
    gradients, bias_gradients = [g.T @ hidden[-1]], [g.sum(axis=0)]
    g = g @ weights[-1]
    hidden_gradients = [g.copy()]
    for layer in range(len(inputs) - 1, -1, -1):
        g = g * (inputs[layer] > 0)
        gradients.append(g.T @ hidden[layer])
        bias_gradients.append(g.sum(axis=0))
        g = g @ weights[layer]
        hidden_gradients.append(g.copy())
    return dict(loss=value, hidden=hidden, inputs=inputs,
                gradients=gradients[::-1], bias_gradients=bias_gradients[::-1],
                hidden_gradients=hidden_gradients[::-1])


def suffix_loss(hidden, y, weights, biases, start, return_masks=False):
    """Finite differences reuse only the unchanged forward prefix, never gradients."""
    masks = []
    for W, b in zip(weights[start:-1], biases[start:-1]):
        u = hidden @ W.T + b
        masks.append(u > 0)
        hidden = np.maximum(0, u)
    scores = hidden @ weights[-1].T + biases[-1]
    shifted = scores - scores.max(axis=1, keepdims=True)
    logp = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    value = float(-logp[np.arange(len(y)), y].mean())
    return (value, masks) if return_masks else value


def sampled_stack_check(X, y, weights, biases, result, layers, samples=25, epsilon=1e-5):
    rng = np.random.default_rng(71)
    gaps, crossings, raw_gaps = [], [], []
    for layer in layers:
        first = rng.choice(weights[layer].size, size=samples, replace=False)
        remaining = np.setdiff1d(np.arange(weights[layer].size), first)
        candidates = np.r_[first, rng.permutation(remaining)]
        analytical, numerical, raw_a, raw_n = [], [], [], []
        rejected = 0
        masks = [u > 0 for u in result["inputs"][layer:]]
        for flat in candidates:
            index = np.unravel_index(flat, weights[layer].shape)
            losses, crossed = [], False
            for sign in (1, -1):
                changed = list(weights)
                changed[layer] = weights[layer].copy()
                changed[layer][index] += sign * epsilon
                value, perturbed = suffix_loss(result["hidden"][layer], y, changed, biases, layer, True)
                losses.append(value)
                crossed |= any(not np.array_equal(a, b) for a, b in zip(masks, perturbed))
            a = result["gradients"][layer][index]
            n = (losses[0] - losses[1]) / (2 * epsilon)
            raw_a.append(a)
            raw_n.append(n)
            if crossed:
                rejected += 1
            else:
                analytical.append(a)
                numerical.append(n)
            if len(analytical) == samples:
                break
        if len(analytical) != samples:
            raise ValueError("not enough sampled weights stay within the same ReLU region")
        gaps.append(relative_gap(np.array(analytical), np.array(numerical)))
        raw_gaps.append(relative_gap(np.array(raw_a), np.array(raw_n)))
        crossings.append(rejected)
    return dict(gaps=gaps, crossing_counts=crossings, unscreened_gaps=raw_gaps)


def depth_scales(width=200, inputs=200, seeds=(7, 8, 9, 10, 11), samples=25):
    rows = []
    for scale in (0.5, 1.0, 2.0):
        forward_sizes, backward_sizes, gaps = [], [], []
        for seed in seeds:
            rng = np.random.default_rng(seed)
            X = rng.normal(size=(inputs, width))
            weights = [rng.normal(size=(width, width)) * np.sqrt(scale * 2 / width) for _ in range(10)]
            weights.append(rng.normal(size=(3, width)) * 0.1 / np.sqrt(width))
            biases = [np.zeros(W.shape[0]) for W in weights]
            y = rng.integers(3, size=inputs)
            result = stack_pass(X, y, weights, biases)
            h = np.array([np.sqrt(np.mean(v**2)) for v in result["hidden"]])
            g = np.array([np.sqrt(np.mean(v**2)) for v in result["hidden_gradients"]])
            forward_sizes.append(h / h[0])
            backward_sizes.append(g / g[-1])
            if seed == seeds[0]:
                gaps = sampled_stack_check(X, y, weights, biases, result, (0, 4, 9), samples)
        h, g = np.mean(forward_sizes, axis=0), np.mean(backward_sizes, axis=0)
        rows.append(dict(scale=scale, variance=scale * 2 / width,
                         hidden_rms=h.tolist(), gradient_rms=g.tolist(),
                         hidden_ratio=float(h[-1]), gradient_ratio=float(g[0]),
                         check_gap=max(gaps["gaps"]), check_gaps=gaps["gaps"],
                         crossing_counts=gaps["crossing_counts"], unscreened_gaps=gaps["unscreened_gaps"],
                         sampled_weights=3 * samples,
                         seeds=list(seeds), width=width, inputs=inputs))
    return rows


def symmetry_check(steps=300):
    """Two hidden units given the same incoming weights and bias, with and without the same outgoing weights."""
    x, target = count_words("My parcel never arrived.", WORDS), [2]
    same_in = hand_weights()
    same_in["W1"][1], same_in["b1"][1] = same_in["W1"][0], same_in["b1"][0]
    same_all = {k: v.copy() for k, v in same_in.items()}
    same_all["W2"][:, 1] = same_all["W2"][:, 0]
    _, X, y = dataset()
    trained = train(X, y, steps=steps, initial=same_all)["theta"]
    apart = max(float(np.max(np.abs(trained["W1"][0] - trained["W1"][1]))),
                float(abs(trained["b1"][0] - trained["b1"][1])),
                float(np.max(np.abs(trained["W2"][:, 0] - trained["W2"][:, 1]))))
    return dict(hidden=forward(x, same_in)["h"][0].tolist(),
                outgoing_differ=backward(x, target, same_in)["b1"].tolist(),
                outgoing_same=backward(x, target, same_all)["b1"].tolist(),
                still_equal=bool(apart < 1e-12), moved=float(np.max(np.abs(trained["W1"] - same_all["W1"]))))


# region book:fnd.neural-networks-backprop.depth-01 output
if __name__ == "__main__":
    symmetry = symmetry_check()
    print("two hidden units given the same incoming weights and bias; the parcel's request")
    print("hidden values: " + ", ".join(f"{v:.6f}" for v in symmetry["hidden"]))
    print("outgoing weights different: gradients of the two biases " + ", ".join(f"{v:.6f}" for v in symmetry["outgoing_differ"]))
    print("outgoing weights the same: gradients of the two biases " + ", ".join(f"{v:.6f}" for v in symmetry["outgoing_same"]))
    print(f"all three the same, after 300 updates on the 300 requests: weights moved by up to {symmetry['moved']:.6f}, the two units still equal to 12 decimal places: {symmetry['still_equal']}")
    depths = depth_scales()
    print("10 ReLU layers, width 200, 200 inputs; three-score softmax head, one target per input")
    print(f"each hidden layer has {depths[0]['width']} units, each reading {depths[0]['width']} values, so 2 over either count is {2 / depths[0]['width']:.6f}")
    print("five seeds 7, 8, 9, 10, 11; RMS ratios averaged over seeds; weights as drawn, before any update")
    print(f"head weight std {0.1 / np.sqrt(200):.9f}, fixed at all scales to limit softmax saturation")
    print("hidden normalized to input; gradients of mean loss normalized to gradient entering hidden layer 10")
    print("check seed 7: 25 sampled weights each in layers 1, 5, 10; float64, epsilon 0.00001")
    print("sample only nudges that preserve every ReLU gate; also report excluded corner crossings")
    for row in depths:
        print(f"scale {row['scale']:g}: variance {row['variance']:.6f}, std {np.sqrt(row['variance']):.6f}, ideal ten-layer factor {row['scale']**5:.6f}")
        print(f"hidden ratio {row['hidden_ratio']:.9f}, gradient ratio {row['gradient_ratio']:.9f}, largest sampled gap {row['check_gap']:.3e}")
        print("sampled gaps in layers 1, 5, 10: " + ", ".join(f"{gap:.3e}" for gap in row["check_gaps"]))
        print("excluded crossing counts in layers 1, 5, 10: " + ", ".join(str(v) for v in row["crossing_counts"]))
        print("unscreened gaps including crossings: " + ", ".join(f"{v:.3e}" for v in row["unscreened_gaps"]))
        print("layer: hidden RMS ratio, gradient RMS ratio")
        for layer, (h, g) in enumerate(zip(row["hidden_rms"], row["gradient_rms"])):
            print(f"{layer}: {h:.9f}, {g:.9f}")
    print(f"Glorot variance for 200 inputs and 200 outputs: {2 / (200 + 200):.6f}")
# endregion book:fnd.neural-networks-backprop.depth-01


def exercise_six():
    x, y, w, epsilon = 2, 1, 0.25, 0.01
    high = 0.5 * ((w + epsilon) * x - y)**2
    low = 0.5 * ((w - epsilon) * x - y)**2
    return dict(plus=high, minus=low, central=(high - low) / (2 * epsilon))


def clinic_check():
    _, X, labels = dataset("clinic", n=100)
    theta = smooth_weights()
    good = backward(X, labels, theta)
    numerical, switched = central_difference(X, labels, theta)
    # The fixture's bias gradients average rows that the mean loss has already averaged.
    gaps = {k: relative_gap(good[k] / len(X) if k in ("b1", "b2") else good[k], numerical[k]) for k in theta}
    return gaps, len(switched)


def exercise_eight():
    """Instructor arithmetic for two hidden layers; the student's code remains their task."""
    _, X, y = dataset()
    theta = smooth_weights()
    weights = [theta["W1"], np.array([[0.8, 0.2], [-0.1, 1.1]]), theta["W2"]]
    biases = [theta["b1"], np.array([0.17, 0.13]), theta["b2"]]
    result = stack_pass(X, y, weights, biases)
    gaps, switched = [], 0
    for layer, (W, b) in enumerate(zip(weights, biases)):
        on = [u > 0 for u in result["inputs"][layer:]]
        for is_bias, array in ((False, W), (True, b)):
            analytical, numerical = [], []
            for index in np.ndindex(array.shape):
                losses, crossed = [], False
                for sign in (1, -1):
                    changed_W = [v.copy() for v in weights]
                    changed_b = [v.copy() for v in biases]
                    (changed_b if is_bias else changed_W)[layer][index] += sign * 1e-5
                    value, nudged_on = suffix_loss(result["hidden"][layer], y, changed_W, changed_b, layer, True)
                    losses.append(value)
                    crossed |= any(not np.array_equal(*pair) for pair in zip(on, nudged_on))
                switched += crossed
                analytical.append((result["bias_gradients"] if is_bias else result["gradients"])[layer][index])
                numerical.append((losses[0] - losses[1]) / 2e-5)
            gaps.append(relative_gap(np.array(analytical), np.array(numerical)))
    eta = 0.001
    moved_W = [v - eta * g for v, g in zip(weights, result["gradients"])]
    moved_b = [v - eta * g for v, g in zip(biases, result["bias_gradients"])]
    predicted = eta * sum(np.sum(g**2) for g in result["gradients"] + result["bias_gradients"])
    delivered = result["loss"] - stack_pass(X, y, moved_W, moved_b)["loss"]
    corner = count_words("I changed my mind.", WORDS)[None, :]
    original = hand_weights()
    corner_weights = [original["W1"], weights[1], original["W2"]]
    corner_biases = [original["b1"], biases[1], original["b2"]]
    corner_result = stack_pass(corner, np.array([1]), corner_weights, corner_biases)
    corner_on = [u > 0 for u in corner_result["inputs"]]
    corner_losses, corner_switched = [], False
    for sign in (1, -1):
        changed = [v.copy() for v in corner_biases]
        changed[0][1] += sign * 1e-5
        value, nudged_on = suffix_loss(corner, np.array([1]), corner_weights, changed, 0, True)
        corner_losses.append(value)
        corner_switched |= any(not np.array_equal(*pair) for pair in zip(corner_on, nudged_on))
    return dict(gaps=gaps, switched=switched, predicted=float(predicted), delivered=float(delivered),
                relative_drop_error=abs(delivered - predicted) / predicted,
                minimum_hidden=float(min(np.min(np.abs(u)) for u in result["inputs"])),
                corner_backward=float(corner_result["bias_gradients"][0][1]),
                corner_central=(corner_losses[0] - corner_losses[1]) / 2e-5,
                corner_switched=bool(corner_switched))


# region book:fnd.neural-networks-backprop.keys-01 output
if __name__ == "__main__":
    print(f"exercise 1: W1 (3, 6), b1 (3,), W2 (4, 3), b2 (4,); {3 * 6 + 3 + 4 * 3 + 4} parameters")
    print("x (6,), u and h (3,), z, p and y (4,); loss scalar; each gradient has its variable's shape")
    print("exercise 2: W1[0,1] and W1[0,2]: absent words; W1 row 1, b1[1], W2 column 1: unit off")
    broken = count_words("It is broken.", WORDS)
    f, g = forward(broken, hand_weights()), backward(broken, [0], hand_weights())
    for name in ("X", "u", "h", "z", "p"):
        show("exercise 3 " + name, f[name][0], 3)
    print(f"exercise 3 loss: {loss(broken, [0], hand_weights()):.9f}")
    for name in ("z", "h", "u", "W1", "b1", "W2", "b2"):
        show("exercise 3 gradient " + name, g[name], 3)
    print("exercise 4: I changed my mind., unit 2 at zero; central averages both slopes, backward takes zero")
    print("a nudge that carries a sum across zero disagrees from a start that is not zero; so can rounding and a poorly chosen epsilon")
    print("passing checks do not validate labels or the loss, or guarantee stable training")
    show("exercise 5 target 1 gradients", scalar_pass()["gradient"])
    print(f"exercise 5 eta 0.05 loss: {scalar_step(0.05)['loss']:.7f}")
    show("exercise 5 target 3 gradients", scalar_pass(target=3)["gradient"])
    print(f"exercise 5 eta 0.1 loss: {scalar_step(0.1)['loss']:.7f}; hidden input {0.0 if abs(scalar_step(0.1)['hidden_input']) < 1e-15 else scalar_step(0.1)['hidden_input']:.6f}, unit off")
    key = exercise_six()
    print(f"exercise 6a: plus loss {key['plus']:.6f}, minus loss {key['minus']:.6f}, central {key['central']:.6f}; reject +1")
    print(f"exercise 6b: 0.707^10 {0.707**10:.9f}, 1.414^10 {1.414**10:.9f}, ReLU variance at 200 inputs {2 / 200:.6f}")
    clinic_gaps, clinic_switched = clinic_check()
    print(f"exercise 7: omitted b1 and b2; 100 generated clinic messages, seed 7; entries whose nudge switched a unit: {clinic_switched}")
    for name, gap in clinic_gaps.items():
        print(f"exercise 7 {name}: largest relative gap {gap:.3e}")
    print("corrected biases: b1 = grad_u.sum(axis=0); b2 = grad_z.sum(axis=0); grad_z already divided by n")
    key8 = exercise_eight()
    print(f"exercise 8 reference: minimum abs(hidden input) {key8['minimum_hidden']:.9f}; entries whose nudge switched a unit: {key8['switched']}")
    print("gaps W1, b1, W_hidden, b_hidden, W2, b2: " + ", ".join(f"{v:.3e}" for v in key8["gaps"]))
    print(f"eta 0.001: predicted drop {key8['predicted']:.9f}, delivered {key8['delivered']:.9f}, relative difference {key8['relative_drop_error']:.9f}")
    print(f"exercise 8 corner: first hidden layer, unit 2; backward {key8['corner_backward']:.6f}, central {key8['corner_central']:.6f}; a nudge switched a unit: {key8['corner_switched']}")
    print("acceptance: no nudge switches a unit; every gap < 0.000001 at epsilon 0.00001 in float64; reproduce the unit 2 corner; drop within 5%")
# endregion book:fnd.neural-networks-backprop.keys-01


def figure_data():
    _, X, y = dataset()
    return dict(
        seed=7, epsilon=epsilon_errors(), depth=depth_scales(),
        training=dict(steps=list(range(301)),
                      gate=train(X, y)["losses"], no_gate=train(X, y, gate=False)["losses"]),
        rates={str(eta): train(X, y, eta)["losses"] for eta in (0.2, 2.0, 16.0)},
    )


if __name__ == "__main__":
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-data", action="store_true")
    args = parser.parse_args()
    if args.write_data:
        path = Path(__file__).with_name("data") / "backprop_figures.json"
        path.write_text(json.dumps(figure_data(), indent=2, allow_nan=False) + "\n")
