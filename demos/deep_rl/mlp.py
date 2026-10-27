"""Two ReLU hidden layers, output-by-input weights, and per-array Adam.

As in chapter 9, examples are rows, W is (outputs, inputs), and biases are
vectors. backward takes the derivative with respect to each output score;
the caller supplies any batch averaging. Adam follows chapter 10's equations.
"""

import numpy as np


def forward(X, theta):
    X = np.atleast_2d(np.asarray(X, dtype=float))
    u1 = X @ theta["W1"].T + theta["b1"]
    h1 = np.maximum(0, u1)
    u2 = h1 @ theta["W2"].T + theta["b2"]
    h2 = np.maximum(0, u2)
    z = h2 @ theta["W3"].T + theta["b3"]
    return dict(X=X, u1=u1, h1=h1, u2=u2, h2=h2, z=z)


def backward(f, theta, grad_z):
    grad_u2 = (grad_z @ theta["W3"]) * (f["u2"] > 0)
    grad_u1 = (grad_u2 @ theta["W2"]) * (f["u1"] > 0)
    return dict(
        W1=grad_u1.T @ f["X"],
        b1=grad_u1.sum(axis=0),
        W2=grad_u2.T @ f["h1"],
        b2=grad_u2.sum(axis=0),
        W3=grad_z.T @ f["h2"],
        b3=grad_z.sum(axis=0),
    )


class MLP:
    def __init__(self, widths=(4, 64, 64, 2), seed=7):
        if len(widths) != 4 or any(w < 1 for w in widths):
            raise ValueError("require input, two hidden widths and output")
        self.widths = tuple(widths)
        rng = np.random.default_rng(seed)
        self.theta = {}
        for layer, (inputs, outputs) in enumerate(zip(widths, widths[1:]), 1):
            scale = np.sqrt((2 if layer < 3 else 1) / inputs)
            self.theta[f"W{layer}"] = rng.normal(0, scale, (outputs, inputs))
            self.theta[f"b{layer}"] = np.zeros(outputs)

    def __call__(self, X):
        return forward(X, self.theta)["z"]

    def copy(self):
        other = object.__new__(MLP)
        other.widths = self.widths
        other.theta = {k: v.copy() for k, v in self.theta.items()}
        return other

    def copy_from(self, other):
        for name in self.theta:
            self.theta[name][...] = other.theta[name]


class Adam:
    def __init__(self, theta, eta=1e-3, beta1=0.9, beta2=0.999, delta=1e-8):
        self.eta, self.beta1, self.beta2, self.delta = eta, beta1, beta2, delta
        self.steps = 0
        self.m = {k: np.zeros_like(v) for k, v in theta.items()}
        self.v = {k: np.zeros_like(v) for k, v in theta.items()}

    def step(self, theta, gradients):
        self.steps += 1
        for name, w in theta.items():
            g = gradients[name]
            self.m[name] = self.beta1 * self.m[name] + (1 - self.beta1) * g
            self.v[name] = self.beta2 * self.v[name] + (1 - self.beta2) * g**2
            mh = self.m[name] / (1 - self.beta1**self.steps)
            vh = self.v[name] / (1 - self.beta2**self.steps)
            update = mh / (np.sqrt(vh) + self.delta)
            w -= self.eta * update


def central_difference(loss, theta, epsilon=1e-5):
    result = {}
    for name, array in theta.items():
        result[name] = np.empty_like(array)
        for index in np.ndindex(array.shape):
            original = array[index]
            try:
                array[index] = original + epsilon
                plus = loss(theta)
                array[index] = original - epsilon
                minus = loss(theta)
                result[name][index] = (plus - minus) / (2 * epsilon)
            finally:
                array[index] = original
    return result


def gradient_check(network, X, targets):
    f = forward(X, network.theta)
    gradients = backward(f, network.theta, (f["z"] - targets) / len(f["X"]))
    estimates = central_difference(
        lambda theta: np.sum((forward(X, theta)["z"] - targets) ** 2) / (2 * len(f["X"])),
        network.theta,
    )
    return max(float(np.max(np.abs(gradients[k] - estimates[k]))) for k in gradients)
