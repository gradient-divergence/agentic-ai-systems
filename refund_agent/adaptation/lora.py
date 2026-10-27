"""Low-rank products beside frozen row-oriented weights; NumPy gradient descent."""

import numpy as np


def trainable_numbers(d_in, d_out, r):
    if any(type(value) is not int or value < 1 for value in (d_in, d_out, r)):
        raise ValueError("positive integer dimensions and rank required")
    if r > min(d_in, d_out):
        raise ValueError("rank exceeds the layer's dimensions")
    return r * (d_in + d_out)


def adapter_bytes(d_in, d_out, r, bytes_per_number=2):
    if type(bytes_per_number) is not int or bytes_per_number < 1:
        raise ValueError("positive integer bytes per number required")
    return trainable_numbers(d_in, d_out, r) * bytes_per_number


class LowRankAdapter:
    def __init__(self, w0, r, nu=32, seed=7):
        w0 = np.array(w0, dtype=float, copy=True)
        if w0.ndim != 2 or not np.isfinite(w0).all() or not np.isfinite(nu) or nu <= 0:
            raise ValueError("finite matrix and positive scale constant required")
        self.trainable_numbers = trainable_numbers(w0.shape[0], w0.shape[1], r)
        self.w0, self.r, self.nu = w0, r, nu
        self.w0.flags.writeable = False
        self.down = np.random.default_rng(seed).normal(0, 0.1, (w0.shape[0], r))
        self.up = np.zeros((r, w0.shape[1]))

    @property
    def scale(self):
        return self.nu / self.r

    def forward(self, x):
        return x @ self.w0 + self.scale * (x @ self.down) @ self.up

    def merged(self):
        return self.w0 + self.scale * self.down @ self.up

    def gradients(self, g):
        return self.scale * g @ self.up.T, self.scale * self.down.T @ g


def fit(x, y, w0, rank=None, *, nu=8, seed=7, steps=4000, step_size=0.1):
    """Training-only MSE over every output number; equal full-batch steps.

    Sufficient statistics avoid repeating the prediction multiply each step.
    """
    x, y, w0 = np.asarray(x, dtype=float), np.asarray(y, dtype=float), np.asarray(w0, dtype=float)
    if (
        x.ndim != 2
        or y.ndim != 2
        or w0.shape != (x.shape[1], y.shape[1])
        or len(x) != len(y)
        or not len(x)
    ):
        raise ValueError("nonempty matched training arrays and layer shape required")
    if type(steps) is not int or steps < 1 or not np.isfinite(step_size) or step_size <= 0:
        raise ValueError("positive steps and finite positive step size required")
    if not all(np.isfinite(a).all() for a in (x, y, w0)):
        raise ValueError("finite training arrays required")
    hessian, target = x.T @ x / len(x), x.T @ y / len(x)
    factor = 2 / y.shape[1]
    if rank is None:
        w = w0.copy()
        for _ in range(steps):
            w -= step_size * factor * (hessian @ w - target)
        return w
    adapter = LowRankAdapter(w0, rank, nu, seed)
    for _ in range(steps):
        g = factor * (hessian @ adapter.merged() - target)
        down_g, up_g = adapter.gradients(g)
        adapter.down -= step_size * down_g
        adapter.up -= step_size * up_g
    return adapter.merged()
