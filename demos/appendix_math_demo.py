#!/usr/bin/env python3
"""Appendix A listings: matrix products, the softmax, cross-entropy, the
attention scale and sinusoidal positions. Each region is printed in the book
with the output it prints; companion/tests/test_appendix_math.py runs them in
print order and compares their output with the book."""
from __future__ import annotations

# region book:fnd.appendix-math-foundations.matmul-01 output
import numpy as np

X = np.array([[1, 2], [0, 1], [3, -1]])  # three rows, two columns
W = np.array([[2, 0], [1, 3]])           # two rows, two columns
print(X.shape, W.shape, (X @ W).shape)
print(X @ W)
print(X[0] @ W)  # row 0 of the product on its own
# the same row as a mixture of the rows of W
print(X[0, 0] * W[0] + X[0, 1] * W[1])
print(np.array_equal((X @ W).T, W.T @ X.T))
# endregion book:fnd.appendix-math-foundations.matmul-01


# region book:fnd.appendix-math-foundations.softmax-01 output
import numpy as np

np.set_printoptions(precision=3, floatmode="fixed", suppress=True)


def softmax(z, tau=1.0):
    """Softmax over the last axis: a matrix is done row by row."""
    s = np.asarray(z, dtype=float) / tau
    e = np.exp(s - s.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def softmax_jacobian(p):
    return np.diag(p) - np.outer(p, p)


def central_difference(f, z, step=1e-6):
    """Derivatives of f along each entry of z, numerically."""
    cols = [(f(z + step * e) - f(z - step * e)) / (2 * step)
            for e in np.eye(len(z))]
    return np.stack(cols, axis=-1)


z = np.array([2.0, 1.0, 0.0])
print(softmax(z), softmax(z + 1000.0))
with np.errstate(over="ignore", invalid="ignore"):
    big = np.exp(z + 1000.0)
    print(big / big.sum())  # no shift: inf / inf
for tau in (0.5, 1.0, 2.0):
    print(tau, softmax(z, tau))
p = softmax(z)
print(softmax_jacobian(p))
numeric = central_difference(softmax, z)
print(np.abs(numeric - softmax_jacobian(p)).max() < 1e-8)
# endregion book:fnd.appendix-math-foundations.softmax-01


# region book:fnd.appendix-math-foundations.crossentropy-01 output
import numpy as np


def log_softmax(z):
    s = z - z.max()
    return s - np.log(np.exp(s).sum())


def cross_entropy(z, label):
    return -log_softmax(z)[label]


z = np.array([2.0, 1.0, 0.0])  # scores for three queues
label = 2                       # the ticket's queue is the third
p = np.exp(log_softmax(z))
grad = p - np.eye(3)[label]
numeric = central_difference(lambda s: cross_entropy(s, label), z)
print(round(cross_entropy(z, label), 3), grad)
print(np.abs(numeric - grad).max() < 1e-8)
z_new = z - 0.5 * grad          # one gradient-descent step
print(z_new, round(cross_entropy(z_new, label), 3))
# endregion book:fnd.appendix-math-foundations.crossentropy-01


# region book:fnd.appendix-math-foundations.scale-01 output
import numpy as np

rng = np.random.default_rng(7)
d_k, n_k, n_q = 512, 8, 4000
q = rng.standard_normal((n_q, d_k))       # one query per row
k = rng.standard_normal((n_q, n_k, d_k))  # eight keys per query
scores = np.einsum("qd,qkd->qk", q, k)
scaled = scores / np.sqrt(d_k)
for name, s in (("unscaled", scores), ("scaled", scaled)):
    a = softmax(s)                              # row by row
    largest = a.max(axis=1).mean()
    trace = (1 - (a ** 2).sum(axis=1)).mean()   # of the Jacobian
    print(f"{name:9}{s.var():6.0f}{largest:6.2f}{trace:6.2f}")
# endregion book:fnd.appendix-math-foundations.scale-01


# region book:fnd.appendix-math-foundations.shift-01 output
import numpy as np


def frequencies(d, base=10000.0):
    if d <= 0 or d % 2:
        raise ValueError(f"width {d} is not positive and even")
    return base ** (-np.arange(0, d, 2) / d)


def positions(n, d):
    """Position vectors for positions 0 to n - 1, one per row."""
    angles = np.arange(n)[:, None] * frequencies(d)
    pe = np.zeros((n, d))
    pe[:, 0::2], pe[:, 1::2] = np.sin(angles), np.cos(angles)
    return pe


def shift(d, k):
    """Block-diagonal matrix that moves a position k places on."""
    omega = frequencies(d)  # checks the width first
    m = np.zeros((d, d))
    for i, w in enumerate(omega):
        c, s = np.cos(k * w), np.sin(k * w)
        m[2 * i:2 * i + 2, 2 * i:2 * i + 2] = [[c, s], [-s, c]]
    return m


pe = positions(8, 4)
print(pe[1], pe[3])
print(shift(4, 2) @ pe[1])
# every row moved on by 2, applied to rows through the transpose
print(np.allclose(pe[:-2] @ shift(4, 2).T, pe[2:]))
# the dot product of each position with the next
print((pe[:-1] * pe[1:]).sum(axis=1))
# endregion book:fnd.appendix-math-foundations.shift-01
