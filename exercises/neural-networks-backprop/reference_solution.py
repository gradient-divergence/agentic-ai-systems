"""Reference solution: mean cross-entropy through two hidden ReLU layers."""

import numpy as np


def forward(X, y, theta):
    X = np.atleast_2d(np.asarray(X, dtype=np.float64))
    u1 = X @ theta["W1"].T + theta["b1"]
    h1 = np.maximum(u1, 0)
    u2 = h1 @ theta["W2"].T + theta["b2"]
    h2 = np.maximum(u2, 0)
    z = h2 @ theta["W3"].T + theta["b3"]
    exp = np.exp(z - z.max(axis=1, keepdims=True))
    p = exp / exp.sum(axis=1, keepdims=True)
    loss = float(-np.log(p[np.arange(len(y)), y]).mean())
    return dict(loss=loss, p=p, u1=u1, u2=u2, h1=h1, h2=h2, X=X)


def backward(X, y, theta):
    f = forward(X, y, theta)
    dz = f["p"].copy()
    dz[np.arange(len(y)), y] -= 1
    dz /= len(y)
    du2 = (dz @ theta["W3"]) * (f["u2"] > 0)
    du1 = (du2 @ theta["W2"]) * (f["u1"] > 0)
    return dict(
        W3=dz.T @ f["h2"],
        b3=dz.sum(axis=0),
        W2=du2.T @ f["h1"],
        b2=du2.sum(axis=0),
        W1=du1.T @ f["X"],
        b1=du1.sum(axis=0),
    )
