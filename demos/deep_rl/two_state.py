"""Shared-weight counterexamples, and the bias of a maximum over noisy estimates."""

import math

import numpy as np


def off_policy_updates(updates=20, theta=10.0, gamma=0.9, eta=0.1):
    path, errors = [theta], []
    for _ in range(updates):
        error = gamma * 2 * theta - theta
        theta += eta * error
        path.append(theta)
        errors.append(error)
    return dict(weights=path, errors=errors, factor=1 + eta * (2 * gamma - 1))


def on_policy_rounds(rounds=20, theta=10.0, gamma=0.9, eta=0.1):
    path = [theta]
    for _ in range(rounds):
        theta += eta * (gamma * 2 * theta - theta)
        theta += eta * (0 - 2 * theta) * 2
        path.append(theta)
    return dict(
        weights=path,
        terminal_factor=1 - 4 * eta,
        factor=(1 + eta * (2 * gamma - 1)) * (1 - 4 * eta),
    )


def return_target_updates(updates=20, theta=10.0, eta=0.1):
    return dict(weights=[theta * (1 - eta) ** t for t in range(updates + 1)], factor=1 - eta)


def held_target_refreshes(refreshes=3, theta=10.0, gamma=0.9, eta=0.1, mix=False, updates=300):
    path, periods = [theta], []
    for _ in range(refreshes):
        held = theta
        losses = []
        for _ in range(updates):
            error = gamma * 2 * held - theta
            if mix:
                losses.append((error**2 + (2 * theta) ** 2) / 4)
                theta += eta * (error - 4 * theta) / 2
            else:
                losses.append(0.5 * error**2)
                theta += eta * error
        periods.append(dict(held=held, start_loss=losses[0], end_loss=losses[-1], weight=theta))
        path.append(theta)
    # Equal-weight, simultaneous regression on the two transitions has its
    # minimum at 2*gamma*held / (1**2 + 2**2). This is distinct from the
    # sequential finite-step rounds in on_policy_rounds.
    factor = 2 * gamma if not mix else 2 * gamma / 5
    return dict(weights=path, periods=periods, factor=factor)


def max_bias(seed=7, samples=100_000):
    maxima = np.max(np.random.default_rng(seed).normal(size=(samples, 2)), axis=1)
    return dict(
        exact=1 / math.sqrt(math.pi),
        simulated=float(maxima.mean()),
        standard_error=float(maxima.std(ddof=1) / np.sqrt(samples)),
        seed=seed,
        samples=samples,
    )
