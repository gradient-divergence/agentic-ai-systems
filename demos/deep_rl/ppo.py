"""A single fixed cart-pole batch, reused through a probability ratio."""

import numpy as np

from demos.deep_rl.cartpole import STATE_SCALE
from demos.deep_rl.mlp import Adam, backward, forward
from demos.deep_rl.policy_gradient import (
    GAMMA,
    SoftmaxPolicy,
    fit_baseline,
    rollout,
    softmax,
    softmax_score,
)
from demos.deep_rl_demo import clipped_objective as clipped_objective
from demos.deep_rl_demo import clipped_slope

CLIP = 0.2
EPOCHS = 10
ETA = 0.01
BATCH_SEEDS = tuple(range(1_300_000_000, 1_300_000_020))


def make_batch():
    policy = SoftmaxPolicy()
    baseline = fit_baseline(policy)
    episodes = [rollout(policy, seed) for seed in BATCH_SEEDS]
    states = np.concatenate([e["states"] for e in episodes])
    actions = np.concatenate([e["actions"] for e in episodes])
    advantages = np.concatenate([e["returns"] - baseline(e["states"]) for e in episodes])
    weights = np.concatenate([GAMMA ** np.arange(len(e["states"])) for e in episodes])
    return dict(
        policy=policy,
        states=states,
        actions=actions,
        advantages=advantages,
        weights=weights,
        old_probabilities=policy.probabilities(states),
        episode_seeds=list(BATCH_SEEDS),
        baseline_seeds=list(baseline.fit_seeds),
    )


def ratio_report(old_probabilities, new_probabilities, actions, eps_clip=CLIP):
    rows = np.arange(len(actions))
    ratios = new_probabilities[rows, actions] / old_probabilities[rows, actions]
    kl = np.sum(old_probabilities * np.log(old_probabilities / new_probabilities), axis=1)
    return dict(
        outside_share=float(np.mean((ratios < 1 - eps_clip) | (ratios > 1 + eps_clip))),
        largest_ratio=float(ratios.max()),
        mean_kl=float(kl.mean()),
        ratios=ratios.tolist(),
    )


def ppo_epochs(batch, epochs=EPOCHS, clipped=True, eta=ETA):
    policy = batch["policy"].copy()
    optimizer = Adam(policy.network.theta, eta)
    states, actions = batch["states"], batch["actions"]
    rows = np.arange(len(actions))
    old = batch["old_probabilities"]
    reports = []
    for epoch in range(1, epochs + 1):
        f = forward(states / STATE_SCALE, policy.network.theta)
        p = softmax(f["z"])
        ratios = p[rows, actions] / old[rows, actions]
        slope = clipped_slope(ratios, batch["advantages"]) if clipped else batch["advantages"]
        weight = batch["weights"] * slope * ratios / len(actions)
        grad_z = -weight[:, None] * softmax_score(p, actions)
        optimizer.step(policy.network.theta, backward(f, policy.network.theta, grad_z))
        report = ratio_report(old, policy.probabilities(states), actions)
        reports.append(dict(epoch=epoch, **report))
    return dict(policy=policy, epochs=reports, clipped=clipped, eta=eta, samples=len(actions))


def batch_experiment():
    batch = make_batch()
    results = {}
    for clipped in (True, False):
        run = ppo_epochs(batch, clipped=clipped)
        results["clipped" if clipped else "unclipped"] = {
            k: v for k, v in run.items() if k != "policy"
        }
    return dict(
        policy_seed=37,
        gamma=GAMMA,
        eps_clip=CLIP,
        epochs=EPOCHS,
        eta=ETA,
        samples=len(batch["actions"]),
        episode_seeds=batch["episode_seeds"],
        baseline_seeds=batch["baseline_seeds"],
        **results,
    )
