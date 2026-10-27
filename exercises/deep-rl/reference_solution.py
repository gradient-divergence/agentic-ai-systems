"""Reference solution: double-target implementation and reproducible five-seed report.

Run with a report path to repeat training. Only the target changes; training,
behavior/replay streams, evaluation episodes and the final checkpoint stay fixed.
"""

import json
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
from acceptance import check_report

from demos.deep_rl import dqn
from demos.deep_rl.cartpole import STATE_SCALE, CartPole
from demos.deep_rl.depot import paired_interval
from demos.deep_rl.mlp import backward, forward


def double_target(reward, terminated, online_next, target_next, gamma):
    chosen = np.argmax(online_next, axis=1)
    next_value = np.asarray(target_next)[np.arange(len(chosen)), chosen]
    return np.asarray(reward) + gamma * (~np.asarray(terminated)) * next_value


def double_update(online, target, batch, optimizer, gamma=0.99, refresh=100):
    y = double_target(
        batch["reward"],
        batch["terminated"],
        online(batch["next_state"]),
        target(batch["next_state"]),
        gamma,
    )
    f = forward(batch["state"], online.theta)
    rows = np.arange(len(y))
    error = f["z"][rows, batch["action"]] - y
    grad_z = np.zeros_like(f["z"])
    grad_z[rows, batch["action"]] = error / len(y)
    optimizer.step(online.theta, backward(f, online.theta, grad_z))
    if target is not online and optimizer.steps % refresh == 0:
        target.copy_from(online)
    return float(np.mean(error**2) / 2)


def starting_estimate(network):
    states = np.array([CartPole().reset(seed) / STATE_SCALE for seed in dqn.EVALUATION_SEEDS])
    return float(network(states).max(axis=1).mean())


def build_report():
    runs = []
    for seed in dqn.TRAINING_SEEDS:
        original = dqn.train_dqn(seed)
        with patch.object(dqn, "dqn_update", double_update):
            double = dqn.train_dqn(seed)
        runs.append(
            dict(
                seed=seed,
                original_returns=original["curve"][-1]["returns"],
                double_returns=double["curve"][-1]["returns"],
                original_starting_estimate=starting_estimate(original["network"]),
                double_starting_estimate=starting_estimate(double["network"]),
            )
        )
    report = dict(
        training_seeds=list(dqn.TRAINING_SEEDS),
        evaluation_seeds=list(dqn.EVALUATION_SEEDS),
        settings=original["settings"],
        runs=runs,
        method=dict(
            checkpoint="final, never selected by evaluation",
            state_scale=STATE_SCALE.tolist(),
            updates="one per 64 new transitions; refresh counted in optimizer updates",
            initialization="MLP seeded He hidden layers, scaled normal output, zero biases",
            optimizer=dict(name="Adam", beta1=0.9, beta2=0.999, delta=1e-8),
            random_streams="behavior SeedSequence([seed, 2]); replay [seed, 5]",
            starting_estimate="mean max Q over the ten evaluation starting states",
        ),
    )
    report["assessment"] = check_report(report)
    report["comparison"] = [
        dict(
            seed=row["seed"],
            original_mean=float(np.mean(row["original_returns"])),
            double_mean=float(np.mean(row["double_returns"])),
            improvement_interval=paired_interval(
                np.array(row["double_returns"]) - row["original_returns"]
            ),
        )
        for row in runs
    ]
    return report


if __name__ == "__main__":
    Path(sys.argv[1]).write_text(
        json.dumps(build_report(), sort_keys=True, indent=2, allow_nan=False) + "\n"
    )
