"""Deterministic, offline experiments for Deep Reinforcement Learning.

The four printed regions run in chapter order without any intervening setup.
They own the small teaching functions; the deep_rl package imports those same
functions. Importing this module prints and writes nothing. Every experiment is
regenerated from seeds, including the long runs; no stored result is an input.
--runs captures outputs with the repository function and writes plot data.
--write-data generates the exercise's deliberately defective PPO artifact.
"""

# region book:fnd.deep-rl.target-01 output
import numpy as np

from demos.output_wrap import print_wrapped


def td_target(reward, ended, next_values, gamma):
    # ended means a terminal state, never the simulator's time limit.
    reward = np.asarray(reward)
    return np.where(
        ended,
        reward,
        reward + gamma * np.max(np.asarray(next_values), axis=-1),
    )


def semi_gradient_step(theta, features, target, eta):
    value = float(np.asarray(theta) @ features)
    error = float(target - value)
    return dict(
        target=float(target),
        error=error,
        loss=0.5 * error**2,
        weights=np.asarray(theta) + eta * error * np.asarray(features),
    )


if __name__ == "__main__":
    print_wrapped(
        "reward 1; gamma 0.9; next values (3, 4); current value 2; eta 0.1"
    )
    for ended in (False, True):
        result = semi_gradient_step(
            np.array([2.0]),
            np.array([1.0]),
            td_target(1, ended, [3, 4], 0.9),
            0.1,
        )
        print_wrapped(
            f"{'ending' if ended else 'continuing'}: target "
            f"{result['target']:.2f}; "
            f"TD error {result['error']:.2f}; half-squared loss "
            f"{result['loss']:.2f}; "
            f"table value {result['weights'][0]:.2f}"
        )
        if ended:
            ending = result
        else:
            continuing = result
# endregion book:fnd.deep-rl.target-01


def print_divergence():
    from demos.deep_rl.two_state import (
        held_target_refreshes,
        max_bias,
        off_policy_updates,
        on_policy_rounds,
        return_target_updates,
    )
    from demos.output_wrap import print_wrapped

    off, mix, observed = off_policy_updates(), on_policy_rounds(), return_target_updates()
    held, held_mix, bias = held_target_refreshes(), held_target_refreshes(mix=True), max_bias()
    print_wrapped("two states: theta and 2*theta; first reward 0; second ends with reward 0")
    print_wrapped("gamma 0.9; eta 0.1; initial theta 10")
    for t in range(2):
        print_wrapped(
            f"first transition update {t + 1}: error {off['errors'][t]:.2f}; "
            f"theta {off['weights'][t + 1]:.3f}"
        )
    print_wrapped(
        f"first-only factor {off['factor']:.3f}; terminal factor {mix['terminal_factor']:.3f}; "
        f"sequential policy-mix round {mix['factor']:.3f}; return-target factor {observed['factor']:.3f}"
    )
    print_wrapped("held targets: 300 regression updates per refresh")
    for i, row in enumerate(held["periods"], 1):
        print_wrapped(
            f"refresh {i}: held {row['held']:.3f}; settled theta {row['weight']:.3f}; "
            f"start loss {row['start_loss']:.2f}; end loss {row['end_loss']:.3e}"
        )
    print_wrapped(
        f"held first-only factor {held['factor']:.3f}; equal-weight two-transition "
        f"regression factor {held_mix['factor']:.3f}"
    )
    print_wrapped("held mixed weights: " + ", ".join(f"{w:.6f}" for w in held_mix["weights"]))
    print_wrapped(
        f"max bias: exact {bias['exact']:.6f}; simulated {bias['simulated']:.6f}; "
        f"SE {bias['standard_error']:.6f}; seed {bias['seed']}; draws {bias['samples']}"
    )


# region book:fnd.deep-rl.diverge-01 output
if __name__ == "__main__":
    print_divergence()
# endregion book:fnd.deep-rl.diverge-01


# region book:fnd.deep-rl.dqn-01
from demos.deep_rl.mlp import MLP, Adam, backward, forward


def dqn_update(online, target, batch, optimizer, gamma=0.99, refresh=100):
    y = td_target(
        batch["reward"],
        batch["terminated"],
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


if __name__ == "__main__":
    from demos.deep_rl.dqn import ReplayBuffer

    online = MLP(seed=7)
    target = online.copy()
    optimizer = Adam(online.theta, eta=1e-3)
    buffer = ReplayBuffer(20_000)
    buffer.add([0, 0, 0.1, 0], 1, 1, [0, 0.1, 0.1, -0.1], False)
    batch = buffer.sample(64, np.random.default_rng(7))
    update_loss = dqn_update(online, target, batch, optimizer)
# endregion book:fnd.deep-rl.dqn-01


def cartpole_runs():
    from demos.deep_rl.dqn import (
        CONDITIONS,
        EVALUATION_SEEDS,
        TRAINING_SEEDS,
        evaluate,
        random_policy,
        rule_policy,
        train_dqn,
    )

    runs = []
    for condition in CONDITIONS:
        for seed in TRAINING_SEEDS:
            run = train_dqn(seed, condition)
            runs.append({k: v for k, v in run.items() if k != "network"})
    return dict(
        runs=runs,
        random=evaluate(random_policy, EVALUATION_SEEDS),
        rule=evaluate(rule_policy, EVALUATION_SEEDS),
        evaluation_seeds=list(EVALUATION_SEEDS),
    )


def print_runs(data):
    import json

    settings = data["runs"][0]["settings"]
    print("cart-pole DQN settings: " + json.dumps(settings, sort_keys=True))
    print("one update per consecutive 64 new transitions; no-replay batches used once")
    print("training seeds: " + ", ".join(str(r["seed"]) for r in data["runs"][:5]))
    print("evaluation seeds: " + ", ".join(map(str, data["evaluation_seeds"])))
    for name in ("random", "rule"):
        print(
            f"{name}: ten episode returns "
            + ", ".join(f"{r:.0f}" for r in data[name])
            + f"; mean {np.mean(data[name]):.2f}"
        )
    print("undiscounted held-out mean return; all policies use the same ten starting states")
    for condition in dict.fromkeys(r["condition"] for r in data["runs"]):
        rows = [r for r in data["runs"] if r["condition"] == condition]
        print(f"condition {condition}; {rows[0]['updates']} updates per seed")
        print("steps | seed 7 | seed 11 | seed 19 | seed 23 | seed 31 | median | min | max")
        for i, point in enumerate(rows[0]["curve"]):
            values = [r["curve"][i]["mean"] for r in rows]
            print(
                f"{point['step']} | "
                + " | ".join(f"{v:.2f}" for v in values)
                + f" | {np.median(values):.2f} | {min(values):.2f} | {max(values):.2f}"
            )
    print("final ten episode returns, in evaluation seed order")
    for row in data["runs"]:
        print(
            f"{row['condition']}, seed {row['seed']}: "
            + ", ".join(f"{r:.0f}" for r in row["curve"][-1]["returns"])
        )


# region book:fnd.deep-rl.runs-01 output
if __name__ == "__main__":
    run_data = cartpole_runs()
    print_runs(run_data)
# endregion book:fnd.deep-rl.runs-01


# region book:fnd.deep-rl.pg-01 output
import numpy as np


def softmax_score(probabilities, action):
    p = np.asarray(probabilities, dtype=float)
    score = -p.copy()
    if p.ndim == 1:
        score[action] += 1
    else:
        score[np.arange(len(p)), np.asarray(action, dtype=int)] += 1
    return score


def reinforce_gradient(
    probabilities, actions, rewards, gamma=0.99, baseline=0.0
):
    from demos.deep_rl.policy_gradient import discounted_returns

    returns = discounted_returns(rewards, gamma)
    weights = gamma ** np.arange(len(returns)) * (
        returns - np.asarray(baseline)
    )
    return np.sum(
        weights[:, None] * softmax_score(probabilities, actions), axis=0
    )


if __name__ == "__main__":
    from demos.deep_rl.policy_gradient import softmax

    probabilities = np.array([0.6, 0.4])
    score = softmax_score(probabilities, 0)
    change = 0.1 * reinforce_gradient(
        probabilities[None, :], [0], [5], baseline=3
    )
    probability = softmax(np.log(probabilities) + change)[0]
    baseline_term = sum(
        p * 3 * softmax_score(probabilities, a)
        for a, p in enumerate(probabilities)
    )
    print("policy (0.6, 0.4); action 0; return 5; baseline 3; eta 0.1")
    print(
        f"score ({score[0]:.3f}, {score[1]:.3f}); score change "
        f"({change[0]:.3f}, {change[1]:.3f}); new probability "
        f"{probability:.6f}"
    )
    print(
        "baseline expectation: ("
        + ", ".join(f"{abs(v):.12f}" for v in baseline_term)
        + ")"
    )
# endregion book:fnd.deep-rl.pg-01


def print_variance(data):
    print(
        f"fixed policy seed {data['policy_seed']}; widths {data['widths']}; gamma {data['gamma']}"
    )
    print(
        f"baseline fitted on {len(data['baseline_seeds'])} separate episodes; "
        f"seeds {data['baseline_seeds'][0]}..{data['baseline_seeds'][-1]}; ridge {data['ridge']}"
    )
    print(
        f"gradient estimates: {len(data['estimation_seeds'])} paired episodes; "
        f"seeds {data['estimation_seeds'][0]}..{data['estimation_seeds'][-1]}; component b3[0]"
    )
    print("| Baseline | Mean gradient | Standard error | Standard deviation | Total variance |")
    print("|---|---:|---:|---:|---:|")
    for name, key in (("None", "plain"), ("Separate fitted state value", "fitted")):
        r = data[key]
        print(
            f"| {name} | {r['mean']:.6f} | {r['standard_error']:.6f} | "
            f"{r['standard_deviation']:.6f} | {r['total_variance']:.6f} |"
        )
    print(f"total variance ratio, fitted / none: {data['total_variance_ratio']:.6f}")
    plain = np.array([softmax_score([0.5, 0.5], a)[0] for a in (0, 1)])
    poor = plain * (1 - 3)
    print(
        f"counterexample, policy (0.5, 0.5), reward 1: variance without baseline "
        f"{plain.var():.2f}; baseline 3: {poor.var():.2f}"
    )


# region book:fnd.deep-rl.variance-01 output
if __name__ == "__main__":
    from demos.deep_rl.policy_gradient import variance_experiment

    variance_data = variance_experiment()
    print_variance(variance_data)
# endregion book:fnd.deep-rl.variance-01


def table_number(value, decimals):
    return f"{value:.{decimals}f}".replace("-", "−")


def print_variance_table(data):
    from demos.output_wrap import print_wrapped

    print_wrapped("| Baseline | Mean gradient | SE | SD | Total variance |")
    print_wrapped("|---|---:|---:|---:|---:|")
    for name, key in (("None", "plain"), ("Separate fitted state value", "fitted")):
        row = data[key]
        values = [
            row[k] for k in ("mean", "standard_error", "standard_deviation", "total_variance")
        ]
        print_wrapped(f"| {name} | " + " | ".join(table_number(v, 6) for v in values) + " |")


# region book:fnd.deep-rl.variance-02 output
if __name__ == "__main__":
    print_variance_table(variance_data)
# endregion book:fnd.deep-rl.variance-02


# region book:fnd.deep-rl.ppo-01 output
import numpy as np


def clipped_objective(ratio, advantage, eps_clip=0.2):
    return np.minimum(
        np.asarray(ratio) * advantage,
        np.clip(ratio, 1 - eps_clip, 1 + eps_clip) * advantage,
    )


def clipped_slope(ratio, advantage, eps_clip=0.2):
    ratio, advantage = np.asarray(ratio), np.asarray(advantage)
    stopped = ((advantage > 0) & (ratio > 1 + eps_clip)) | (
        (advantage < 0) & (ratio < 1 - eps_clip)
    )
    return np.where(stopped, 0.0, advantage)


if __name__ == "__main__":
    print("epsilon_clip 0.2")
    print(
        "| Advantage | Ratio | Ratio * advantage | Clipped term | Objective | "
        "Slope |"
    )
    print("|---:|---:|---:|---:|---:|---:|")
    for advantage, ratio in ((2, 1.3), (2, 0.7), (-2, 1.3), (-2, 0.7)):
        print(
            f"| {advantage} | {ratio:.1f} | {ratio * advantage:.1f} | "
            f"{np.clip(ratio, 0.8, 1.2) * advantage:.1f} | "
            f"{clipped_objective(ratio, advantage):.1f} | "
            f"{clipped_slope(ratio, advantage):.0f} |"
        )
# endregion book:fnd.deep-rl.ppo-01


def print_ppo_table():
    print("| Advantage | Ratio | Ratio × advantage | Clipped term | Objective | Slope |")
    print("|---:|---:|---:|---:|---:|---:|")
    for advantage, ratio in ((2, 1.3), (2, 0.7), (-2, 1.3), (-2, 0.7)):
        values = (
            advantage,
            ratio,
            ratio * advantage,
            np.clip(ratio, 0.8, 1.2) * advantage,
            clipped_objective(ratio, advantage),
            clipped_slope(ratio, advantage),
        )
        print(
            "| " + " | ".join(table_number(v, d) for v, d in zip(values, (0, 1, 1, 1, 1, 0))) + " |"
        )


# region book:fnd.deep-rl.ppo-02 output
if __name__ == "__main__":
    print_ppo_table()
# endregion book:fnd.deep-rl.ppo-02


def print_batch(data):
    from demos.output_wrap import print_wrapped

    print_wrapped(
        f"policy seed {data['policy_seed']}; gamma {data['gamma']}; {data['samples']} samples "
        f"from {len(data['episode_seeds'])} episodes; eta {data['eta']}; epsilon_clip {data['eps_clip']}"
    )
    print_wrapped(
        f"batch seeds {data['episode_seeds'][0]}..{data['episode_seeds'][-1]}; "
        f"separate baseline seeds {data['baseline_seeds'][0]}..{data['baseline_seeds'][-1]}"
    )
    print_wrapped("old probabilities, returns, baseline and gamma**t weights frozen for all epochs")
    print_wrapped(
        "objective | epoch | share ratios outside [0.8, 1.2] | largest ratio | mean KL(old || new)"
    )
    for name in ("clipped", "unclipped"):
        for r in data[name]["epochs"]:
            print_wrapped(
                f"{name} | {r['epoch']} | {r['outside_share']:.6f} | "
                f"{r['largest_ratio']:.6f} | {r['mean_kl']:.6f}"
            )


# region book:fnd.deep-rl.batch-01 output
if __name__ == "__main__":
    from demos.deep_rl.ppo import batch_experiment

    batch_data = batch_experiment()
    print_batch(batch_data)
# endregion book:fnd.deep-rl.batch-01


def print_depot(data):
    import json

    print("depot settings: " + json.dumps(data["config"], sort_keys=True))
    print(
        f"DQN widths {data['widths']}; gamma {data['gamma']}; eta {data['eta']}; "
        f"batch {data['batch_size']}; buffer {data['buffer_capacity']}; "
        f"update every {data['update_every']} steps; refresh every {data['refresh']} updates"
    )
    print(
        f"epsilon {data['epsilon']}; training seeds {data['training_seeds']}; "
        f"budget {data['budget']} steps; checkpoints every {data['checkpoint_every']} updates"
    )
    print(
        f"fixed-rule threshold {data['rule_threshold']}; {len(data['shifts'])} shared held-out shifts; "
        f"seeds {data['shifts'][0]}..{data['shifts'][-1]}"
    )
    print(
        f"agent's rule benchmark: {data['report']['baseline_cost']:.6f}, from "
        f"{len(data['report']['baseline_seeds'])} training shifts; seeds "
        f"{data['report']['baseline_seeds'][0]}..{data['report']['baseline_seeds'][-1]}"
    )
    for row in data["report"]["checkpoint_estimates"]:
        print(
            f"agent seed {data['report']['seed']}, checkpoint {row['updates']}: "
            f"estimated starting cost {row['estimated_cost']:.6f}"
        )
    print(
        f"selected checkpoint {data['report']['updates']}; reported saving "
        f"{data['reported_saving']:.6f} ({data['reported_saving_percent']:.6f}%) against training rule benchmark"
    )
    print(
        "| Policy or report | Mean cost a shift | Saving against rule | Paired 95% interval of saving |"
    )
    print("|---|---:|---:|---:|")
    print(
        f"| Agent estimate (not realized) | {data['report']['estimated_cost']:.6f} | "
        f"{data['reported_saving']:.6f} | not an interval |"
    )
    for r in data["policies"]:
        print(
            f"| {r['name']} | {r['cost']:.6f} | {r['saving']:.6f} | "
            f"[{r['interval'][0]:.6f}, {r['interval'][1]:.6f}] |"
        )
    print(f"exact expected cost by backward induction: {data['exact_expected_cost']:.6f}")
    print(
        "exact dispatch thresholds (periods left: waiting parcels): "
        + ", ".join(f"{left}: {threshold}" for left, threshold in data["exact_thresholds"].items())
    )
    print(f"five final seeds pass the paired adoption rule: {data['passes_paired_rule']}")
    print("decision: " + data["decision"])


# region book:fnd.deep-rl.depot-01 output
if __name__ == "__main__":
    from demos.deep_rl.depot import depot_experiment

    depot_data = depot_experiment()
    print_depot(depot_data)
# endregion book:fnd.deep-rl.depot-01


def print_depot_table(data):
    from demos.output_wrap import print_wrapped

    print_wrapped(
        "| Policy or report | Mean cost | Saving | Paired 95% interval |"
    )
    print_wrapped("|---|---:|---:|---:|")
    print_wrapped(
        "| Agent's estimate, not realized | "
        + table_number(data["report"]["estimated_cost"], 3)
        + " | "
        + table_number(data["reported_saving"], 3)
        + " | Not an interval |"
    )
    for row in data["policies"]:
        name = row["name"].replace("DQN seed", "DQN, seed")
        name = name[:1].upper() + name[1:]
        low, high = (table_number(v, 2) for v in row["interval"])
        print_wrapped(
            f"| {name} | {table_number(row['cost'], 3)} | "
            f"{table_number(row['saving'], 3)} | [{low}, {high}] |"
        )


# region book:fnd.deep-rl.depot-02 output
if __name__ == "__main__":
    print_depot_table(depot_data)
# endregion book:fnd.deep-rl.depot-02


def print_keys():
    from demos.deep_rl.policy_gradient import softmax

    print("exercise 3: reward -2; gamma 0.9; next values (-10, -6); current -8; eta 0.1")
    for name, ended in (("continuing", False), ("resolved", True), ("time limit", False)):
        r = semi_gradient_step(
            np.array([-8.0]), np.array([1.0]), td_target(-2, ended, [-10, -6], 0.9), 0.1
        )
        print(
            f"{name}: target {r['target']:.2f}; error {r['error']:.2f}; "
            f"loss {r['loss']:.2f}; new table value {r['weights'][0]:.2f}"
        )
    p = softmax([1, 0])
    score = softmax_score(p, 0)
    change = 0.5 * score
    print(
        f"exercise 4: score ({score[0]:.6f}, {score[1]:.6f}); "
        f"step ({change[0]:.6f}, {change[1]:.6f}); "
        f"new probability {softmax(np.array([1, 0]) + change)[0]:.6f}"
    )
    print(
        f"without baseline, return 4: new probability "
        f"{softmax(np.array([1, 0]) + 0.5 * 4 * score)[0]:.6f}"
    )
    term = sum(p[a] * 3 * softmax_score(p, a) for a in (0, 1))
    print("baseline expectation: (" + ", ".join(f"{abs(v):.12f}" for v in term) + ")")
    for advantage in (1, -1):
        print(
            f"ratio 1.25, advantage {advantage}: objective {clipped_objective(1.25, advantage):.2f}; "
            f"slope {clipped_slope(1.25, advantage):.0f}"
        )


# region book:fnd.deep-rl.keys-01 output
if __name__ == "__main__":
    print_keys()
# endregion book:fnd.deep-rl.keys-01


# region book:fnd.deep-rl.offer-cells-01 output
from demos.deep_rl.offer_cells import offer_cells, print_offer_cells

if __name__ == "__main__":
    print_offer_cells(offer_cells())
# endregion book:fnd.deep-rl.offer-cells-01


def figure_data(runs):
    from demos.deep_rl.two_state import off_policy_updates, on_policy_rounds

    return dict(
        cartpole=runs, two_state=dict(off_policy=off_policy_updates(), on_policy=on_policy_rounds())
    )


def exercise_fixture():
    from demos.deep_rl.cartpole import STATE_SCALE
    from demos.deep_rl.policy_gradient import softmax
    from demos.deep_rl.ppo import make_batch, ratio_report

    batch = make_batch()
    policy = batch["policy"].copy()
    optimizer = Adam(policy.network.theta, eta=0.01)
    rows, actions = np.arange(len(batch["actions"])), batch["actions"]
    old = batch["old_probabilities"]
    reports = []
    for epoch in range(1, 11):
        f = forward(batch["states"] / STATE_SCALE, policy.network.theta)
        p = softmax(f["z"])
        ratio = p[rows, actions] / old[rows, actions]
        # Planted error: clip * advantage, omitting PPO's minimum and its pull back.
        slope = np.where((ratio > 0.8) & (ratio < 1.2), batch["advantages"], 0.0)
        weight = batch["weights"] * slope * ratio / len(actions)
        grad_z = -weight[:, None] * softmax_score(p, actions)
        optimizer.step(policy.network.theta, backward(f, policy.network.theta, grad_z))
        reports.append(
            dict(epoch=epoch, **ratio_report(old, policy.probabilities(batch["states"]), actions))
        )
    cases = [
        dict(ratio=r, advantage=a, agent_objective=float(np.clip(r, 0.8, 1.2) * a))
        for a, r in ((2, 1.3), (2, 0.7), (-2, 1.3), (-2, 0.7))
    ]
    return dict(
        scenario="constructed cart-pole PPO update",
        eps_clip=0.2,
        eta=0.01,
        policy_seed=37,
        episode_seeds=batch["episode_seeds"],
        worked_cases=cases,
        samples=[
            dict(
                action=int(a),
                old_probabilities=old[i].tolist(),
                ratio=1.0,
                advantage=float(batch["advantages"][i]),
                weight=float(batch["weights"][i]),
                agent_objective=float(batch["advantages"][i]),
                ratio_after_ten_epochs=reports[-1]["ratios"][i],
            )
            for i, a in enumerate(actions)
        ],
        agent_code="clip(ratio, 1 - eps_clip, 1 + eps_clip) * advantage",
        epochs=reports,
        agent_claim="clipping kept every update within 20%",
    )


if __name__ == "__main__":
    import argparse
    import json
    import sys
    from pathlib import Path

    from demos import regions as _regions

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs", action="store_true", help="write plot data and capture chapter outputs"
    )
    parser.add_argument("--write-data", action="store_true", help="write the PPO exercise fixture")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.runs:
        path = root / "demos/data/deep_rl_figures.json"
        path.write_text(
            json.dumps(figure_data(run_data), indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
        capture = vars(_regions)
        regions = capture["extract_snippets_from_text"](Path(__file__).read_text(), Path(__file__))
        outputs = capture["capture_outputs"](
            {s.snippet_id: s for s in regions}, sys.executable, timeout=600
        )
        for name, content in outputs.items():
            (root / "book_outputs" / f"_{name}.txt").write_text(content)
    if args.write_data:
        path = root / "exercises/deep-rl/ppo-05.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(exercise_fixture(), indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
