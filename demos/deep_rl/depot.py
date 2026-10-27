"""A finite parcel-depot simulator and its exact, numerical backward induction.

At each ten-minute decision, the van may take up to its capacity and return at
once. Remaining parcels pay one period's waiting cost. Then independent Poisson
arrivals join the queue, and overflow above the waiting cap goes by courier.
After the last period, mandatory trips clear every remaining parcel. The initial
queue is zero. All arrival draws are fixed by the shift seed before any action.

The state is (parcels waiting, periods left). No discount is used: every shift
ends after 36 decisions. Backward induction aggregates the Poisson tail exactly
in float64, including its expected overflow, rather than truncating arrivals at
the van or queue capacity. Optimal means optimal for these constructed inputs.
"""

from dataclasses import asdict, dataclass
from functools import lru_cache
from statistics import NormalDist

import numpy as np

from demos.deep_rl.dqn import ReplayBuffer, dqn_update
from demos.deep_rl.mlp import MLP, Adam

SHIFT_SEEDS = tuple(range(1_400_000_000, 1_400_000_200))
TRAINING_SEEDS = (7, 11, 19, 23, 31)
TRAINING_BUDGET = 16_060
RULE_THRESHOLD = 12


@dataclass(frozen=True)
class DepotConfig:
    periods: int = 36
    minutes: int = 10
    arrival_mean: float = 2.0
    trip_cost: float = 40.0
    wait_cost: float = 1.0
    capacity: int = 30
    waiting_cap: int = 60
    courier_cost: float = 5.0


@dataclass
class ExactPolicy:
    values: np.ndarray
    actions: np.ndarray

    def __call__(self, state):
        waiting, left = state
        return int(self.actions[left, waiting])


def transition(state, action, arrivals, config):
    waiting, left = state
    if action not in (0, 1) or left < 1:
        raise ValueError("require an active shift and a binary action")
    dispatched = min(waiting, config.capacity) if action else 0
    remaining = waiting - dispatched
    cost = (config.trip_cost if dispatched else 0) + config.wait_cost * remaining
    waiting = remaining + arrivals
    overflow = max(0, waiting - config.waiting_cap)
    cost += config.courier_cost * overflow
    waiting -= overflow
    left -= 1
    if left == 0:
        cost += config.trip_cost * ((waiting + config.capacity - 1) // config.capacity)
        waiting = 0
    return (waiting, left), float(cost), left == 0


def simulate_shift(policy, seed, config=DepotConfig()):
    arrivals = np.random.default_rng(seed).poisson(config.arrival_mean, config.periods)
    state, total = (0, config.periods), 0.0
    for count in arrivals:
        state, cost, _ = transition(state, policy(state), int(count), config)
        total += cost
    return total


def rule_policy(state, threshold=RULE_THRESHOLD):
    return int(state[0] >= threshold)


@lru_cache(maxsize=8)
def exact_policy(config=DepotConfig()):
    probabilities = np.empty(256)
    probabilities[0] = np.exp(-config.arrival_mean)
    for k in range(1, len(probabilities)):
        probabilities[k] = probabilities[k - 1] * config.arrival_mean / k
    tails = np.cumsum(probabilities[::-1])[::-1]
    values = np.zeros((config.periods + 1, config.waiting_cap + 1))
    actions = np.zeros_like(values, dtype=int)
    final = config.trip_cost * np.ceil(np.arange(config.waiting_cap + 1) / config.capacity)
    for left in range(1, config.periods + 1):
        future = final if left == 1 else values[left - 1]
        for waiting in range(config.waiting_cap + 1):
            costs = []
            for action in (0, 1):
                dispatched = min(waiting, config.capacity) if action else 0
                remaining = waiting - dispatched
                k = config.waiting_cap - remaining
                expectation = float(probabilities[:k] @ future[remaining : config.waiting_cap])
                expectation += tails[k] * future[-1]
                overflow = (
                    config.arrival_mean
                    if k == 0
                    else (config.arrival_mean * tails[k - 1] - k * tails[k])
                )
                costs.append(
                    (config.trip_cost if dispatched else 0)
                    + config.wait_cost * remaining
                    + config.courier_cost * overflow
                    + expectation
                )
            actions[left, waiting] = int(np.argmin(costs))
            values[left, waiting] = min(costs)

    return ExactPolicy(values, actions)


def depot_policy(network, config=DepotConfig()):
    scale = np.array([config.waiting_cap, config.periods])
    return lambda state: int(np.argmax(network(np.asarray(state) / scale)[0]))


def train_depot_dqn(seed, budget=TRAINING_BUDGET, config=DepotConfig()):
    online = MLP((2, 32, 32, 2), seed)
    target, optimizer = online.copy(), Adam(online.theta)
    buffer = ReplayBuffer(20_000, 2)
    rng = np.random.default_rng(np.random.SeedSequence([seed, 4]))
    scale = np.array([config.waiting_cap, config.periods])
    shift, period, state = 0, 0, (0, config.periods)

    def arrival_stream(index):
        return np.random.default_rng(seed * 1_000_000 + index).poisson(
            config.arrival_mean, config.periods
        )

    arrivals = arrival_stream(shift)
    checkpoints = []
    for step in range(1, budget + 1):
        epsilon = 1 - 0.95 * min(1, (step - 1) / 10_000)
        action = (
            int(rng.integers(2))
            if rng.random() < epsilon
            else int(np.argmax(online(np.asarray(state) / scale)[0]))
        )
        next_state, cost, terminated = transition(state, action, int(arrivals[period]), config)
        buffer.add(
            np.asarray(state) / scale, action, -cost, np.asarray(next_state) / scale, terminated
        )
        if buffer.size >= 64 and step % 4 == 0:
            dqn_update(online, target, buffer.sample(64, rng), optimizer, gamma=1.0, refresh=250)
            if optimizer.steps % 2_000 == 0:
                estimate = -float(np.max(online([0, 1])))
                checkpoints.append(
                    dict(updates=optimizer.steps, estimated_cost=estimate, network=online.copy())
                )
        state, period = next_state, period + 1
        if terminated:
            shift += 1
            arrivals, period, state = arrival_stream(shift), 0, (0, config.periods)
    if not checkpoints or checkpoints[-1]["updates"] != optimizer.steps:
        checkpoints.append(
            dict(
                updates=optimizer.steps,
                estimated_cost=-float(np.max(online([0, 1]))),
                network=online.copy(),
            )
        )
    return dict(
        seed=seed,
        network=online,
        checkpoints=checkpoints,
        updates=optimizer.steps,
        budget=budget,
        config=asdict(config),
    )


def agent_report(seed=7, training=None):
    training = train_depot_dqn(seed) if training is None else training
    chosen = min(training["checkpoints"], key=lambda c: c["estimated_cost"])
    baseline_seeds = list(range(seed * 1_000_000, seed * 1_000_000 + 100))
    baseline_cost = float(np.mean([simulate_shift(rule_policy, s) for s in baseline_seeds]))
    return dict(
        seed=seed,
        updates=chosen["updates"],
        estimated_cost=chosen["estimated_cost"],
        baseline_cost=baseline_cost,
        baseline_seeds=baseline_seeds,
        network=chosen["network"],
        checkpoint_estimates=[
            dict(updates=c["updates"], estimated_cost=c["estimated_cost"])
            for c in training["checkpoints"]
        ],
    )


def paired_interval(savings):
    savings = np.asarray(savings, dtype=float)
    if len(savings) < 2 or not np.all(np.isfinite(savings)):
        raise ValueError("a paired interval needs at least two finite shifts")
    radius = NormalDist().inv_cdf(0.975) * savings.std(ddof=1) / np.sqrt(len(savings))
    mean = float(savings.mean())
    return [mean - float(radius), mean + float(radius)]


def held_out_comparison(policies, shift_seeds=SHIFT_SEEDS, config=DepotConfig()):
    baseline = np.array([simulate_shift(rule_policy, seed, config) for seed in shift_seeds])
    rows = []
    for name, policy in policies.items():
        costs = np.array([simulate_shift(policy, seed, config) for seed in shift_seeds])
        savings = baseline - costs
        rows.append(
            dict(
                name=name,
                cost=float(costs.mean()),
                saving=float(savings.mean()),
                interval=paired_interval(savings),
                costs=costs.tolist(),
            )
        )
    return dict(shifts=list(shift_seeds), baseline_costs=baseline.tolist(), policies=rows)


def adopt(results):
    seeds = results["seeds"]
    baseline = np.asarray(results["baseline_costs"])
    return len(seeds) == 5 and all(
        np.mean(baseline - np.asarray(costs)) > 0
        and paired_interval(baseline - np.asarray(costs))[0] > 0
        for costs in seeds
    )


def depot_experiment():
    runs = [train_depot_dqn(seed) for seed in TRAINING_SEEDS]
    report = agent_report(TRAINING_SEEDS[0], runs[0])
    policies = {"selected checkpoint": depot_policy(report["network"])}
    policies.update({f"DQN seed {r['seed']}": depot_policy(r["network"]) for r in runs})
    policies.update({"fixed rule": rule_policy, "exact policy": exact_policy()})
    comparison = held_out_comparison(policies)
    rows = comparison["policies"]
    adoption = adopt(
        dict(
            baseline_costs=comparison["baseline_costs"],
            seeds=[row["costs"] for row in rows if row["name"].startswith("DQN")],
        )
    )
    estimated = report["estimated_cost"]
    exact = exact_policy()
    thresholds = {
        str(left): int(np.flatnonzero(exact.actions[left])[0]) for left in (36, 18, 6, 3, 2, 1)
    }
    return dict(
        config=asdict(DepotConfig()),
        widths=[2, 32, 32, 2],
        gamma=1.0,
        eta=1e-3,
        batch_size=64,
        buffer_capacity=20_000,
        update_every=4,
        refresh=250,
        epsilon=[1.0, 0.05, 10_000],
        training_seeds=list(TRAINING_SEEDS),
        budget=TRAINING_BUDGET,
        checkpoint_every=2_000,
        rule_threshold=RULE_THRESHOLD,
        report={k: v for k, v in report.items() if k != "network"},
        reported_saving=float(report["baseline_cost"] - estimated),
        reported_saving_percent=float(100 * (1 - estimated / report["baseline_cost"])),
        exact_thresholds=thresholds,
        exact_expected_cost=float(exact_policy().values[DepotConfig().periods, 0]),
        passes_paired_rule=bool(adoption),
        decision="do not adopt: exact policy available",
        **comparison,
    )
