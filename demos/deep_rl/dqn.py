"""DQN and matched, sequential-batch ablations, without an external environment.

No-replay runs use each consecutive batch once, then discard it. All conditions
use the same batch size and number of updates; full runs instead draw that many
transitions uniformly from their history. Evaluation never selects a checkpoint.
"""

from dataclasses import asdict, dataclass

import numpy as np

from demos.deep_rl.cartpole import STATE_SCALE, CartPole
from demos.deep_rl.mlp import MLP, Adam
from demos.deep_rl_demo import dqn_update
from demos.deep_rl_demo import semi_gradient_step as semi_gradient_step
from demos.deep_rl_demo import td_target as td_target

CONDITIONS = ("full", "no_target", "no_replay", "neither")
TRAINING_SEEDS = (7, 11, 19, 23, 31)
EVALUATION_SEEDS = tuple(range(1_000_000_000, 1_000_000_010))


@dataclass(frozen=True)
class DQNConfig:
    widths: tuple = (4, 64, 64, 2)
    eta: float = 1e-3
    gamma: float = 0.99
    batch_size: int = 64
    capacity: int = 20_000
    refresh: int = 100
    epsilon_end: float = 0.05
    epsilon_steps: int = 10_000
    evaluation_every: int = 2_500
    budget: int = 50_000


class ReplayBuffer:
    def __init__(self, capacity, state_dim=4):
        self.capacity, self.size, self.cursor = capacity, 0, 0
        self.data = dict(
            state=np.empty((capacity, state_dim)),
            action=np.empty(capacity, dtype=int),
            reward=np.empty(capacity),
            next_state=np.empty((capacity, state_dim)),
            terminated=np.empty(capacity, dtype=bool),
        )

    def add(self, state, action, reward, next_state, terminated):
        for name, value in zip(self.data, (state, action, reward, next_state, terminated)):
            self.data[name][self.cursor] = value
        self.cursor = (self.cursor + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def index_probabilities(self):
        return np.full(self.size, 1 / self.size) if self.size else np.array([])

    def sample(self, size, rng):
        if not self.size:
            raise ValueError("cannot sample an empty buffer")
        indices = rng.integers(self.size, size=size)
        return {**{k: v[indices].copy() for k, v in self.data.items()}, "indices": indices}

    def consecutive_batch(self):
        indices = np.arange(self.size)
        return {k: v[indices].copy() for k, v in self.data.items()}


def greedy_policy(network):
    return lambda state, rng: int(np.argmax(network(state / STATE_SCALE)[0]))


def random_policy(state, rng):
    return int(rng.integers(2))


def rule_policy(state, rng):
    return int(state[2] > 0)


def evaluate(policy, episode_seeds):
    returns = []
    for seed in episode_seeds:
        env = CartPole()
        state = env.reset(seed)
        rng = np.random.default_rng(np.random.SeedSequence([seed, 1]))
        total = 0.0
        while not env.finished:
            state, reward, _, _ = env.step(policy(state, rng))
            total += reward
        returns.append(total)
    return returns


def train_dqn(seed, condition="full", budget=50_000, config=None, episode_seeds=EVALUATION_SEEDS):
    if condition not in CONDITIONS:
        raise ValueError("unknown DQN condition")
    config = DQNConfig() if config is None else config
    online = MLP(config.widths, seed)
    target = online.copy() if condition in ("full", "no_replay") else online
    optimizer = Adam(online.theta, config.eta)
    rng = np.random.default_rng(np.random.SeedSequence([seed, 2]))
    replay_rng = np.random.default_rng(np.random.SeedSequence([seed, 5]))
    replay = condition in ("full", "no_target")
    buffer = ReplayBuffer(config.capacity if replay else config.batch_size)
    env, episode = CartPole(), 0
    training_episode_seeds = [seed * 1_000_000]
    state = env.reset(training_episode_seeds[-1])
    curve = []

    def measure(step):
        rewards = evaluate(greedy_policy(online), episode_seeds)
        curve.append(dict(step=step, returns=rewards, mean=float(np.mean(rewards))))

    measure(0)
    for step in range(1, budget + 1):
        epsilon = 1 - (1 - config.epsilon_end) * min(1, (step - 1) / config.epsilon_steps)
        action = (
            int(rng.integers(2))
            if rng.random() < epsilon
            else int(np.argmax(online(state / STATE_SCALE)[0]))
        )
        next_state, reward, terminated, truncated = env.step(action)
        buffer.add(state / STATE_SCALE, action, reward, next_state / STATE_SCALE, terminated)
        if step % config.batch_size == 0:
            batch = (
                buffer.sample(config.batch_size, replay_rng)
                if replay
                else buffer.consecutive_batch()
            )
            dqn_update(online, target, batch, optimizer, config.gamma, config.refresh)
            if not replay:
                buffer.size = buffer.cursor = 0
        state = next_state
        if terminated or truncated:
            episode += 1
            training_episode_seeds.append(seed * 1_000_000 + episode)
            state = env.reset(training_episode_seeds[-1])
        if step % config.evaluation_every == 0 or step == budget:
            measure(step)
    return dict(
        network=online,
        seed=seed,
        condition=condition,
        settings={**asdict(config), "widths": list(config.widths), "budget": budget},
        updates=optimizer.steps,
        curve=curve,
        evaluation_seeds=list(episode_seeds),
        training_episode_seeds=training_episode_seeds,
    )
