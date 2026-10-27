"""REINFORCE with reward-to-go, gamma**t, and an independently fitted baseline."""

import numpy as np

from demos.deep_rl.cartpole import STATE_SCALE, CartPole
from demos.deep_rl.mlp import MLP, backward, forward
from demos.deep_rl_demo import reinforce_gradient as reinforce_gradient
from demos.deep_rl_demo import softmax_score

GAMMA = 0.99
BASELINE_SEEDS = tuple(range(1_100_000_000, 1_100_000_400))
VARIANCE_SEEDS = tuple(range(1_200_000_000, 1_200_002_000))


def softmax(scores):
    scores = np.asarray(scores, dtype=float)
    shifted = scores - scores.max(axis=-1, keepdims=True)
    p = np.exp(shifted)
    return p / p.sum(axis=-1, keepdims=True)


def discounted_returns(rewards, gamma=GAMMA):
    result = np.empty(len(rewards))
    future = 0.0
    for t in range(len(rewards) - 1, -1, -1):
        future = rewards[t] + gamma * future
        result[t] = future
    return result


class SoftmaxPolicy:
    def __init__(self, seed=37, widths=(4, 16, 16, 2)):
        self.network = MLP(widths, seed)

    def probabilities(self, states):
        return softmax(self.network(np.asarray(states) / STATE_SCALE))

    def copy(self):
        other = object.__new__(SoftmaxPolicy)
        other.network = self.network.copy()
        return other


def rollout(policy, seed, gamma=GAMMA):
    env = CartPole()
    state = env.reset(seed)
    rng = np.random.default_rng(np.random.SeedSequence([seed, 3]))
    states, actions, probabilities, rewards = [], [], [], []
    while not env.finished:
        p = policy.probabilities(state)[0]
        action = int(rng.choice(2, p=p))
        states.append(state)
        actions.append(action)
        probabilities.append(p)
        state, reward, _, _ = env.step(action)
        rewards.append(reward)
    return dict(
        states=np.array(states),
        actions=np.array(actions),
        probabilities=np.array(probabilities),
        rewards=np.array(rewards),
        returns=discounted_returns(rewards, gamma),
        seed=seed,
    )


def value_features(states):
    X = np.atleast_2d(np.asarray(states) / STATE_SCALE)
    return np.column_stack((np.ones(len(X)), X, X**2))


class StateValueBaseline:
    def __init__(self, weights, fit_seeds):
        self.weights, self.fit_seeds = weights, tuple(fit_seeds)

    def __call__(self, states):
        return value_features(states) @ self.weights


def fit_baseline(policy, episode_seeds=BASELINE_SEEDS, gamma=GAMMA, ridge=1e-3):
    episodes = [rollout(policy, seed, gamma) for seed in episode_seeds]
    X = value_features(np.concatenate([e["states"] for e in episodes]))
    y = np.concatenate([e["returns"] for e in episodes])
    weights = np.linalg.solve(X.T @ X + ridge * np.eye(X.shape[1]), X.T @ y)
    return StateValueBaseline(weights, episode_seeds)


def episode_gradient(policy, episode, baseline=None, gamma=GAMMA):
    states = episode["states"]
    f = forward(states / STATE_SCALE, policy.network.theta)
    b = np.zeros(len(states)) if baseline is None else baseline(states)
    weights = gamma ** np.arange(len(states)) * (episode["returns"] - b)
    grad_z = weights[:, None] * softmax_score(episode["probabilities"], episode["actions"])
    gradients = backward(f, policy.network.theta, grad_z)
    return np.concatenate([gradients[k].ravel() for k in policy.network.theta])


def gradient_variance(policy, episodes, baseline=None, gamma=GAMMA):
    if baseline is not None and set(baseline.fit_seeds).intersection(e["seed"] for e in episodes):
        raise ValueError("baseline fit and gradient-estimation episodes must be disjoint")
    gradients = np.array([episode_gradient(policy, e, baseline, gamma) for e in episodes])
    # b3[0], the first action's output bias, is the penultimate flattened entry.
    component = gradients[:, -2]
    return dict(
        mean=float(component.mean()),
        standard_error=float(component.std(ddof=1) / np.sqrt(len(component))),
        standard_deviation=float(component.std(ddof=1)),
        total_variance=float(gradients.var(axis=0, ddof=1).sum()),
        episodes=len(episodes),
        component="b3[0]",
    )


def variance_experiment():
    policy = SoftmaxPolicy()
    baseline = fit_baseline(policy)
    episodes = [rollout(policy, seed) for seed in VARIANCE_SEEDS]
    plain = gradient_variance(policy, episodes)
    fitted = gradient_variance(policy, episodes, baseline)
    return dict(
        policy_seed=37,
        gamma=GAMMA,
        widths=list(policy.network.widths),
        baseline_seeds=list(BASELINE_SEEDS),
        estimation_seeds=list(VARIANCE_SEEDS),
        ridge=1e-3,
        plain=plain,
        fitted=fitted,
        total_variance_ratio=fitted["total_variance"] / plain["total_variance"],
    )
