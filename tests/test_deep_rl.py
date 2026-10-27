"""Numerical contracts, chapter-order listings and seeded deep RL experiments."""

import contextlib
import importlib
import io
import itertools
import json
import math
import os
import runpy
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from demos import regions as _regions

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "demos/deep_rl_demo.py"
TOOLS = vars(_regions)
PREFIX = "fnd.deep-rl."
PRINTED = ("target-01", "dqn-01", "pg-01", "ppo-01")
OUTPUTS = (
    "target-01",
    "diverge-01",
    "runs-01",
    "pg-01",
    "variance-01",
    "variance-02",
    "ppo-01",
    "ppo-02",
    "batch-01",
    "depot-01",
    "depot-02",
    "keys-01",
    "offer-cells-01",
)


def module(name):
    return importlib.import_module("demos.deep_rl." + name)


def snippets():
    return TOOLS["extract_snippets_from_text"](DEMO.read_text(), DEMO)


def test_printed_listings_run_in_chapter_order_without_hidden_setup():
    namespace: dict[str, Any] = {"__name__": "__main__"}
    found = {s.snippet_id: s for s in snippets()}
    for name in PRINTED:
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(found[PREFIX + name].content, name, "exec"), namespace)
    assert namespace["continuing"]["target"] == 4.6
    assert namespace["ending"]["target"] == 1
    assert namespace["optimizer"].steps == 1
    assert namespace["probability"] > 0.6
    assert namespace["clipped_objective"](1.3, -2, 0.2) == -2.6


@pytest.mark.parametrize("name", ["pg-01", "ppo-01", "offer-cells-01"])
def test_graduate_path_listings_run_independently_without_hidden_setup(name):
    namespace: dict[str, Any] = {"__name__": "__main__"}
    found = {s.snippet_id: s for s in snippets()}
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        exec(compile(found[PREFIX + name].content, name, "exec"), namespace)
    stored = ROOT / "book_outputs" / f"_{PREFIX}{name}.txt"
    assert stdout.getvalue().encode() == stored.read_bytes()


@pytest.mark.parametrize(
    "ended,expected", [(False, (4.6, 2.6, 3.38, 2.26)), (True, (1, -1, 0.5, 1.9))]
)
def test_hand_target_error_loss_and_one_hot_step(ended, expected):
    dqn = module("dqn")
    result = dqn.semi_gradient_step([2, 8], [1, 0], dqn.td_target(1, ended, [3, 4], 0.9), 0.1)
    np.testing.assert_allclose(
        [result["target"], result["error"], result["loss"], result["weights"][0]], expected
    )
    assert result["weights"][1] == 8


@pytest.mark.parametrize(
    "ended,expected",
    [
        (False, (-7.4, 0.6, 0.18, -7.94)),
        (True, (-2, 6, 18, -7.4)),
    ],
)
def test_incident_exercise_target_error_loss_and_step(ended, expected):
    dqn = module("dqn")
    result = dqn.semi_gradient_step([-8], [1], dqn.td_target(-2, ended, [-10, -6], 0.9), 0.1)
    np.testing.assert_allclose(
        [result["target"], result["error"], result["loss"], result["weights"][0]], expected
    )


def test_hospital_exercise_step_baseline_and_ppo_cases():
    pg, ppo = module("policy_gradient"), module("ppo")
    p = pg.softmax([1, 0])
    score = pg.softmax_score(p, 0)
    np.testing.assert_allclose(score, [1 / (1 + math.e), -1 / (1 + math.e)])
    step = 0.5 * pg.reinforce_gradient(p[None, :], [0], [4], baseline=3)
    assert pg.softmax(np.array([1, 0]) + step)[0] == pytest.approx(0.781, abs=0.0005)
    plain = 0.5 * pg.reinforce_gradient(p[None, :], [0], [4])
    assert pg.softmax(np.array([1, 0]) + plain)[0] == pytest.approx(0.889, abs=0.0005)
    assert ppo.clipped_objective(1.25, 1) == pytest.approx(1.2)
    assert ppo.clipped_slope(1.25, 1) == 0
    assert ppo.clipped_objective(1.25, -1) == pytest.approx(-1.25)
    assert ppo.clipped_slope(1.25, -1) == -1


def test_terminal_and_time_limit_are_separate_and_only_terminal_masks_target():
    cart, dqn = module("cartpole"), module("dqn")
    env = cart.CartPole(limit=1)
    env.reset(7)
    _, reward, terminal, truncated = env.step(1)
    assert truncated and not terminal
    assert dqn.td_target(reward, terminal, [3, 4], 0.9) == 4.6
    env.reset(7)
    env.state = np.array([cart.POSITION_LIMIT + 0.1, 0, 0, 0])
    _, reward, terminal, truncated = env.step(1)
    assert terminal and not truncated
    assert dqn.td_target(reward, terminal, [3, 4], 0.9) == 1
    with pytest.raises(ValueError, match="reset"):
        env.step(1)


@pytest.mark.parametrize("state,action", [([0, 0, 0, 0], 1), ([0.1, -0.2, 0.05, 0.3], 0)])
def test_cartpole_step_against_documented_equations(state, action):
    cart = module("cartpole")
    env = cart.CartPole()
    env.reset(7)
    env.state = np.array(state, dtype=float)
    x, velocity, angle, omega = state
    # Independent scalar evaluation of the documented force elimination.
    mass = 1.1
    intermediate = (10 * (2 * action - 1) + 0.05 * omega**2 * math.sin(angle)) / mass
    angular = (9.8 * math.sin(angle) - math.cos(angle) * intermediate) / (
        0.5 * (4 / 3 - 0.1 * math.cos(angle) ** 2 / mass)
    )
    linear = intermediate - 0.05 * angular * math.cos(angle) / mass
    expected = [
        x + 0.02 * velocity,
        velocity + 0.02 * linear,
        angle + 0.02 * omega,
        omega + 0.02 * angular,
    ]
    np.testing.assert_allclose(env.step(action)[0], expected, rtol=1e-13, atol=1e-14)
    if state == [0, 0, 0, 0]:
        np.testing.assert_allclose(expected, [0, 0.1951219512195122, 0, -0.2926829268292683])


def test_cartpole_seed_and_validation():
    env = module("cartpole").CartPole()
    first = env.reset(7)
    np.testing.assert_array_equal(first, env.reset(7))
    assert not np.array_equal(first, env.reset(11))
    with pytest.raises(ValueError, match="action"):
        env.step(2)


def test_network_backward_agrees_with_central_differences():
    mlp = module("mlp")
    network = mlp.MLP((4, 5, 3, 2), 7)
    X = np.random.default_rng(19).normal(size=(3, 4))
    targets = np.array([[2, -1], [1, 3], [-1, 0]])
    assert mlp.gradient_check(network, X, targets) < 1e-8


def test_adam_is_chapter_tens_per_array_update():
    mlp = module("mlp")
    from demos.training_control_demo import adam

    X, y = np.array([[2, 0.2], [2, -0.2], [-2, 0.2], [-2, -0.2]]), np.array([2, 0, 0, -2])
    theta = {"w": np.zeros(2)}
    optimizer = mlp.Adam(theta, eta=0.1)
    path = [theta["w"].copy()]
    hessian, target = X.T @ X / len(y), X.T @ y / len(y)
    for _ in range(10):
        optimizer.step(theta, {"w": hessian @ theta["w"] - target})
        path.append(theta["w"].copy())
    np.testing.assert_array_equal(path, adam(X, y, 0.1, 10))


@pytest.mark.parametrize("same_values", [True, False])
def test_frozen_target_gradient_is_ordinary_even_when_copies_have_equal_values(same_values):
    mlp = module("mlp")
    online = mlp.MLP((2, 3, 3, 2), 7)
    target = online.copy() if same_values else mlp.MLP((2, 3, 3, 2), 11)
    X, next_X = np.array([[0.7, -0.4], [0.2, 0.3]]), np.array([[0.6, 0.8], [0.9, -0.3]])
    actions = np.array([0, 1])
    y = module("dqn").td_target([1, -2], [False, True], target(next_X), 0.9)
    f = mlp.forward(X, online.theta)
    grad_z = np.zeros_like(f["z"])
    grad_z[np.arange(2), actions] = (f["z"][np.arange(2), actions] - y) / 2
    analytic = mlp.backward(f, online.theta, grad_z)
    numerical = mlp.central_difference(
        lambda theta: np.mean((mlp.forward(X, theta)["z"][np.arange(2), actions] - y) ** 2) / 2,
        online.theta,
    )
    for name in analytic:
        np.testing.assert_allclose(analytic[name], numerical[name], atol=1e-8)
        assert not np.shares_memory(online.theta[name], target.theta[name])


def test_moving_target_full_gradient_includes_the_term_semigradient_drops():
    # Q(first)=theta, Q(next)=2*theta. delta=0.8*theta; d(delta)/dtheta=0.8.
    theta = {"weight": np.array([10.0])}
    mlp = module("mlp")
    estimate = mlp.central_difference(
        lambda w: 0.5 * (0.9 * 2 * w["weight"][0] - w["weight"][0]) ** 2, theta
    )
    semi = -8.0
    omitted = 8 * 1.8
    assert estimate["weight"][0] == pytest.approx(semi + omitted)
    assert estimate["weight"][0] != pytest.approx(semi)


def test_target_ignores_online_changes_until_the_scheduled_refresh():
    mlp, dqn = module("mlp"), module("dqn")
    online = mlp.MLP((4, 4, 3, 2), 7)
    target = online.copy()
    optimizer = mlp.Adam(online.theta)
    buffer = dqn.ReplayBuffer(8)
    buffer.add([1, 0, 0, 0], 0, 1, [0, 1, 0, 0], False)
    batch = buffer.sample(4, np.random.default_rng(7))
    original = target(batch["next_state"]).copy()
    dqn.dqn_update(online, target, batch, optimizer, refresh=2)
    np.testing.assert_array_equal(target(batch["next_state"]), original)
    assert any(not np.array_equal(online.theta[k], target.theta[k]) for k in online.theta)
    dqn.dqn_update(online, target, batch, optimizer, refresh=2)
    for name in online.theta:
        np.testing.assert_array_equal(online.theta[name], target.theta[name])


def test_buffer_has_exact_uniform_index_probabilities_and_seeded_sampling():
    buffer = module("dqn").ReplayBuffer(3)
    for i in range(5):
        buffer.add(np.full(4, i), i % 2, i, np.full(4, i + 1), i == 4)
    np.testing.assert_array_equal(buffer.index_probabilities(), np.ones(3) / 3)
    sampled = buffer.sample(30_000, np.random.default_rng(7))
    indices = np.random.default_rng(7).integers(3, size=30_000)
    np.testing.assert_array_equal(sampled["indices"], indices)
    assert np.max(np.abs(np.bincount(indices) / len(indices) - 1 / 3)) < 0.01
    assert set(sampled["reward"]) == {2, 3, 4}
    with pytest.raises(ValueError, match="empty"):
        module("dqn").ReplayBuffer(1).sample(1, np.random.default_rng(7))


def test_two_state_factors_and_within_period_losses():
    two = module("two_state")
    off, mix = two.off_policy_updates(), two.on_policy_rounds()
    np.testing.assert_allclose(off["weights"][:3], [10, 10.8, 11.664])
    np.testing.assert_allclose(off["errors"][:2], [8, 8.64])
    assert off["factor"] == pytest.approx(1.08)
    assert mix["terminal_factor"] == pytest.approx(0.6)
    assert mix["factor"] == pytest.approx(0.648)
    assert two.return_target_updates()["factor"] == pytest.approx(0.9)
    held = two.held_target_refreshes()
    np.testing.assert_allclose(held["weights"], [10, 18, 32.4, 58.32])
    assert held["factor"] == pytest.approx(1.8)
    np.testing.assert_allclose([r["start_loss"] for r in held["periods"]], [32, 103.68, 335.9232])
    assert all(r["end_loss"] < 1e-20 for r in held["periods"])
    assert two.held_target_refreshes(mix=True)["factor"] == pytest.approx(0.36)
    np.testing.assert_allclose(
        two.held_target_refreshes(mix=True)["weights"], [10, 3.6, 1.296, 0.46656]
    )


def test_maximization_bias_matches_exact_value_within_its_standard_error():
    result = module("two_state").max_bias()
    assert result["exact"] == 1 / math.sqrt(math.pi)
    assert abs(result["simulated"] - result["exact"]) < result["standard_error"]


def test_softmax_score_and_baseline_cancel_by_enumeration():
    from fractions import Fraction

    pg = module("policy_gradient")
    p = np.array([0.6, 0.4])
    np.testing.assert_allclose(pg.softmax_score(p, 0), [0.4, -0.4])
    np.testing.assert_allclose(pg.softmax_score(p, 1), [-0.6, 0.6])
    baseline = sum(p[a] * 3 * pg.softmax_score(p, a) for a in (0, 1))
    rational = [Fraction(3, 5), Fraction(2, 5)]
    exact = [sum(rational[a] * 3 * ((a == i) - rational[i]) for a in (0, 1)) for i in (0, 1)]
    assert exact == [0, 0]
    np.testing.assert_allclose(baseline, [0.0, 0.0], atol=float(np.finfo(float).eps), rtol=0)
    gradient = pg.reinforce_gradient(p[None, :], [0], [5], baseline=3)
    np.testing.assert_allclose(0.1 * gradient, [0.08, -0.08])
    expected = 0.6 * math.exp(0.16) / (0.6 * math.exp(0.16) + 0.4)
    assert pg.softmax(np.log(p) + 0.1 * gradient)[0] == pytest.approx(expected)


def test_reinforce_mean_equals_exact_two_step_gradient_with_gamma_power():
    pg = module("policy_gradient")
    logits = np.array([0.3, -0.2])
    gamma, baseline = 0.7, np.array([1.2, -0.3])
    p = pg.softmax(logits)
    expected = np.zeros(2)

    # The first reward and the second transition both depend on actions, but
    # their probabilities and the rewards have no trainable parameters.
    def rewards(a, b):
        return np.array([2 * a - 1, 3 * (a == b) - b], dtype=float)

    def objective(z):
        q = pg.softmax(z)
        return sum(
            q[a] * q[b] * (rewards(a, b)[0] + gamma * rewards(a, b)[1])
            for a, b in itertools.product((0, 1), repeat=2)
        )

    for a, b in itertools.product((0, 1), repeat=2):
        expected += (
            p[a]
            * p[b]
            * pg.reinforce_gradient(np.tile(p, (2, 1)), [a, b], rewards(a, b), gamma, baseline)
        )
    exact = np.array(
        [
            (objective(logits + np.eye(2)[i] * 1e-5) - objective(logits - np.eye(2)[i] * 1e-5))
            / 2e-5
            for i in range(2)
        ]
    )
    np.testing.assert_allclose(expected, exact, atol=1e-10)
    # Dropping gamma**t does change this discounted objective's mean gradient.
    wrong = np.zeros(2)
    for a, b in itertools.product((0, 1), repeat=2):
        returns = pg.discounted_returns(rewards(a, b), gamma)
        wrong += (
            p[a]
            * p[b]
            * np.sum(
                (returns - baseline)[:, None] * pg.softmax_score(np.tile(p, (2, 1)), [a, b]), axis=0
            )
        )
    assert not np.allclose(wrong, exact)


def test_poor_baseline_can_increase_variance_without_changing_mean():
    pg = module("policy_gradient")
    plain = np.array([pg.softmax_score([0.5, 0.5], a)[0] for a in (0, 1)])
    poor = plain * (1 - 3)
    assert plain.mean() == poor.mean() == 0
    assert plain.var() == 0.25
    assert poor.var() == 1


def test_baseline_fit_is_separate_and_frozen_for_gradient_estimation():
    pg = module("policy_gradient")
    policy = pg.SoftmaxPolicy()
    fitted = pg.fit_baseline(policy, (101, 102, 103))
    weights = fitted.weights.copy()
    estimates = [pg.rollout(policy, s) for s in (201, 202, 203)]
    result = pg.gradient_variance(policy, estimates, fitted)
    assert result["episodes"] == 3 and result["total_variance"] > 0
    np.testing.assert_array_equal(weights, fitted.weights)
    with pytest.raises(ValueError, match="disjoint"):
        pg.gradient_variance(policy, [pg.rollout(policy, 101)], fitted)


@pytest.mark.parametrize(
    "advantage,ratio,objective,slope",
    [
        (2, 1.3, 2.4, 0),
        (2, 0.7, 1.4, 2),
        (-2, 1.3, -2.6, -2),
        (-2, 0.7, -1.6, 0),
    ],
)
def test_ppo_four_cases_and_their_slopes(advantage, ratio, objective, slope):
    ppo = module("ppo")
    assert ppo.clipped_objective(ratio, advantage, 0.2) == pytest.approx(objective)
    assert ppo.clipped_slope(ratio, advantage, 0.2) == slope
    numerical = (
        ppo.clipped_objective(ratio + 1e-6, advantage)
        - ppo.clipped_objective(ratio - 1e-6, advantage)
    ) / 2e-6
    assert numerical == pytest.approx(slope, abs=1e-9)


def test_ppo_reuses_a_frozen_batch_and_ratios_pass_the_clip_band():
    ppo = module("ppo")
    batch = ppo.make_batch()
    original = {k: v.copy() for k, v in batch.items() if isinstance(v, np.ndarray)}
    clipped, plain = ppo.ppo_epochs(batch), ppo.ppo_epochs(batch, clipped=False)
    assert clipped["epochs"][0] == plain["epochs"][0]
    assert clipped["epochs"][-1]["outside_share"] > 0
    assert clipped["epochs"][-1]["largest_ratio"] > 1.2
    assert clipped["epochs"][-1]["mean_kl"] > 0
    for name, value in original.items():
        np.testing.assert_array_equal(batch[name], value)


def test_depot_exact_policy_matches_independent_tiny_tree_enumeration():
    depot = module("depot")
    config = depot.DepotConfig(
        periods=3, arrival_mean=0.4, trip_cost=4, capacity=2, waiting_cap=3, courier_cost=2
    )
    policy = depot.exact_policy(config)
    # Independent transition and chance tree; >30 arrivals have <1e-44 mass.
    masses = [math.exp(-0.4) * 0.4**a / math.factorial(a) for a in range(31)]

    def branch(waiting, left, action, arrivals):
        take = min(waiting, 2) if action else 0
        remaining = waiting - take
        cost = (4 if take else 0) + remaining
        following = remaining + arrivals
        cost += 2 * max(0, following - 3)
        following = min(following, 3)
        if left == 1:
            return cost + 4 * math.ceil(following / 2)
        return cost + best(following, left - 1)[0]

    from functools import lru_cache

    @lru_cache(None)
    def best(waiting, left):
        costs = [
            sum(p * branch(waiting, left, action, a) for a, p in enumerate(masses))
            for action in (0, 1)
        ]
        return min(costs), int(np.argmin(costs))

    for left, waiting in itertools.product(range(1, 4), range(4)):
        value, action = best(waiting, left)
        assert policy.values[left, waiting] == pytest.approx(value, abs=1e-11)
        assert policy((waiting, left)) == action


def test_shift_arrivals_are_fixed_before_the_policy_acts():
    depot = module("depot")

    def always_wait(state):
        return 0

    first = depot.simulate_shift(always_wait, 7)
    assert first == depot.simulate_shift(always_wait, 7)
    assert first != depot.simulate_shift(always_wait, 11)
    config = depot.DepotConfig(periods=2, waiting_cap=3, capacity=2)
    state, cost, terminal = depot.transition((3, 1), 0, 2, config)
    assert state == (0, 0) and terminal
    assert cost == 3 + 10 + 80


def test_agent_report_is_selected_only_from_reproducible_training_estimates():
    depot = module("depot")
    first, second = depot.agent_report(7), depot.agent_report(7)
    assert first["estimated_cost"] == second["estimated_cost"]
    assert first["updates"] == second["updates"]
    assert first["checkpoint_estimates"] == second["checkpoint_estimates"]
    assert first["estimated_cost"] == min(
        r["estimated_cost"] for r in first["checkpoint_estimates"]
    )
    for name in first["network"].theta:
        np.testing.assert_array_equal(first["network"].theta[name], second["network"].theta[name])


def test_adoption_rejects_four_of_five_and_shared_noise_and_requires_paired_intervals():
    depot = module("depot")
    baseline = np.full(200, 100.0)
    wins = (baseline - 2).tolist()
    assert not depot.adopt(dict(baseline_costs=baseline, seeds=[wins] * 4))
    assert not depot.adopt(dict(baseline_costs=baseline, seeds=[wins] * 4 + [baseline.tolist()]))
    shared_noise = np.tile([-10.0, 11.0], 100)
    noisy = (baseline - shared_noise).tolist()
    assert np.mean(shared_noise) > 0
    assert depot.paired_interval(shared_noise)[0] < 0
    assert not depot.adopt(dict(baseline_costs=baseline, seeds=[noisy] * 5))
    assert depot.adopt(dict(baseline_costs=baseline, seeds=[wins] * 5))


def test_dqn_seed_is_deterministic_and_evaluation_does_not_influence_training():
    dqn = module("dqn")
    first = dqn.train_dqn(7, budget=512, episode_seeds=(100_001, 100_002))
    second = dqn.train_dqn(7, budget=512, episode_seeds=(100_001, 100_002))
    other_evaluation = dqn.train_dqn(7, budget=512, episode_seeds=(200_001, 200_002))
    assert first["curve"] == second["curve"]
    assert first["training_episode_seeds"] == other_evaluation["training_episode_seeds"]
    for name in first["network"].theta:
        np.testing.assert_array_equal(first["network"].theta[name], second["network"].theta[name])
        np.testing.assert_array_equal(
            first["network"].theta[name], other_evaluation["network"].theta[name]
        )
    assert not set(first["training_episode_seeds"]).intersection(first["evaluation_seeds"])


def test_depot_training_and_report_never_consult_held_out_shifts(monkeypatch):
    depot = module("depot")
    original = depot.simulate_shift

    def guarded(policy, seed, *args, **kwargs):
        assert seed not in depot.SHIFT_SEEDS, "training/report read held-out shifts"
        return original(policy, seed, *args, **kwargs)

    monkeypatch.setattr(depot, "simulate_shift", guarded)
    report = depot.agent_report(7)
    assert report["baseline_seeds"] == list(range(7_000_000, 7_000_100))
    assert report["baseline_cost"] > report["estimated_cost"]


def test_chapter_never_reads_chapter_four_or_stored_outputs_as_inputs():
    # Exercise every computational path with an audit hook. The separate full
    # capture checks the 20 long runs; this guard uses a short DQN run so the
    # default suite spends its time on one complete capture, not repeated ones.
    code = """
import contextlib, io, sys
def audit(event, args):
    if event == 'open' and isinstance(args[0], (str, bytes)):
        path = str(args[0]).replace('\\\\', '/')
        assert '/book_outputs/' not in path and 'deep_rl_figures.json' not in path, path
        assert '/refund-history/' not in path and '/refund-data/' not in path, path
        assert not (('/refund_agent/' in path) and path.endswith(('.json', '.csv', '.jsonl'))), path
    if event in ('socket.connect', 'socket.getaddrinfo'):
        raise AssertionError('chapter attempted network access')
sys.addaudithook(audit)
from refund_agent.data import generate
from refund_agent import split
def forbidden(*args, **kwargs):
    raise AssertionError('chapter touched chapter four data')
generate.generate = forbidden
split.freeze = split.time_split = forbidden
with contextlib.redirect_stdout(io.StringIO()):
    from demos import deep_rl_demo as demo
    from demos.deep_rl import dqn, policy_gradient, ppo, depot
    dqn.train_dqn(7, budget=1024)
    policy_gradient.variance_experiment()
    ppo.batch_experiment()
    depot.depot_experiment()
    demo.exercise_fixture()
"""
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, result.stderr


def test_import_is_silent_and_does_not_write_files(tmp_path):
    code = "import demos.deep_rl_demo; import demos.deep_rl.dqn; import demos.deep_rl.policy_gradient; import demos.deep_rl.ppo; import demos.deep_rl.depot"
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=tmp_path, capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == ""
    assert list(tmp_path.iterdir()) == []


@pytest.fixture(scope="module")
def captured():
    return TOOLS["capture_outputs"](
        {s.snippet_id: s for s in snippets()}, sys.executable, timeout=120
    )


@pytest.mark.parametrize("name", OUTPUTS)
def test_every_stored_output_regenerates_byte_for_byte(name, captured):
    stored = ROOT / "book_outputs" / f"_{PREFIX}{name}.txt"
    assert captured[PREFIX + name].encode() == stored.read_bytes()


@pytest.mark.slow
@pytest.mark.skipif(
    os.environ.get("DEEP_RL_SLOW") != "1", reason="set DEEP_RL_SLOW=1 for repeated full capture"
)
def test_capture_is_deterministic_in_a_fresh_process(captured):
    repeated = TOOLS["capture_outputs"](
        {s.snippet_id: s for s in snippets()}, sys.executable, timeout=120
    )
    assert captured == repeated


def test_output_regions_are_in_outline_order_and_printed_functions_are_canonical():
    regions = snippets()
    assert [s.snippet_id.removeprefix(PREFIX) for s in regions if s.output] == list(OUTPUTS)
    demo = importlib.import_module("demos.deep_rl_demo")
    dqn, pg, ppo = module("dqn"), module("policy_gradient"), module("ppo")
    assert demo.td_target is dqn.td_target
    assert demo.dqn_update is dqn.dqn_update
    assert demo.reinforce_gradient is pg.reinforce_gradient
    assert demo.softmax_score is pg.softmax_score
    assert demo.clipped_objective is ppo.clipped_objective


def test_every_plotted_result_matches_fresh_capture_and_conditions_keep_the_same_update_count(
    captured,
):
    demo = importlib.import_module("demos.deep_rl_demo")
    path = ROOT / "demos/data/deep_rl_figures.json"
    generated = json.loads(path.read_text())
    printed = io.StringIO()
    with contextlib.redirect_stdout(printed):
        demo.print_runs(generated["cartpole"])
    assert printed.getvalue() == captured[PREFIX + "runs-01"]
    assert generated["two_state"] == demo.figure_data(None)["two_state"]
    rows = generated["cartpole"]["runs"]
    assert len(rows) == 20
    assert len({r["updates"] for r in rows}) == 1
    for row in rows:
        assert row["settings"]["budget"] == 50_000
        assert [p["step"] for p in row["curve"]] == list(range(0, 50_001, 2_500))
        assert not set(row["training_episode_seeds"]).intersection(row["evaluation_seeds"])
        assert all(len(p["returns"]) == 10 for p in row["curve"])


def test_exercise_fixture_regenerates_and_contains_only_the_planned_clip_error():
    demo = importlib.import_module("demos.deep_rl_demo")
    fixture = json.loads((ROOT / "exercises/deep-rl/ppo-05.json").read_text())
    assert fixture == demo.exercise_fixture()
    correct = module("ppo").clipped_objective
    differences = []
    for row in fixture["worked_cases"]:
        ratio, advantage = row["ratio"], row["advantage"]
        assert row["agent_objective"] == np.clip(ratio, 0.8, 1.2) * advantage
        if not math.isclose(row["agent_objective"], correct(ratio, advantage)):
            differences.append((ratio, advantage))
    assert differences == [(0.7, 2), (1.3, -2)]
    assert len(fixture["epochs"]) == 10
    assert any(not 0.8 <= r["ratio_after_ten_epochs"] <= 1.2 for r in fixture["samples"])


@pytest.fixture
def double_learning_report():
    dqn = module("dqn")
    lengths = [200] + [499] * 9
    gamma = dqn.DQNConfig().gamma
    original = np.mean([sum(gamma**t for t in range(length)) for length in lengths])
    improved = np.mean([sum(gamma**t for t in range(length + 1)) for length in lengths])
    return dict(
        training_seeds=list(dqn.TRAINING_SEEDS),
        evaluation_seeds=list(dqn.EVALUATION_SEEDS),
        settings=dqn.DQNConfig().__dict__,
        runs=[
            dict(
                seed=s,
                original_returns=lengths.copy(),
                double_returns=[length + 1 for length in lengths],
                original_starting_estimate=float(original),
                double_starting_estimate=float(improved),
            )
            for s in dqn.TRAINING_SEEDS
        ],
    )


def test_double_learning_exercise_acceptance_checks_distinct_actions_and_paired_report(
    double_learning_report,
):
    acceptance = runpy.run_path(str(ROOT / "exercises/deep-rl/acceptance.py"))

    def correct(reward, terminated, online, target, gamma):
        values = target[np.arange(len(target)), np.argmax(online, axis=1)]
        return np.where(terminated, reward, reward + gamma * values)

    acceptance["check_target"](correct)
    with pytest.raises(AssertionError):
        acceptance["check_target"](lambda r, t, o, q, g: module("dqn").td_target(r, t, q, g))
    result = acceptance["check_report"](double_learning_report)
    assert result["adopt"]
    for row in result["estimates"]:
        assert row["original_realized"] == pytest.approx(row["original_estimate"])
        assert row["double_realized"] == pytest.approx(row["double_estimate"])
    # A one-push improvement has a zero-width positive paired interval in raw
    # lengths. Discounting the mixed lengths would make that interval cross zero.
    double_learning_report["runs"][-1]["double_returns"] = double_learning_report["runs"][-1][
        "original_returns"
    ].copy()
    assert not acceptance["check_report"](double_learning_report)["adopt"]
    starter = runpy.run_path(str(ROOT / "exercises/deep-rl/starter.py"))
    with pytest.raises(NotImplementedError, match="choose with the online network"):
        acceptance["check_target"](starter["double_target"])


def test_double_learning_correct_solution_passes_cli_and_starter_fails_for_its_todo(
    tmp_path,
    double_learning_report,
):
    candidate = tmp_path / "candidate.py"
    candidate.write_text(
        "import numpy as np\n\n"
        "def double_target(reward, terminated, online_next, target_next, gamma):\n"
        "    actions = np.argmax(online_next, axis=1)\n"
        "    values = target_next[np.arange(len(actions)), actions]\n"
        "    return np.where(terminated, reward, reward + gamma * values)\n"
    )
    report = tmp_path / "report.json"
    report.write_text(json.dumps(double_learning_report))
    command = [sys.executable, str(ROOT / "exercises/deep-rl/acceptance.py")]
    result = subprocess.run(command + [str(candidate), str(report)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    checked = json.loads(result.stdout)
    assert checked["adopt"]
    for row in checked["estimates"]:
        assert row["original_realized"] == pytest.approx(row["original_estimate"])
        assert row["double_realized"] == pytest.approx(row["double_estimate"])
    result = subprocess.run(
        command + [str(ROOT / "exercises/deep-rl/starter.py"), str(report)],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "NotImplementedError: choose with the online network" in result.stderr


@pytest.mark.parametrize("gamma,length", [(0.99, 200), (0.99, 500), (0.5, 3), (0, 10)])
def test_double_learning_realized_value_discounts_each_unit_reward(
    gamma,
    length,
    double_learning_report,
):
    double_learning_report["settings"]["gamma"] = gamma
    expected = sum(gamma**t for t in range(length))
    for row in double_learning_report["runs"]:
        row["original_returns"] = row["double_returns"] = [length] * 10
        row["original_starting_estimate"] = row["double_starting_estimate"] = expected
    acceptance = runpy.run_path(str(ROOT / "exercises/deep-rl/acceptance.py"))
    result = acceptance["check_report"](double_learning_report)
    assert not result["adopt"]
    for row in result["estimates"]:
        assert row["original_realized"] == pytest.approx(expected)
        assert row["double_realized"] == pytest.approx(expected)


def test_plots_render_in_the_book_formats_when_matplotlib_is_available(tmp_path, monkeypatch):
    pytest.importorskip("matplotlib")
    plot = runpy.run_path(str(ROOT.parent / "figures/plots/fig_deep_rl.py"))
    original_save = plot["save"]

    def check_axes_then_save(fig, name, out):
        for ax in fig.axes:
            assert ax.xaxis.label.get_color() == plot["SLATE"]
            assert ax.yaxis.label.get_color() == plot["SLATE"]
        original_save(fig, name, out)

    monkeypatch.setitem(original_save.__globals__, "save", check_axes_then_save)
    plot["main"](tmp_path)
    for name in ("cartpole-runs", "two-state"):
        for suffix in (".pdf", ".png", "-gray.png"):
            assert (tmp_path / f"fig-deep-rl-{name}{suffix}").stat().st_size > 0


@pytest.mark.slow
@pytest.mark.skipif(
    os.environ.get("DEEP_RL_SLOW") != "1", reason="set DEEP_RL_SLOW=1 for full-run proof"
)
def test_one_full_seed_run_regenerates_byte_for_byte():
    expected = json.loads((ROOT / "demos/data/deep_rl_figures.json").read_text())
    expected = expected["cartpole"]["runs"][0]
    actual = module("dqn").train_dqn(
        expected["seed"], expected["condition"], expected["settings"]["budget"]
    )
    actual = {k: v for k, v in actual.items() if k != "network"}
    assert (
        json.dumps(actual, sort_keys=True).encode() == json.dumps(expected, sort_keys=True).encode()
    )


@pytest.fixture(scope="module")
def offer_learning_run():
    from refund_agent.offers.evaluation import EVALUATION_START
    from refund_agent.offers.policies import Capped, EpsilonGreedy
    from refund_agent.offers.replay import replay, training_rows

    rows = [row for row in training_rows() if row.at < EVALUATION_START]
    return rows, replay(rows, Capped(EpsilonGreedy(0.3), 0.05), seed=7)


def test_offer_cells_partition_one_learning_replay_and_include_empty_cells(offer_learning_run):
    from collections import Counter

    cells = module("offer_cells")
    rows, run = offer_learning_run
    result = cells.count_offer_cells(rows, run.log.rows)
    by_id = {row.request_id: row for row in rows}
    first_at = {}
    for row in rows:
        first_at[row.requester_id] = min(row.at, first_at.get(row.requester_id, row.at))
    coarse, refined = Counter(), Counter()
    for sent in run.log.rows:
        row = by_id[sent.request_id]
        prior = first_at[row.requester_id] < row.at
        coarse[row.reason, sent.offer] += 1
        refined[row.reason, row.amount_cents >= 2500, prior, sent.offer] += 1
    assert len(result["coarse"]) == 9
    assert len(result["refined"]) == 36
    assert dict(+Counter(result["coarse"])) == dict(coarse)
    assert dict(+Counter(result["refined"])) == dict(refined)
    assert sum(result["coarse"].values()) == sum(result["refined"].values()) == len(rows)
    assert result["requests"] == len(rows)
    assert result["requesters"] == len({row.requester_id for row in rows})
    # This generator has unique requesters: the additional flag is unobserved.
    assert result["earlier_refunds"] == 0
    assert all(count == 0 for (_, _, prior, _), count in result["refined"].items() if prior)


def test_offer_earlier_refund_uses_strict_time_and_only_learning_rows(offer_learning_run):
    from dataclasses import replace
    from datetime import timedelta

    rows, run = offer_learning_run
    cells = module("offer_cells")
    first = rows[0]
    # Test-only transformations check recurrence and simultaneous requests;
    # the printed run always uses the original generated learning requests.
    simultaneous = replace(rows[1], requester_id=first.requester_id, at=first.at)
    later = replace(rows[2], requester_id=first.requester_id, at=first.at + timedelta(days=1))
    requests = [later, simultaneous, first]
    flags = cells.earlier_refund_flags(requests)
    assert flags == {first.request_id: False, simultaneous.request_id: False, later.request_id: True}
    assert cells.earlier_refund_flags([first]) == {first.request_id: False}
    sent = [decision for decision in run.log.rows if decision.request_id in flags]
    result = cells.count_offer_cells(requests, sent)
    assert result["requesters"] == 1
    assert result["earlier_refunds"] == 1
    counts_by_prior = {
        prior: sum(count for (_, _, flag, _), count in result["refined"].items() if flag == prior)
        for prior in (False, True)
    }
    assert counts_by_prior == {False: 2, True: 1}


@pytest.mark.parametrize("change", ["before", "evaluation", "unpaid", "tier2", "duplicate"])
def test_offer_cells_refuse_requests_outside_learning_scope(change, offer_learning_run):
    from dataclasses import replace
    from datetime import timedelta

    from refund_agent.offers.evaluation import EVALUATION_START
    from refund_agent.offers.replay import TRAIN_START

    rows, run = offer_learning_run
    first = rows[0]
    changes = {
        "before": replace(first, at=TRAIN_START - timedelta(seconds=1)),
        "evaluation": replace(first, at=EVALUATION_START),
        "unpaid": replace(first, paid=False),
        "tier2": replace(first, amount_cents=5000),
        "duplicate": first,
    }
    invalid = [*rows, first] if change == "duplicate" else [changes[change], *rows[1:]]
    with pytest.raises(ValueError, match="learning|unique"):
        module("offer_cells").count_offer_cells(invalid, run.log.rows)


@pytest.mark.parametrize("change", ["missing", "duplicate"])
def test_offer_cells_count_sent_decisions_once_not_maturity_events(change, offer_learning_run):
    rows, run = offer_learning_run
    sent = run.log.rows
    invalid = sent[:-1] if change == "missing" else (sent[0], *sent[:-1])
    with pytest.raises(ValueError, match="one sent notice"):
        module("offer_cells").count_offer_cells(rows, invalid)
    assert len(run.log.events) == 2 * len(sent)


def test_offer_logistic_design_has_seven_independent_coefficients(offer_learning_run):
    from dataclasses import replace

    cells = module("offer_cells")
    rows, _ = offer_learning_run
    design = [
        cells.logistic_features(replace(rows[0], reason=reason, amount_cents=amount), offer, prior)
        for reason, amount, prior, offer in itertools.product(
            cells.REASONS, (2499, 2500), (False, True), cells.OFFERS
        )
    ]
    assert np.asarray(design).shape == (36, 7)
    assert np.linalg.matrix_rank(design) == 7
    assert design[0] == (1, 0, 0, 0, 0, 0, 0)
    assert design[-1] == (1, 0, 1, 0, 1, 1, 1)


def test_offer_demo_reads_no_records_outputs_or_network():
    code = r'''
import sys
def audit(event, args):
    if event == "open" and isinstance(args[0], (str, bytes)):
        path = str(args[0]).replace("\\", "/")
        assert "/demos/data/" not in path and "/book_outputs/" not in path, path
        assert not ("/refund_agent/" in path and path.endswith((".json", ".csv", ".jsonl"))), path
    if event in ("socket.connect", "socket.getaddrinfo"):
        raise AssertionError("offer cells attempted network access")
sys.addaudithook(audit)
from demos.deep_rl.offer_cells import offer_cells
result = offer_cells()
assert sum(result["coarse"].values()) == result["requests"]
assert result["settings"] == {"seed": 7, "epsilon": 0.3, "cap": 0.05}
'''
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
