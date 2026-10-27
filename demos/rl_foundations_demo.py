"""Constructed late-parcel decisions for Reinforcement Learning Foundations.

Reward follows the action: a transition at step t carries R_{t+1}. The parcel
uses discount 1 because day 10 forces payment; the continuing one-state example
uses a discount below 1. No project history, frozen set or customer is read.
Imports do no work. --write-data writes plot data and the constructed exercise
report; marked stdout is captured by scripts/snippets/extract_snippets.py.
"""

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class LateParcel:
    lost_share: float = 0.1
    delivery_probability: float = 0.6
    last_day: int = 10
    amount: float = 49.99
    wait_cost: float = 3.0

    def __post_init__(self):
        if not 0 <= self.lost_share <= 1 or not 0 <= self.delivery_probability <= 1:
            raise ValueError("parcel probabilities must lie in [0, 1]")
        if not isinstance(self.last_day, int) or self.last_day < 1:
            raise ValueError("last day must be a positive integer")
        if (
            not np.isfinite([self.amount, self.wait_cost]).all()
            or min(self.amount, self.wait_cost) < 0
        ):
            raise ValueError("costs must be finite and nonnegative")

    def survival(self, s):
        if not isinstance(s, int) or not 0 <= s <= self.last_day:
            raise ValueError("day outside the constructed horizon")
        return self.lost_share + (1 - self.lost_share) * (1 - self.delivery_probability) ** s

    def lost_belief(self, s):
        surviving = self.survival(s)
        if surviving == 0:
            raise ValueError("no unscanned parcel can reach this day")
        return self.lost_share / surviving

    def scan_probability(self, s):
        return (1 - self.lost_belief(s)) * self.delivery_probability


@dataclass(frozen=True)
class FiniteMDP:
    states: tuple
    actions: dict
    transitions: dict
    rewards: dict
    terminals: tuple
    discount: float

    def __post_init__(self):
        if len(set(self.states)) != len(self.states) or not set(self.terminals) <= set(self.states):
            raise ValueError("states must be unique and include every terminal")
        if not 0 <= self.discount <= 1:
            raise ValueError("discount must lie in [0, 1]")
        for s in self.states:
            if s in self.terminals:
                if self.actions.get(s, ()):
                    raise ValueError("terminal states have no actions")
                continue
            if not self.actions.get(s):
                raise ValueError("each nonterminal state needs actions")
            for a in self.actions[s]:
                row = self.transitions[s, a]
                if not row or not np.isclose(sum(p for p, _ in row), 1, atol=1e-12, rtol=0):
                    raise ValueError("transition probabilities must sum to one")
                if any(
                    not np.isfinite(p) or p < 0 or next_s not in self.states for p, next_s in row
                ):
                    raise ValueError("invalid transition")
                if not np.isfinite(self.rewards[s, a]):
                    raise ValueError("expected rewards must be finite")


def parcel_mdp(model, scans=None):
    days = tuple(range(model.last_day + 1))
    scans = [model.scan_probability(s) for s in days[:-1]] if scans is None else scans
    if len(scans) != model.last_day:
        raise ValueError("one scan probability per waiting day is required")
    actions, transitions, rewards = {}, {}, {}
    for s in days:
        actions[s] = ("pay", "wait") if s < model.last_day else ("pay",)
        transitions[s, "pay"] = ((1.0, "paid"),)
        rewards[s, "pay"] = -model.amount
        if s < model.last_day:
            transitions[s, "wait"] = ((scans[s], "scanned"), (1 - scans[s], s + 1))
            rewards[s, "wait"] = -model.wait_cost
    return FiniteMDP(
        days + ("paid", "scanned"), actions, transitions, rewards, ("paid", "scanned"), 1.0
    )


def policy_evaluation(mdp, policy):
    """Solve the Bellman expectation equations for a fixed (possibly random) rule."""
    states = [s for s in mdp.states if s not in mdp.terminals]
    index = {s: i for i, s in enumerate(states)}
    matrix, rewards = np.eye(len(states)), np.zeros(len(states))
    for s in states:
        row = {policy[s]: 1.0} if isinstance(policy[s], str) else policy[s]
        if (
            not set(row) <= set(mdp.actions[s])
            or any(p < 0 for p in row.values())
            or not np.isclose(sum(row.values()), 1)
        ):
            raise ValueError("invalid decision policy")
        for a, chance in row.items():
            rewards[index[s]] += chance * mdp.rewards[s, a]
            for p, next_s in mdp.transitions[s, a]:
                if next_s in index:
                    matrix[index[s], index[next_s]] -= mdp.discount * chance * p
    solved = np.linalg.solve(matrix, rewards)
    return {**dict(zip(states, solved.tolist())), **dict.fromkeys(mdp.terminals, 0.0)}


def wait_rule(model):
    return {
        s: "wait"
        if s < model.last_day and model.scan_probability(s) * model.amount > model.wait_cost
        else "pay"
        for s in range(model.last_day + 1)
    }


def one_state_example(discount):
    if not 0 <= discount < 1:
        raise ValueError("the continuing example needs a discount below 1")
    return FiniteMDP(
        ("open", "finished"),
        {"open": ("finish", "wait")},
        {("open", "finish"): ((1.0, "finished"),), ("open", "wait"): ((1.0, "open"),)},
        {("open", "finish"): 3.0, ("open", "wait"): 1.0},
        ("finished",),
        discount,
    )


def first_pay_day(policy):
    return min(s for s, a in policy.items() if a == "pay")


def policy_cost_parts(model, k):
    survival = model.survival(k)
    days = sum(model.survival(s) for s in range(k))
    return {
        "cost": model.wait_cost * days + model.amount * survival,
        "waiting_days": days,
        "waiting_cost": model.wait_cost * days,
        "lost_refunds": model.amount * model.lost_share,
        "late_refunds": model.amount
        * (1 - model.lost_share)
        * (1 - model.delivery_probability) ** k,
    }


# region book:fnd.rl-foundations.model-01 output
if __name__ == "__main__":
    model = LateParcel()
    print(
        f"constructed parcel: lost {model.lost_share:.3f}, daily delivery {model.delivery_probability:.3f}"
    )
    print(
        f"amount {model.amount:.2f}, waiting price {model.wait_cost:.2f}, compulsory pay day {model.last_day}, discount 1"
    )
    print("day | lost belief | next-day scan probability")
    for s in range(7):
        print(f"{s} | {model.lost_belief(s):.3f} | {model.scan_probability(s):.3f}")
# endregion book:fnd.rl-foundations.model-01


# region book:fnd.rl-foundations.values-01 output
if __name__ == "__main__":
    model = LateParcel()
    print("pay day | expected claim cost")
    for k in range(8):
        rule = {s: "pay" if s >= k else "wait" for s in range(model.last_day + 1)}
        cost = -policy_evaluation(parcel_mdp(model), rule)[0]
        print(f"{k} | {cost:.2f}")
    parts = policy_cost_parts(model, 5)
    print(
        f"day 5: waiting days {parts['waiting_days']:.6f}, waiting cost {parts['waiting_cost']:.2f}, lost refunds {parts['lost_refunds']:.2f}, late refunds {parts['late_refunds']:.2f}"
    )
    print(
        f"no-scan path to day 5: waiting {5 * model.wait_cost:.2f}, payment {model.amount:.2f}, return {-5 * model.wait_cost - model.amount:.2f}"
    )
# endregion book:fnd.rl-foundations.values-01


# region book:fnd.rl-foundations.vi-01 output
if __name__ == "__main__":
    from demos import rl_foundations_demo


def bellman_backup(mdp, values, s):
    return {
        a: mdp.rewards[s, a]
        + mdp.discount
        * sum(p * values[next_s] for p, next_s in mdp.transitions[s, a])
        for a in mdp.actions[s]
    }


def value_iteration(mdp, tolerance=1e-12, max_sweeps=10000):
    values = dict.fromkeys(mdp.states, 0.0)
    history = [{"sweep": 0, "values": values.copy(), "policy": {}}]
    for sweep in range(1, max_sweeps + 1):
        updated, policy = dict.fromkeys(mdp.terminals, 0.0), {}
        for s in mdp.states:
            if s in mdp.terminals:
                continue
            q = bellman_backup(mdp, values, s)
            policy[s] = max(q, key=q.get)
            updated[s] = q[policy[s]]
        gap = max(abs(updated[s] - values[s]) for s in mdp.states)
        if gap <= tolerance:
            return values, policy, history
        history.append(
            {"sweep": sweep, "values": updated.copy(), "policy": policy.copy()}
        )
        values = updated
    raise ValueError("value iteration did not converge within max_sweeps")


if __name__ == "__main__":
    rl_foundations_demo.print_vi_results(value_iteration)
# endregion book:fnd.rl-foundations.vi-01


# region book:fnd.rl-foundations.stopping-01 output
if __name__ == "__main__":
    from demos.rl_foundations_demo import (
        LateParcel,
        first_pay_day,
        parcel_mdp,
        policy_evaluation,
        value_iteration,
        wait_rule,
    )

    scans = (0, 0.9, 0)

    class RisingScans(LateParcel):
        def scan_probability(self, s):
            return scans[s]

    model = RisingScans(last_day=2, amount=10, wait_cost=1)
    mdp = parcel_mdp(model)
    values, optimal, _ = value_iteration(mdp)
    shortcut = wait_rule(model)
    print(
        f"rising scans {scans}, "
        f"amount {model.amount:g}, wait {model.wait_cost:g}: "
        f"one-stage rule pays day {first_pay_day(shortcut)}, "
        f"cost {-policy_evaluation(mdp, shortcut)[0]:.2f}; "
        f"optimal {optimal[0]}s, cost {-values[0]:.2f}"
    )
# endregion book:fnd.rl-foundations.stopping-01


# region book:fnd.rl-foundations.td-01 output
if __name__ == "__main__":
    from demos import rl_foundations_demo


def td0_update(current, reward, next_value, discount, eta, terminated):
    target = reward if terminated else reward + discount * next_value
    error = target - current
    return current + eta * error, target, error


def q_learning_update(current, reward, next_values, discount, eta, terminated):
    target = reward if terminated else reward + discount * max(next_values)
    error = target - current
    return current + eta * error, target, error


if __name__ == "__main__":
    rl_foundations_demo.print_td_results(td0_update, q_learning_update)
# endregion book:fnd.rl-foundations.td-01


def simulate_claim(model, policy, rng):
    """An action changes termination; carrier arrivals are independent of the action."""
    cost, days, steps = 0.0, 0, []
    for s in range(model.last_day + 1):
        action = policy(s) if callable(policy) else policy[s]
        if action == "pay":
            reward, next_s, terminated = -model.amount, "paid", True
        elif action == "wait" and s < model.last_day:
            reward = -model.wait_cost
            terminated = rng.random() < model.scan_probability(s)
            next_s = "scanned" if terminated else s + 1
            days += 1
        else:
            raise ValueError("invalid action, including waiting past the compulsory boundary")
        steps.append((s, action, reward, next_s, bool(terminated)))
        cost -= reward
        if terminated:
            return {"cost": cost, "waiting_days": days, "steps": steps}
    raise AssertionError("claim did not terminate")


def q_learning_run(model, claims=1000, seed=0):
    rng = np.random.default_rng(seed)
    q = np.zeros((model.last_day + 1, 2))
    counts = np.zeros_like(q, dtype=int)
    costs, excess, last_bad, total_cost = [], [], -1, 0.0
    mdp = parcel_mdp(model)
    planned, rule, _ = value_iteration(mdp)
    planned_cost = -planned[0]
    planned_q = {s: bellman_backup(mdp, planned, s) for s in range(model.last_day + 1)}
    active_days = [s for s, a in rule.items() if a == "wait"]
    advantages = 0.0

    def behavior(s):
        if s == model.last_day:
            return "pay"
        a = int(rng.integers(2)) if rng.random() < 0.1 else int(np.argmax(q[s]))
        return ("pay", "wait")[a]

    for claim in range(claims):
        episode = simulate_claim(model, behavior, rng)
        for s, action, reward, next_s, terminated in episode["steps"]:
            a = ("pay", "wait").index(action)
            counts[s, a] += 1
            next_values = () if terminated else q[next_s, : len(mdp.actions[next_s])]
            q[s, a], _, _ = q_learning_update(
                q[s, a], reward, next_values, 1, 1 / counts[s, a], terminated
            )
            advantages += planned[s] - planned_q[s][action]
        if any(("pay", "wait")[int(np.argmax(q[s]))] != rule[s] for s in active_days):
            last_bad = claim
        costs.append(episode["cost"])
        total_cost += episode["cost"]
        excess.append(total_cost - (claim + 1) * planned_cost)
    settled = last_bad + 2 if last_bad < claims - 1 else None
    final_rule = {
        s: ("pay", "wait")[int(np.argmax(q[s, : len(mdp.actions[s])]))]
        for s in range(model.last_day + 1)
    }
    return {
        "seed": seed,
        "claims": claims,
        "q": q.tolist(),
        "counts": counts.tolist(),
        "costs": costs,
        "excess": excess,
        "settled_claim": settled,
        "planned_advantages": advantages,
        "final_rule": list(final_rule.values()),
        "final_greedy_cost": -policy_evaluation(mdp, final_rule)[0],
        "full_rule_agrees": final_rule == rule,
    }


def exploration_cost(model):
    mdp = parcel_mdp(model)
    values, rule, _ = value_iteration(mdp)
    behavior = {}
    for s, action in rule.items():
        behavior[s] = {
            a: 0.1 / len(mdp.actions[s]) + (0.9 if a == action else 0) for a in mdp.actions[s]
        }
    cost = -policy_evaluation(mdp, behavior)[0]
    day0 = 0.05 * (model.amount + values[0])
    return {
        "planned_cost": -values[0],
        "behavior_cost": cost,
        "extra_cost": cost + values[0],
        "day0_extra": day0,
    }


def plan_from_logs(model, claims=200, seed=0):
    """Construct pay-at-once logs with carrier follow-up, then estimate only from counts.

    scan_day is a decision-day index: a scan after the wait action on day s is
    observed on day s+1. None means no scan through the ten-day follow-up.
    No customer response to waiting is supplied by these logs.
    """
    rng = np.random.default_rng(seed)
    lost = rng.random(claims) < model.lost_share
    scan_day = np.full(claims, model.last_day, dtype=int)
    if model.delivery_probability > 0:
        scan_day[~lost] = np.minimum(
            rng.geometric(model.delivery_probability, (~lost).sum()) - 1, model.last_day
        )
    at_risk = [int(np.count_nonzero(scan_day >= s)) for s in range(model.last_day)]
    scans = [int(np.count_nonzero(scan_day == s)) for s in range(model.last_day)]
    hazards = [n / risk if risk else 0.0 for n, risk in zip(scans, at_risk)]
    values, policy, _ = value_iteration(parcel_mdp(model, scans=hazards))
    true_cost = -policy_evaluation(parcel_mdp(model), policy)[0]
    return {
        "seed": seed,
        "claims": claims,
        "paid_day": 0,
        "followup_days": model.last_day,
        "at_risk": at_risk,
        "scans": scans,
        "scan_probabilities": hazards,
        "unsupported_days": [s for s, n in enumerate(at_risk) if n == 0],
        "policy": policy,
        "estimated_cost": -values[0],
        "true_cost": true_cost,
    }


@lru_cache(maxsize=1)
def learning_experiment():
    model = LateParcel()
    live = [q_learning_run(model, claims=1000, seed=seed) for seed in range(20)]
    exact = exploration_cost(model)
    plans = [plan_from_logs(model, claims=200, seed=seed) for seed in range(200)]
    gaps = [row["true_cost"] - exact["planned_cost"] for row in plans]
    settled = [row["settled_claim"] for row in live if row["settled_claim"] is not None]
    return {
        "construction": "simulated claims only; no historical requests or customer effects",
        "live_seeds": list(range(20)),
        "claims_per_live_seed": 1000,
        "log_seeds": list(range(200)),
        "claims_per_log_seed": 200,
        "exact": exact,
        "live": live,
        "summary": {
            "settled_seeds": len(settled),
            "settled_median": float(np.median(settled)) if settled else None,
            "settled_max": max(settled) if settled else None,
            "live_excess_median": float(np.median([row["excess"][-1] for row in live])),
            "live_advantages_median": float(np.median([row["planned_advantages"] for row in live])),
            "q0_wait_median": float(np.median([row["q"][0][1] for row in live])),
            "planned_q0_wait": -exact["planned_cost"],
            "full_rule_seeds": sum(row["full_rule_agrees"] for row in live),
            "final_greedy_cost_median": float(
                np.median([row["final_greedy_cost"] for row in live])
            ),
            "log_gap_mean": float(np.mean(gaps)),
            "log_gap_p95": float(np.quantile(gaps, 0.95)),
            "log_gap_max": max(gaps),
        },
        "log_plans": [
            {
                "seed": row["seed"],
                "pay_day": first_pay_day(row["policy"]),
                "cost_above_planned": gap,
            }
            for row, gap in zip(plans, gaps)
        ],
    }


# region book:fnd.rl-foundations.learn-01 output
if __name__ == "__main__":
    result = learning_experiment()
    summary, exact = result["summary"], result["exact"]
    print(
        "constructed live Q-learning: seeds 0..19, 1000 claims each, random action one time in ten, eta 1/n per pair, zero start"
    )
    print(
        f"days 0..4 stay greedy on planned rule: seeds {summary['settled_seeds']}/20, median claim {summary['settled_median']:.1f}, latest {summary['settled_max']}"
    )
    print(
        f"sampled cost above planned over 1000 claims: median {summary['live_excess_median']:.2f}"
    )
    print(
        f"sum of planned action advantages: median {summary['live_advantages_median']:.2f} (sampled paths; distinct from realized cost)"
    )
    print(
        f"continuing exploration at exact planned values: extra per claim {exact['extra_cost']:.6f}, over 1000 {1000 * exact['extra_cost']:.2f}, day 0 component {exact['day0_extra']:.6f}"
    )
    print(
        f"day 0 wait estimate median {summary['q0_wait_median']:.2f}; exact value {summary['planned_q0_wait']:.2f}"
    )
    rare = result["live"][0]
    print(
        f"seed 0 day 5 visits: pay {rare['counts'][5][0]}, wait {rare['counts'][5][1]}; Q pay {rare['q'][5][0]:.2f}, Q wait {rare['q'][5][1]:.2f}"
    )
    print(
        f"whole final greedy rule agrees: {summary['full_rule_seeds']}/20 seeds; final greedy claim cost median {summary['final_greedy_cost_median']:.2f}"
    )
    print(
        f"planning from 200 pay-at-once claims with 10-day carrier follow-up, seeds 0..199: extra per claim mean {summary['log_gap_mean']:.6f}, p95 {summary['log_gap_p95']:.6f}, max {summary['log_gap_max']:.6f}"
    )
    print("carrier arrivals are independent of payment; customer reactions to waiting are absent")
# endregion book:fnd.rl-foundations.learn-01


def two_carriers():
    models = [LateParcel(), LateParcel(lost_share=0.02, delivery_probability=0.25)]
    scans = []
    for s in range(10):
        weights = [m.survival(s) for m in models]
        scans.append(sum(w * m.scan_probability(s) for w, m in zip(weights, models)) / sum(weights))
    _, pooled, _ = value_iteration(parcel_mdp(models[0], scans=scans))
    rules = [value_iteration(parcel_mdp(m))[1] for m in models]
    aware_cost = np.mean(
        [-policy_evaluation(parcel_mdp(m), rule)[0] for m, rule in zip(models, rules)]
    )
    pooled_cost = np.mean([-policy_evaluation(parcel_mdp(m), pooled)[0] for m in models])
    return {
        "day5_beliefs": [m.lost_belief(5) for m in models],
        "carrier_pay_days": [first_pay_day(rule) for rule in rules],
        "pooled_pay_day": first_pay_day(pooled),
        "aware_cost": float(aware_cost),
        "pooled_cost": float(pooled_cost),
    }


# region book:fnd.rl-foundations.state-01 output
if __name__ == "__main__":
    result = two_carriers()
    print(
        "carriers X and Y: mixture 0.5 each; X lost 0.10, delivery 0.60; Y lost 0.02, delivery 0.25"
    )
    for name, belief, day in zip(("X", "Y"), result["day5_beliefs"], result["carrier_pay_days"]):
        print(f"{name}: day 5 lost belief {belief:.3f}, pay day {day}")
    print(
        f"pooled day-only rule pays day {result['pooled_pay_day']}; cost {result['pooled_cost']:.2f}, carrier-aware {result['aware_cost']:.2f}, gap {result['pooled_cost'] - result['aware_cost']:.2f}"
    )
# endregion book:fnd.rl-foundations.state-01


def price_of_waiting(amount=49.99):
    rows = []
    for price in (0, 1, 3, 10, 27):
        model = LateParcel(amount=amount, wait_cost=price)
        values, rule, _ = value_iteration(parcel_mdp(model))
        rows.append(
            {
                "price": price,
                "pay_day": first_pay_day(rule),
                "cost": -values[0],
                "one_step_agrees": rule == wait_rule(model),
            }
        )
    return rows


def shaped_mdp(mdp, potential):
    if any(potential[s] != 0 for s in mdp.terminals):
        raise ValueError("the potential must be zero at every terminal")
    rewards = {
        (s, a): reward
        + mdp.discount * sum(p * potential[next_s] for p, next_s in mdp.transitions[s, a])
        - potential[s]
        for (s, a), reward in mdp.rewards.items()
    }
    return FiniteMDP(mdp.states, mdp.actions, mdp.transitions, rewards, mdp.terminals, mdp.discount)


def shaping_check():
    model = LateParcel()
    mdp = parcel_mdp(model)
    potentials = [
        {s: -model.amount * model.lost_belief(s) for s in range(11)},
        {s: -float(s) for s in range(11)},
        {s: float(np.sin(s)) for s in range(11)},
    ]
    planned, rule, _ = value_iteration(mdp)
    results = []
    for phi in potentials:
        phi.update(dict.fromkeys(mdp.terminals, 0.0))
        values, shaped_rule, _ = value_iteration(shaped_mdp(mdp, phi))
        if shaped_rule != rule:
            raise AssertionError("potential changed the best policy")
        results.append((values, first_pay_day(shaped_rule)))
    bonus = {key: value + (1 if key[1] == "wait" else 0) for key, value in mdp.rewards.items()}
    bonus_mdp = FiniteMDP(
        mdp.states, mdp.actions, mdp.transitions, bonus, mdp.terminals, mdp.discount
    )
    return {
        "raw_cost": -planned[0],
        "shaped_cost": -results[0][0][0],
        "potential_pay_days": [day for _, day in results],
        "flat_bonus_pay_day": first_pay_day(value_iteration(bonus_mdp)[1]),
    }


def student_support_rewards():
    outcomes = {"spam_help": (0.20, 6, 0, 4), "targeted": (0.16, 2, 0, 1)}
    return {
        name: (progress, round(progress - 0.4 * interventions - 2.0 * high_risk - 0.5 * hours, 2))
        for name, (progress, interventions, high_risk, hours) in outcomes.items()
    }


def incident_mdp(reward):
    if reward not in ("planted", "corrected"):
        raise ValueError("reward must be planted or corrected")
    states = ("unchecked", "healthy", "failing", "closed")
    actions = {s: ("close", "wait") for s in states[:-1]}
    transitions, rewards = {}, {}
    for s in states[:-1]:
        transitions[s, "close"] = ((1.0, "closed"),)
        transitions[s, "wait"] = ((0.8, "healthy"), (0.2, "failing"))
        rewards[s, "close"] = 5.0 if reward == "planted" or s == "healthy" else -5.0
        rewards[s, "wait"] = -0.5
    return FiniteMDP(states, actions, transitions, rewards, ("closed",), 0.9)


def incident_report():
    mdp = incident_mdp("planted")
    values, rule, _ = value_iteration(mdp)
    wait_first = {"unchecked": "wait", "healthy": "close", "failing": "close"}
    low = policy_evaluation(mdp, wait_first)["unchecked"]
    return {
        "constructed": True,
        "reward_specification": "close: +5 whether checked or not; wait for check: -0.5; discount 0.9",
        "check_probabilities": {"healthy": 0.8, "failing": 0.2},
        "learned_rule": rule,
        "reported_value": values["unchecked"],
        "curve_construction": "expected score of a mixture that increasingly closes before checking; illustrative, not measured training",
        "reward_curve": [
            {"close_before_check_probability": p, "reward": (1 - p) * low + p * values["unchecked"]}
            for p in (0.0, 0.25, 0.5, 0.75, 1.0)
        ],
        "agent_claim": "The rising reward curve is offered as evidence that the incident policy improved.",
        "exercise": "Find what the reward pays for and recompute the rule with confirmed resolution.",
    }


# region book:fnd.rl-foundations.reward-01 output
if __name__ == "__main__":
    zero = price_of_waiting()[0]
    print(
        f"zero waiting price: pay day {zero['pay_day']}, claim cost {zero['cost']:.2f}; lost customer waits {zero['pay_day']} days"
    )
    result = shaping_check()
    print(
        f"potential -49.99 * lost belief, terminal potential 0: pay day {result['potential_pay_days'][0]}, original cost {result['raw_cost']:.2f}, shaped cost {result['shaped_cost']:.2f}"
    )
    print(
        "three potentials' pay days (-49.99 * lost belief, -1 * s, sin(s)): "
        f"{', '.join(str(day) for day in result['potential_pay_days'])}"
    )
    print(f"flat waiting bonus 1.00: pay day {result['flat_bonus_pay_day']}")
    for name, (progress, full) in student_support_rewards().items():
        print(f"student support {name}: progress-only {progress:.2f}, fuller reward {full:.2f}")
    for reward in ("planted", "corrected"):
        values, rule, _ = value_iteration(incident_mdp(reward))
        print(
            f"incident {reward}: unchecked action {rule['unchecked']}, expected discounted reward {values['unchecked']:.6f}"
        )
# endregion book:fnd.rl-foundations.reward-01


# region book:fnd.rl-foundations.keys-01 output
if __name__ == "__main__":
    discount = 0.8
    _, _, history = value_iteration(one_state_example(discount))
    optimum = 1 / (1 - discount)
    sweeps = next(row["sweep"] for row in history if abs(optimum - row["values"]["open"]) < 0.01)
    print(
        f"exercise 4: finish {3:.1f}, wait once then finish {1 + discount * 3:.1f}, wait forever {optimum:.1f}, optimum {optimum:.1f}, tie discount {1 - 1 / 3:.6f} (2/3), sweeps within 0.01 {sweeps}"
    )
    print(
        f"exercise 5: eta 0.5, discount 1, reward -3, current -20, next -15: no scan {td0_update(-20, -3, -15, 1, 0.5, False)[0]:.1f}, scan {td0_update(-20, -3, -15, 1, 0.5, True)[0]:.1f}"
    )
    mean = 0.0
    rewards = [1, -2, 4, 0]
    for n, reward in enumerate(rewards, 1):
        mean, _, _ = q_learning_update(mean, reward, [100], 0, 1 / n, False)
    print(f"exercise 5 running mean: rewards {rewards}, discount 0, final {mean:.2f}")
    print(
        "exercise 8 acceptance: 5 unique evidenced rows; JSON round-trip; verdict learn iff every answer is yes; rendered table; autonomy unchanged"
    )
    practice = LateParcel(amount=24.99)
    values, rule, _ = value_iteration(parcel_mdp(practice))
    print(
        f"exercise 6: parcel amount {practice.amount:.2f}, waiting price {practice.wait_cost:.2f}: pay day {first_pay_day(rule)}, cost {-values[0]:.2f}"
    )
    carriers = two_carriers()
    print(
        f"practice carriers: pooled cost {carriers['pooled_cost']:.2f}, aware cost {carriers['aware_cost']:.2f}, gap {carriers['pooled_cost'] - carriers['aware_cost']:.2f}"
    )
# endregion book:fnd.rl-foundations.keys-01


def print_vi_results(iterate):
    from demos.output_wrap import print_wrapped

    model = LateParcel()
    values, rule, history = iterate(parcel_mdp(model))
    print_wrapped(
        f"parcel discount 1, last day {model.last_day}: wait days 0..{first_pay_day(rule) - 1}, pay days {first_pay_day(rule)}..{model.last_day}, V*(0) {values[0]:.2f}"
    )
    print_wrapped("sweep | first pay day | V(0)")
    for row in history[1:]:
        print_wrapped(f"{row['sweep']} | {first_pay_day(row['policy'])} | {row['values'][0]:.6f}")
    print_wrapped(
        f"last changed sweep {history[-1]['sweep']}; the next backup confirms equality; no discounted contraction claimed"
    )
    print_wrapped("amount | waiting price | pay day | one-stage agrees")
    for amount in (49.99, 24.99):
        for row in price_of_waiting(amount):
            print_wrapped(
                f"{amount:.2f} | {row['price']:.2f} | {row['pay_day']} | {row['one_step_agrees']}"
            )
    print_wrapped(
        f"pay-at-once threshold: {model.scan_probability(0) * model.amount:.6f} per waiting day (tie pays)"
    )
    for discount in (0.5, 0.9):
        values, _, path = iterate(one_state_example(discount))
        optimum = max(3, 1 / (1 - discount))
        errors = [abs(optimum - row["values"]["open"]) for row in path[1:4]]
        within = next(row["sweep"] for row in path if abs(optimum - row["values"]["open"]) < 0.01)
        print_wrapped(
            f"one state discount {discount:.1f}: finish 3.00, wait once then finish {1 + discount * 3:.2f}, wait forever {1 / (1 - discount):.2f}, optimum {values['open']:.2f}"
        )
        print_wrapped(
            f"  distances after first sweeps: {', '.join(f'{e:.6f}' for e in errors)}; within 0.01 at sweep {within}"
        )


def print_td_results(td, q_update):
    from demos.output_wrap import print_wrapped

    examples = [
        ("TD nonterminal", td(2, 1, 4, 0.9, 0.1, False)),
        ("TD terminal (next value 4 ignored)", td(2, 1, 4, 0.9, 0.1, True)),
        ("Q day 4 wait no scan", q_update(-46, -3, [-49.99, -51.20], 1, 0.1, False)),
        ("Q day 4 wait scan", q_update(-46, -3, [-49.99, -51.20], 1, 0.1, True)),
    ]
    print_wrapped("TD: current 2, reward 1, next 4, discount 0.9, eta 0.1")
    print_wrapped(
        "Q: day 4 wait, current -46.00, reward -3.00, next pay -49.99 and wait -51.20, discount 1, eta 0.1"
    )
    for name, (value, target, error) in examples:
        print_wrapped(f"{name}: target {target:.2f}, error {error:.2f}, new value {value:.2f}")


def vi_figure_data():
    model = LateParcel()
    _, _, history = value_iteration(parcel_mdp(model))
    return {
        "constructed": True,
        "discount": 1.0,
        "amount": model.amount,
        "wait_cost": model.wait_cost,
        "days": list(range(model.last_day + 1)),
        "sweeps": [
            {
                "sweep": row["sweep"],
                "values": [row["values"][s] for s in range(model.last_day + 1)],
                "actions": [row["policy"].get(s) for s in range(model.last_day + 1)],
            }
            for row in history
        ],
        "belief": [model.lost_belief(s) for s in range(model.last_day + 1)],
        "scan_probability": [model.scan_probability(s) for s in range(model.last_day + 1)],
    }


def write_data(directory=None):
    directory = Path(directory) if directory is not None else Path(__file__).with_name("data")
    directory.mkdir(parents=True, exist_ok=True)
    for name, data in (
        ("vi", vi_figure_data()),
        ("learning", learning_experiment()),
        ("incident_05", incident_report()),
    ):
        (directory / f"rl_foundations_{name}.json").write_text(
            json.dumps(data, indent=2, sort_keys=True) + "\n"
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-data", action="store_true")
    if parser.parse_args().write_data:
        write_data()
