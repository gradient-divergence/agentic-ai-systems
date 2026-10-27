"""Deterministic exploration outputs and a sealed offer package for chapter 21.

The two printed regions run in order without hidden setup. Imports do no work.
The reward is the decision's one-step reward; its observation time is a field.
Only the learning window supplies a finding. Evaluation outcomes stay sealed.
"""

import itertools
import json
import math
from collections import Counter
from datetime import timedelta
from pathlib import Path


def print_trace():
    offers = ("standard", "dated", "standard", "referenced", "dated")
    rates = {"standard": 0.80, "dated": 0.86, "referenced": 0.84}
    regret = 0.0
    print("round offer true_no_contact gap cumulative_pseudo_regret")
    for t, offer in enumerate(offers, 1):
        gap = max(rates.values()) - rates[offer]
        regret += gap
        print(f"{t} {offer} {rates[offer]:.2f} {gap:.2f} {regret:.2f}")
    counts = Counter(offers)
    total = sum(counts[a] * (max(rates.values()) - rate) for a, rate in rates.items())
    print(
        f"decomposition: {counts['standard']} * 0.06 + {counts['referenced']} * 0.02 = {total:.2f}"
    )


# region book:fnd.exploration-bandits.trace-01 output
if __name__ == "__main__":
    print_trace()
# endregion book:fnd.exploration-bandits.trace-01


# region book:fnd.exploration-bandits.epsilon-01 output
import numpy as np


def epsilon_greedy_probabilities(estimates, epsilon):
    values = np.asarray(estimates, dtype=float)
    if (
        values.ndim != 1
        or not len(values)
        or not np.all(np.isfinite(values))
        or not 0 <= epsilon <= 1
    ):
        raise ValueError("finite estimates and epsilon in [0, 1] required")
    probabilities = np.full(len(values), epsilon / len(values))
    probabilities[np.argmax(values)] += 1 - epsilon
    return probabilities


def choose(rng, probabilities):
    p = np.asarray(probabilities, dtype=float)
    if (
        p.ndim != 1
        or np.any(p < 0)
        or not np.all(np.isfinite(p))
        or not np.isclose(p.sum(), 1)
    ):
        raise ValueError("probabilities must be nonnegative and sum to 1")
    index = int(rng.choice(len(p), p=p))
    return index, float(p[index])


def incremental_mean(mean, count, reward):
    return mean + (reward - mean) / (count + 1)


if __name__ == "__main__":
    print(
        f"four answers averaging 0.3; fifth answer 1: "
        f"{incremental_mean(0.3, 4, 1):.2f}"
    )
    probabilities = epsilon_greedy_probabilities([0.80, 0.86, 0.84], 0.1)
    print(
        "epsilon 0.1, probabilities (standard, dated, referenced):",
        ", ".join(f"{p:.6f}" for p in probabilities),
    )
    print(f"away from the leader: {1 - probabilities[1]:.6f}")
    index, propensity = choose(np.random.default_rng(7), probabilities)
    print(
        f"seed 7 choice: {('standard', 'dated', 'referenced')[index]}; "
        f"propensity {propensity:.6f}"
    )
# endregion book:fnd.exploration-bandits.epsilon-01


def print_ucb():
    from refund_agent.offers.policies import hoeffding_radius

    n, t, alpha = 25, 100, 0.05
    fixed = hoeffding_radius(n, alpha)
    union = hoeffding_radius(n, alpha, counts=t)
    bonus = hoeffding_radius(n, t**-4)
    print(f"n {n}, alpha {alpha:.2f}: fixed-count radius {fixed:.4f}")
    print(
        f"all {t} counts, overall alpha {alpha:.2f}: radius {union:.4f}; ratio {union / fixed:.2f}"
    )
    print(
        f"UCB1, t {t}: bonus {bonus:.4f}; ratio to fixed-count {bonus / fixed:.2f}; per-count failure {t**-4:.8f}; union failure {t**-3:.6f}"
    )
    counts, means = np.array([60, 25, 15]), np.array([0.80, 0.84, 0.73])
    indices = means + np.sqrt(2 * math.log(t) / counts)
    print("indices (standard, dated, referenced):", ", ".join(f"{x:.4f}" for x in indices))
    print("next notice:", ("standard", "dated", "referenced")[np.argmax(indices)])


# region book:fnd.exploration-bandits.ucb-01 output
if __name__ == "__main__":
    print_ucb()
# endregion book:fnd.exploration-bandits.ucb-01


def print_thompson():
    from refund_agent.offers.policies import best_probabilities

    counts, successes = np.array([40, 10, 5]), np.array([32, 9, 4])
    p = best_probabilities(successes, counts)
    samples = np.random.default_rng(7).beta(1 + successes, 1 + counts - successes, size=(300000, 3))
    frequencies = np.bincount(np.argmax(samples, axis=1), minlength=3) / len(samples)
    print("seed 7 Monte Carlo: 300000 draws; numerical quadrature beside it")
    print("offer successes answers posterior mean probability_best monte_carlo")
    for i, offer in enumerate(("standard", "dated", "referenced")):
        a, b = 1 + successes[i], 1 + counts[i] - successes[i]
        print(
            f"{offer} {successes[i]} {counts[i]} Beta({a},{b}) {a / (a + b):.3f} {p[i]:.3f} {frequencies[i]:.3f}"
        )
    print(f"maximum quadrature/Monte Carlo difference: {max(abs(p - frequencies)):.6f}")


# region book:fnd.exploration-bandits.thompson-01 output
if __name__ == "__main__":
    print_thompson()
# endregion book:fnd.exploration-bandits.thompson-01


# region book:fnd.exploration-bandits.cap-01 output
def capped_probabilities(bandit, cap, baseline=0):
    p = np.asarray(bandit, dtype=float)
    if not 0 <= cap <= 1 or not np.isclose(p.sum(), 1) or np.any(p < 0):
        raise ValueError("valid cap and probability vector required")
    result = cap * p
    result[baseline] += 1 - cap
    return result


def cap_from_budget(rho_budget, mu_baseline, mu_floor=0.0):
    if not all(
        np.isfinite(x) and 0 <= x <= 1
        for x in (rho_budget, mu_baseline, mu_floor)
    ):
        raise ValueError("budget and rates must be finite and in [0, 1]")
    loss = mu_baseline - mu_floor
    return min(1.0, rho_budget / loss) if loss > 0 else 1.0


if __name__ == "__main__":
    print(
        f"budget 0.01; baseline 0.80; unsigned floor 0: cap "
        f"{cap_from_budget(0.01, 0.80):.4f}"
    )
    cap = cap_from_budget(0.01, 0.80, 0.60)
    print(f"signed floor 0.60: cap {cap:.4f}")
    for estimates in ([0.85, 0.80, 0.80], [0.80, 0.85, 0.80]):
        p = capped_probabilities(
            epsilon_greedy_probabilities(estimates, 0.3), cap
        )
        print(
            "capped epsilon 0.3 probabilities:",
            ", ".join(f"{x:.3f}" for x in p),
        )
    floor = cap * 0.3 / 3
    print(f"propensity floor {floor:.3f}; largest weight {1 / floor:.0f}")
# endregion book:fnd.exploration-bandits.cap-01


def delay_batch():
    # Twenty of one hundred contacts, spread as evenly as integers allow over 14 days.
    contact_days = np.ceil(np.arange(1, 21) * 14 / 20).astype(int)
    rows = []
    for day in range(15):
        contacts = int(sum(contact_days <= day))
        rows.append(
            {
                "day": day,
                "requests": 100,
                "contacts": contacts,
                "silent_success": contacts / 100,
                "arrived_only": contacts / (contacts + 80)
                if day >= 14
                else 1.0
                if contacts
                else None,
                "matured": 0.2 if day >= 14 else None,
            }
        )
    return rows


def print_delay():
    print("requests 100; true repeat-contact rate 0.20; window 14 days")
    print("day contacts silent_as_success arrived_only matured")
    for row in delay_batch():
        if row["day"] in (0, 7, 14):
            values = [
                "undefined" if row[key] is None else f"{row[key]:.2f}"
                for key in ("silent_success", "arrived_only", "matured")
            ]
            print(row["day"], row["contacts"], *values)


# region book:fnd.exploration-bandits.delay-01 output
if __name__ == "__main__":
    print_delay()
# endregion book:fnd.exploration-bandits.delay-01


def enumerated_means():
    averages = []
    for first, second, third in itertools.product((0, 1), repeat=3):
        leader = 0 if first >= second else 1
        sums, counts = [first, second], [1, 1]
        sums[leader] += third
        counts[leader] += 1
        averages.append([k / n for k, n in zip(sums, counts)])
    return tuple(np.mean(averages, axis=0))


def adaptive_experiment(seed=7, runs=100000, rounds=20, epsilon=0.2):
    rng = np.random.default_rng(seed)
    counts = np.ones((runs, 2), dtype=int)
    successes = rng.integers(2, size=(runs, 2))
    terms = np.zeros((runs, 2))
    which = np.arange(runs)
    # Both offers are initialized once. Those deterministic rounds lack support
    # for the other arm; the reweighted means use only the remaining 18 rounds.
    for _ in range(rounds - 2):
        leaders = np.argmax(successes / counts, axis=1)
        p = np.full((runs, 2), epsilon / 2)
        p[which, leaders] += 1 - epsilon
        selected = (rng.random(runs) >= p[:, 0]).astype(int)
        reward = rng.integers(2, size=runs)
        counts[which, selected] += 1
        successes[which, selected] += reward
        terms[which, selected] += reward / p[which, selected]
    weighted = terms / (rounds - 2)
    return {
        "seed": seed,
        "runs": runs,
        "rounds": rounds,
        "epsilon": epsilon,
        "reweighted_rounds": rounds - 2,
        "plain": np.mean(successes / counts, axis=0).tolist(),
        "reweighted": np.mean(weighted, axis=0).tolist(),
        "standard_errors": (np.std(weighted, axis=0, ddof=1) / math.sqrt(runs)).tolist(),
    }


def print_adaptive():
    print("two offers, true rates 0.5, eight equally likely three-answer outcomes")
    print("enumerated expectations:", ", ".join(f"{x:.4f}" for x in enumerated_means()))
    result = adaptive_experiment()
    print(
        f"seed {result['seed']}; runs {result['runs']}; rounds {result['rounds']}; epsilon {result['epsilon']:.1f}; supported reweighted rounds {result['reweighted_rounds']}"
    )
    for i in range(2):
        print(
            f"offer {i + 1}: plain {result['plain'][i]:.6f}; reweighted {result['reweighted'][i]:.6f}; standard error {result['standard_errors'][i]:.6f}"
        )


# region book:fnd.exploration-bandits.adaptive-01 output
if __name__ == "__main__":
    print_adaptive()
# endregion book:fnd.exploration-bandits.adaptive-01


def build_experiment(seed=7, source=None):
    from refund_agent.offers.evaluation import (
        CappedUniform,
        OfferPackage,
        cap_zero_fixture,
        evaluation_log,
        fixed_rule,
        freeze_candidate,
        split_windows,
    )
    from refund_agent.offers.policies import Capped, EpsilonGreedy
    from refund_agent.offers.replay import replay, report, training_rows

    rows = training_rows(source)
    learning, evaluation = split_windows(rows)
    run = replay(learning, Capped(EpsilonGreedy(0.3), 0.05), seed)
    candidate = freeze_candidate(run.tables, run.finished_at, last_decision_at=learning[-1].at)
    logger = CappedUniform(0.05)
    package = OfferPackage(
        run.log,
        candidate,
        fixed_rule(candidate),
        logger,
        evaluation_log(evaluation, logger, seed, candidate=candidate),
        cap_zero_fixture(evaluation, seed),
    )
    return {
        "seed": seed,
        "rows": rows,
        "learning_rows": learning,
        "evaluation_rows": evaluation,
        "run": run,
        "report": report(run.log),
        "package": package,
        "accesses": [],
    }


def print_run(experiment):
    from refund_agent.offers.evaluation import CappedUniform
    from refund_agent.offers.notices import OFFERS

    package, card = experiment["package"], experiment["report"]
    print(
        f"seed {experiment['seed']}; training {len(experiment['rows'])}; learning {len(experiment['learning_rows'])}; evaluation {len(experiment['evaluation_rows'])}"
    )
    print(
        "decision windows: learning [2025-04-01, 2026-02-01); evaluation [2026-02-01, 2026-06-01)"
    )
    mix = Counter(row.reason for row in experiment["learning_rows"])
    for reason, count in sorted(mix.items()):
        print(f"learning mix {reason}: {count}; share {count / sum(mix.values()):.6f}")
    print("reason offer sent matured no_contact_rate Wilson_95 true_rate")
    for cell in card["cells"]:
        rate = f"{cell['rate']:.4f}" if cell["rate"] is not None else "undefined"
        low, high = cell["wilson"]
        print(
            f"{cell['reason']} {cell['offer']} {cell['sent']} {cell['matured']} {rate} [{low:.4f}, {high:.4f}] {cell['true_rate']:.2f}"
        )
    coin = card["coin"]
    print(
        f"coin baseline: {coin['baseline_no_contacts']}/{coin['baseline_count']}; learner: {coin['learner_no_contacts']}/{coin['learner_count']}"
    )
    if coin["interval"] is None:
        print("coin difference/interval undefined: a coin group has no matured answers")
    else:
        low, high = coin["interval"]
        print(
            f"learner minus baseline no-contact rate {coin['difference']:.6f}; approximate 95% interval [{low:.6f}, {high:.6f}]"
        )
    print(
        f"cumulative pseudo-regret {card['regret']:.2f}; per request {card['regret'] / len(package.learning.rows):.6f}"
    )
    print(
        f"expected extra requests with a repeat contact vs standard {card['expected_extra_contacts']:.2f}; budget {card['budget']:.2f}; expectation, not realized harm"
    )
    print(
        f"learning months {card['months']}; formula count per offer for 0.82 vs 0.88: {card['sample_size_per_offer']}; comparison target 550"
    )
    for reason, offers in card["waits"].items():
        for offer in OFFERS:
            wait = offers[offer]
            months = (
                f"{wait['months_to_550']:.2f}"
                if wait["months_to_550"] is not None
                else "undefined (zero sends)"
            )
            print(
                f"{reason} {offer}: sends/month {wait['sent_per_month']:.2f}; months to 550 {months}"
            )
    print(
        "five log rows: decision request customer decided_at reason amount_cents offers probabilities offer propensity coin version fingerprint seed override contact_at matures_at reward"
    )
    for row in package.learning.rows[:5]:
        p = ",".join(f"{x:.6f}" for x in row.probabilities)
        contact = row.contact_at.isoformat() if row.contact_at else "none"
        print(
            f"{row.decision_id} {row.request_id} {row.customer_id} {row.decided_at.isoformat()} {row.reason} {row.amount_cents} {','.join(row.eligible_offers)} [{p}] {row.offer} {row.propensity:.6f} {int(row.coin)} {row.policy_version} {row.policy_fingerprint} {row.seed} {row.override} {contact} {row.matures_at.isoformat()} {row.reward}"
        )
    print(
        f"candidate frozen at {package.candidate.frozen_at.isoformat()}; fingerprint {package.candidate.fingerprint}"
    )
    for key, offer, *_ in package.candidate.tables:
        print(f"frozen candidate {key}: {offer}")
    print(f"fixed standard rule fingerprint {package.baseline.fingerprint}")
    print(
        f"evaluation logger {package.logger.version}; fingerprint {package.logger.fingerprint}; seed {experiment['seed']}; cap {package.logger.cap:.4f}"
    )
    p = package.logger.probabilities()
    print(
        "evaluation probabilities:",
        ", ".join(f"{x:.10f}" for x in p),
        f"; floor {min(p):.10f}; largest weight {1 / min(p):.0f}",
    )
    print(
        "evaluation probabilities at cap 0.0125:",
        ", ".join(f"{x:.10f}" for x in CappedUniform(0.0125).probabilities()),
    )
    alternative = capped_probabilities([0, 0.5, 0.5], package.logger.cap)
    print(
        "cap-only max-min counterexample, inner [0, 0.5, 0.5]:",
        ", ".join(f"{x:.6f}" for x in alternative),
        f"; floor {min(alternative):.6f}; largest weight {1 / min(alternative):.0f}",
    )
    expected_matches = sum(
        row.probabilities[OFFERS.index(package.candidate.offer_for(row))]
        for row in package.evaluation.log.rows
    )
    print(
        f"expected candidate-matching evaluation decisions {expected_matches:.2f}; not a realized count or effective sample size"
    )
    print(f"cap-zero fixture rows {len(package.cap_zero.log.rows)}; probabilities 1, 0, 0")
    print(
        f"evaluation outcome accesses {len(package.evaluation.outcomes.log)}; truth accesses {len(package.evaluation.truth.log)}; held-out/golden consultations {len(experiment['accesses'])}"
    )
    print(
        "finding: the cap leaves new-notice cells thin; no live run or notice adoption is justified by this simulation; stage 3 still drafts for a person"
    )


# region book:fnd.exploration-bandits.run-01 output
if __name__ == "__main__":
    experiment = build_experiment()
    print_run(experiment)
# endregion book:fnd.exploration-bandits.run-01


def _seed_paths(rows, strategy, seeds, grouped=True):
    from refund_agent.offers.response import NO_CONTACT_RATE, answers

    reasons = tuple(NO_CONTACT_RATE)
    reason_indices = np.array([reasons.index(row.reason) for row in rows])
    rates = np.array([NO_CONTACT_RATE[row.reason] for row in rows])
    rewards = np.array([[a.reward for a in answers(row).values()] for row in rows])
    times = np.array([row.at.timestamp() for row in rows])
    ready = np.searchsorted(times, times - timedelta(days=14).total_seconds(), side="right")
    checkpoints = np.unique(np.linspace(0, len(rows), 201, dtype=int))
    paths = np.zeros((len(seeds), len(checkpoints)))
    counts = np.zeros((len(seeds), len(reasons) if grouped else 1, 3), dtype=int)
    successes = np.zeros_like(counts)
    selected = np.zeros((len(seeds), len(rows)), dtype=int)
    segment_sends = np.zeros((len(seeds), 3), dtype=int)
    rngs = [np.random.default_rng(seed) for seed in seeds]
    cumulative = np.zeros(len(seeds))
    which, pending, checkpoint = np.arange(len(seeds)), 0, 1
    for i, reason in enumerate(reason_indices):
        while pending < ready[i]:
            key = reason_indices[pending] if grouped else 0
            arm = selected[:, pending]
            counts[which, key, arm] += 1
            successes[which, key, arm] += rewards[pending, arm]
            pending += 1
        key = reason if grouped else 0
        n, k = counts[:, key], successes[:, key]
        means = np.divide(k, n, out=np.zeros_like(k, dtype=float), where=n > 0)
        if strategy == "ucb1":
            indices = means + np.sqrt(2 * math.log(max(i + 1, 1)) / np.maximum(n, 1))
            indices[n == 0] = np.inf
            choices = np.argmax(indices, axis=1)
        elif strategy == "thompson":
            choices = np.array(
                [int(np.argmax(rng.beta(1 + ks, 1 + ns - ks))) for rng, ks, ns in zip(rngs, k, n)]
            )
        else:
            epsilon = 0.3 if strategy == "capped" else 0.1
            p = np.full(means.shape, epsilon / 3)
            p[which, np.argmax(means, axis=1)] += 1 - epsilon
            choices = np.array(
                [
                    int(rng.choice(3, p=q)) if strategy != "capped" or rng.random() < 0.05 else 0
                    for rng, q in zip(rngs, p)
                ]
            )
        selected[:, i] = choices
        if reasons[reason] == "not_delivered":
            segment_sends[which, choices] += 1
        cumulative += max(rates[i]) - rates[i, choices]
        if checkpoint < len(checkpoints) and i + 1 == checkpoints[checkpoint]:
            paths[:, checkpoint] = cumulative
            checkpoint += 1
    return {
        "x": checkpoints.tolist(),
        "seeds": list(seeds),
        "paths": paths.tolist(),
        "mean": paths.mean(axis=0).tolist(),
        "p10": np.quantile(paths, 0.1, axis=0).tolist(),
        "p90": np.quantile(paths, 0.9, axis=0).tolist(),
        "not_delivered_sends": segment_sends.mean(axis=0).tolist(),
    }


def figure_data(experiment):
    rows, seeds = experiment["learning_rows"], tuple(range(7, 27))
    strategies = {
        key: _seed_paths(rows, key, seeds) for key in ("epsilon", "ucb1", "thompson", "capped")
    }
    one_table = _seed_paths(rows, "thompson", seeds, grouped=False)
    return {
        "version": "exploration-figures-1",
        "learning_window": ["2025-04-01", "2026-02-01"],
        "seed_spread": "policy randomness on the same constructed potential answers",
        "regret_measure": "cumulative pseudo-regret against the best offer per reason",
        "strategies": strategies,
        "not_delivered": {
            "one_table_thompson": one_table["not_delivered_sends"],
            "per_reason_thompson": strategies["thompson"]["not_delivered_sends"],
            "capped": strategies["capped"]["not_delivered_sends"],
        },
        "offers": ["standard", "dated", "referenced"],
        "delay": delay_batch(),
    }


def print_regret(data):
    print(
        "learning window; 20 policy seeds 7 to 26; shared planted potential answers; epsilon 0.1, capped epsilon 0.3 and cap 0.05"
    )
    for name, row in data["strategies"].items():
        print(
            f"{name}: final pseudo-regret mean {row['mean'][-1]:.2f}; 10th to 90th percentiles [{row['p10'][-1]:.2f}, {row['p90'][-1]:.2f}]"
        )
    for name, sends in data["not_delivered"].items():
        rate = np.dot(sends, [0.80, 0.82, 0.88]) / sum(sends)
        print(
            f"not_delivered {name}: mean sends (standard, dated, referenced) "
            + ", ".join(f"{n:.2f}" for n in sends)
            + f"; true mean no-contact rate of those sends {rate:.6f}; loss vs best {0.88 - rate:.6f}"
        )


# region book:fnd.exploration-bandits.regret-01 output
if __name__ == "__main__":
    figures = figure_data(experiment)
    print_regret(figures)
# endregion book:fnd.exploration-bandits.regret-01


def exercise_five(correct=False):
    q = epsilon_greedy_probabilities([0.80, 0.86, 0.84], 0.3)
    p = capped_probabilities(q, 0.05)
    rng, rows = np.random.default_rng(7), []
    for i in range(2000):
        coin = bool(rng.random() < 0.05)
        selected = choose(rng, q)[0] if coin else 0
        logged = p if correct else q
        rows.append(
            {
                "decision_id": f"exercise-05-{i:04d}",
                "cap": 0.05,
                "coin": coin,
                "offer": ("standard", "dated", "referenced")[selected],
                "bandit_probabilities": q.tolist(),
                "probabilities": logged.tolist(),
                "propensity": float(logged[selected]),
            }
        )
    return {
        "seed": 7,
        "statement": "the logged propensities are ready for evaluation",
        "rows": rows,
    }


def print_keys():
    print(
        "exercise 1: all five dated; pseudo-regret 0.00; a live desk does not know the true rates"
    )
    print("exercise 2: UCB1 probabilities 0 or 1; mix a known positive coin to restore support")
    counts, means = np.array([250, 100, 50]), np.array([0.62, 0.68, 0.60])
    indices = means + np.sqrt(2 * math.log(400) / counts)
    print(
        "exercise 3: indices "
        + ", ".join(f"{x:.4f}" for x in indices)
        + f"; C; fixed-count radius {math.sqrt(math.log(1 / 0.05) / (2 * 50)):.4f}"
    )
    unsigned, signed = cap_from_budget(0.015, 0.85), cap_from_budget(0.015, 0.85, 0.70)
    floor = signed * 0.2 / 3
    print(
        f"exercise 4 (instructor): caps {unsigned:.6f}, {signed:.2f}; learner/month {1200 * signed:.0f}; floor {floor:.6f}; largest weight {1 / floor:.0f}"
    )
    fixture = exercise_five()
    for i, offer in enumerate(("standard", "dated", "referenced")):
        count = sum(row["offer"] == offer for row in fixture["rows"])
        q = fixture["rows"][0]["probabilities"][i]
        applied = capped_probabilities(fixture["rows"][0]["bandit_probabilities"], 0.05)[i]
        print(
            f"exercise 5 (instructor): {offer}; sent {count}/2000; frequency {count / 2000:.4f}; logged {q:.3f}; applied {applied:.3f}"
        )
    print("exercise 5: new-notice probabilities overstated 20 times; weights understated 20 times")
    print(
        "exercise 6: amount contexts <2500 and >=2500 cents; caps 0.05 and 0.20; validate both logs, maturity and reproducibility"
    )


# region book:fnd.exploration-bandits.keys-01 output
if __name__ == "__main__":
    print_keys()
# endregion book:fnd.exploration-bandits.keys-01


def write_data(experiment, figures, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for name, data in (
        ("exploration_bandits_figures.json", figures),
        ("exploration_bandits_log_05.json", exercise_five()),
    ):
        (directory / name).write_text(json.dumps(data, sort_keys=True, indent=2) + "\n")
    experiment["package"].persist(directory / "exploration_bandits_package")


def amount_context(row):
    return f"{row.reason}|{'under25' if row.amount_cents < 2500 else '25plus'}"


def exercise_six_runs(seed=7, context_key=amount_context):
    from refund_agent.offers.evaluation import split_windows
    from refund_agent.offers.policies import Capped, EpsilonGreedy
    from refund_agent.offers.replay import replay, report, training_rows

    learning, _ = split_windows(training_rows())
    result = {}
    for cap in (0.05, 0.20):
        run = replay(learning, Capped(EpsilonGreedy(0.3), cap), seed, table_key=context_key)
        result[cap] = {"run": run, "report": report(run.log)}
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-data", action="store_true")
    if parser.parse_args().write_data:
        write_data(experiment, figures, Path(__file__).resolve().parent / "data")
