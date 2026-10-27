"""Listings and outputs for RLHF and preference alignment, with no language model.

The three printed regions run in chapter order without intervening setup. The
five-kind telecom experiment generates independent labeled, comparison and
held-out windows. Only training windows fit the frozen reference and the reward
gaps; DPO reads comparisons and that reference. Held-out outcomes measure what
preference accuracy leaves unanswered. --write-data regenerates the figures and
the two constructed agent reports. keys-01 contains instructor exercise answers.
"""

# ruff: noqa: E402

# region book:fnd.rlhf-alignment.pair-01 output
import numpy as np
from pydantic import BaseModel


class PreferencePair(BaseModel):
    chosen_score: float
    rejected_score: float


def preference_probability(u):
    """Stable logistic of a preference gap (a scalar or an array)."""
    return np.exp(-np.logaddexp(0, -np.asarray(u)))


def bradley_terry_loss(pair: PreferencePair) -> float:
    return float(np.logaddexp(0, -(pair.chosen_score - pair.rejected_score)))


if __name__ == "__main__":
    print("chosen rejected gap preference_probability loss")
    for chosen, rejected in ((1.2, 0.4), (6.2, 5.4)):
        pair = PreferencePair(chosen_score=chosen, rejected_score=rejected)
        u = chosen - rejected
        print(
            f"{chosen:.1f} {rejected:.1f} {u:.1f} "
            f"{preference_probability(u):.4f} {bradley_terry_loss(pair):.4f}"
        )
    print(
        f"70:30 split: gap {np.log(0.7 / 0.3):.4f}; agreement ceiling "
        f"{max(0.7, 0.3):.4f}"
    )
# endregion book:fnd.rlhf-alignment.pair-01


# region book:fnd.rlhf-alignment.optimum-01 output
from demos.data_science_strategy_demo import zero_failure_bound


def kl_optimum(ref_probs, rewards, beta):
    """Reweight; return probabilities, reward, KL and beta log Z."""
    ref, r = (
        np.asarray(ref_probs, dtype=float),
        np.asarray(rewards, dtype=float),
    )
    if (
        ref.ndim != 1
        or r.shape != ref.shape
        or not np.isfinite(ref).all()
        or not np.isfinite(r).all()
        or (ref < 0).any()
        or not np.isclose(ref.sum(), 1, rtol=0, atol=1e-12)
        or not np.isfinite(beta)
        or beta <= 0
    ):
        raise ValueError(
            "a normalized reference, finite rewards and beta > 0 are required"
        )
    support = ref > 0
    peak = r[support].max()
    centered = (r[support] - peak) / beta
    # log1p/expm1 keep beta log Z accurate even when beta is very large.
    log_z = (
        np.log1p(ref[support] @ np.expm1(centered))
        if np.max(abs(centered)) < 0.5
        else np.log(ref[support] @ np.exp(centered))
    )
    log_ratio = centered - log_z
    p = np.zeros_like(ref)
    p[support] = np.exp(np.log(ref[support]) + log_ratio)
    if (p[support] == 0).any():
        raise ValueError(
            "positive support underflows at this beta; use a larger beta"
        )
    kl = max(0.0, float(p[support] @ log_ratio))
    return dict(
        probabilities=p,
        expected_reward=float(p @ r),
        kl=kl,
        beta_log_z=float(peak + beta * log_z),
    )


def clean_test_chance(rate, n):
    """Chance that n independent drafts contain no occurrence at this rate."""
    if not 0 <= rate <= 1 or n < 0 or int(n) != n:
        raise ValueError("rate must be in [0, 1] and n a nonnegative integer")
    return float((1 - rate) ** n)


if __name__ == "__main__":
    ref, rewards = np.array([0.50, 0.45, 0.05]), np.array([1.0, 0.5, -2.0])
    print("given rewards: fault 1.0, check 0.5, credit -2.0")
    print(
        "beta        fault      check       credit    score       KL "
        "beta_log_Z credit/10000 clean/300"
    )
    print(
        f"reference {ref[0]:.6f} {ref[1]:.6f} {ref[2]:.8f} "
        f"{ref @ rewards:.6f} 0.000000 -- {10000 * ref[2]:.4f} "
        f"{clean_test_chance(ref[2], 300):.6g}"
    )
    for beta in (1.0, 0.5, 0.25):
        result = kl_optimum(ref, rewards, beta)
        p = result["probabilities"]
        print(
            f"{beta:<9g} {p[0]:.6f} {p[1]:.6f} {p[2]:.8f} "
            f"{result['expected_reward']:.6f} {result['kl']:.6f} "
            f"{result['beta_log_z']:.6f} {10000 * p[2]:.4f} "
            f"{clean_test_chance(p[2], 300):.6g}"
        )
    credit = kl_optimum(ref, rewards, 0.5)["probabilities"][2]
    print(f"beta 0.5 credit: one in {1 / credit:.2f} drafts")
    print(f"300 clean drafts: 95% rate bound {zero_failure_bound(300):.6f}")
# endregion book:fnd.rlhf-alignment.optimum-01


# region book:fnd.rlhf-alignment.dpo-01 output
import math

import numpy as np
from pydantic import BaseModel, Field

from demos.output_wrap import print_wrapped


class DPOConfig(BaseModel):
    model_config = {"extra": "forbid"}
    beta: float = Field(..., gt=0.0)


class PreferenceBatch(BaseModel):
    chosen_logprob: float
    rejected_logprob: float
    ref_chosen_logprob: float
    ref_rejected_logprob: float


def sequence_logprob(token_logprobs):
    """The chain rule sums token log-probabilities; it never averages them."""
    return math.fsum(token_logprobs)


def implicit_rewards(batch, config):
    return (
        config.beta * (batch.chosen_logprob - batch.ref_chosen_logprob),
        config.beta * (batch.rejected_logprob - batch.ref_rejected_logprob),
    )


def dpo_loss(batch: PreferenceBatch, config: DPOConfig) -> float:
    policy_delta = batch.chosen_logprob - batch.rejected_logprob
    ref_delta = batch.ref_chosen_logprob - batch.ref_rejected_logprob
    margin = policy_delta - ref_delta
    scaled = config.beta * margin
    return float(np.logaddexp(0, -scaled))


def preference_accuracy(batch: PreferenceBatch) -> float:
    policy_delta = batch.chosen_logprob - batch.rejected_logprob
    ref_delta = batch.ref_chosen_logprob - batch.ref_rejected_logprob
    margin = policy_delta - ref_delta
    return (
        0.5
        if math.isclose(margin, 0.0, rel_tol=0.0, abs_tol=1e-12)
        else float(margin > 0)
    )


if __name__ == "__main__":
    tokens = (
        [-0.2, -0.4, -0.5],
        [-0.3, -0.7, -0.8],
        [-0.5, -0.5, -0.4],
        [-0.4, -0.6, -0.6],
    )
    logs = [sequence_logprob(t) for t in tokens]
    batch = PreferenceBatch(**dict(zip(PreferenceBatch.model_fields, logs)))
    config = DPOConfig(beta=0.2)
    rw, rl = implicit_rewards(batch, config)
    print_wrapped(
        "token log-probabilities (chosen, rejected, reference chosen, "
        "reference rejected):"
    )
    for t in tokens:
        print_wrapped(" ".join(f"{v:.1f}" for v in t))
    print_wrapped("sequence sums: " + " ".join(f"{v:.1f}" for v in logs))
    policy_gap, reference_gap = logs[0] - logs[1], logs[2] - logs[3]
    print_wrapped(
        f"policy gap {policy_gap:.1f}; reference gap {reference_gap:.1f}; "
        f"margin {policy_gap - reference_gap:.1f}"
    )
    print_wrapped(
        f"beta {config.beta:g}; implicit rewards {rw:.4f} {rl:.4f}; gap "
        f"{rw - rl:.4f}"
    )
    print_wrapped(
        f"DPO loss {dpo_loss(batch, config):.4f}; implicit ordering "
        f"{preference_accuracy(batch):.1f}"
    )
    same = PreferenceBatch(
        chosen_logprob=logs[2],
        rejected_logprob=logs[3],
        ref_chosen_logprob=logs[2],
        ref_rejected_logprob=logs[3],
    )
    print_wrapped(
        f"policy = reference: loss {dpo_loss(same, config):.4f}; implicit "
        f"ordering {preference_accuracy(same):.1f}"
    )
# endregion book:fnd.rlhf-alignment.dpo-01


import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from demos.supervised_learning_demo import calibration_error, logit, platt, reliability


@dataclass(frozen=True)
class FaultKinds:
    shares: tuple = (0.2, 0.4, 0.6, 0.8, 0.9)
    drafts: tuple = ("Check the local equipment fault.", "A network fault is being checked.")


KINDS = FaultKinds()
DATA = Path(__file__).with_name("data")


def generate(seed=7, n_labeled=2000, n_pairs=2000, n_heldout_reports=2000, n_heldout_pairs=2000):
    """Counts are per kind; each window has its own seeded random stream."""
    sizes = (n_labeled, n_pairs, n_heldout_reports, n_heldout_pairs)
    if any(n < 0 or int(n) != n for n in sizes):
        raise ValueError("window sizes must be nonnegative integers")
    data = {}
    names = ("labeled", "pairs", "heldout_reports", "heldout_pairs")
    for name, n, stream in zip(names, sizes, np.random.SeedSequence(seed).spawn(4)):
        rng = np.random.default_rng(stream)
        kind = np.repeat(np.arange(len(KINDS.shares)), n)
        cause = (rng.random(len(kind)) < np.asarray(KINDS.shares)[kind]).astype(int)
        if name.endswith("pairs"):
            first = rng.integers(0, 2, len(kind))
            rows = dict(
                kind=kind,
                first=first,
                second=1 - first,
                chosen=cause,
                chosen_position=(first != cause).astype(int),
            )
        else:
            rows = dict(kind=kind, true_cause=cause)
        for array in rows.values():
            array.setflags(write=False)
        data[name] = rows
    return data


def frequencies(rows, label):
    counts = np.bincount(rows["kind"], minlength=len(KINDS.shares))
    if (counts == 0).any():
        raise ValueError("each kind needs training observations")
    positive = np.bincount(rows["kind"], weights=rows[label], minlength=len(counts))
    return positive / counts, counts / counts.sum()


def fit_reference(labeled):
    """Exact per-kind log-loss fit; the resulting reference stays frozen."""
    q, _ = frequencies(labeled, "true_cause")
    if ((q == 0) | (q == 1)).any():
        raise ValueError("a finite log-loss optimum needs both causes for every kind")
    theta = logit(q)
    theta.setflags(write=False)
    return theta


def fit_reward_gaps(pairs):
    """Bradley–Terry MLE: the gap is the log-odds of choosing the network draft."""
    q, _ = frequencies(pairs, "chosen")
    if ((q == 0) | (q == 1)).any():
        raise ValueError("a finite reward gap needs both choices for every kind")
    return logit(q)


def rlhf_optimum(reference, reward_gaps, beta):
    ref = preference_probability(reference)
    return np.array(
        [kl_optimum([p, 1 - p], [r, 0], beta)["probabilities"][0] for p, r in zip(ref, reward_gaps)]
    )


def dpo_objective(theta, reference, pairs, beta):
    q, weights = frequencies(pairs, "chosen")
    u = beta * (theta - reference)
    return float(weights @ (q * np.logaddexp(0, -u) + (1 - q) * np.logaddexp(0, u)))


def dpo_gradient(theta, reference, pairs, beta):
    q, weights = frequencies(pairs, "chosen")
    return beta * weights * (preference_probability(beta * (theta - reference)) - q)


def train_dpo(beta=1, steps=100000, rate=10, *, reference=None, pairs=None):
    """Full-batch gradient descent on comparison likelihood, from the reference."""
    DPOConfig(beta=beta)
    if steps < 1 or int(steps) != steps or not np.isfinite(rate) or rate <= 0:
        raise ValueError("positive steps and a finite positive rate are required")
    if reference is None or pairs is None:
        training = generate(n_heldout_reports=0, n_heldout_pairs=0)
        reference = fit_reference(training["labeled"]) if reference is None else reference
        pairs = training["pairs"] if pairs is None else pairs
    q, weights = frequencies(pairs, "chosen")
    theta = reference.copy()
    for step in range(steps + 1):
        gradient = beta * weights * (preference_probability(beta * (theta - reference)) - q)
        norm = float(np.linalg.norm(gradient))
        if norm < 1e-10 or step == steps:
            break
        theta -= rate * gradient
    return dict(
        logits=theta,
        probabilities=preference_probability(theta),
        steps=step,
        gradient_norm=norm,
        converged=norm < 1e-10,
        loss=dpo_objective(theta, reference, pairs, beta),
    )


def fit_training(data):
    """This function's two inputs are the only windows allowed to affect fitting."""
    reference = fit_reference(data["labeled"])
    gaps = fit_reward_gaps(data["pairs"])
    fits = {
        str(beta): train_dpo(beta, reference=reference, pairs=data["pairs"]) for beta in (1.0, 0.25)
    }
    if not all(fit["converged"] for fit in fits.values()):
        raise RuntimeError("DPO did not reach the specified gradient tolerance")
    return dict(reference=reference, reward_gaps=gaps, fits=fits)


def evaluate(probabilities, reference, reports, pairs):
    """Separate denominators: held-out comparisons for preferences, reports for truth."""
    kind, y = reports["kind"], reports["true_cause"]
    if (np.bincount(kind, minlength=len(KINDS.shares)) == 0).any():
        raise ValueError("evaluation needs at least one report for every kind")
    if len(pairs["kind"]) == 0:
        raise ValueError("evaluation needs at least one comparison")
    p = probabilities[kind]
    kl = np.zeros_like(p, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        for mass, log_ref in (
            (p, -np.logaddexp(0, -reference[kind])),
            (1 - p, -np.logaddexp(0, reference[kind])),
        ):
            nonzero = mass != 0
            kl[nonzero] += mass[nonzero] * (np.log(mass[nonzero]) - log_ref[nonzero])
        mean_kl = float(kl.mean())
    if not np.isfinite(mean_kl):
        raise ValueError("evaluation KL must be finite")
    correct = (p >= 0.5) == y
    confidence = np.maximum(p, 1 - p)
    high = confidence > 0.9
    # A deterministic endpoint has infinite log-odds, which still orders a pair.
    with np.errstate(divide="ignore"):
        margins = (logit(probabilities) - reference)[pairs["kind"]] * (2 * pairs["chosen"] - 1)
    agreement = np.where(np.abs(margins) <= 1e-12, 0.5, margins > 0)
    table = [reliability(y[kind == k], p[kind == k], bins=1)[0] for k in range(len(KINDS.shares))]
    return dict(
        preference_accuracy=float(agreement.mean()),
        correctness=float(correct.mean()),
        calibration_error=calibration_error(table),
        mean_kl=max(0.0, mean_kl),
        mean_confidence=float(confidence.mean()),
        reliability=table,
        reports=len(y),
        comparisons=len(margins),
        high_count=int(high.sum()),
        high_share=float(high.mean()),
        high_confidence=float(confidence[high].mean()) if high.any() else None,
        high_correctness=float(correct[high].mean()) if high.any() else None,
    )


def build_experiment(seed=7):
    data = generate(seed)
    trained = fit_training(data)
    policies = {
        "reference": preference_probability(trained["reference"]),
        **{key: fit["probabilities"] for key, fit in trained["fits"].items()},
    }
    evaluations = {
        key: evaluate(p, trained["reference"], data["heldout_reports"], data["heldout_pairs"])
        for key, p in policies.items()
    }
    return dict(seed=seed, data=data, **trained, evaluations=evaluations)


def print_training(run):
    from demos.output_wrap import print_wrapped

    print_wrapped(
        f"seed {run['seed']}; 5 kinds; per kind: labeled 2000, training pairs 2000, held-out reports 2000, held-out pairs 2000"
    )
    print_wrapped("full-batch DPO: rate 10; max steps 100000; gradient norm < 1e-10")
    print_wrapped("kind mu reference reward_gap DPO_beta_1 DPO_beta_0.25")
    ref = preference_probability(run["reference"])
    for k, mu in enumerate(KINDS.shares):
        print_wrapped(
            f"{k} {mu:.1f} {ref[k]:.6f} {run['reward_gaps'][k]:.6f} {run['fits']['1.0']['probabilities'][k]:.6f} {run['fits']['0.25']['probabilities'][k]:.6f}"
        )
    print_wrapped("policy preference_accuracy correctness calibration_error mean_KL mean_confidence")
    for name, result in run["evaluations"].items():
        print_wrapped(
            f"{name} {result['preference_accuracy']:.6f} {result['correctness']:.6f} {result['calibration_error']:.6f} {result['mean_kl']:.6f} {result['mean_confidence']:.6f}"
        )
    for beta, fit in run["fits"].items():
        optimum = rlhf_optimum(run["reference"], run["reward_gaps"], float(beta))
        print_wrapped(
            f"beta {beta}: steps {fit['steps']}; gradient norm {fit['gradient_norm']:.3e}; max DPO/RLHF probability gap {np.max(abs(fit['probabilities'] - optimum)):.3e}"
        )
    for mu in (0.5, 0.6):
        print_wrapped(
            f"population mu {mu:.1f}: beta 1 {preference_probability(2 * logit(mu)):.6f}; beta 0.25 {preference_probability(5 * logit(mu)):.6f}"
        )


def delivered_record(run=None):
    run = build_experiment() if run is None else run
    ref, tuned = run["evaluations"]["reference"], run["evaluations"]["0.25"]
    return dict(
        scenario="constructed telecom support delivery",
        seed=run["seed"],
        beta=0.25,
        claims=[
            f"Preference accuracy {tuned['preference_accuracy']:.4f}, up from {ref['preference_accuracy']:.4f}.",
            f"Mean confidence {tuned['mean_confidence']:.4f} on the stated cause, so replies above 0.9 can go without review.",
            "The KL penalty keeps it close to the reference.",
            "No credit promised in 300 test drafts, so enable the credit tool.",
        ],
        clean_credit_test=dict(
            drafts=300, observed_credits=0, status="constructed clean observation, not a permission"
        ),
    )


def delivered(run=None):
    run = build_experiment() if run is None else run
    record = json.loads((DATA / "rlhf_alignment_delivered.json").read_text())
    if record != delivered_record(run):
        raise ValueError("delivery fixture is stale: regenerate it with --write-data")
    ref, tuned = run["evaluations"]["reference"], run["evaluations"]["0.25"]
    credit = kl_optimum([0.5, 0.45, 0.05], [1, 0.5, -2], 1)["probabilities"][2]
    checks = [
        f"Correctness {ref['correctness']:.4f} before and {tuned['correctness']:.4f} after; {tuned['reports']} held-out reports, {tuned['comparisons']} held-out pairs.",
        f"Above 0.9: {tuned['high_count']} reports ({tuned['high_share']:.4f}); mean confidence {tuned['high_confidence']:.6f}, correctness {tuned['high_correctness']:.6f}. Calibration error {tuned['calibration_error']:.6f}, reference {ref['calibration_error']:.6f}.",
        f"Mean KL {tuned['mean_kl']:.6f}; penalty prices distance and forbids nothing.",
        f"95% credit-rate bound {zero_failure_bound(300):.6f}; beta 1 credit probability {credit:.8f}, clean-test chance {clean_test_chance(credit, 300):.6f}; permission belongs at execution.",
    ]
    print(f"agent report: seed {record['seed']}, DPO beta {record['beta']}")
    for claim, check in zip(record["claims"], checks):
        print(f"claim: {claim}")
        print(f"check: {check}")


def exercise_five_record(seed=7):
    """Constructed hospital records; the single defect selects true-urgent labels."""
    rng = np.random.default_rng(seed)
    p = np.repeat([0.70, 0.85, 0.91, 0.95, 0.99], 200)
    truth = np.repeat([0.35, 0.50, 0.65, 0.75, 0.80], 200)
    y = (rng.random(len(p)) < truth).astype(int)
    margins = rng.permutation(np.r_[np.ones(780), -np.ones(220)])
    return dict(
        scenario="constructed hospital portal urgency report",
        seed=seed,
        beta=0.25,
        agent_report=dict(
            preference_accuracy=float(np.mean(margins > 0)),
            mean_probability_on_urgent=float(p[y == 1].mean()),
            calibration_rows="true_urgent == 1",
            calibration_verdict="calibrated",
            proposal="Route messages above 0.9 without a nurse.",
        ),
        heldout_messages=[
            dict(message_id=f"m-{i:04d}", probability_urgent=float(q), true_urgent=int(t))
            for i, (q, t) in enumerate(zip(p, y))
        ],
        heldout_preferences=[
            dict(pair_id=f"p-{i:04d}", margin=float(m)) for i, m in enumerate(margins)
        ],
    )


def exercise_windows(seed=7):
    """Training first, a labeled map-fitting window, then a separate later check."""
    training = generate(seed, n_heldout_reports=0, n_heldout_pairs=0)
    calibration = generate(seed + 100, 0, 0, 2000, 0)["heldout_reports"]
    later = generate(seed + 200, 0, 0, 2000, 0)["heldout_reports"]
    return training, calibration, later


def exercise_six(seed=7):
    """Accept the fitted-sample identity; keys-01 prints population diagnostics."""
    training, calibration, later = exercise_windows(seed)
    ref = fit_reference(training["labeled"])
    gaps = fit_reward_gaps(training["pairs"])
    fit = train_dpo(0.5, reference=ref, pairs=training["pairs"])
    p = fit["probabilities"]
    mapped, slope, intercept = platt(p[calibration["kind"]], calibration["true_cause"])
    population_gaps = abs(fit["logits"] - 3 * logit(np.asarray(KINDS.shares)))

    def error(q):
        return calibration_error(
            [
                reliability(later["true_cause"][later["kind"] == k], np.full(2000, q[k]), bins=1)[0]
                for k in range(5)
            ]
        )

    return dict(
        seed=seed,
        beta=0.5,
        logits=fit["logits"].tolist(),
        target_logits=(ref + gaps / 0.5).tolist(),
        population_check=dict(
            logit_gaps=population_gaps.tolist(),
            max_logit_gap=float(population_gaps.max()),
            tolerance=0.02,
            passed=bool(population_gaps.max() <= 0.02),
        ),
        slope=slope,
        intercept=intercept,
        reference_error=error(preference_probability(ref)),
        before_error=error(p),
        after_error=error(mapped(p)),
        reference_choices=(ref >= 0).astype(int).tolist(),
        tuned_choices=(p >= 0.5).astype(int).tolist(),
        mapped_choices=(mapped(p) >= 0.5).astype(int).tolist(),
        window_order=["training", "calibration", "later check"],
        calibration_reports=len(calibration["kind"]),
        later_reports=len(later["kind"]),
    )


def print_keys():
    print("exercise 1 inputs: scores 2.1 1.3, then 0.4 -0.4")
    print(f"exercise 1: both gaps 0.8; preference probability {preference_probability(0.8):.4f}")
    print(f"exercise 2: 300 clean drafts; 95% rate bound {zero_failure_bound(300):.6f}")
    pair = PreferencePair(chosen_score=0.3, rejected_score=1.1)
    print(f"exercise 3(a) inputs: chosen {pair.chosen_score:g}, rejected {pair.rejected_score:g}")
    print(
        f"exercise 3(a): probability {preference_probability(-0.8):.4f}; loss {bradley_terry_loss(pair):.4f}"
    )
    tokens = ([-0.4, -0.9, -0.2], [-0.3, -0.6], [-0.5, -1.0, -0.4], [-0.2, -0.5])
    logs = [sequence_logprob(t) for t in tokens]
    batch = PreferenceBatch(**dict(zip(PreferenceBatch.model_fields, logs)))
    config = DPOConfig(beta=0.5)
    rw, rl = implicit_rewards(batch, config)
    print(
        f"exercise 3(b) inputs: beta {config.beta:g}; policy chosen/rejected, reference chosen/rejected tokens"
    )
    for t in tokens:
        print(" ".join(f"{v:.1f}" for v in t))
    print(
        "exercise 3(b), policy chosen/rejected, reference chosen/rejected: "
        + " ".join(f"{v:.1f}" for v in logs)
    )
    print(
        f"implicit rewards {rw:.4f} {rl:.4f}; gap {rw - rl:.4f}; DPO loss {dpo_loss(batch, config):.4f}"
    )
    same = PreferenceBatch(
        chosen_logprob=logs[2],
        rejected_logprob=logs[3],
        ref_chosen_logprob=logs[2],
        ref_rejected_logprob=logs[3],
    )
    print(f"exercise 3(c), policy = reference: loss {dpo_loss(same, config):.4f}")
    ref, rewards, beta = [0.6, 0.3, 0.1], [0.5, 1.5, -1], 0.5
    print(f"exercise 4 inputs: reference {ref}; scores {rewards}; beta {beta:g}")
    optimum = kl_optimum(ref, rewards, beta)
    print("exercise 4: probabilities " + " ".join(f"{v:.6f}" for v in optimum["probabilities"]))
    print(
        f"score {optimum['expected_reward']:.6f}; KL {optimum['kl']:.6f}; beta log Z {optimum['beta_log_z']:.6f}"
    )
    print(
        f"two answers: mu 0.7, beta 0.5, tuned probability {preference_probability(3 * logit(0.7)):.6f}"
    )
    fixture = json.loads((DATA / "rlhf_alignment_report_05.json").read_text())
    rows = fixture["heldout_messages"]
    y, p = (
        np.array([r["true_urgent"] for r in rows]),
        np.array([r["probability_urgent"] for r in rows]),
    )
    print(
        f"exercise 5: preference accuracy {fixture['agent_report']['preference_accuracy']:.4f}; mean probability on {int(y.sum())} true-urgent messages {p[y == 1].mean():.6f}"
    )
    table = reliability(y, p, bins=5)
    print("all held-out messages: count mean_probability true_urgent observed_rate")
    for count, mean, urgent in table:
        print(f"{count} {mean:.6f} {urgent} {urgent / count:.6f}")
    print(
        f"all {len(y)} messages: calibration error {calibration_error(table):.6f}; refuse routing inferred from the selected-label check"
    )
    report = exercise_six()
    print(
        f"exercise 6: seed {report['seed']}, beta {report['beta']}; calibration {report['calibration_reports']}, later {report['later_reports']} reports"
    )
    print("acceptance tolerances: logit gap 0.02; slope gap 0.05; later calibration-error gap 0.02")
    print(
        f"max fitted-sample logit gap {np.max(abs(np.array(report['logits']) - report['target_logits'])):.3e}; Platt slope {report['slope']:.6f}, intercept {report['intercept']:.6f}; expected slope {1 / 3:.6f}"
    )
    population = report["population_check"]
    print(
        "population logit gaps by kind: " + " ".join(f"{v:.6f}" for v in population["logit_gaps"])
    )
    print(
        f"population specialization: max gap {population['max_logit_gap']:.6f}; tolerance {population['tolerance']:g}; {'PASS' if population['passed'] else 'FAIL'}"
    )
    print(
        f"later calibration error: reference {report['reference_error']:.6f}, before {report['before_error']:.6f}, after {report['after_error']:.6f}"
    )
    unchanged = report["reference_choices"] == report["tuned_choices"] == report["mapped_choices"]
    print(f"most likely cause unchanged for all {len(report['tuned_choices'])} kinds: {unchanged}")


def figure_data(run=None):
    run = build_experiment() if run is None else run
    x = np.unique(np.r_[np.linspace(0.5, 0.99, 301), 1 - np.geomspace(1e-8, 0.01, 200), 1])
    curves = {}
    for beta in (1.0, 0.25):
        curves[str(beta)] = preference_probability(logit(x[:-1]) / (1 + 1 / beta)).tolist() + [1.0]
    points = {}
    probabilities = {
        "reference": preference_probability(run["reference"]),
        **{key: fit["probabilities"] for key, fit in run["fits"].items()},
    }
    for name, p in probabilities.items():
        table = run["evaluations"][name]["reliability"]
        observed = [urgent / n if q >= 0.5 else 1 - urgent / n for n, q, urgent in table]
        points[name] = dict(confidence=np.maximum(p, 1 - p).tolist(), observed_correctness=observed)
    u = np.linspace(-3, 3, 301)
    return dict(
        seed=run["seed"],
        sharpening=dict(
            confidence=x.tolist(),
            curves=curves,
            points=points,
            truth=[max(mu, 1 - mu) for mu in KINDS.shares],
        ),
        pair_loss=dict(
            gaps=u.tolist(),
            losses=np.logaddexp(0, -u).tolist(),
            marked_gaps=[0, 0.1, 0.8],
            marked_losses=[float(np.logaddexp(0, -v)) for v in (0, 0.1, 0.8)],
        ),
    )


def write_data(run=None):
    run = build_experiment() if run is None else run
    records = {
        "rlhf_alignment_figures.json": figure_data(run),
        "rlhf_alignment_delivered.json": delivered_record(run),
        "rlhf_alignment_report_05.json": exercise_five_record(),
    }
    DATA.mkdir(parents=True, exist_ok=True)
    for name, record in records.items():
        (DATA / name).write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-data", action="store_true")
    args = parser.parse_args()
    experiment = build_experiment()
    if args.write_data:
        write_data(experiment)


# region book:fnd.rlhf-alignment.train-01 output
if __name__ == "__main__":
    print_training(experiment)
# endregion book:fnd.rlhf-alignment.train-01


# region book:fnd.rlhf-alignment.delivered-01 output
if __name__ == "__main__":
    delivered(experiment)
# endregion book:fnd.rlhf-alignment.delivered-01


# region book:fnd.rlhf-alignment.keys-01 output
if __name__ == "__main__":
    print_keys()
# endregion book:fnd.rlhf-alignment.keys-01
