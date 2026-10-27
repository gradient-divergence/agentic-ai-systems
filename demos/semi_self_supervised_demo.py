"""Unlabeled structure, tested against the frozen months of chapter 5.

The original three printed listings run in chapter order on their own; choose-01
also runs independently, using only pre-August inputs. All score fits use training
inputs and final training labels, with validation used for the audit and comparisons.
The open and refused pools expose only decision-time inputs. August and September
are frozen before fitting and each is consulted once, after every fit.

`holdout-01` resumes a durable journal, replaying chapter 5's first read from its
stored printed record on first use. By default the journal lives in the working
directory, beside its manifest; set BOOK_GOLDEN_DIRECTORY to keep it across runs.
Output capture runs in a fresh directory, reproducing the two-entry book record.
Repeated runs against an existing directory append further reads, never reset it.

The budget experiments are fixed before either month is read. Generated tickets
have three queues drawn as clusters. Random labels are judged with and without
self-training, and labels on the tickets nearest the k-means centers with and
without spreading, so each pair shares its labels. Refund budgets hide training
labels and are judged on validation, including distance from the generator's
planted risk. No encoder is trained, and no result changes the refund agent or
its proposed score.
Run with --write-fixture to regenerate the deliberately contaminated exercise.
"""

# ruff: noqa: E402

import json
import os
import re
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if __name__ == "__main__":
    for thread_pool in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[thread_pool] = "1"

from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from demos.supervised_learning_demo import (
    GO_LIVE,
    MINUS,
    bootstrap,
    consulted,
    frozen,
    gate,
    planted_risk,
    print_record,
    realized_cost,
    risk,
    wilson_interval,
    windows,
)
from refund_agent.data.history import generate_history
from refund_agent.model.history_baseline import feature_matrix

OUTPUTS = Path(__file__).resolve().parents[1] / "book_outputs"
FIXTURE = Path(__file__).resolve().parent / "data" / "semi_self_supervised_selftrain_05.json"
INPUTS = (
    "amount_cents",
    "damaged",
    "changed_mind",
    "delivery_age_days",
    "prior_refund_count",
    "paid_cents",
    "order_evidence_age_seconds",
    "shipment_evidence_age_seconds",
)
PURPOSE = (
    "K=4, ten starts, seed 7, training-standardized eight inputs; nearest-center "
    "cutoff at training percentile 98; boosted comparator and self-training on open "
    "and refused tier 1 pools, confidence 0.9, three rounds, gate 0.080; all fits fixed"
)


def inputs(rows):
    """Inputs only; an open or unpaid outcome is never read even as a placeholder."""
    return np.nan_to_num(feature_matrix(rows))


def prepare():
    """Freeze both months and remove them from the lists any fit may receive."""
    history = generate_history(seed=7)
    window = windows([r for r in history if r.paid])
    august = frozen(window.pop("test"), "history-august")
    september = frozen(window.pop("shifted"), "history-september")
    refused = [r for r in history if not r.paid and r.at < GO_LIVE]
    pools = {
        "open": window["open"],
        "refused tier 1": [r for r in refused if r.amount_cents < 5000],
    }
    counts = {
        "history": len(history),
        "never_paid": sum(not r.paid for r in history),
        "refused_before": len(refused),
        "refused_above": sum(r.amount_cents >= 5000 for r in refused),
    }
    return window, pools, august, september, counts


def binary_losses(y, p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -(y * np.log(p) + (1 - y) * np.log1p(-p))


def classification_loss(model, X, y):
    p = model.predict_proba(X)
    column = {label: k for k, label in enumerate(model.classes_)}
    return float(
        -np.log(np.clip([p[i, column[label]] for i, label in enumerate(y)], 1e-12, 1)).mean()
    )


def audit_zeros(model, X, y, confidence=0.9):
    from demos.output_wrap import print_wrapped

    accepted = risk(model, X) <= 1 - confidence
    n, wrong = int(accepted.sum()), int(y[accepted].sum())
    low, high = wilson_interval(wrong, n)
    print_wrapped(
        f"validation accepted as 0 before round 1: {n}, wrong {wrong}, "
        f"rate {wrong / n:.4f} ({low:.4f} to {high:.4f})"
    )


def print_rounds(name, rows, base_rate):
    from demos.output_wrap import print_wrapped

    print_wrapped(f"{name}: labeled base rate {base_rate:.4f}; confidence 0.9, three rounds")
    print_wrapped("round  accepted  as 1  share 1  total pseudo  train loss  valid loss")
    for row in rows:
        print_wrapped(
            f"{row['round']:5} {row['accepted']:9} {row['by_class'].get(1, 0):5} "
            f"{row['share_one']:.4f} {row['total']:12} "
            f"{row['training_loss']:11.4f} {row['validation_loss']:11.4f}"
        )


# region book:fnd.semi-self-supervised.kmeans-01 output
import numpy as np


def lloyd_step(points, centers):
    """Assign to nearest centers; move nonempty centers to their case mean."""
    x, c = np.asarray(points, dtype=float), np.asarray(centers, dtype=float)
    group = ((x[:, None] - c) ** 2).argmin(axis=1)
    before = float(((x - c[group]) ** 2).sum())
    members = [x[group == k] for k in range(len(c))]
    moved = np.array(
        [m.mean() if len(m) else c[k] for k, m in enumerate(members)]
    )
    after = float(((x - moved[group]) ** 2).sum())
    return [m.tolist() for m in members], moved.tolist(), before, after


if __name__ == "__main__":
    points, centers = [1, 2, 8, 9], [1, 8]
    for step in (1, 2):
        groups, centers, before, after = lloyd_step(points, centers)
        print(f"step {step}: groups {groups}, objective {before:g}")
        print(f"  centers {centers}, objective {after:g}")
    one, four = lloyd_step(points, [1])[3], lloyd_step(points, points)[3]
    print(f"one group: {one:g}; four groups: {four:g}")
# endregion book:fnd.semi-self-supervised.kmeans-01


# region book:fnd.semi-self-supervised.selftrain-01 output
from demos import semi_self_supervised_demo as setup
from demos.output_wrap import print_wrapped
from demos.supervised_learning_demo import arrays, boosted


def self_train(X, y, pool, factory=boosted, confidence=0.9, rounds=3):
    """Keep accepted labels fixed; remove each accepted case from the pool."""
    model = factory().fit(X, y)
    remaining = np.arange(len(pool))
    report = []
    total = 0
    for step in range(1, rounds + 1):
        probabilities = model.predict_proba(pool[remaining])
        keep = probabilities.max(axis=1) >= confidence
        labels = model.classes_[probabilities[keep].argmax(axis=1)]
        accepted = int(keep.sum())
        if accepted:
            X = np.concatenate([X, pool[remaining[keep]]])
            y = np.concatenate([y, labels])
            model = factory().fit(X, y)
        total += accepted
        report.append(
            {
                "round": step,
                "accepted": accepted,
                "total": total,
                "by_class": {
                    int(c): int((labels == c).sum()) for c in model.classes_
                },
                "share_one": float((labels == 1).mean()) if accepted else 0.0,
                "training_loss": setup.classification_loss(model, X, y),
                "model": model,
            }
        )
        remaining = remaining[~keep]
        if not len(remaining):
            break
    return model, report


if __name__ == "__main__":
    window, pools, august, september, counts = setup.prepare()
    Xt, yt, _ = arrays(window["training"])
    Xv, yv, _ = arrays(window["validation"])
    supervised = boosted().fit(Xt, yt)
    setup.audit_zeros(supervised, Xv, yv)
    loss = setup.classification_loss
    print_wrapped(
        f"supervised: training loss {loss(supervised, Xt, yt):.4f}, "
        f"validation loss {loss(supervised, Xv, yv):.4f}"
    )
    self_models, traces = {}, {}
    for pool_name, pool_rows in pools.items():
        pool = setup.inputs(pool_rows)
        self_models[pool_name], traces[pool_name] = self_train(Xt, yt, pool)
        for row in traces[pool_name]:
            row["validation_loss"] = loss(row["model"], Xv, yv)
        setup.print_rounds(pool_name, traces[pool_name], yt.mean())
# endregion book:fnd.semi-self-supervised.selftrain-01


# region book:fnd.semi-self-supervised.contrast-01 output
import math


def contrastive_loss(scores, temperature=1.0):
    """Candidate 0 is the partner; differentiate by the unscaled scores."""
    z = np.asarray(scores, dtype=float) / temperature
    z -= z.max()  # the softmax is unchanged, and exponentials stay finite
    p = np.exp(z) / np.exp(z).sum()
    loss = float(-z[0] + math.log(np.exp(z).sum()))
    target = np.zeros(len(z))
    target[0] = 1
    return loss, p, (p - target) / temperature


if __name__ == "__main__":
    for scores, temperature in (
        ([2, 0, 0], 1),
        ([0, 0, 0], 1),
        ([2, 0, 0], 0.5),
    ):
        loss, p, gradient = contrastive_loss(scores, temperature)
        print(f"scores {scores}, temperature {temperature}: loss {loss:.4f}")
        print(
            f"  probabilities {[round(float(v), 3) for v in p]}, "
            f"gradient {[round(float(v), 3) for v in gradient]}"
        )
    for candidates in (3, 8, 64):
        loss, _, _ = contrastive_loss(np.zeros(candidates))
        print(f"collapsed, {candidates} candidates: {loss:.4f}")
# endregion book:fnd.semi-self-supervised.contrast-01


# region book:fnd.semi-self-supervised.pools-01 output
# Counts of labels and the two reasons a request has no label yet; output only.
if __name__ == "__main__":
    print(f"history: {counts['history']:,}; never paid: {counts['never_paid']:,}")
    print(
        f"labeled by 1 August: {len(yt) + len(yv):,} (training {len(yt):,}, validation {len(yv):,})"
    )
    print(f"open paid pool before 1 August: {len(pools['open']):,}")
    print(
        f"refused before 1 August: {counts['refused_before']:,}; "
        f"tier 1: {len(pools['refused tier 1']):,}; above tier 1: {counts['refused_above']:,}"
    )
    print("Refused requests above tier 1 are outside every labeled request's amount range.")
# endregion book:fnd.semi-self-supervised.pools-01


def unit_partitions(window):
    rows = window["training"] + window["validation"] + window["open"]
    all_inputs = inputs(rows)
    two = all_inputs[:, [0, 3]]
    training = inputs(window["training"])
    scaler = StandardScaler().fit(training[:, [0, 3]])
    raw = KMeans(n_clusters=4, n_init=10, random_state=7).fit(two)
    scaled = KMeans(n_clusters=4, n_init=10, random_state=7).fit(scaler.transform(two))
    another = KMeans(n_clusters=4, n_init=10, random_state=17).fit(scaler.transform(two))
    pick = np.linspace(0, len(rows) - 1, 1000, dtype=int)
    figure = {
        "points": (two[pick] / [100, 1]).round(4).tolist(),
        "cents_groups": raw.labels_[pick].tolist(),
        "scaled_groups": scaled.labels_[pick].tolist(),
    }
    return two, scaler, raw, scaled, adjusted_rand_score(scaled.labels_, another.labels_), figure


# region book:fnd.semi-self-supervised.units-01 output
# Two partitions of the same requests, and the eight-input training centers; output only.
if __name__ == "__main__":
    two, two_scaler, raw_groups, scaled_groups, stability, unit_figure = unit_partitions(window)
    print(f"paid tier 1 before 1 August: {len(two):,}; K=4, ten starts, seed 7")
    for name, model, centers in (
        ("amount in cents, days as recorded", raw_groups, raw_groups.cluster_centers_),
        (
            "both standardized on training",
            scaled_groups,
            two_scaler.inverse_transform(scaled_groups.cluster_centers_),
        ),
    ):
        print(name)
        for k, center in enumerate(centers):
            print(
                f"  group {k + 1}: {int((model.labels_ == k).sum())}, "
                f"amount ${center[0] / 100:.2f}, delivery age {center[1]:.2f} days"
            )
    print(f"standardized groups, seed 7 against seed 17: adjusted Rand {stability:.4f}")
    group_scaler = StandardScaler().fit(Xt)
    group_model = KMeans(n_clusters=4, n_init=10, random_state=7).fit(group_scaler.transform(Xt))
    group_centers = group_scaler.inverse_transform(group_model.cluster_centers_)
    group_centers[np.abs(group_centers) < 1e-12] = 0
    print("eight inputs: " + ", ".join(INPUTS))
    for k, center in enumerate(group_centers):
        print(
            f"  group {k + 1}: "
            + ", ".join(f"{name}={value:.2f}" for name, value in zip(INPUTS, center))
        )
    standardized_spread = np.ptp(group_model.cluster_centers_, axis=0)
    print(
        "center separation, standardized range: "
        + ", ".join(
            f"{INPUTS[k]}={standardized_spread[k]:.2f}" for k in np.argsort(-standardized_spread)
        )
    )
    anomaly_cut = float(
        np.quantile(group_model.transform(group_scaler.transform(Xt)).min(axis=1), 0.98)
    )
    print(f"nearest-center distance, training percentile 98: {anomaly_cut:.4f}")
# endregion book:fnd.semi-self-supervised.units-01


def print_density_choices(x):
    """A neighbourhood includes its own point; -1 marks noise, never a group."""
    for count in (2, 3):
        labels = DBSCAN(eps=1.5, min_samples=count).fit_predict(x)
        groups = [x[labels == k, 0].tolist() for k in np.unique(labels) if k >= 0]
        noise = x[labels == -1, 0].tolist()
        print(f"density radius 1.5, min count {count}: {groups}; noise {noise}")


def refund_choices():
    """Measure a cut's geometry and repeatability, without outcomes or frozen reads."""
    paid = [r for r in generate_history(seed=7) if r.paid and r.at < GO_LIVE]
    # Reuse the plot's ordering and training-only scaler, without freezing any month.
    two, scaler, _, _, _, _ = unit_partitions(windows(paid))
    plane = scaler.transform(two)
    pick = np.linspace(0, len(plane) - 1, 1000, dtype=int)
    choices = []
    for k in range(2, 9):
        first = KMeans(n_clusters=k, n_init=10, random_state=7).fit(plane)
        second = KMeans(n_clusters=k, n_init=10, random_state=17).fit(plane)
        choices.append(
            {
                "K": k,
                "objective": float(first.inertia_),
                "silhouette": float(silhouette_score(plane[pick], first.labels_[pick])),
                "adjusted_rand": float(adjusted_rand_score(first.labels_, second.labels_)),
            }
        )
    return {"fit_rows": len(plane), "plot_rows": len(pick), "choices": choices}


def print_refund_choices():
    report = refund_choices()
    print(
        f"refund: fit {report['fit_rows']}; silhouette {report['plot_rows']} plotted rows; "
        "ten starts"
    )
    print("K  objective(full)  silhouette(plot)  adjusted Rand(full; seeds 7,17)")
    for row in report["choices"]:
        print(
            f"{row['K']}  {row['objective']:15.4f}  {row['silhouette']:16.4f}  "
            f"{row['adjusted_rand']:.4f}"
        )


# region book:fnd.semi-self-supervised.choose-01 output
import numpy as np
from scipy.special import softmax
from sklearn.metrics import silhouette_samples

from demos import semi_self_supervised_demo as demo

if __name__ == "__main__":
    x = np.array([1, 2, 8, 9]).reshape(-1, 1)
    for groups in ([0, 0, 1, 1], [0, 0, 1, 2]):
        v = silhouette_samples(x, groups)
        print(f"silhouettes: {v.round(3).tolist()}; mean {v.mean():.3f}")
    p = softmax(-(np.array([[4], [5]]) - [1.5, 8.5]) ** 2 / 8, axis=1)
    print(f"Pr(first | x=4, x=5): {p[:, 0].round(3).tolist()}")
    demo.print_density_choices(x)
    demo.print_refund_choices()
# endregion book:fnd.semi-self-supervised.choose-01


def tickets(seed=7):
    """Three fictional service queues; their clusters and labels coincide by construction."""
    rng = np.random.default_rng(seed)
    centers = np.array([[-2.5, 0.0], [2.5, 0.0], [0.0, 3.5]])
    y = rng.integers(0, 3, 4000)
    X = centers[y] + rng.normal(0, 1.1, size=(4000, 2))
    # A random, disjoint holdout is fixed before any acquisition or fit.
    order = rng.permutation(len(X))
    pool, holdout = order[:3000], order[3000:]
    Xh, yh = X[holdout].copy(), y[holdout].copy()
    Xh.flags.writeable = yh.flags.writeable = False
    return X[pool], y[pool], Xh, yh


def ticket_classifier():
    return KNeighborsClassifier(n_neighbors=3)


def ticket_run(seed, budget, data):
    X, y, Xh, yh = data
    chosen = np.random.default_rng(seed).choice(len(X), budget, replace=False)
    unlabeled = np.setdiff1d(np.arange(len(X)), chosen)
    supervised = ticket_classifier().fit(X[chosen], y[chosen])
    trained, trace = self_train(X[chosen], y[chosen], X[unlabeled], ticket_classifier)
    groups = KMeans(n_clusters=budget, n_init=10, random_state=seed).fit(X)
    # Each representative belongs to its group; distinct groups cannot reuse a label.
    representatives = np.array(
        [
            np.flatnonzero(groups.labels_ == k)[
                ((X[groups.labels_ == k] - center) ** 2).sum(axis=1).argmin()
            ]
            for k, center in enumerate(groups.cluster_centers_)
        ]
    )
    # The same labels twice: on their own, then spread through each ticket's group.
    central = ticket_classifier().fit(X[representatives], y[representatives])
    spread = y[representatives][groups.labels_]
    propagated = ticket_classifier().fit(X, spread)
    accuracy = {
        name: float((model.predict(Xh) == yh).mean())
        for name, model in (
            ("random labels", supervised),
            ("self-trained", trained),
            ("center labels", central),
            ("center labels, spread", propagated),
        )
    }
    return accuracy, trace, (X, y, Xh, yh, chosen, unlabeled, representatives)


def spread_summary(values):
    """Spread across the twenty draws, not an interval on an individual ticket."""
    values = np.asarray(values)
    return [round(float(values.mean()), 6), *np.quantile(values, [0.025, 0.975]).round(6).tolist()]


def budget_experiments(Xt, yt, Xv, yv, truth):
    data = tickets()
    ticket_rows, refund_rows = [], []
    for budget in (6, 12, 24, 48, 96):
        results = [ticket_run(seed, budget, data)[0] for seed in range(20)]
        ticket_rows.append(
            {
                "labels": budget,
                **{name: spread_summary([r[name] for r in results]) for name in results[0]},
            }
        )
    for budget in (300, 1000, 3000):
        values = {
            name: {key: [] for key in ("log_loss", "risk_mae", "mean")}
            for name in ("supervised", "self-trained")
        }
        for seed in range(20):
            chosen = np.random.default_rng(seed).choice(len(Xt), budget, replace=False)
            remaining = np.setdiff1d(np.arange(len(Xt)), chosen)
            base = boosted().fit(Xt[chosen], yt[chosen])
            trained, _ = self_train(Xt[chosen], yt[chosen], Xt[remaining])
            for name, model in (("supervised", base), ("self-trained", trained)):
                p = risk(model, Xv)
                values[name]["log_loss"].append(float(binary_losses(yv, p).mean()))
                values[name]["risk_mae"].append(float(np.abs(p - truth).mean()))
                values[name]["mean"].append(float(p.mean()))
        refund_rows.append(
            {
                "labels": budget,
                **{
                    name: {key: spread_summary(v) for key, v in metrics.items()}
                    for name, metrics in values.items()
                },
            }
        )
    return {
        "tickets": ticket_rows,
        "refunds": refund_rows,
        "ticket_pool": 3000,
        "ticket_holdout": 1000,
        "draws": 20,
        "refund_validation": len(yv),
        "validation_risk": round(float(truth.mean()), 6),
    }


def contaminated_report():
    """Exercise 5's single defect: a post-fit pool slice with mixed label provenance."""
    data = tickets()
    X, y, Xh, yh = data
    chosen = np.random.default_rng(7).choice(len(X), 12, replace=False)
    unlabeled = np.setdiff1d(np.arange(len(X)), chosen)
    _, trace = self_train(X[chosen], y[chosen], X[unlabeled], ticket_classifier)
    # The cut is made after round 1; half the targets are each model's own predictions.
    # This single evaluation defect inflates the reported accuracy in every round.
    cut = unlabeled[:1000]
    rounds = []
    for row in trace:
        prediction = row["model"].predict(X[cut])
        targets = y[cut].copy()
        targets[:500] = prediction[:500]
        rounds.append(
            {
                "round": row["round"],
                "accepted": row["accepted"],
                "accepted_by_queue": [row["by_class"].get(k, 0) for k in range(3)],
                "holdout_accuracy": round(float((prediction == targets).mean()), 4),
            }
        )
    honest = [round(float((row["model"].predict(Xh) == yh).mean()), 4) for row in trace]
    report = {
        "task": "three service-desk queues",
        "seed": 7,
        "labeled": 12,
        "confidence": 0.9,
        "rounds": rounds,
        "holdout": {
            "requests": len(cut),
            "source": "unlabeled pool",
            "cut_after_round": 1,
            "outcome_labels": 500,
            "pseudo_labels": 500,
            "pseudo_label_source": "the model being judged in each round",
        },
    }
    return report, honest


# region book:fnd.semi-self-supervised.budget-01 output
# Label-budget comparisons, fixed before either frozen month is read; output only.
if __name__ == "__main__":
    truth_v = np.array([planted_risk(r) for r in window["validation"]])
    budgets = budget_experiments(Xt, yt, Xv, yv, truth_v)
    print("twenty draws, seeds 0 to 19; mean (2.5th to 97.5th percentile across draws)")
    print("tickets: pool 3000, frozen random holdout 1000; three queues, 3-nearest neighbors")
    ticket_names = ("random labels", "self-trained", "center labels", "center labels, spread")
    print("labels" + "".join(f"  {name:26}" for name in ticket_names).rstrip() + "  (accuracy)")
    for row in budgets["tickets"]:
        print(
            f"{row['labels']:6}"
            + "".join(
                f"  {v[0]:.4f} ({v[1]:.4f} to {v[2]:.4f})" for v in map(row.get, ticket_names)
            )
        )
    print(
        f"refunds: training labels hidden; validation {len(yv)}, "
        f"mean planted risk {truth_v.mean():.4f}"
    )
    print("labels  model          validation log loss     planted-risk MAE        mean prediction")
    for row in budgets["refunds"]:
        for name in ("supervised", "self-trained"):
            print(
                f"{row['labels']:6}  {name:12}"
                + "".join(
                    f"  {v[0]:.4f} ({v[1]:.4f} to {v[2]:.4f})"
                    for v in (row[name][n] for n in ("log_loss", "risk_mae", "mean"))
                )
            )
    planted, honest_ticket = contaminated_report()
# endregion book:fnd.semi-self-supervised.budget-01


def resume_chapter_five(golden, source, directory):
    """Replay the actual printed consultation, never an invented earlier experiment."""
    text = (OUTPUTS / f"_fnd.supervised-learning.{source}.txt").read_text(encoding="utf-8")
    fingerprint = re.search(r"^sha256: ([0-9a-f]+)$", text, re.M)
    if fingerprint is None or not golden.manifest["sha256"].startswith(fingerprint[1]):
        raise ValueError("chapter 5's printed fingerprint differs from this frozen month")
    entries = re.findall(r"^  (\d+): ([^,]+), ([^,]+), (.+)$", text, re.M)
    prior = [
        {"entry": int(n), "set": name, "reader": reader, "purpose": purpose}
        for n, name, reader, purpose in entries
        if name == golden.manifest["name"]
    ]
    if len(prior) != 1 or prior[0]["entry"] != 1:
        raise ValueError("chapter 5's printed record must contain exactly its first read")
    golden.persist(directory, prior_log=prior)


def anomaly_row(name, X, y):
    distance = group_model.transform(group_scaler.transform(X)).min(axis=1)
    flagged = distance > anomaly_cut
    n, wrong = int(flagged.sum()), int(y[flagged].sum())
    low, high = wilson_interval(wrong, n) if n else (0, 1)
    return (
        f"| {name} | {n}/{len(y)} ({flagged.mean():.4f}) | {wrong} "
        f"({wrong / n:.4f}, {low:.4f} to {high:.4f}) | "
        f"{int(y.sum())}/{len(y)} ({y.mean():.4f}) |"
    )


def estimate(values):
    mean, low, high = bootstrap(values)
    return f"{mean:.4f} ({low:.4f} to {high:.4f})"


def holdout_tables(X, y, amount):
    group = group_model.predict(group_scaler.transform(X))
    print(
        "| Group | August count | Center in recorded units | Wrong payments | Rate, 95% Wilson interval |"
    )
    print("|---|---|---|---|---|")
    for k, center in enumerate(group_centers):
        chosen = group == k
        n, wrong = int(chosen.sum()), int(y[chosen].sum())
        low, high = wilson_interval(wrong, n)
        description = ", ".join(f"{name}={value:.2f}" for name, value in zip(INPUTS, center))
        print(
            f"| {k + 1} | {n} | {description} | {wrong} | "
            f"{wrong / n:.4f} ({low:.4f} to {high:.4f}) |"
        )
    print(
        "\n| Month and inputs | Flagged | Wrong among flagged, rate and 95% interval | Overall wrong |"
    )
    print("|---|---|---|---|")
    print(anomaly_row("August, seconds", X, y))
    changed = X.copy()
    changed[:, [6, 7]] *= 1000
    print(anomaly_row("August, milliseconds", changed, y))
    print(anomaly_row("September, seconds", X_s, y_s))
    print(
        "\nPaired differences are method minus supervised; 95% percentile bootstrap, 2000 draws, seed 7."
    )
    print(
        "| Score | Mean prediction (difference) | Log loss (difference) | Referred (share difference) | Wrong payments paid (share difference) | Cost per request (difference) | Combined training loss |"
    )
    print("|---|---|---|---|---|---|---|")
    p0 = risk(supervised, X)
    loss0, pay0 = binary_losses(y, p0), gate(p0)
    cost0 = realized_cost(y, amount, pay0)
    for name, model in {"supervised": supervised, **self_models}.items():
        p = risk(model, X)
        loss, pay = binary_losses(y, p), gate(p)
        cost = realized_cost(y, amount, pay)
        training_loss = (
            classification_loss(model, Xt, yt)
            if name == "supervised"
            else traces[name][-1]["training_loss"]
        )
        print(
            f"| {name} | {p.mean():.4f} ({estimate(p - p0)}) | "
            f"{loss.mean():.4f} ({estimate(loss - loss0)}) | "
            f"{int((~pay).sum())} ({estimate(pay0.astype(float) - pay)}) | "
            f"{int(y[pay].sum())} ({estimate(y * (pay.astype(float) - pay0))}) | "
            f"{cost.mean():.4f} ({estimate(cost - cost0)}) | {training_loss:.4f} |"
        )


# region book:fnd.semi-self-supervised.holdout-01 output
# Every fit is over; one logged consultation of each frozen month feeds all three tables.
if __name__ == "__main__":
    journal = Path(os.environ.get("BOOK_GOLDEN_DIRECTORY", "golden-sets"))
    resume_chapter_five(august, "august-01", journal)
    resume_chapter_five(september, "numbers-01", journal)
    X_a, y_a, a_a = consulted(august, PURPOSE)
    X_s, y_s, a_s = consulted(september, PURPOSE)
    print_record(august)
    print_record(september)
    print(f"August: {len(y_a)}, wrong {int(y_a.sum())} ({y_a.mean():.4f})")
    print(f"September: {len(y_s)}, wrong {int(y_s.sum())} ({y_s.mean():.4f})")
    holdout_tables(X_a, y_a, a_a)
    print(f"fingerprints intact: August {august.intact()}, September {september.intact()}")
# endregion book:fnd.semi-self-supervised.holdout-01


# The four regions below print holdout-01's record and tables again at the width of a
# printed page. They read nothing new: each uses the arrays the two consultations above
# returned, and the fitted groups, cutoff and scores as they stood before those reads.
SEPARATING = {
    "damaged": "Share of damage claims",
    "changed_mind": "Share of changes of mind",
    "delivery_age_days": "Mean days since delivery",
    "prior_refund_count": "Mean earlier refunds",
}


def print_wrapped_record(golden, width=66):
    """A frozen set's name, size and fingerprint, then its whole log, wrapped."""
    for key in ("name", "requests", "sha256"):
        value = golden.manifest[key]
        print(f"{key}: {value[:16] if key == 'sha256' else value}")
    print("access log, every entry:")
    for entry in golden.log:
        line = f"{entry['entry']}: {entry['set']}, {entry['reader']}, {entry['purpose']}"
        print(textwrap.fill(line, width, initial_indent="  ", subsequent_indent="     "))


def groups_table(X, y):
    """One column a group: its center in the inputs that separate the centers, and its rate."""
    from demos.output_wrap import print_wrapped

    group = group_model.predict(group_scaler.transform(X))
    counts = [
        (int((group == k).sum()), int(y[group == k].sum())) for k in range(len(group_centers))
    ]
    intervals = [wilson_interval(wrong, n) for n, wrong in counts]

    def row(label, cells):
        print_wrapped(f"| {label} | " + " | ".join(cells) + " |")

    row("Measure", [f"Group {k + 1}" for k in range(len(counts))])
    print_wrapped("|---" * (len(counts) + 1) + "|")
    for name, label in SEPARATING.items():
        row(label, [f"{center[INPUTS.index(name)]:.2f}" for center in group_centers])
    row("Wrong payments", [f"{wrong} of {n}" for n, wrong in counts])
    row("Rate", [f"{wrong / n:.3f}" for n, wrong in counts])
    row("95% lower", [f"{low:.3f}" for low, high in intervals])
    row("Upper", [f"{high:.3f}" for low, high in intervals])


def flags_table(months):
    """The nearest-center flags of each month: how many, and how many were wrong payments."""
    from demos.output_wrap import print_wrapped

    print_wrapped("| Month and units | Flagged | Wrong flagged | Precision | 95% interval |")
    print_wrapped("|---|---|---|---|---|")
    for name, X, y in months:
        distance = group_model.transform(group_scaler.transform(X)).min(axis=1)
        flagged = distance > anomaly_cut
        n, wrong = int(flagged.sum()), int(y[flagged].sum())
        low, high = wilson_interval(wrong, n) if n else (0, 1)
        print_wrapped(
            f"| {name} | {n} of {len(y)} | {wrong} of {int(y.sum())} "
            f"| {wrong / n:.3f} | {low:.3f} to {high:.3f} |"
        )


def scores_table(X, y, amount):
    """One row a measure, one column a score; each change is paired with the supervised."""
    models = {"supervised": supervised, **self_models}
    fitted = {
        "supervised": (
            classification_loss(supervised, Xt, yt),
            classification_loss(supervised, Xv, yv),
        )
    }
    for name in self_models:
        last = traces[name][-1]
        fitted[name] = (last["training_loss"], last["validation_loss"])
    p = {name: risk(model, X) for name, model in models.items()}
    losses = {name: binary_losses(y, p[name]) for name in models}
    pay = {name: gate(p[name]) for name in models}
    cost = {name: realized_cost(y, amount, pay[name]) for name in models}

    def row(label, cells):
        print(f"| {label} | " + " | ".join(cells) + " |")

    def change(values):
        paired = (estimate(values[n] - values["supervised"]) for n in self_models)
        return ["", *(text.replace("-", MINUS) for text in paired)]

    print("| Measure | Supervised | Open pool | Refused pool |")
    print("|---|---|---|---|")
    row("Training loss", [f"{fitted[n][0]:.4f}" for n in models])
    row("Validation loss", [f"{fitted[n][1]:.4f}" for n in models])
    row("Mean prediction", [f"{p[n].mean():.4f}" for n in models])
    row("change", change(p))
    row("Log loss", [f"{losses[n].mean():.4f}" for n in models])
    row("change", change(losses))
    row("Referred", [str(int((~pay[n]).sum())) for n in models])
    row("Wrong payments paid", [str(int(y[pay[n]].sum())) for n in models])
    row("Cost per request", [f"{cost[n].mean():.4f}" for n in models])
    row("change", change(cost))


# region book:fnd.semi-self-supervised.record-01 output
# August's record after this chapter's read, its log wrapped to the page; output only.
if __name__ == "__main__":
    print_wrapped_record(august)
# endregion book:fnd.semi-self-supervised.record-01


# region book:fnd.semi-self-supervised.groups-01 output
# The four groups against August's labels; output only.
if __name__ == "__main__":
    groups_table(X_a, y_a)
# endregion book:fnd.semi-self-supervised.groups-01


# region book:fnd.semi-self-supervised.flags-01 output
# The anomaly flags in August, in August with two inputs in milliseconds, and in September.
if __name__ == "__main__":
    in_milliseconds = X_a.copy()
    in_milliseconds[:, [6, 7]] *= 1000
    flags_table(
        (
            ("August, seconds", X_a, y_a),
            ("August, milliseconds", in_milliseconds, y_a),
            ("September, seconds", X_s, y_s),
        )
    )
# endregion book:fnd.semi-self-supervised.flags-01


# region book:fnd.semi-self-supervised.scores-01 output
# The supervised score and the two self-trained scores on three windows; output only.
if __name__ == "__main__":
    scores_table(X_a, y_a, a_a)
# endregion book:fnd.semi-self-supervised.scores-01


# region book:fnd.semi-self-supervised.numbers-01 output
# Arithmetic and qualifications the writer needs beside the measured tables; output only.
if __name__ == "__main__":
    print(f"soft target: 0.12 - 0.12 = {0.12 - 0.12:g}, gradient zero")
    print(f"dilution: 300 x 0.04 / (300 + 2700) = {300 * 0.04 / 3000:.4f}")
    theoretical = 1 / (1 + math.exp(-(-5 + 0.7 + 0.6 * 13 / 14 + 0.4 + 0.7 + 1.5)))
    print(f"planted risk at paid-input extremes before September: {theoretical:.4f}")
    before = window["training"] + window["validation"] + window["open"]
    print(
        f"largest planted risk among actual paid requests before 1 August: "
        f"{max(planted_risk(r) for r in before):.4f}"
    )
    for pool_name, pool_rows in pools.items():
        # Round 1 is scored by the supervised fit, each later round by the refit before it.
        scorers = [supervised, *(row["model"] for row in traces[pool_name][:-1])]
        largest = max(float(risk(model, inputs(pool_rows)).max()) for model in scorers)
        print(f"largest predicted risk in the {pool_name} pool, any round: {largest:.4f}")
    for tau in (1, 0.5):
        scores, eps = np.array([2.0, 0.0, 0.0]), 1e-6
        gradient = contrastive_loss(scores, tau)[2]
        numeric = np.array(
            [
                (
                    contrastive_loss(scores + eps * row, tau)[0]
                    - contrastive_loss(scores - eps * row, tau)[0]
                )
                / (2 * eps)
                for row in np.eye(3)
            ]
        )
        print(
            f"contrastive finite difference, temperature {tau}: "
            f"max error {np.abs(gradient - numeric).max():.2e}"
        )
    print("Scores (2, 0, 0) and (3, 1, 0) are arithmetic inputs, not dot products of unit vectors.")
    print("Ticket confidence 0.9 requires all three nearest labeled neighbors to agree.")
    print("Refund rounds are a fixed diagnostic, not a validation-selected stopping rule.")
# endregion book:fnd.semi-self-supervised.numbers-01


# region book:fnd.semi-self-supervised.keys-01 output
# Exercise answers and the honest figure for the contaminated report; output only.
if __name__ == "__main__":
    print(
        "exercise 1: groups [1, 2], [8, 9]; centers [1.5, 8.5]; objective 2 then 1, four groups 0"
    )
    group = group_model.predict(group_scaler.transform(X_a))
    rates = [y_a[group == k].mean() for k in range(4)]
    highest = int(np.argmax(rates))
    print(
        f"exercise 1b: group {highest + 1}, observed rate {rates[highest]:.4f}; the labels chose it"
    )
    print(f"exercise 2: soft-target gradient 0; equal three-candidate loss {math.log(3):.4f}")
    c = [0, 2]
    for step in (1, 2, 3):
        groups, c, before, after = lloyd_step([0, 2, 10, 12], c)
        print(
            f"exercise 3, step {step}: groups {groups}, objective {before:g}; "
            f"centers {c}, objective {after:g}"
        )
    print(
        f"exercise 4a: 40 / (400 + 1600) = {40 / 2000:.3f}; "
        "at gate 0.080, referred at 0.100, paid at 0.020"
    )
    for tau in (1, 0.5):
        loss, _, gradient = contrastive_loss([3, 1, 0], tau)
        print(
            f"exercise 4b: temperature {tau}, loss {loss:.4f}, "
            f"gradient {[round(float(v), 3) for v in gradient]}"
        )
    print(
        f"exercise 5: contaminated accuracies "
        f"{[row['holdout_accuracy'] for row in planted['rounds']]}; "
        f"honest frozen-holdout accuracies {honest_ticket}"
    )
    print(
        "exercise 5: the pool slice was taken after round 1 and half its labels are guesses; "
        "replace that one evaluation defect with the frozen random holdout."
    )
    first = traces["refused tier 1"][0]
    low, high = 0.5 * yt.mean(), 1.5 * yt.mean()
    print(
        f"exercise 6 reference, band [{low:.4f}, {high:.4f}]: "
        f"proposed {first['accepted']}, share 1 {first['share_one']:.4f}, "
        f"base rate {yt.mean():.4f}: stop before adding any; journal gains 0 entries"
    )
    if "--write-fixture" in sys.argv:
        FIXTURE.write_text(json.dumps(planted, indent=2) + "\n", encoding="utf-8")
# endregion book:fnd.semi-self-supervised.keys-01


# region book:fnd.semi-self-supervised.figures-01 output
# Data for the two plots and the margin curve; no draw.io diagram is made here.
if __name__ == "__main__":
    leads = np.linspace(-2, 6, 81)
    figures = {
        "units": unit_figure,
        "budget": budgets,
        "contrastive": {
            "temperature": 1,
            "candidates": 3,
            "lead": leads.round(4).tolist(),
            "loss": [round(contrastive_loss([lead, 0, 0])[0], 6) for lead in leads],
        },
    }
    print(json.dumps(figures, indent=1))
# endregion book:fnd.semi-self-supervised.figures-01
