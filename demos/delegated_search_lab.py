"""Teaching runs on development rows: equal trials, EI, inputs and stacking.

Nothing here enters the selection record or consults a frozen month. The grid
is a union of conditional products; random uses the chapter's own seeded draw.
All three arms fit full configurations: trials are equal, not CPU time.
The stored trace is plotted by figures/plots/fig_delegated_search.py.
"""

import itertools
from functools import lru_cache

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit

from demos.delegated_model_demo import (
    GO_LIVE,
    SPACE,
    arrays,
    binary_losses,
    boosted,
    commission_for,
    configurations,
    fit_delivery,
    fresh_truth,
    log_loss,
    make_model,
    prepare,
    risk,
)
from demos.delegated_model_pipeline import GroupRisk
from demos.supervised_learning_demo import FIT_BY

# Fixed before running: four logistic points plus a 2^4 boosted product, L2 fixed.
GRID = {
    "logistic": {"C": [0.01, 0.1, 1.0, 10.0]},
    "boosted": {
        "max_iter": [50, 200],
        "learning_rate": [0.025, 0.1],
        "max_leaf_nodes": [3, 15],
        "min_samples_leaf": [50, 200],
        "l2_regularization": [1.0],
    },
}
GRID_DEFINITION = (
    "grid: logistic C={0.01,0.1,1,10}; boosted "
    "max_iter={50,200} x learning_rate={0.025,0.1} x "
    "max_leaf_nodes={3,15} x min_samples_leaf={50,200}; l2_regularization=1"
)


def development_only(window):
    """Reject frozen/open rows before any fit or label extraction."""
    if set(window) != {"training", "validation"} or any(
        r.at >= GO_LIVE or r.label_final_at is None or r.label_final_at > cutoff
        for name, cutoff in (("training", FIT_BY), ("validation", GO_LIVE))
        for r in window[name]
    ):
        raise ValueError("lab receives development rows only")


def catalog(space=SPACE):
    return [
        {"family": family, **dict(zip(params, values))}
        for family, params in space.items()
        for values in itertools.product(*params.values())
    ]


def encoding(settings):
    keys = ["C", *SPACE["boosted"]]
    scales = [100, 200, 0.1, 15, 200, 10]
    return [
        settings["family"] == "boosted",
        *[settings.get(k, 0) / s for k, s in zip(keys, scales)],
    ]


# region book:fnd.automl-and-selection.ei-01 output
import numpy as np
from scipy.stats import norm


def expected_improvement(mean, std, best):
    mean, std = np.asarray(mean), np.asarray(std)
    gain = best - mean
    scaled = np.divide(gain, std, out=np.zeros_like(gain), where=std > 0)
    normal_ei = gain * norm.cdf(scaled) + std * norm.pdf(scaled)
    return np.where(std > 0, normal_ei, np.maximum(gain, 0))


if __name__ == "__main__":
    ei = expected_improvement([0.1660, 0.1672], [0.0002, 0.0030], 0.1666)
    print(f"given best 0.1666: EI(A)={ei[0]:.5f}; EI(B)={ei[1]:.5f}")
    print(f"try {'AB'[ei.argmax()]} next (larger expected improvement)")
# endregion book:fnd.automl-and-selection.ei-01


def compare(window, budget=20, seed=7):
    development_only(window)
    grid = catalog(GRID)
    if not 1 <= budget <= len(grid):
        raise ValueError("budget must fit the twenty-point grid")
    commission = commission_for(window, budget, seed)
    space = catalog()
    X = np.array([encoding(s) for s in space])
    random_order = [space.index(s) for s in configurations(budget, seed)]
    grid_order = [space.index(s) for s in grid[:budget]]
    # This is the chapter's baseline, outside each arm's twenty search trials.
    baseline = make_model(
        {"family": "boosted", **{k: boosted().get_params()[k] for k in SPACE["boosted"]}},
        seed,
    ).fit(*arrays(window["training"])[:2])
    Xv, yv, _ = arrays(window["validation"])
    baseline_loss = log_loss(yv, risk(baseline, Xv))
    pool, mu, _ = fresh_truth()
    base_truth = binary_losses(mu, risk(baseline, arrays(pool)[0])).mean()
    records = {}
    for method in ("grid", "random", "model-based"):
        chosen, losses, trials = [], [], []
        winner = None
        for n in range(budget):
            if method == "grid":
                index = grid_order[n]
            elif method == "random" or n < 3:
                index = random_order[n]
            else:
                gp = GaussianProcessRegressor(
                    kernel=Matern(nu=2.5),
                    alpha=1e-6,
                    normalize_y=True,
                    optimizer=None,
                    random_state=seed,
                )
                gp.fit(X[chosen], losses)
                mean, std = gp.predict(X, return_std=True)
                acquisition = expected_improvement(mean, std, min(losses))
                acquisition[chosen] = -np.inf
                index = int(acquisition.argmax())
            delivery = fit_delivery(f"{method} {n + 1}", method, space[index], window, commission)
            chosen.append(index)
            losses.append(delivery.reported_loss)
            if winner is None or delivery.reported_loss < winner.reported_loss:
                winner = delivery
            trials.append(
                {
                    "entry": n + 1,
                    "settings": space[index],
                    "loss": losses[-1],
                    "best_so_far": min(losses),
                    "commission": commission.sha256,
                }
            )
        best = min(trials, key=lambda t: t["loss"])
        best.update(
            baseline_loss=baseline_loss,
            measured_lead=float(baseline_loss - winner.reported_loss),
            true_lead=float(base_truth - binary_losses(mu, winner.predict(pool, mapped=False)).mean()),
        )
        records[method] = trials
    return records


@lru_cache(maxsize=1)
def search_run():
    """Run once per process; the printable table and JSON share these exact fits."""
    window, _ = prepare()
    records = compare(window)
    return {
        "seed": 7,
        "budget": 20,
        "training_requests": len(window["training"]),
        "validation_requests": len(window["validation"]),
        "fresh_seed": 71,
        "fresh_requests": len(fresh_truth()[0]),
        "grid_definition": GRID_DEFINITION,
        "model_based_start": "random arm's first three configurations, in order",
        "trials": records,
    }


def print_comparison(result):
    from demos.output_wrap import print_wrapped

    print_wrapped("20 full trials per arm, seed 7; model-based starts with random's first 3")
    print_wrapped(result["grid_definition"])
    print_wrapped("best raw validation log loss:     5       10       20")
    for method, trials in result["trials"].items():
        best = min(trials, key=lambda t: t["loss"])
        checkpoints = "  ".join(f"{trials[n - 1]['best_so_far']:.4f}" for n in (5, 10, 20))
        print_wrapped(f"{method:28} {checkpoints}")
        print_wrapped("winner: " + ", ".join(f"{k}={v}" for k, v in best["settings"].items()))
        print_wrapped(
            f"lead over baseline {best['baseline_loss']:.4f}: {best['measured_lead']:.6f}; "
            f"estimated true lead ({result['fresh_requests']:,}, seed 71): {best['true_lead']:.6f}"
        )


def stacking(window, seed=7):
    """Fit the meta-model on time-safe out-of-fold predictions of training rows."""
    development_only(window)
    rows = window["training"]
    X, y, _ = arrays(rows)
    settings = (
        {"family": "logistic", "C": 1.0},
        {
            "family": "boosted",
            "max_iter": 100,
            "learning_rate": 0.05,
            "max_leaf_nodes": 7,
            "min_samples_leaf": 100,
            "l2_regularization": 1.0,
        },
    )
    features, labels, folds = [], [], []
    for training, held in TimeSeriesSplit(n_splits=4).split(X):
        training = np.array([i for i in training if rows[i].label_final_at <= rows[held[0]].at])
        columns = [
            risk(make_model(s, seed).fit(X[training], y[training]), X[held]) for s in settings
        ]
        features.extend(np.column_stack(columns))
        labels.extend(y[held])
        folds.append(
            {
                "training": len(training),
                "held": len(held),
                "latest_label": max(rows[i].label_final_at for i in training).isoformat(),
                "first_held": rows[held[0]].at.isoformat(),
            }
        )
    meta = LogisticRegression(max_iter=1000).fit(features, labels)
    Xv, yv, _ = arrays(window["validation"])
    members = [make_model(s, seed).fit(X, y) for s in settings]
    predictions = np.column_stack([risk(member, Xv) for member in members])
    own = np.column_stack([risk(member, X) for member in members])
    in_sample = LogisticRegression(max_iter=1000).fit(own, y)

    def weights(model):
        return dict(zip(("intercept", "logistic", "boosted"), [float(model.intercept_[0]), *map(float, model.coef_[0])]))

    return {
        "folds": folds,
        "meta_rows": len(labels),
        "member_losses": {s["family"]: log_loss(yv, predictions[:, i]) for i, s in enumerate(settings)},
        "weights": weights(meta),
        "in_sample_weights": weights(in_sample),
        "validation_loss": log_loss(yv, risk(meta, predictions)),
        "in_sample_loss": log_loss(yv, risk(in_sample, predictions)),
    }


def print_stacking(result):
    from demos.output_wrap import print_wrapped

    for i, fold in enumerate(result["folds"], 1):
        print_wrapped(f"fold {i}: train {fold['training']}, held {fold['held']}; latest final label {fold['latest_label']} <= first held {fold['first_held']}")
    print_wrapped(f"meta rows: {result['meta_rows']}")
    print_wrapped("member validation log loss: " + "; ".join(f"{k}={v:.6f}" for k, v in result["member_losses"].items()))
    for key, name in (("weights", "honest"), ("in_sample_weights", "in-sample")):
        print_wrapped(name + " combiner: " + "; ".join(f"{k}={v:.6f}" for k, v in result[key].items()))
    print_wrapped(f"validation log loss: honest={result['validation_loss']:.6f}; in-sample={result['in_sample_loss']:.6f}")
    relation = "higher" if result["in_sample_loss"] > result["validation_loss"] else "lower" if result["in_sample_loss"] < result["validation_loss"] else "equal"
    print_wrapped(f"The in-sample stack's validation loss is {relation} than the honest stack's.")


def feature_construction(window):
    """A label-built feature's statistics use training only and omit its own label."""
    development_only(window)
    model = GroupRisk(clean=True).fit(window["training"], window["validation"])
    return model.provenance()


def screen_noise(X, y, noise, Xh, yh, held_noise):
    """Rank 200 single-column additions, then select five; no joint subset search."""
    losses = []
    for column in range(noise.shape[1]):
        model = make_model({"family": "logistic", "C": 1.0}).fit(
            np.column_stack((X, noise[:, column])), y
        )
        p = risk(model, np.column_stack((Xh, held_noise[:, column])))
        losses.append(log_loss(yh, p))
    return {"columns": np.argsort(losses, kind="stable")[:5].tolist(), "losses": losses}


def select_inside_training(training, noise):
    """The right selector's interface admits training rows and training noise only."""
    development_only({"training": training, "validation": []})
    held = np.arange(len(training) * 4 // 5, len(training))
    fit = np.array([i for i in range(held[0]) if training[i].label_final_at <= training[held[0]].at])
    X, y, _ = arrays(training)
    result = screen_noise(X[fit], y[fit], noise[fit], X[held], y[held], noise[held])
    return {
        **result,
        "training": len(fit),
        "held": len(held),
        "latest_label": max(training[i].label_final_at for i in fit).isoformat(),
        "first_held": training[held[0]].at.isoformat(),
    }


def selection(window, seed=7):
    development_only(window)
    X, y, _ = arrays(window["training"])
    Xv, yv, _ = arrays(window["validation"])
    noise = np.random.default_rng(seed).standard_normal((len(X) + len(Xv), 200))
    nt, nv = noise[:len(X)], noise[len(X):]
    wrong = screen_noise(X, y, nt, Xv, yv, nv)
    right = select_inside_training(window["training"], nt)
    pool, mu, _ = fresh_truth()
    Xfresh = arrays(pool)[0]
    fresh_noise = np.random.default_rng(71).standard_normal((len(pool), 200))
    models = {}
    for name, columns in (("baseline", []), ("validation-chosen", wrong["columns"]), ("training-chosen", right["columns"])):
        model = make_model({"family": "logistic", "C": 1.0}, seed).fit(
            np.column_stack((X, nt[:, columns])), y
        )
        p = risk(model, np.column_stack((Xv, nv[:, columns])))
        fresh_p = risk(model, np.column_stack((Xfresh, fresh_noise[:, columns])))
        models[name] = {
            "noise_selected": len(columns),
            "columns": columns,
            "validation_loss": log_loss(yv, p),
            "expected_loss": float(binary_losses(mu, fresh_p).mean()),
        }
    return {"models": models, "noise_columns": 200, "fresh_requests": len(pool), "wrong": wrong, "right": right}


def print_selection(result):
    from demos.output_wrap import print_wrapped

    print_wrapped("200 noise columns, seed 7: rank single-column additions; refit the best five jointly")
    print_wrapped("wrong: rank on validation; right: last fifth of training, earlier labels final before it")
    reference = result["models"]["baseline"]
    for name, row in result["models"].items():
        label = "logistic reference" if name == "baseline" else name
        lead = "" if name == "baseline" else f"lead over logistic reference {reference['validation_loss'] - row['validation_loss']:+.6f}; "
        print_wrapped(f"{label:18} validation log loss {row['validation_loss']:.6f}; {lead}noise selected {row['noise_selected']}/5")
    print_wrapped(
        f"generator's expectation on a fresh June draw ({result['fresh_requests']:,} requests, seed 71; planted risk), lead over logistic reference: "
        + "; ".join(f"{name}={reference['expected_loss'] - row['expected_loss']:.6f}" for name, row in result["models"].items() if name != "baseline")
    )
    chosen = result["models"]["validation-chosen"]["validation_loss"]
    print_wrapped("Validation-chosen inputs " + ("lower" if chosen < reference["validation_loss"] else "do not lower") + " validation loss in this run.")


# region book:fnd.automl-and-selection.lab-01 output
from demos import delegated_search_lab as lab

if __name__ == "__main__":
    lab.print_comparison(lab.search_run())
# endregion book:fnd.automl-and-selection.lab-01


# region book:fnd.automl-and-selection.lab-trace-01 output
import json

from demos import delegated_search_lab as lab

if __name__ == "__main__":
    print(json.dumps(lab.search_run(), indent=2, sort_keys=True))
# endregion book:fnd.automl-and-selection.lab-trace-01


# region book:fnd.automl-and-selection.select-01 output
from demos import delegated_search_lab as lab

if __name__ == "__main__":
    development, _ = lab.prepare()
    lab.print_selection(lab.selection(development, seed=7))
# endregion book:fnd.automl-and-selection.select-01


# region book:fnd.automl-and-selection.stack-01 output
from demos import delegated_search_lab as lab

if __name__ == "__main__":
    development, _ = lab.prepare()
    lab.print_stacking(lab.stacking(development, seed=7))
# endregion book:fnd.automl-and-selection.stack-01
