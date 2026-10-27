#!/usr/bin/env python3
from __future__ import annotations

# region book:fnd.semi-self-supervised.core-01
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression


def self_train(seed: int = 0) -> dict[str, int]:
    X, y = make_classification(
        n_samples=120,
        n_features=6,
        n_informative=4,
        random_state=seed,
    )
    labeled_mask = np.zeros(len(y), dtype=bool)
    labeled_mask[:24] = True
    model = LogisticRegression(max_iter=200)
    model.fit(X[labeled_mask], y[labeled_mask])
    pseudo_labels = model.predict(X[~labeled_mask])
    return {
        "labeled": int(labeled_mask.sum()),
        "pseudo": int((~labeled_mask).sum()),
        "pseudo_positive": int((pseudo_labels == 1).sum()),
    }
# endregion book:fnd.semi-self-supervised.core-01


# region book:fnd.semi-self-supervised.example-01
print(self_train())
# endregion book:fnd.semi-self-supervised.example-01


# region book:fnd.causal-inference.core-01
from pydantic import BaseModel


class DiDInput(BaseModel):
    treatment_pre: float
    treatment_post: float
    control_pre: float
    control_post: float


def difference_in_differences(data: DiDInput) -> float:
    return (data.treatment_post - data.treatment_pre) - (
        data.control_post - data.control_pre
    )
# endregion book:fnd.causal-inference.core-01


# region book:fnd.causal-inference.example-01
effect = difference_in_differences(
    DiDInput(
        treatment_pre=0.12,
        treatment_post=0.18,
        control_pre=0.11,
        control_post=0.13,
    )
)
print({"did_effect": round(effect, 4)})
# endregion book:fnd.causal-inference.example-01


# region book:fnd.causal-inference.sota-01
def dowhy_ate_demo(seed: int = 0) -> dict[str, float | str]:
    import numpy as np
    import pandas as pd

    try:
        from dowhy import CausalModel
    except Exception as exc:
        if isinstance(exc, ModuleNotFoundError) and exc.name == "dowhy":
            return {"status": "skipped (install dowhy)"}
        return {"status": "installed but failed to import", "reason": f"{type(exc).__name__}: {exc}"}

    rng = np.random.default_rng(seed)
    n = 200
    x = rng.normal(size=n)
    treatment = (x + rng.normal(scale=0.5, size=n) > 0).astype(int)
    outcome = 0.6 * treatment + 0.3 * x + rng.normal(scale=0.2, size=n)
    df = pd.DataFrame({"treatment": treatment, "outcome": outcome, "x": x})
    model = CausalModel(
        data=df,
        treatment="treatment",
        outcome="outcome",
        common_causes=["x"],
    )
    estimand = model.identify_effect()
    estimate = model.estimate_effect(estimand, method_name="backdoor.linear_regression")
    return {"status": "ok", "ate": float(estimate.value)}
# endregion book:fnd.causal-inference.sota-01


print(dowhy_ate_demo())


from sklearn.datasets import make_blobs
from sklearn.ensemble import IsolationForest


def unsupervised_sota_demo(seed: int = 0) -> dict[str, float | int | str]:
    X, _ = make_blobs(n_samples=240, n_features=6, centers=4, random_state=seed)
    iso = IsolationForest(contamination=0.05, random_state=seed)
    anomaly_flags = iso.fit_predict(X)
    anomaly_count = int((anomaly_flags == -1).sum())
    try:
        from sklearn.mixture import BayesianGaussianMixture, GaussianMixture

        gmm = GaussianMixture(n_components=6, random_state=seed)
        gmm.fit(X)
        gmm_resp = gmm.predict_proba(X)
        avg_resp = float(np.mean(np.max(gmm_resp, axis=1)))

        bgmm = BayesianGaussianMixture(
            n_components=10,
            weight_concentration_prior_type="dirichlet_process",
            random_state=seed,
        )
        bgmm.fit(X)
        active_components = int((bgmm.weights_ > 1e-3).sum())
    except ImportError:
        avg_resp = -1.0
        active_components = -1

    try:
        import hdbscan  # type: ignore
        import umap  # type: ignore

        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=seed)
        reduced = reducer.fit_transform(X)
        clusterer = hdbscan.HDBSCAN(min_cluster_size=12)
        labels = clusterer.fit_predict(reduced)
        cluster_count = len(set(labels)) - (1 if -1 in labels else 0)
        noise_points = int((labels == -1).sum())
        status = "umap+hdbscan"
    except ImportError:
        cluster_count = -1
        noise_points = -1
        status = "skipped (install umap-learn + hdbscan)"

    return {
        "status": status,
        "anomalies": anomaly_count,
        "clusters": cluster_count,
        "noise_points": noise_points,
        "avg_gmm_responsibility": round(avg_resp, 4) if avg_resp >= 0 else -1,
        "bayesian_gmm_components": active_components,
    }

print(unsupervised_sota_demo())
