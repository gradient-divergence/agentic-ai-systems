"""Clustering utilities for the spine embedding pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from pydantic import BaseModel, Field, field_validator


class ClusteringConfig(BaseModel):
    output_dir: Path = Field(...)
    n_clusters: int = Field(12, ge=2)
    seed: int = Field(13, ge=0)
    silhouette_sample: int = Field(5000, ge=200)

    @field_validator("output_dir")
    @classmethod
    def _ensure_output_dir(cls, value: Path) -> Path:
        value.mkdir(parents=True, exist_ok=True)
        return value

    model_config = {"validate_assignment": True}


class ClusterSummary(BaseModel):
    n_clusters: int
    silhouette: float | None
    cluster_sizes: list[int]
    output_dir: Path


def _write_cluster_sizes(output_dir: Path, sizes: list[int]) -> None:
    lines = ["cluster_id,size"]
    for idx, size in enumerate(sizes):
        lines.append(f"{idx},{size}")
    (output_dir / "cluster_sizes.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")


def cluster_embeddings(
    embeddings: np.ndarray,
    review_ids: list[str],
    cfg: ClusteringConfig,
) -> ClusterSummary:
    if embeddings.shape[0] < cfg.n_clusters:
        raise ValueError("Not enough samples to cluster.")

    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    model = KMeans(n_clusters=cfg.n_clusters, random_state=cfg.seed, n_init="auto")
    labels = model.fit_predict(embeddings)

    counts = np.bincount(labels, minlength=cfg.n_clusters).tolist()
    _write_cluster_sizes(cfg.output_dir, counts)

    silhouette = None
    if embeddings.shape[0] > cfg.n_clusters:
        rng = np.random.default_rng(cfg.seed)
        sample_size = min(cfg.silhouette_sample, embeddings.shape[0])
        sample_idx = rng.choice(embeddings.shape[0], size=sample_size, replace=False)
        silhouette = float(silhouette_score(embeddings[sample_idx], labels[sample_idx]))

    label_path = cfg.output_dir / "cluster_labels.csv"
    with label_path.open("w", encoding="utf-8") as handle:
        handle.write("review_id,cluster_id\n")
        for review_id, label in zip(review_ids, labels):
            handle.write(f"{review_id},{label}\n")

    summary = ClusterSummary(
        n_clusters=cfg.n_clusters,
        silhouette=silhouette,
        cluster_sizes=counts,
        output_dir=cfg.output_dir,
    )
    (cfg.output_dir / "cluster_summary.json").write_text(
        json.dumps(summary.model_dump(mode="json"), indent=2), encoding="utf-8"
    )
    return summary
