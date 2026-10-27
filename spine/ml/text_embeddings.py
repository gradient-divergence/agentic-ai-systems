"""Embedding pipeline for Yelp review text."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

import numpy as np
from pydantic import BaseModel, Field, field_validator


class EmbeddingConfig(BaseModel):
    subset_dir: Path = Field(...)
    output_dir: Path = Field(...)
    max_rows: int | None = Field(50_000, ge=1)
    min_chars: int = Field(20, ge=0)
    seed: int = Field(13, ge=0)
    use_gpu: bool = False
    use_sentence_transformer: bool = False
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    tfidf_max_features: int = Field(5000, ge=100)
    svd_components: int = Field(128, ge=2)

    @field_validator("subset_dir")
    @classmethod
    def _validate_subset_dir(cls, value: Path) -> Path:
        if not value.exists():
            raise ValueError(f"Subset directory does not exist: {value}")
        return value

    @field_validator("output_dir")
    @classmethod
    def _ensure_output_dir(cls, value: Path) -> Path:
        value.mkdir(parents=True, exist_ok=True)
        return value

    model_config = {"validate_assignment": True}


class EmbeddingResult(BaseModel):
    model_type: str
    embedding_dim: int
    rows: int
    output_dir: Path
    embeddings_path: Path
    review_ids_path: Path


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def load_review_texts(
    subset_dir: Path, max_rows: int | None, min_chars: int
) -> tuple[list[str], list[str]]:
    review_path = subset_dir / "review.json"
    review_ids: list[str] = []
    texts: list[str] = []
    for idx, row in enumerate(iter_jsonl(review_path)):
        if max_rows is not None and idx >= max_rows:
            break
        text = row.get("text") or ""
        if len(text) < min_chars:
            continue
        review_ids.append(str(row.get("review_id", "")))
        texts.append(text)
    return review_ids, texts


def _save_embeddings(
    output_dir: Path, review_ids: list[str], embeddings: np.ndarray, model_type: str
) -> EmbeddingResult:
    embeddings_path = output_dir / "embeddings.npy"
    review_ids_path = output_dir / "review_ids.json"
    np.save(embeddings_path, embeddings)
    review_ids_path.write_text(json.dumps(review_ids, indent=2), encoding="utf-8")

    metadata = {
        "model_type": model_type,
        "rows": len(review_ids),
        "embedding_dim": int(embeddings.shape[1]),
    }
    (output_dir / "embedding_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    return EmbeddingResult(
        model_type=model_type,
        embedding_dim=int(embeddings.shape[1]),
        rows=len(review_ids),
        output_dir=output_dir,
        embeddings_path=embeddings_path,
        review_ids_path=review_ids_path,
    )


# region book:fnd.unsupervised-learning.sota-01
def _embed_with_sentence_transformer(
    cfg: EmbeddingConfig, texts: list[str]
) -> tuple[np.ndarray, str]:
    from sentence_transformers import SentenceTransformer

    device = "cuda" if cfg.use_gpu else "cpu"
    model = SentenceTransformer(cfg.model_name, device=device)
    embeddings = model.encode(texts, show_progress_bar=False)
    return np.asarray(embeddings, dtype=float), f"sentence_transformer:{cfg.model_name}"


def umap_hdbscan_clusters(
    embeddings: np.ndarray, seed: int = 13, min_cluster_size: int = 12
) -> dict[str, int | str]:
    try:
        import hdbscan  # type: ignore
        import umap  # type: ignore
    except ImportError:
        return {
            "method": "umap+hdbscan",
            "status": "skipped",
            "reason": "install umap-learn and hdbscan",
        }

    reducer = umap.UMAP(
        n_neighbors=15,
        min_dist=0.1,
        metric="cosine",
        random_state=seed,
    )
    reduced = reducer.fit_transform(embeddings)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
    labels = clusterer.fit_predict(reduced)
    clusters = len(set(labels)) - (1 if -1 in labels else 0)
    noise = int((labels == -1).sum())
    return {"method": "umap+hdbscan", "clusters": clusters, "noise_points": noise}


def isolation_forest_anomalies(
    embeddings: np.ndarray, seed: int = 13, contamination: float = 0.05
) -> dict[str, int]:
    from sklearn.ensemble import IsolationForest

    model = IsolationForest(random_state=seed, contamination=contamination)
    preds = model.fit_predict(embeddings)
    return {"anomalies": int((preds == -1).sum())}


def gmm_soft_clusters(
    embeddings: np.ndarray, seed: int = 13, n_components: int = 6
) -> dict[str, float | int]:
    from sklearn.mixture import GaussianMixture

    gmm = GaussianMixture(n_components=n_components, random_state=seed)
    gmm.fit(embeddings)
    responsibilities = gmm.predict_proba(embeddings)
    max_resp = float(np.mean(np.max(responsibilities, axis=1)))
    return {"components": n_components, "avg_max_responsibility": max_resp}


def bayesian_gmm_soft_clusters(
    embeddings: np.ndarray, seed: int = 13, max_components: int = 10
) -> dict[str, int]:
    from sklearn.mixture import BayesianGaussianMixture

    bgmm = BayesianGaussianMixture(
        n_components=max_components,
        random_state=seed,
        weight_concentration_prior_type="dirichlet_process",
    )
    bgmm.fit(embeddings)
    active = int((bgmm.weights_ > 1e-3).sum())
    return {"active_components": active, "max_components": max_components}
# endregion


# region book:fnd.unsupervised-learning.core-01
from numpy import ndarray


def _embed_with_tfidf(cfg: EmbeddingConfig, texts: list[str]) -> tuple[ndarray, str]:
    import numpy as np
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer

    vectorizer = TfidfVectorizer(
        max_features=cfg.tfidf_max_features,
        stop_words="english",
    )
    tfidf = vectorizer.fit_transform(texts)
    if tfidf.shape[1] < 2:
        return tfidf.toarray(), "tfidf"

    n_components = min(cfg.svd_components, max(2, tfidf.shape[1] - 1))
    reducer = TruncatedSVD(n_components=n_components, random_state=cfg.seed)
    embeddings = reducer.fit_transform(tfidf)
    return np.asarray(embeddings, dtype=float), "tfidf_svd"
# endregion


def build_embeddings(cfg: EmbeddingConfig) -> EmbeddingResult:
    review_ids, texts = load_review_texts(cfg.subset_dir, cfg.max_rows, cfg.min_chars)
    if not texts:
        raise ValueError("No texts found for embedding.")

    if cfg.use_sentence_transformer:
        try:
            embeddings, model_type = _embed_with_sentence_transformer(cfg, texts)
        except ModuleNotFoundError:
            embeddings, model_type = _embed_with_tfidf(cfg, texts)
    else:
        embeddings, model_type = _embed_with_tfidf(cfg, texts)

    return _save_embeddings(cfg.output_dir, review_ids, embeddings, model_type)
