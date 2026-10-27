"""CLI runner for embedding + clustering pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

from pydantic import ValidationError

from spine.core.config import SpineConfig
from spine.ml.clustering import ClusteringConfig
from spine.ml.text_embeddings import EmbeddingConfig


def parse_args() -> tuple[EmbeddingConfig, ClusteringConfig]:
    parser = argparse.ArgumentParser(description="Run embeddings + clustering for spine.")
    parser.add_argument("--subset-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--max-rows", type=int, default=50_000)
    parser.add_argument("--min-chars", type=int, default=20)
    parser.add_argument("--seed", type=int, default=SpineConfig().seed)
    parser.add_argument("--use-gpu", action="store_true")
    parser.add_argument("--use-sentence-transformer", action="store_true")
    parser.add_argument("--tfidf-max-features", type=int, default=5000)
    parser.add_argument("--svd-components", type=int, default=128)
    parser.add_argument("--n-clusters", type=int, default=12)
    parser.add_argument("--silhouette-sample", type=int, default=5000)
    args = parser.parse_args()

    subset_dir = Path(args.subset_dir) if args.subset_dir else SpineConfig().subset_dir
    base_output = (
        Path(args.output_dir)
        if args.output_dir
        else SpineConfig().output_dir / "embeddings"
    )
    embed_output = base_output / "embeddings"
    cluster_output = base_output / "clusters"

    try:
        embed_cfg = EmbeddingConfig(
            subset_dir=subset_dir,
            output_dir=embed_output,
            max_rows=args.max_rows,
            min_chars=args.min_chars,
            seed=args.seed,
            use_gpu=bool(args.use_gpu),
            use_sentence_transformer=bool(args.use_sentence_transformer),
            tfidf_max_features=args.tfidf_max_features,
            svd_components=args.svd_components,
        )
        cluster_cfg = ClusteringConfig(
            output_dir=cluster_output,
            n_clusters=args.n_clusters,
            seed=args.seed,
            silhouette_sample=args.silhouette_sample,
        )
    except ValidationError as exc:
        raise SystemExit(str(exc)) from exc

    return embed_cfg, cluster_cfg


# region book:fnd.unsupervised-learning.example-01
import json

import numpy as np

from spine.ml.clustering import cluster_embeddings
from spine.ml.text_embeddings import build_embeddings


def main() -> None:
    embed_cfg, cluster_cfg = parse_args()
    embed_result = build_embeddings(embed_cfg)
    summary = cluster_embeddings(
        embeddings=np.load(embed_result.embeddings_path),
        review_ids=json.loads(embed_result.review_ids_path.read_text()),
        cfg=cluster_cfg,
    )
    payload = {
        "embedding_model": embed_result.model_type,
        "embedding_dim": embed_result.embedding_dim,
        "rows": embed_result.rows,
        "cluster_summary": summary.model_dump(mode="json"),
    }
    print(json.dumps(payload, indent=2))
# endregion


if __name__ == "__main__":
    from spine.core.seeds import set_global_seed

    embed_cfg, _ = parse_args()
    set_global_seed(embed_cfg.seed)
    main()
