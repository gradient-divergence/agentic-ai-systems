"""CLI entrypoint to train the Yelp GBM baseline."""

from __future__ import annotations

# region book:fnd.supervised-learning.example-01
import argparse
import json
from pathlib import Path

from pydantic import ValidationError

from spine.core.config import SpineConfig
from spine.ml.gbm_baseline import TrainConfig, run_training


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train GBM baseline for the spine.")
    parser.add_argument("--subset-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--seed", type=int, default=SpineConfig().seed)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--label-threshold", type=float, default=4.0)
    parser.add_argument("--no-lightgbm", action="store_true")
    parser.add_argument("--max-rows", type=int, default=None)
    args = parser.parse_args()

    subset_dir = Path(args.subset_dir) if args.subset_dir else SpineConfig().subset_dir
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else SpineConfig().output_dir / "gbm_baseline"
    )

    try:
        return TrainConfig(
            subset_dir=subset_dir,
            output_dir=output_dir,
            seed=args.seed,
            test_fraction=args.test_fraction,
            label_threshold=args.label_threshold,
            use_lightgbm=not args.no_lightgbm,
            max_rows=args.max_rows,
        )
    except ValidationError as exc:
        raise SystemExit(str(exc)) from exc


def main() -> None:
    cfg = parse_args()
    metrics = run_training(cfg)
    print(json.dumps(metrics.model_dump(), indent=2))
# endregion


if __name__ == "__main__":
    from spine.core.seeds import set_global_seed

    cfg = parse_args()
    set_global_seed(cfg.seed)
    main()
