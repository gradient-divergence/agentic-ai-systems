"""Smoke-run entrypoint for the spine scaffold."""

from __future__ import annotations

# region book:fnd.agentic-loop.example-01
import argparse
import json
from pathlib import Path

from spine.core.config import SpineConfig
from spine.core.logging import setup_json_logger


def parse_args() -> SpineConfig:
    parser = argparse.ArgumentParser(description="Spine smoke run.")
    parser.add_argument("--subset-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--use-gpu", action="store_true")
    args = parser.parse_args()

    subset_dir = Path(args.subset_dir) if args.subset_dir else None
    output_dir = Path(args.output_dir) if args.output_dir else None

    return SpineConfig(
        subset_dir=subset_dir or SpineConfig().subset_dir,
        output_dir=output_dir or SpineConfig().output_dir,
        use_gpu=bool(args.use_gpu),
    )


def main() -> None:
    cfg = parse_args()
    logger = setup_json_logger()
    metadata_path = cfg.subset_dir / "subset_metadata.json"
    metadata = {}
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())

    payload = {
        "event": "spine_smoke_run",
        "dataset": cfg.dataset_slug,
        "subset_dir": str(cfg.subset_dir),
        "run_dir": str(cfg.run_dir),
        "use_gpu": cfg.use_gpu,
        "metadata": metadata,
    }
    logger.info("spine_smoke_run", extra={"payload": payload})
# endregion


if __name__ == "__main__":
    from spine.core.seeds import set_global_seed

    cfg = parse_args()
    set_global_seed(cfg.seed)
    main()
