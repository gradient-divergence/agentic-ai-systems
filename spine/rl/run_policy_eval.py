"""CLI runner for baseline policy evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path

from pydantic import ValidationError

from spine.core.config import SpineConfig
from spine.rl.policy_eval import PolicyEvalConfig


def parse_args() -> PolicyEvalConfig:
    parser = argparse.ArgumentParser(description="Evaluate baseline policy.")
    parser.add_argument("--subset-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--label-threshold", type=float, default=4.0)
    parser.add_argument("--max-steps", type=int, default=1000)
    args = parser.parse_args()

    subset_dir = Path(args.subset_dir) if args.subset_dir else SpineConfig().subset_dir
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else SpineConfig().output_dir / "rl_eval"
    )

    try:
        return PolicyEvalConfig(
            subset_dir=subset_dir,
            output_dir=output_dir,
            label_threshold=args.label_threshold,
            max_steps=args.max_steps,
        )
    except ValidationError as exc:
        raise SystemExit(str(exc)) from exc


import json

from spine.rl.policy_eval import evaluate_baseline_policy


def main() -> None:
    cfg = parse_args()
    result = evaluate_baseline_policy(cfg)
    print(json.dumps(result.model_dump(mode="json"), indent=2))


if __name__ == "__main__":
    from spine.core.seeds import set_global_seed

    set_global_seed(SpineConfig().seed)
    main()
