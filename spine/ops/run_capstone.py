"""Capstone runner linking data, model, and policy evaluation."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError, field_validator

from spine.core.config import SpineConfig
from spine.core.logging import setup_json_logger
from spine.data.yelp_features import build_feature_rows
from spine.ml.gbm_baseline import TrainConfig, run_training
from spine.ops.governance import GovernanceChecklist, write_checklist
from spine.ops.metrics import RunMetrics, write_run_metrics
from spine.rl.policy_eval import PolicyEvalConfig, evaluate_baseline_policy


class CapstoneConfig(BaseModel):
    subset_dir: Path = Field(...)
    output_dir: Path = Field(...)
    max_rows: int = Field(20_000, ge=1000)
    label_threshold: float = Field(4.0, ge=1.0, le=5.0)
    max_eval_steps: int = Field(2000, ge=100)
    model_auc_margin: float = Field(0.05, gt=0.0, le=0.5)
    use_lightgbm: bool = True
    run_id: str = Field(default_factory=lambda: datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"))

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


def parse_args() -> CapstoneConfig:
    parser = argparse.ArgumentParser(description="Run spine capstone pipeline.")
    parser.add_argument("--subset-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--max-rows", type=int, default=20_000)
    parser.add_argument("--label-threshold", type=float, default=4.0)
    parser.add_argument("--max-eval-steps", type=int, default=2000)
    parser.add_argument("--model-auc-margin", type=float, default=0.05)
    parser.add_argument("--no-lightgbm", action="store_true")
    args = parser.parse_args()

    subset_dir = Path(args.subset_dir) if args.subset_dir else SpineConfig().subset_dir
    output_dir = Path(args.output_dir) if args.output_dir else SpineConfig().output_dir

    try:
        return CapstoneConfig(
            subset_dir=subset_dir,
            output_dir=output_dir,
            max_rows=args.max_rows,
            label_threshold=args.label_threshold,
            max_eval_steps=args.max_eval_steps,
            model_auc_margin=args.model_auc_margin,
            use_lightgbm=not args.no_lightgbm,
        )
    except ValidationError as exc:
        raise SystemExit(str(exc)) from exc




def main() -> None:
    cfg = parse_args()
    logger = setup_json_logger()
    run_dir = cfg.output_dir / f"capstone_{cfg.run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    rows = build_feature_rows(
        cfg.subset_dir,
        label_threshold=cfg.label_threshold,
        max_rows=cfg.max_rows,
    )
    positive_rate = (sum(row.label for row in rows) / len(rows)) if rows else 0.0

    train_cfg = TrainConfig(
        subset_dir=cfg.subset_dir,
        output_dir=run_dir / "ml",
        label_threshold=cfg.label_threshold,
        use_lightgbm=cfg.use_lightgbm,
        max_rows=cfg.max_rows,
    )
    model_metrics = run_training(train_cfg)

    eval_cfg = PolicyEvalConfig(
        subset_dir=cfg.subset_dir,
        output_dir=run_dir / "policy_eval",
        label_threshold=cfg.label_threshold,
        max_steps=cfg.max_eval_steps,
    )
    policy_metrics = evaluate_baseline_policy(eval_cfg)

    run_metrics = RunMetrics(
        rows=len(rows),
        positive_rate=positive_rate,
        model_accuracy=model_metrics.accuracy,
        model_roc_auc=model_metrics.roc_auc,
        policy_accuracy=policy_metrics.accuracy,
        average_reward=policy_metrics.average_reward,
    )
    metrics_path = write_run_metrics(run_dir / "ops", run_metrics)

    checklist = GovernanceChecklist(
        data_quality_ok=len(rows) > 0,
        model_eval_ok=0.5 + cfg.model_auc_margin <= model_metrics.roc_auc <= 1.0,
        policy_eval_ok=(
            0.0 < policy_metrics.promote_rate < 1.0
            and policy_metrics.accuracy > policy_metrics.majority_accuracy
        ),
        notes=["Capstone run generated metrics and policy eval."],
    )
    checklist_path = write_checklist(run_dir / "ops", checklist)
    approved = all((checklist.data_quality_ok, checklist.model_eval_ok, checklist.policy_eval_ok))

    report = {
        "approved": approved,
        "data_rows": len(rows),
        "positive_rate": positive_rate,
        "model_auc_margin": cfg.model_auc_margin,
        "model_metrics": model_metrics.model_dump(mode="json"),
        "policy_metrics": policy_metrics.model_dump(mode="json"),
        "metrics_path": str(metrics_path),
        "checklist_path": str(checklist_path),
    }
    (run_dir / "capstone_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    logger.info(
        "capstone_run_complete" if approved else "capstone_gate_failed",
        extra={"payload": {"run_dir": str(run_dir), "rows": len(rows), "approved": approved}},
    )
    print(json.dumps(report, indent=2))
    if not approved:
        raise SystemExit(1)


if __name__ == "__main__":
    from spine.core.seeds import set_global_seed

    set_global_seed(SpineConfig().seed)
    main()
