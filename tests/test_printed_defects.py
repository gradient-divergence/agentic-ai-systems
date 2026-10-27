import ast
import builtins
import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from pydantic import BaseModel, ConfigDict, Field
from test_regressions import ROOT, region


def test_bradley_terry_loss_handles_large_score_gap() -> None:
    ns = region("demos/preference_demo.py", "fnd.rlhf-alignment.pair-01")
    pair = ns["PreferencePair"](chosen_score=0, rejected_score=1000)
    assert ns["bradley_terry_loss"](pair) == pytest.approx(1000)


def test_dpo_loss_handles_large_log_ratio_gap() -> None:
    ns = region("demos/preference_demo.py", "fnd.rlhf-alignment.dpo-01")
    batch = ns["PreferenceBatch"](chosen_logprob=-2000, rejected_logprob=0,
                                  ref_chosen_logprob=0, ref_rejected_logprob=0)
    assert ns["dpo_loss"](batch, ns["DPOConfig"](beta=1)) == pytest.approx(2000)


@pytest.mark.parametrize("reference_rejected, expected", [(-1.5, 0.0), (-1.2, 0.5), (-1.1, 1.0)])
def test_preference_accuracy_compares_log_ratios(reference_rejected: float, expected: float) -> None:
    ns = region("demos/preference_demo.py", "fnd.rlhf-alignment.dpo-01")
    batch = ns["PreferenceBatch"](chosen_logprob=-1, rejected_logprob=-1.2,
                                  ref_chosen_logprob=-1, ref_rejected_logprob=reference_rejected)
    assert ns["preference_accuracy"](batch) == expected


def test_preference_accuracy_tolerates_roundoff_in_log_ratio_tie() -> None:
    ns = region("demos/preference_demo.py", "fnd.rlhf-alignment.dpo-01")
    batch = ns["PreferenceBatch"](chosen_logprob=-0.3, rejected_logprob=-0.6,
                                  ref_chosen_logprob=-1.1, ref_rejected_logprob=-1.4)
    assert ns["preference_accuracy"](batch) == 0.5


def test_dpo_has_only_standard_beta_coefficient() -> None:
    ns = region("demos/preference_demo.py", "fnd.rlhf-alignment.dpo-01")
    assert set(ns["DPOConfig"].model_fields) == {"beta"}
    batch = ns["PreferenceBatch"](chosen_logprob=-1.1, rejected_logprob=-1.8,
                                  ref_chosen_logprob=-1.4, ref_rejected_logprob=-1.6)
    assert ns["dpo_loss"](batch, ns["DPOConfig"](beta=0.2)) == pytest.approx(np.logaddexp(0, -0.1))


def test_consensus_rejects_duplicate_voter_identity() -> None:
    ns = region("demos/agentic_systems_demo.py", "fnd.multi-agent-systems.sota-01")
    vote = ns["Vote"](agent="planner", approve=True, weight=0.9)
    config = ns["ConsensusConfig"](quorum=2, min_weight=1.5)
    with pytest.raises(ValueError, match="once"):
        ns["weighted_consensus"]([vote, vote], config)
    other = ns["Vote"](agent="auditor", approve=True, weight=0.7)
    assert ns["weighted_consensus"]([vote, other], config)["approved"]


def test_tool_request_forbids_extra_envelope_fields() -> None:
    from spine.agent.tool_schema import ToolRequest

    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        ToolRequest.model_validate({"tool_name": "refund.execute", "input": {"amount": "not-a-number"},
                                    "approval_id": "approval-001", "risk_tier": 3})


def test_tool_request_validates_selected_payload_schema() -> None:
    from spine.agent.tool_schema import ToolRequest

    class RefundInput(BaseModel):
        model_config = ConfigDict(extra="forbid", strict=True)
        amount: float = Field(gt=0, allow_inf_nan=False)

    context = {"tool_schemas": {"refund.execute": RefundInput}}
    for payload in ({"amount": "not-a-number"}, {"amount": -1}, {"amount": 1, "unexpected": True}):
        with pytest.raises(ValueError):
            ToolRequest.model_validate({"tool_name": "refund.execute", "input": payload}, context=context)
    with pytest.raises(ValueError):
        ToolRequest(tool_name="unknown", input={})
    request = ToolRequest.model_validate(
        {"tool_name": "refund.execute", "input": {"amount": 1.5}}, context=context,
    )
    assert request.input == {"amount": 1.5}


@pytest.mark.parametrize("context", [None, {}, {"other": {}}])
def test_tool_request_reports_missing_schema_context(context: Any) -> None:
    from spine.agent.tool_schema import ToolRequest

    with pytest.raises(ValueError, match="Missing tool schema context"):
        ToolRequest.model_validate({"tool_name": "unknown", "input": {}}, context=context)
    with pytest.raises(ValueError, match="No input schema for tool: unknown"):
        ToolRequest.model_validate({"tool_name": "unknown", "input": {}},
                                   context={"tool_schemas": {}})


@pytest.mark.parametrize("scale", [1e200, 1e-200, 1.7e308])
def test_cosine_similarity_is_stable_at_extreme_scales(scale: float) -> None:
    ns = region("demos/part7_8_demo.py", "fnd.appendix-math-foundations.core-01")
    assert ns["cosine_similarity"]([scale, scale], [scale, scale]) == pytest.approx(1)
    assert ns["cosine_similarity"]([scale, 0], [0, scale]) == 0


def test_cosine_similarity_rejects_zero_norm() -> None:
    ns = region("demos/part7_8_demo.py", "fnd.appendix-math-foundations.core-01")
    with pytest.raises(ValueError, match="non-zero"):
        ns["cosine_similarity"]([0, 0], [1, 0])


def test_optuna_uses_non_deprecated_logistic_regression_parameters() -> None:
    pytest.importorskip("optuna")
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        ns = region("demos/supervised_ml_demo.py", "fnd.optimization-regularization.optuna-01")
        assert "best_value" in ns["optuna_tune"](23)


def test_lightning_training_has_working_optimizer() -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("lightning")
    ns = region("demos/deep_learning_demo.py", "fnd.transformers-foundation-models.frameworks-01")
    model = ns["TinyLightningModule"](dim=2)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.fill_(0.1)
    optimizer = model.configure_optimizers()
    assert isinstance(optimizer, torch.optim.Optimizer)
    before = model.block.fc1.weight.detach().clone()
    loss = model.training_step((torch.ones(2, 2), torch.zeros(2, 2)), 0)
    loss.backward()
    optimizer.step()
    assert not torch.equal(before, model.block.fc1.weight)


def test_lightning_harness_is_optional(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("torch")
    original_import = builtins.__import__

    def missing_lightning(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "lightning":
            raise ModuleNotFoundError("No module named 'lightning'", name="lightning")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing_lightning)
    ns = region("demos/deep_learning_demo.py", "fnd.transformers-foundation-models.frameworks-01")
    assert "TinyBlock" in ns
    assert "TinyLightningModule" not in ns


def test_regularization_config_requires_only_applied_l2() -> None:
    ns = region("demos/deep_learning_demo.py", "fnd.optimization-regularization.core-01")
    cfg = ns["RegularizationConfig"](l2=0.1)
    assert set(type(cfg).model_fields) == {"l2"}
    assert ns["l2_penalty"](np.array([2.0]), cfg) == pytest.approx(0.4)


def test_attention_kernel_does_not_print_unused_config() -> None:
    ns = region("demos/deep_learning_demo.py", "fnd.transformers-foundation-models.core-01")
    assert "AttentionConfig" not in ns


@pytest.mark.parametrize("auc, promote_rate, margin, approved", [
    (0.4, 1.0, 0.05, False), (0.58, 0.5, 0.1, False), (0.65, 0.5, 0.1, True),
])
def test_capstone_gates_use_configured_margin_and_exit_decision(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, auc: float, promote_rate: float,
    margin: float, approved: bool,
) -> None:
    from types import SimpleNamespace

    from spine.ml.gbm_baseline import ModelMetrics
    from spine.ops import run_capstone
    from spine.rl.policy_eval import PolicyEvalResult

    cfg = run_capstone.CapstoneConfig(subset_dir=tmp_path, output_dir=tmp_path,
                                     run_id="test", model_auc_margin=margin)
    monkeypatch.setattr(run_capstone, "parse_args", lambda: cfg)
    monkeypatch.setattr(run_capstone, "build_feature_rows", lambda *args, **kwargs: [SimpleNamespace(label=1)])
    monkeypatch.setattr(run_capstone, "run_training",
                        lambda cfg: ModelMetrics(accuracy=0.4, roc_auc=auc, model_type="test"))
    monkeypatch.setattr(run_capstone, "evaluate_baseline_policy", lambda cfg: PolicyEvalResult(
        steps=1, average_reward=0, promote_rate=promote_rate, accuracy=0.9,
        majority_accuracy=0.5, output_dir=tmp_path,
    ))
    if approved:
        run_capstone.main()
    else:
        with pytest.raises(SystemExit) as error:
            run_capstone.main()
        assert error.value.code == 1
    report = json.loads((tmp_path / "capstone_test/capstone_report.json").read_text())
    assert report["approved"] == approved
    assert report["model_auc_margin"] == margin


def test_timeseries_example_data_is_seeded() -> None:
    # Execute only the example with a recording trainer; no optional backend needed.
    path = ROOT / "demos/timeseries_demo.py"
    example = next(node for node in ast.parse(path.read_text()).body
                   if isinstance(node, ast.FunctionDef) and node.name == "_example")
    import pandas as pd

    frames = []
    ns: dict[str, Any] = {"pd": pd, "np": np, "TimeSeriesConfig": lambda **kwargs: None,
                         "train_lgbm_forecaster": lambda data, cfg: (frames.append(data), 0)}
    exec(compile(ast.Module(body=[example], type_ignores=[]), str(path), "exec"), ns)
    ns["_example"]()
    ns["_example"]()
    pd.testing.assert_frame_equal(frames[0], frames[1])
