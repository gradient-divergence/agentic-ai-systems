import ast
import builtins
import json
import math
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Annotated, Any

import numpy as np
import pandas as pd
import pytest
from pydantic import BaseModel, Field

from demos import regions as _regions

ROOT = Path(__file__).resolve().parents[1]
extract = vars(_regions)[
    "extract_snippets_from_text"
]


def region(source: str, snippet_id: str, namespace: dict[str, Any] | None = None) -> dict[str, Any]:
    path = ROOT / source
    snippet = next(s for s in extract(path.read_text(), path) if s.snippet_id == snippet_id)
    namespace = {} if namespace is None else namespace
    exec(compile(snippet.content, str(path), "exec"), namespace)
    return namespace


def test_gqa_cache_uses_head_dimension_and_layer_count() -> None:
    ns = region("demos/deep_learning_demo.py", "fnd.transformers-foundation-models.sota-01")
    # Four key/value heads, each 4096 / 16 = 256 numbers wide.
    assert ns["kv_cache_bytes"](1, 4, 256, 2048, 2) == 8_388_608
    assert ns["kv_cache_bytes"](3, 4, 256, 2048, 2) == 3 * 8_388_608


def test_brier_score_is_finite_probability_loss() -> None:
    ns = region("demos/supervised_ml_demo.py", "fnd.ml-data-and-evaluation.sota-01")
    scores = ns["calibrated_cv_scores"]()
    assert math.isfinite(scores["brier"])
    assert 0.0 <= scores["brier"] <= 1.0


def timeseries_features() -> dict[str, Any]:
    # Load the feature helper without requiring the optional training backend.
    path = ROOT / "demos/timeseries_demo.py"
    nodes: list[ast.stmt] = [
        node
        for node in ast.parse(path.read_text()).body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef))
        and node.name in {"TimeSeriesConfig", "_make_features"}
    ]
    ns = {
        "pd": pd,
        "np": np,
        "BaseModel": BaseModel,
        "Field": Field,
        "Annotated": Annotated,
        "List": list,
        "Tuple": tuple,
    }
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), ns)
    return ns


def test_rolling_features_exclude_current_target() -> None:
    ns = timeseries_features()
    cfg = ns["TimeSeriesConfig"](lags=[1], rolling_window=2)
    df = pd.DataFrame(
        {"ds": pd.date_range("2024-01-01", periods=5), "y": [1.0, 2.0, 3.0, 4.0, 5.0]}
    )
    X, _ = ns["_make_features"](df, cfg)
    changed = df.copy()
    changed.loc[3, "y"] = 1000.0
    altered, _ = ns["_make_features"](changed, cfg)
    assert X.loc[3, "y_roll_mean_2"] == altered.loc[3, "y_roll_mean_2"] == 2.5
    for lag in [0, -1]:
        with pytest.raises(ValueError):
            ns["TimeSeriesConfig"](lags=[lag])


def test_forecaster_features_exclude_datetime() -> None:
    ns = timeseries_features()
    cfg = ns["TimeSeriesConfig"](lags=[1], rolling_window=2)
    df = pd.DataFrame(
        {"ds": pd.date_range("2024-01-01", periods=5), "y": [1.0, 2.0, 3.0, 4.0, 5.0]}
    )
    X, _ = ns["_make_features"](df, cfg)
    assert cfg.time_col not in X.columns
    assert all(pd.api.types.is_numeric_dtype(dtype) for dtype in X.dtypes)


def test_policy_can_monitor_low_scores(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from spine.rl import policy_eval
    from spine.rl.environment import ReviewEnvironment

    rows: list[Any] = [
        SimpleNamespace(
            review_id=str(i),
            review_stars=stars,
            user_average_stars=stars,
            business_stars=stars,
            label=label,
        )
        for i, (stars, label) in enumerate([(1.0, 0), (5.0, 1)])
    ]
    monkeypatch.setattr(policy_eval, "load_environment", lambda cfg: ReviewEnvironment(rows, 2))
    cfg = policy_eval.PolicyEvalConfig(subset_dir=tmp_path, output_dir=tmp_path, max_steps=2)
    result = policy_eval.evaluate_baseline_policy(cfg)
    assert result.promote_rate == 0.5
    assert result.accuracy == 1.0


@pytest.mark.parametrize("auc", [0.5, 0.5001])
def test_governance_rejects_constant_policy_and_chance_model(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    auc: float,
) -> None:
    from spine.ml.gbm_baseline import ModelMetrics
    from spine.ops import run_capstone
    from spine.rl.policy_eval import PolicyEvalResult

    cfg = run_capstone.CapstoneConfig(subset_dir=tmp_path, output_dir=tmp_path, run_id="test")
    monkeypatch.setattr(run_capstone, "parse_args", lambda: cfg)
    monkeypatch.setattr(
        run_capstone,
        "build_feature_rows",
        lambda *args, **kwargs: [SimpleNamespace(label=x) for x in [1, 1, 1, 1, 0]],
    )
    monkeypatch.setattr(
        run_capstone,
        "run_training",
        lambda cfg: ModelMetrics(accuracy=0.8, roc_auc=auc, model_type="test"),
    )
    monkeypatch.setattr(
        run_capstone,
        "evaluate_baseline_policy",
        lambda cfg: PolicyEvalResult(
            steps=5,
            average_reward=0.6,
            promote_rate=1.0,
            accuracy=0.8,
            majority_accuracy=0.8,
            output_dir=tmp_path,
        ),
    )
    with pytest.raises(SystemExit) as error:
        run_capstone.main()
    assert error.value.code == 1
    checklist = json.loads((tmp_path / "capstone_test/ops/governance_checklist.json").read_text())
    assert not checklist["model_eval_ok"]
    assert not checklist["policy_eval_ok"]


def test_brier_scorer_minimizes_loss_and_reports_positive_value() -> None:
    ns = region("demos/supervised_ml_demo.py", "fnd.ml-data-and-evaluation.sota-01")

    def cross_validate(*args: Any, **kwargs: Any) -> dict[str, np.ndarray]:
        assert kwargs["scoring"]["brier"]._sign == -1
        return {"test_roc_auc": np.array([0.8]), "test_brier": np.array([-0.25])}

    ns["cross_validate"] = cross_validate
    assert ns["calibrated_cv_scores"]()["brier"] == 0.25


def test_routing_preserves_original_sender() -> None:
    from demos.agentic_systems_demo import AgentMessage, route_message

    message = AgentMessage(sender="planner", content="validate risk")
    routed = route_message(message, ["executor", "auditor"])
    assert [item.sender for item in routed] == ["planner", "planner"]
    assert [item.recipient for item in routed] == ["executor", "auditor"]


# region chapter34:observability-reliability-nearest-rank
def test_p95_uses_nearest_rank() -> None:
    from refund_agent.observability_reliability.objectives import nearest_rank

    assert nearest_rank((320, 280, 360, 410, 295, 330, 310, 500), .95) == 500
# endregion chapter34:observability-reliability-nearest-rank


def test_dowhy_import_failure_is_not_reported_as_missing_install(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ns = region("demos/unsupervised_causal_automl_demo.py", "fnd.causal-inference.sota-01")
    original_import = builtins.__import__

    def broken_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "dowhy":
            raise ModuleNotFoundError("No module named 'numpy.distutils'", name="numpy.distutils")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", broken_import)
    result = ns["dowhy_ate_demo"]()
    assert result["status"] == "installed but failed to import"
    assert "numpy.distutils" in result["reason"]
    assert "ate" not in result

    def missing_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "dowhy":
            raise ModuleNotFoundError("No module named 'dowhy'", name="dowhy")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing_import)
    assert ns["dowhy_ate_demo"]() == {"status": "skipped (install dowhy)"}


def test_optuna_study_receives_seeded_sampler(monkeypatch: pytest.MonkeyPatch) -> None:
    kwargs = {}
    optuna: Any = ModuleType("optuna")
    optuna.__dict__["Trial"] = object
    optuna.__dict__["samplers"] = SimpleNamespace(
        TPESampler=lambda seed: SimpleNamespace(seed=seed)
    )

    def create_study(**options: Any) -> Any:
        kwargs.update(options)
        return SimpleNamespace(
            optimize=lambda *args, **kwargs: None, best_params={}, best_value=0.0
        )

    optuna.__dict__["create_study"] = create_study
    monkeypatch.setitem(sys.modules, "optuna", optuna)
    region("demos/supervised_ml_demo.py", "fnd.optimization-regularization.optuna-01")
    assert kwargs.get("sampler") is not None
    assert kwargs["sampler"].seed == 23


def test_lightgbm_forecaster_runs() -> None:
    pytest.importorskip("lightgbm")
    from demos.timeseries_demo import TimeSeriesConfig, train_lgbm_forecaster

    rng = np.random.default_rng(13)
    df = pd.DataFrame({"ds": pd.date_range("2024-01-01", periods=80), "y": rng.normal(size=80)})
    _, mae = train_lgbm_forecaster(df, TimeSeriesConfig())
    assert math.isfinite(mae)


# chapter30:data-systems retirement: the toy aggregation's three-line check
# leaves with its listing; the chapter's source-contract tests replace it.


def test_optuna_tuning_is_reproducible() -> None:
    pytest.importorskip("optuna")
    ns = region("demos/supervised_ml_demo.py", "fnd.optimization-regularization.optuna-01")
    assert ns["optuna_tune"](23) == ns["optuna_tune"](23)
