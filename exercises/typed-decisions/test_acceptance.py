"""Exercise 6 checks, separate from the default suite; the starter fails deliberately.

Set TYPED_DECISIONS_SOLUTION to a completed Python submission. The reference
solution is public; the supplied starter does not contain it.
"""

import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from demos.typed_decisions_demo import july_rows  # noqa: E402
from refund_agent.typed import ACTION_MAP, ScriptedVendorModel  # noqa: E402

PATH = Path(os.environ.get("TYPED_DECISIONS_SOLUTION", Path(__file__).with_name("starter.py")))
SPEC = importlib.util.spec_from_file_location("typed_decisions_submission", PATH)
if SPEC is None or SPEC.loader is None:
    raise ValueError("TYPED_DECISIONS_SOLUTION must name a Python submission")
SOLUTION = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SOLUTION)


@pytest.fixture(scope="module")
def rows():
    return july_rows()


def test_refuses_both_planted_faults_and_preserves_the_no_denial_map(rows):
    report = SOLUTION.accept(rows, ScriptedVendorModel())
    assert report["decision"] == "refuse"
    assert set(report["reasons"]) == {"reliability", "negation"}
    assert report["reliability"] > .0005
    assert report["negation_violations"] == len(rows)
    assert "deny" not in ACTION_MAP.values()


def test_brier_is_measured_on_real_outcomes_and_parts_include_the_binning_residual(rows):
    report = SOLUTION.accept(rows, ScriptedVendorModel())
    p = np.array(ScriptedVendorModel().predict(rows))
    y = np.array([r.found_false_30d for r in rows])
    reliability, resolution, residual = 0., 0., 0.
    for group in np.array_split(np.argsort(p, kind="stable"), 5):
        forecast, observed = p[group].mean(), y[group].mean()
        reliability += len(group) * (forecast - observed) ** 2 / len(rows)
        resolution += len(group) * (observed - y.mean()) ** 2 / len(rows)
        centered_p, centered_y = p[group] - forecast, y[group] - observed
        residual += np.sum(centered_p ** 2 - 2 * centered_p * centered_y) / len(rows)
    expected = dict(brier=np.mean((p - y) ** 2), reliability=reliability,
                    resolution=resolution, uncertainty=y.mean() * (1 - y.mean()),
                    residual=residual)
    for name, value in expected.items():
        assert report[name] == pytest.approx(value, rel=0, abs=1e-12), name
    assert report["brier"] == pytest.approx(report["reliability"] - report["resolution"]
        + report["uncertainty"] + report["residual"], abs=1e-12)


@pytest.mark.parametrize("incoherent", [False, True])
def test_negation_probe_uses_the_supplied_models_responses(rows, incoherent):
    class ProbeModel(ScriptedVendorModel):
        negation_calls = 0

        def predict(self, rows, negated=False):
            p = np.array(super().predict(rows))
            if not negated:
                return p.tolist()
            self.negation_calls += 1
            complement = 1 - p
            if incoherent:
                complement[::2] = p[::2]
            return complement.tolist()

    model = ProbeModel()
    report = SOLUTION.accept(rows, model)
    assert model.negation_calls > 0
    p = np.array(model.predict(rows))
    negated = np.array(model.predict(rows, negated=True))
    violations = int(np.sum(np.abs(p + negated - 1) > 1e-9))
    assert violations < len(rows)
    assert report["negation_violations"] == violations
    assert report["decision"] == "refuse"
    assert set(report["reasons"]) == ({"reliability", "negation"} if violations else {"reliability"})


def test_seed_twenty_reproduces_every_digit_and_passes_only_july(rows, monkeypatch):
    from demos import typed_decisions_demo
    from refund_agent.data import history
    from refund_agent.split import GoldenSet

    def forbidden(*args, **kwargs):
        raise AssertionError("exercise 6 must never consult a frozen month")

    monkeypatch.setattr(GoldenSet, "consult", forbidden)
    second_rows = july_rows()
    monkeypatch.setattr(history, "generate_history", forbidden)
    monkeypatch.setattr(typed_decisions_demo, "generate_history", forbidden)
    first = SOLUTION.accept(rows, ScriptedVendorModel())
    second = SOLUTION.accept(second_rows, ScriptedVendorModel())
    assert first == second
    assert first["seed"] == 20 and first["requests"] == len(rows)
    assert all(r.at.year == 2026 and r.at.month == 7 for r in rows)


@pytest.mark.parametrize("outcomes", [[0] * 20, [1] * 20, [0, 1] * 10])
@pytest.mark.parametrize("prediction", [0., .02, np.sqrt(.0005), .03, .5, 1.])
@pytest.mark.parametrize("incoherent", [False, True])
def test_changed_outcomes_and_predictions_apply_both_sides_of_ceiling(outcomes, prediction, incoherent):
    rows = [SimpleNamespace(found_false_30d=y) for y in outcomes]

    class Model:
        def predict(self, rows, negated=False):
            return [(.25 if incoherent else 1 - prediction) if negated else prediction] * len(rows)

    report = SOLUTION.accept(rows, Model())
    mean = sum(outcomes) / len(outcomes)
    reliability = (prediction - mean) ** 2
    violations = len(rows) if incoherent else 0
    reasons = ({"reliability"} if reliability > .0005 else set()) | ({"negation"} if violations else set())
    assert report["decision"] == ("refuse" if reasons else "accept")
    assert set(report["reasons"]) == reasons
    assert report["requests"] == len(rows)
    assert report["negation_violations"] == violations
    assert report["reliability"] == pytest.approx(reliability, abs=1e-12)
    assert report["brier"] == pytest.approx(np.mean((prediction - np.array(outcomes)) ** 2))
    assert report["resolution"] == pytest.approx(0.)
    assert report["uncertainty"] == pytest.approx(mean * (1 - mean))
    assert report["residual"] == pytest.approx(0., abs=1e-12)
    assert json.loads(json.dumps(report, allow_nan=False)) == report
