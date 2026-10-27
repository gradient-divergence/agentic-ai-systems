"""Unsealed exercise checks; the full Candidate C evaluation requires a separate evaluation run.

Only candidate_c.json and the shipped constructed exercise-5 rows are read.
No OfferPackage, outcome store or simulation truth is opened.
"""

import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pytest

from refund_agent.data.generate import generate
from refund_agent.labels import LABEL_POLICY
from refund_agent.offers.evaluation import OfferPackage, SealedRecords
from refund_agent.offers.notices import OFFERS
from refund_agent.offpolicy.estimators import (
    Record,
    direct_method,
    doubly_robust,
    effective_sample_size,
    ips,
    policy_ratios,
    self_normalized_ips,
)
from refund_agent.offpolicy.policies import Policy
from refund_agent.offpolicy.reward_model import CellModel
from refund_agent.offpolicy.support import support_table, supported
from refund_agent.split import GoldenSet, freeze

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def golden_evidence(tmp_path):
    data = generate()
    golden = freeze(data, data.requests, LABEL_POLICY, name="chapter4-exercise-guard")
    directory = tmp_path / "golden-sets"
    golden.persist(
        directory,
        prior_log=[
            dict(entry=1, set=golden.manifest["name"], reader="chapter4", purpose="prior evidence")
        ],
    )
    return golden, directory, {p: p.read_bytes() for p in directory.glob("*")}


@pytest.fixture
def solution(monkeypatch, golden_evidence):
    def forbidden(*args, **kwargs):
        raise AssertionError("unsealed exercise checks must not open a package or sealed store")

    monkeypatch.setattr(OfferPackage, "load", forbidden)
    monkeypatch.setattr(SealedRecords, "__init__", forbidden)
    monkeypatch.setattr(SealedRecords, "load", forbidden)
    monkeypatch.setattr(SealedRecords, "consult", forbidden)
    monkeypatch.setattr(GoldenSet, "consult", forbidden)
    monkeypatch.setattr(GoldenSet, "persist", forbidden)
    path = Path(os.environ.get("PRACTICAL_RL_SOLUTION", Path(__file__).with_name("starter.py"))).resolve()
    monkeypatch.chdir(golden_evidence[1].parent)
    spec = importlib.util.spec_from_file_location("unsealed_practical_submission", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_candidate_c_specification_and_unchanged_chapter4_golden_journal(solution, golden_evidence):
    golden, directory, before = golden_evidence
    specification = json.loads(Path(__file__).with_name("candidate_c.json").read_text())
    frozen_at = "2026-02-01T00:00:00+00:00"
    policy = solution.candidate_c(frozen_at)
    assert policy.version == specification["version"] and policy.frozen_at == frozen_at
    assert {cell for cell, _ in policy.tables} == set(specification["choices"])
    for cell, chosen in specification["choices"].items():
        assert policy.probabilities(cell) == {action: float(action == chosen) for action in OFFERS}
    assert before == {p: p.read_bytes() for p in directory.glob("*")}
    assert len(golden.log) == 1


def test_shipped_exercise_support_and_estimates_are_independently_recomputed():
    fixture = json.loads((ROOT / "demos/data/practical_rl_reports_05.json").read_text())
    raw = fixture["records"]
    records = [Record(**row) for row in raw]
    policy = Policy(**fixture["candidate"])
    model = CellModel(tuple(tuple(row) for row in fixture["model"]))
    table = support_table(records, policy)
    expected = []
    for cell in sorted({row["context"] for row in raw}):
        group = [row for row in raw if row["context"] == cell]
        for action in sorted(OFFERS):
            probabilities = [row["logged"].get(action, 0.0) for row in group]
            positives = [p for p in probabilities if p > 0]
            expected.append(
                dict(
                    cell=cell,
                    offer=action,
                    records=len(group),
                    positive_share=len(positives) / len(group),
                    min_positive=min(positives) if positives else None,
                    acts=policy.probabilities(cell).get(action, 0.0) > 0,
                    eligible=all(not row["eligible"] or action in row["eligible"] for row in group),
                    matches=sum(row["action"] == action for row in group),
                )
            )
    assert table == expected
    assert not supported(table), "not-delivered dated notices have no support"
    assert supported([row for row in table if row["cell"] != "not_delivered"])

    supported_raw = [row for row in raw if row["context"] != "not_delivered"]
    selected = [row for row in records if row.context != "not_delivered"]
    weights = [
        policy.probabilities(row["context"])[row["action"]] / row["logged"][row["action"]]
        for row in supported_raw
    ]
    rewards = [row["reward"] for row in supported_raw]
    predictions = {(cell, action): mean for cell, action, mean, _ in fixture["model"]}
    dm_terms = [
        sum(
            p * predictions[(row["context"], action)]
            for action, p in policy.probabilities(row["context"]).items()
        )
        for row in supported_raw
    ]
    dr_terms = [
        dm + weight * (row["reward"] - predictions[(row["context"], row["action"])])
        for dm, weight, row in zip(dm_terms, weights, supported_raw)
    ]
    total = sum(w * y for w, y in zip(weights, rewards))
    np.testing.assert_allclose(policy_ratios(policy, selected), weights)
    assert ips(weights, rewards) == pytest.approx(total / len(selected)) == pytest.approx(0.875)
    assert self_normalized_ips(weights, rewards) == pytest.approx(total / sum(weights))
    assert effective_sample_size(weights) == pytest.approx(
        sum(weights) ** 2 / sum(w * w for w in weights)
    )
    assert direct_method(policy, model, [r.context for r in selected]) == pytest.approx(
        sum(dm_terms) / len(selected)
    )
    assert doubly_robust(policy, model, selected) == pytest.approx(sum(dr_terms) / len(selected))
