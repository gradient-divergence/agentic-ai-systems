"""Exercise 6 acceptance; run explicitly, with the unfinished starter by default.

Set AUTOENCODERS_NLP_SOLUTION=/path/to/submission.py for a completed solution.
The historical questions are fixed before either encoder is evaluated.
"""

import importlib.util
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from demos.autoencoders_nlp_demo import build_experiment
from refund_agent.data.generate import generate
from refund_agent.retrieval.clauses import policy_clauses
from refund_agent.retrieval.encoders import CountEncoder
from refund_agent.retrieval.index import PolicyIndex

PATH = Path(os.environ.get("AUTOENCODERS_NLP_SOLUTION", Path(__file__).with_name("starter.py")))
SPEC = importlib.util.spec_from_file_location("autoencoders_nlp_solution", PATH)
SOLUTION = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SOLUTION)
CHANGE = datetime(2026, 10, 1, tzinfo=timezone.utc)


@pytest.fixture
def policies():
    original = generate(7).policies
    before = [policy.model_dump_json() for policy in original]
    extended = SOLUTION.extend_policies(original)
    assert [policy.model_dump_json() for policy in original] == before
    assert [policy.model_dump_json() for policy in extended[:5]] == before
    return extended


@pytest.fixture(scope="module")
def historical():
    return build_experiment(7)["golden_questions"]["policy"]


def test_sixth_version_has_the_constructed_window_and_effective_date(policies):
    assert len(policies) == 6
    sixth = policies[-1]
    assert sixth.effective_at == CHANGE
    assert sixth.version == "2026-10-01" and sixth.return_days == 60
    assert "within 60 days" in sixth.text and "2026-10-01" in sixth.text


@pytest.mark.parametrize("rarity", [False, True])
def test_twenty_boundary_questions_and_every_printed_hit_are_dated(policies, rarity, capsys):
    questions = SOLUTION.boundary_questions()
    assert len(questions) == len({q.request_id for q in questions}) == 20
    assert any(q.as_of < CHANGE for q in questions)
    assert any(q.as_of >= CHANGE for q in questions)
    for question in questions:
        version = max(
            (p for p in policies if p.effective_at <= question.as_of), key=lambda p: p.effective_at
        )
        expected = f"{version.policy_id}@{version.version}#{question.section}"
        assert question.relevant == expected
        index = SOLUTION.build_index(policies, question.as_of - timedelta(hours=1), rarity)
        result = SOLUTION.print_hits(index, question)
        assert result.refusal is None and len(result.hits) == 1
        assert result.hits[0].clause.citation == expected
        assert all(hit.clause.in_force(question.as_of) for hit in result.hits)
        printed = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
        assert printed == [hit.clause.record() for hit in result.hits]
        for record in printed:
            assert {"citation", "effective_from", "effective_to", "observed_at"} <= record.keys()


@pytest.mark.parametrize("rarity", [False, True])
def test_index_manifest_and_encoder_match_the_policy_text_only_fit(policies, rarity):
    read_at = CHANGE - timedelta(hours=1)
    index = SOLUTION.build_index(policies, read_at, rarity)
    encoder = CountEncoder(rarity=rarity).fit([c.text for c in policy_clauses(policies, read_at)])
    expected = PolicyIndex.build(policies, read_at, encoder)
    assert index.encoder.manifest() == encoder.manifest()
    assert index.manifest == expected.manifest
    assert [clause.record() for clause in index.clauses] == [
        clause.record() for clause in expected.clauses
    ]


@pytest.mark.parametrize("rarity", [False, True])
def test_thirty_one_day_old_read_returns_nothing(policies, rarity):
    stale = SOLUTION.build_index(policies, CHANGE - timedelta(days=31), rarity)
    result = stale.retrieve("damaged 55 days after delivery", CHANGE, 1)
    assert not result.hits and result.refusal == "stale_policy"
    fresh = SOLUTION.build_index(policies, CHANGE - timedelta(days=30), rarity)
    assert fresh.retrieve("damaged 55 days after delivery", CHANGE, 1).hits


@pytest.mark.parametrize("rarity", [False, True])
def test_golden_recall_at_one_is_unchanged(policies, historical, rarity):
    original = generate(7).policies
    encoder = CountEncoder(rarity=rarity).fit([c.text for c in policy_clauses(original, CHANGE)])
    before, after = [], []
    for question in historical:
        read_at = question.as_of - timedelta(hours=1)
        baseline = PolicyIndex.build(original, read_at, encoder)
        updated = SOLUTION.build_index(policies, read_at, rarity)
        base_hits = baseline.retrieve(question.text, question.as_of, 1).hits
        hits = updated.retrieve(question.text, question.as_of, 1).hits
        assert all(hit.clause.in_force(question.as_of) for hit in hits)
        before.append(any(hit.clause.citation == question.relevant for hit in base_hits))
        after.append(any(hit.clause.citation == question.relevant for hit in hits))
    assert sum(after) == sum(before) == len(historical)
