"""Five evidenced answers determine the written case, never execution authority."""

import json

import pytest
from pydantic import ValidationError


def types():
    from refund_agent.learning_case import FitRow, LearningCase, stage3_cases, verdict

    return FitRow, LearningCase, stage3_cases, verdict


def test_stage3_cases_hold_both_candidates_and_keep_owner():
    _, _, build, _ = types()
    cases = build()
    assert len(cases) == 2
    assert [case.verdict for case in cases] == ["not yet", "not yet"]
    assert cases[0].rows[0].answer == "no"
    assert cases[0].rows[2].answer == "no"
    assert cases[1].rows[0].answer == "yes"
    assert all(case.owner == "refund_cap_v4 policy owner" for case in cases)
    assert all("autonomy unchanged" in case.render() for case in cases)


@pytest.mark.parametrize("answer", ("no", "unknown", "yes"))
def test_learn_requires_all_five_yes(answer):
    FitRow, LearningCase, build, verdict = types()
    fields = build()[0].model_dump()
    fields.pop("verdict")
    fields["rows"] = [
        FitRow(
            question=question,
            answer="yes",
            evidence="observed result",
            source="companion/demos/example.py:1",
        ).model_dump()
        for question in ("sequential", "feedback", "choice", "safety", "value")
    ]
    fields["rows"][2]["answer"] = answer
    result = LearningCase.model_validate(fields)
    assert result.verdict == ("learn" if answer == "yes" else "not yet")
    assert verdict(result.rows) == result.verdict


@pytest.mark.parametrize("field", ("evidence", "source"))
@pytest.mark.parametrize("text", ("", "   "))
def test_rows_refuse_missing_evidence_or_source(field, text):
    FitRow, _, _, _ = types()
    data = dict(question="sequential", answer="yes", evidence="observed", source="file:1")
    data[field] = text
    with pytest.raises(ValidationError):
        FitRow(**data)


def test_case_round_trips_json_and_refuses_forged_verdict_or_duplicate_question():
    _, LearningCase, build, _ = types()
    case = build()[1]
    assert LearningCase.model_validate_json(case.model_dump_json()) == case
    data = json.loads(case.model_dump_json())
    data["verdict"] = "learn"
    with pytest.raises(ValidationError):
        LearningCase.model_validate(data)
    data.pop("verdict")
    data["rows"][4]["question"] = data["rows"][0]["question"]
    with pytest.raises(ValidationError):
        LearningCase.model_validate(data)


def test_validated_record_cannot_mutate_answers_or_verdict():
    _, _, build, _ = types()
    case = build()[1]
    with pytest.raises(ValidationError):
        case.verdict = "learn"
    with pytest.raises(ValidationError):
        case.rows[0].answer = "no"
    with pytest.raises(TypeError):
        case.rows[0] = case.rows[1]
    assert case.verdict == "not yet"
    assert "verdict: not yet" in case.render()


def test_exercise6_threshold_candidate_has_five_cited_rows_and_rule_verdict():
    from refund_agent.learning_case import LearningCase, threshold_case, verdict

    case = threshold_case()
    assert isinstance(case, LearningCase)
    assert case.verdict == verdict(case.rows) == "not yet"
    assert "threshold" in case.decision
    assert len(case.rows) == 5
    assert all(row.source.startswith("companion/") and ":" in row.source for row in case.rows)
    assert "| sequential |" in case.render()
