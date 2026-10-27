"""The requirement is signed before its comparison, not merely before its gate."""

import pytest

from refund_agent.launch import validate, validate_row
from refund_agent.launch.capstone_blueprint_requirement import (
    Period,
    refund_requirement,
)

ROLES = {
    "decision_owner": ("morgan",),
    "release_owner": ("morgan",),
    "reviewer": ("alex",),
    "named_approver": ("morgan",),
}


@pytest.mark.parametrize(
    "row_id,question",
    [
        ("4", "What evidence would show that the agent is improving?"),
        ("6", "What can a person override, and how fast?"),
        ("7", "Which rollback has actually been rehearsed?"),
    ],
)
def test_requirement_uses_the_authors_questions(row_id, question):
    row = next(row for row in refund_requirement("B6").rows if row.row_id == row_id)
    assert row.question == question


def test_comparison_cannot_open_on_signature_day():
    req = refund_requirement(
        "B6", signed_day=1, comparison_period=Period(start=1, end=14)
    )
    with pytest.raises(ValueError, match="comparison.*signature"):
        validate(req, ROLES)


def test_default_requirement_and_transferred_roles():
    req = refund_requirement("B6")
    assert validate(req, ROLES) == req
    assert len(req.rows) == 7
    assert (
        validate_row(req.rows[0], {k: "domain's named holder" for k in ROLES})
        == ()
    )


@pytest.mark.parametrize(
    "update,reason",
    [
        ({"signed_by": "alex"}, "decision_owner"),
        ({"signed_day": 64}, "precede gate"),
        ({"comparison_period": Period(start=50, end=50)}, "empty comparison"),
        (
            {"comparison_period": Period(start=50, end=65)},
            "comparison closes after gate",
        ),
    ],
)
def test_refuse_invalid_requirement(update, reason):
    with pytest.raises(ValueError, match=reason):
        validate(refund_requirement("B6").model_copy(update=update), ROLES)


def test_comparison_is_not_refreshable():
    row = refund_requirement("B6").rows[0]
    changed = row.model_copy(
        update={
            "records": (
                row.records[0].model_copy(update={"rerunnable": True}),
            )
        }
    )
    assert "comparison is never rerunnable" in validate_row(changed, ROLES)
