"""Hold persistence, chronology and exact release evidence."""

import json
from datetime import datetime, timedelta, timezone

import pytest

from refund_agent.audit.audits_explanations_holds import AuditHolds
from refund_agent.audit.audits_explanations_rates import (
    AuditObservation,
    AuditRule,
    Period,
    audit_period,
    minimum_release_sample,
)

AT = datetime(2026, 5, 1, tzinfo=timezone.utc)


def record(day, *, wrong=22, read=80):
    start = AT + timedelta(days=day)
    rows = tuple(
        AuditObservation(
            request_id=f"audits-explanations-{day}-{i}",
            order_id=f"audits-explanations-o-{i}",
            segment="C",
            at=start + timedelta(days=1),
            outcome="refer" if i < 239 else "approve",
            held_proposal=i < 239,
            reason="already_delivered" if i < 239 else "eligible",
            owed=i < wrong if i < 239 else True,
            found_false_if_paid=None if i < 239 else False,
            label_mature_at=start + timedelta(days=32),
        )
        for i in range(634)
    )
    return audit_period(
        rows,
        Period(start=start, end=start + timedelta(days=28)),
        AuditRule(signed_at=AT - timedelta(days=1), read_per_segment=read),
        7,
        at=start + timedelta(days=33),
    )


def test_restart_small_clean_sample_keeps_hold_large_sample_releases(tmp_path):
    path = tmp_path / "audits-explanations-holds.jsonl"
    store = AuditHolds(path)
    first = record(0)
    assert first.segments[2].breach
    events = store.apply(first)
    assert len(events) == 1 and events[0].action == "open"
    assert store.apply(first) == ()
    assert not store.is_held(
        "C", "already_delivered", first.at - timedelta(seconds=1)
    )
    assert AuditHolds(path).is_held("C", "already_delivered", first.at)
    clean = record(34, wrong=0)
    assert (
        not clean.segments[2].breach and not clean.segments[2].release_allowed
    )
    assert store.apply(clean) == ()
    assert AuditHolds(path).is_held("C", "already_delivered", clean.at)
    needed = minimum_release_sample(239, 395, 0.01)
    clean = record(68, wrong=0, read=needed)
    assert clean.segments[2].release_allowed
    assert store.apply(clean)[0].action == "release"
    assert not AuditHolds(path).is_held("C", "already_delivered", clean.at)
    assert AuditHolds(path).is_held("C", "already_delivered", first.at)


def test_tampering_late_audit_and_wrong_owner_refused(tmp_path):
    path = tmp_path / "audits-explanations-holds.jsonl"
    store = AuditHolds(path)
    store.apply(record(34))
    with pytest.raises(ValueError, match="out of order"):
        store.apply(record(0))
    with pytest.raises(ValueError, match="owner"):
        store.apply(
            record(68).model_copy(
                update={
                    "rule": record(68).rule.model_copy(
                        update={"signed_by": "alex"}
                    )
                }
            )
        )
    lines = path.read_text().splitlines()
    row = json.loads(lines[0])
    row["audit_fingerprint"] = "tampered"
    path.write_text(json.dumps(row) + "\n" + "\n".join(lines[1:]) + "\n")
    with pytest.raises(ValueError, match="integrity"):
        AuditHolds(path).events()
