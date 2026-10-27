import csv
from collections import defaultdict
from datetime import timedelta
from hashlib import sha256
from pathlib import Path

import pytest

from refund_agent.contracts import Request, write_json
from refund_agent.data import history
from refund_agent.data.generate import generate


@pytest.fixture(scope="module")
def rows() -> list[history.HistoryRow]:
    return history.generate_history()


def test_deterministic_csv_and_seed(tmp_path: Path) -> None:
    first = history.generate_history(rows=200)
    assert first == history.generate_history(rows=200)
    assert first != history.generate_history(seed=8, rows=200)
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    history.write_history(first, a)
    history.write_history(history.generate_history(rows=200), b)
    assert a.read_bytes() == b.read_bytes()
    assert history.load_history(a) == first
    with a.open(newline="") as stream:
        assert next(csv.reader(stream)) == list(history.HistoryRow.model_fields)


def test_schema_and_decision_time_features(rows: list[history.HistoryRow]) -> None:
    assert len(rows) == 36000
    assert len({row.request_id for row in rows}) == len(rows)
    assert len({row.requester_id for row in rows}) == len(rows)
    assert rows == sorted(rows, key=lambda row: (row.at, row.request_id))
    assert {row.at.strftime("%Y-%m") for row in rows} == {
        f"2025-{month:02}" for month in range(4, 13)
    } | {f"2026-{month:02}" for month in range(1, 10)}
    for row in rows:
        request = Request(
            **{
                name: getattr(row, name)
                for name in (
                    "request_id",
                    "requester_id",
                    "order_reference",
                    "at",
                    "amount_cents",
                    "reason",
                )
            }
        )
        assert request.at == row.at and row.at.utcoffset() == timedelta(0)
        assert type(row.amount_cents) is int and row.amount_cents > 0
        assert type(row.paid_cents) is int and type(row.prior_refund_count) is int
        assert type(row.paid) is bool
        assert row.policy_version <= row.at.date().isoformat()
        assert row.paid == (row.policy_decision == "approve")
        if row.paid:
            assert type(row.found_false_30d) is int and row.found_false_30d in (0, 1)
            assert row.label_final_at == row.at + timedelta(days=30)
            assert row.amount_cents < 5000
        else:
            assert row.found_false_30d is None and row.label_final_at is None
        assert (row.delivery_age_days is None) == (not row.shipment_delivered)
    assert len({row.amount_cents for row in rows if row.paid}) > 1000


def test_base_rate_and_drift(rows: list[history.HistoryRow]) -> None:
    before = [row for row in rows if row.paid and row.at < history.DRIFT_AT]
    after = [row for row in rows if row.paid and row.at >= history.DRIFT_AT]
    pre = sum(row.found_false_30d == 1 for row in before) / len(before)
    post = sum(row.found_false_30d == 1 for row in after) / len(after)
    assert 0.015 <= pre <= 0.04
    assert 0.03 <= post <= 0.08
    assert post > 1.4 * pre
    row = before[0]
    later = row.model_copy(update={"at": history.DRIFT_AT})
    assert history._false_probability(later) == 2 * history._false_probability(row)


def test_label_draw_cannot_change_features(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(history, "_outcome_draw", lambda seed, index: 0.0)
    positive = history.generate_history(rows=300)
    monkeypatch.setattr(history, "_outcome_draw", lambda seed, index: 1.0)
    negative = history.generate_history(rows=300)
    for a, b in zip(positive, negative):
        assert a.model_dump(exclude={"found_false_30d"}) == b.model_dump(
            exclude={"found_false_30d"}
        )
        if a.paid:
            assert a.found_false_30d == 1 and b.found_false_30d == 0


def test_no_feature_determines_label(rows: list[history.HistoryRow]) -> None:
    assert set(history.FEATURE_COLUMNS).isdisjoint(
        {
            "found_false_30d",
            "paid",
            "policy_decision",
            "label_final_at",
            "request_id",
            "requester_id",
            "order_reference",
            "at",
            "policy_version",
        }
    )
    for feature in history.FEATURE_COLUMNS:
        labels: dict[object, set[int | None]] = defaultdict(set)
        for row in rows:
            if row.paid:
                labels[getattr(row, feature)].add(row.found_false_30d)
        assert any(values == {0, 1} for values in labels.values()), feature


def test_signal_and_noise_roles(rows: list[history.HistoryRow]) -> None:
    paid = [row for row in rows if row.paid and row.at < history.DRIFT_AT]
    with_prior = [row for row in paid if row.prior_refund_count]
    without_prior = [row for row in paid if not row.prior_refund_count]

    def rate(group: list[history.HistoryRow]) -> float:
        return sum(row.found_false_30d == 1 for row in group) / len(group)

    assert rate(with_prior) > 4 * rate(without_prior)
    row = paid[0]
    changed_noise = row.model_copy(
        update={
            "paid_cents": row.paid_cents + 100,
            "order_evidence_age_seconds": row.order_evidence_age_seconds + 1,
            "shipment_evidence_age_seconds": row.shipment_evidence_age_seconds + 1,
        }
    )
    assert history._false_probability(row) == history._false_probability(changed_noise)


def test_golden_corpus_bytes_unchanged(tmp_path: Path) -> None:
    path = tmp_path / "golden.json"
    write_json(path, generate(seed=7))
    # Canonical write_json bytes captured at base 77d1ec2f before this change.
    assert sha256(path.read_bytes()).hexdigest() == (
        "aaa172c9b3cda32fd58fa92db49c5978936248bf6de0e107fbc989bc902bd49f"
    )
