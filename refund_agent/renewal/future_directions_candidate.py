"""Rebuild the sealed selection without generating or consulting frozen months."""

import json
from dataclasses import dataclass
from datetime import timedelta
from functools import lru_cache
from pathlib import Path

import numpy as np

from demos.delegated_model_demo import BASELINE, build
from demos.supervised_learning_demo import GO_LIVE, arrays, platt, windows
from refund_agent.data.generate import _policies
from refund_agent.data.history import DEFAULT_ROWS, END_AT, START_AT, _row
from refund_agent.future_directions_risk import RiskComponent, features
from refund_agent.selection import fingerprint


@dataclass(frozen=True)
class SelectedScore:
    component: RiskComponent
    selection_fingerprint: str
    sealed_receipt: str
    rebuild_fingerprint: str


def development_rows(seed=7):
    policies = _policies()
    first = policies[0].model_dump()
    first.update(
        version="2025-04-01",
        effective_at=START_AT,
        text=policies[0].text.replace("2026-01-20", "2025-04-01"),
    )
    from refund_agent.contracts import Policy

    policies.insert(0, Policy(**first))
    rows = []
    for index in range(DEFAULT_ROWS):
        earliest = START_AT + timedelta(days=index * (END_AT - START_AT).days // DEFAULT_ROWS)
        if earliest >= GO_LIVE:
            break  # No row of either frozen month is generated here.
        row = _row(seed, index, DEFAULT_ROWS, policies)
        if row.paid:
            rows.append(row)
    split = windows(sorted(rows, key=lambda r: (r.at, r.request_id)))
    return {name: split[name] for name in ("training", "validation")}


@lru_cache(maxsize=1)
def selected_score(seed=7):
    record_path = (
        Path(__file__).resolve().parents[2]
        / "book_outputs/_fnd.automl-and-selection.record-full-01.txt"
    )
    record = json.loads(record_path.read_text())
    stored_hash = record.pop("sha256")
    if fingerprint(record) != stored_hash or record["selected"] != BASELINE:
        raise ValueError("stored selection record is changed or selected another score")
    development = development_rows(seed)
    built = build(window=development, seed=seed)
    if built.receipt != record["receipt"]:
        raise ValueError("selected score rebuild differs from the sealed selection fingerprint")
    delivery = built.rebuilds[BASELINE]
    mapping = delivery.statistics["calibration"]
    component = RiskComponent(delivery.model, mapping["slope"], mapping["intercept"])
    if not np.allclose(
        component.predict(development["validation"]),
        delivery.predict(development["validation"]),
        rtol=0,
        atol=1e-12,
    ):
        raise ValueError("risk component differs from the inspector's selected score")
    return SelectedScore(component, stored_hash, built.receipt, component.fingerprint)


def recalibrate(score, rows, *, evaluated_at):
    if not rows or any(
        not r.paid or r.label_final_at is None or r.label_final_at > evaluated_at for r in rows
    ):
        raise ValueError("recalibration requires matured paid outcomes only")
    if len({r.request_id for r in rows}) != len(rows):
        raise ValueError("recalibration requests must be unique")
    raw = np.clip(score.model.predict_proba(features(rows))[:, 1], 1e-12, 1 - 1e-12)
    _, slope, intercept = platt(raw, arrays(rows)[1])
    return RiskComponent(score.model, slope, intercept, score.threshold)
