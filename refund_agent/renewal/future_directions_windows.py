"""Fresh deterministic requests after history, with disjoint hash draws."""

from dataclasses import dataclass
from datetime import datetime
from math import floor

from refund_agent.data.generate import _policies
from refund_agent.data.history import (
    DEFAULT_ROWS,
    END_AT,
    HistoryRow,
    _false_probability,
    _features,
    _order,
    _outcome_draw,
    _request,
    _shipment,
)
from refund_agent.rules import risk_score
from refund_agent.selection import fingerprint

SPACING = (END_AT - datetime(2025, 4, 1, tzinfo=END_AT.tzinfo)) / DEFAULT_ROWS


def generate_forward(seed: int, start: datetime, paid: int) -> list[HistoryRow]:
    if start < END_AT or type(paid) is not int or paid < 1:
        raise ValueError("forward start and positive paid count required")
    first = DEFAULT_ROWS + floor((start - END_AT) / SPACING) + 1
    rows, paid_count = [], 0
    policy = _policies()[-1]
    index = first
    while paid_count < paid:
        at = END_AT + (index - DEFAULT_ROWS) * SPACING
        order = _order(seed, index, at)
        row = _features(
            _request(seed, index, at, order), order, _shipment(seed, index, at, order), policy
        )
        if row.paid:
            row = row.model_copy(
                update={
                    "found_false_30d": int(_outcome_draw(seed, index) < _false_probability(row))
                }
            )
            paid_count += 1
        rows.append(row)
        index += 1
    return rows


def incumbent_decision(row: HistoryRow) -> str:
    from refund_agent.contracts import Request, State

    request = Request(
        request_id=row.request_id,
        requester_id=row.requester_id,
        order_reference=row.order_reference,
        at=row.at,
        amount_cents=row.amount_cents,
        reason=row.reason,
        text="Please review my refund request.",
    )
    if not row.paid:
        return row.policy_decision
    return "refer" if risk_score(State(request=request, trace_id=row.request_id)) >= 0.2 else "pay"


def candidate_decision(row: HistoryRow, score: float) -> str:
    incumbent = incumbent_decision(row)
    return "refer" if incumbent == "pay" and score >= 0.080 else incumbent


@dataclass(frozen=True)
class Window:
    window_id: str
    start: datetime
    rows: tuple[HistoryRow, ...]
    paid_ids: tuple[str, ...]
    seed: int

    @classmethod
    def generate(cls, window_id, seed, start, paid):
        rows = tuple(generate_forward(seed, start, paid))
        return cls(window_id, start, rows, tuple(r.request_id for r in rows if r.paid), seed)

    @property
    def fingerprint(self):
        return fingerprint(
            {
                "id": self.window_id,
                "start": self.start.isoformat(),
                "seed": self.seed,
                "rows": [r.model_dump(mode="json") for r in self.rows],
            }
        )

    @property
    def mature_at(self):
        dates = [r.label_final_at for r in self.rows if r.paid]
        if not dates or any(at is None for at in dates):
            raise ValueError("paid window requires final observation times")
        return max(at for at in dates if at is not None)


def validate_window(window, rows, *, paid, evaluated_at, expected_start, seed):
    if window.start != expected_start or window.seed != seed:
        raise ValueError("window does not follow the signed selection rule")
    chosen = tuple(r.request_id for r in window.rows if incumbent_decision(r) == "pay")
    if (
        chosen != window.paid_ids
        or len(chosen) != paid
        or len(set(chosen)) != paid
        or (tuple(r.request_id for r in rows) != chosen)
    ):
        raise ValueError("window needs exactly the declared first unique paid requests")
    if any(r.label_final_at is None or r.label_final_at > evaluated_at for r in rows):
        raise ValueError("window has an immature paid outcome")
    # Reconstruct from the signed draw rule, preventing a replacement of case contents.
    canonical = tuple(generate_forward(seed, expected_start, paid))
    if window.rows != canonical or tuple(r for r in canonical if r.paid) != tuple(rows):
        raise ValueError("window differs from its declared generated cases")
