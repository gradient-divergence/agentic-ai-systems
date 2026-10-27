"""Every tier 3 outcome names the approver, including denials."""

from pathlib import Path
from typing import Literal

from refund_agent import contracts as c
from refund_agent.launch.capstone_blueprint_requirement import (
    LaunchRecord,
    Period,
)
from refund_agent.launch.capstone_blueprint_scan import (
    LaunchLogs,
    _review_matches,
)
from refund_agent.rules import tier_for


class Tier3Record(LaunchRecord):
    period: Period
    requests: int
    outcomes: tuple[dict[str, str], ...]
    decided_otherwise: tuple[str, ...]
    pending: tuple[str, ...]
    verdict: Literal["ok", "hold"]


def inspect_tier3(
    logs: LaunchLogs, policy: c.Policy, period: Period, *, request_ids=None
) -> Tier3Record:
    actions = [
        a
        for a in logs.decisions
        if period.start <= a.requested_day < period.end
        and tier_for(a.request.amount_cents) == c.Tier.NAMED_APPROVER
    ]
    if len({a.request.request_id for a in actions}) != len(actions):
        raise ValueError(
            "tier 3 requires one terminal/pending state per request"
        )
    ids = {a.request.request_id for a in actions}
    if request_ids is not None and ids - set(request_ids):
        raise ValueError(
            "tier 3 log contains a request outside its intake period"
        )
    missing = set(request_ids or ()) - ids
    bad, pending, rows = [], list(missing), []
    for action in actions:
        rid = action.request.request_id
        if action.decision.outcome == c.Outcome.REFER:
            pending.append(rid)
        elif (
            action.review is None
            or not _review_matches(action)
            or action.review.actor != policy.named_approver
        ):
            bad.append(rid)
        rows.append(
            {
                "request_id": rid,
                "outcome": action.decision.outcome.value,
                "approver": action.review.actor
                if action.review is not None
                else "absent",
            }
        )
    return Tier3Record(
        period=period,
        requests=len(request_ids) if request_ids is not None else len(actions),
        outcomes=tuple(rows),
        decided_otherwise=tuple(bad),
        pending=tuple(pending),
        verdict="hold" if bad or pending else "ok",
    )


def tier3_path(run_dir: Path, policy: c.Policy, period: Period) -> Tier3Record:
    from refund_agent.data.generate import Dataset
    from refund_agent.launch.capstone_blueprint_timeline import START

    logs = LaunchLogs.model_validate_json(
        (Path(run_dir) / "capstone-blueprint-logs.json").read_text()
    )
    stream = Dataset.model_validate_json(
        (Path(run_dir) / "capstone-blueprint-stream.json").read_text()
    )
    ids = tuple(
        r.request_id
        for r in stream.requests
        if tier_for(r.amount_cents) == c.Tier.NAMED_APPROVER
        and period.start
        <= 1 + (r.at - START).total_seconds() / 86400
        < period.end
    )
    return inspect_tier3(logs, policy, period, request_ids=ids)
