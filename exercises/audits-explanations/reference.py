"""Reference solution: complete record and blind-sample audit response."""

import json
from datetime import datetime

from refund_agent.audit.audits_explanations_explain import explain
from refund_agent.audit.audits_explanations_rates import (
    AuditObservation,
    AuditRule,
    Period,
    audit_period,
)

__all__ = ["explain", "analyze_report"]


def analyze_report(report):
    rows = tuple(
        AuditObservation.model_validate_json(json.dumps(row))
        for row in report["observations"]
    )
    rule = AuditRule.model_validate_json(json.dumps(report["rule"]))
    period = Period.model_validate_json(json.dumps(report["period"]))

    audit = audit_period(
        rows,
        period,
        rule,
        report["seed"],
        at=datetime.fromisoformat(report["at"]),
    )
    return {
        "holds": {
            r.segment: list(r.held_reasons) for r in audit.segments if r.breach
        },
        "notify": rule.source_owner,
        "reply": "Appeals are self-selected. Use blind readings to estimate wrongful denials.",
    }
