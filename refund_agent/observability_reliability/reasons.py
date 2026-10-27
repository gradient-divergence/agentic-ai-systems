"""Chapter 34 owns these classes; chapter 35 contributes through reasons hooks.

Unknown codes raise, so a new runtime reason cannot silently become good service.
"""

from types import MappingProxyType

POLICY = (
    "eligible",
    "delivery_unknown",
    "unknown_reason",
    "suspicious_request",
    "already_delivered",
    "outside_return_window",
    "amount_over_policy",
    "order_not_found",
    "approval_required",
    "approval_recorded",
)
SYSTEM = (
    "model_error",
    "tool_error",
    "run_error",
    "missing_evidence",
    "evidence_mismatch",
    "non_current_policy",
    "future_evidence",
    "context_over_budget",
    "policy_not_in_bundle",
    "deadline_exceeded",
    "proposal_required",
    "invalid_approval",
    "amount_mismatch",
    "reason_mismatch",
    "execution_before_proposal",
    "execution_before_request",
    "invalid_execution_time",
    "step_budget_exhausted",
    "runtime_turn_limit",
    "audit_unavailable",
    "unexpected_tool",
    "unknown_tool",
    "invalid_decision",
    "execution_not_recorded",
    "request_content_conflict",
    "request_already_executed",
    "approval_conflict",
    "open_review_required",
    "mutable_state_required",
    "invalid_tool_arguments",
    "order_mismatch",
    "ambiguous_order",
    "future_prior_refund",
    "missing_order",
    "shipment_not_found",
    "policy_not_found",
    "decision_required",
    "idempotency_key_required",
    "missing_execution_arguments",
    "idempotency_conflict",
    "verifier_timeout",
    "verifier_disagreement",
    "verifier_error",
    "canary_shadow",
    "canary_running",
    "canary_paused",
    "canary_ended",
    "canary_rollback",
    "canary_amount",
    "canary_outside_cohort",
    "canary_daily_cap",
)
_classes = {code: "policy" for code in POLICY}
_classes.update({code: "system" for code in SYSTEM})
_classes["shipment_incomplete"] = "hold"
for source in ("order", "shipment", "policy"):
    for pattern in (
        "stale_{}",
        "future_{}",
        "missing_{}_freshness",
        "invalid_{}_freshness",
    ):
        _classes[pattern.format(source)] = "system"
for source in ("order", "shipment", "policy", "ledger"):
    for problem in (
        "version_unknown",
        "owner_missing",
        "budget_unapproved",
        "budget",
    ):
        _classes[f"contract_{problem}:{source}"] = "system"
for code in (
    "contract_vocabulary:shipment.status",
    "contract_invariant:order.paid_equals_items",
    "contract_invariant:order.prior_within_paid",
    "contract_invariant:shipment.time_order",
    "contract_invariant:shipment.delivery_after_shipping",
    "contract_invariant:shipment.not_after_observation",
    "contract_invariant:policy.version_equals_effective_date",
    "contract_invariant:ledger.positive_integer_cents",
    "contract_invariant:ledger.unique_idempotency_key",
):
    _classes[code] = "system"
for store in ("ledger", "audits", "proposals", "reviews", "approvals"):
    _classes[store + "_write_not_recorded"] = "system"
REASON_CLASS = MappingProxyType(_classes)


def reason_class(code: str, *, extra=None) -> str:
    classes = {**REASON_CLASS, **({} if extra is None else extra)}
    if any(
        value not in ("policy", "system", "hold") for value in classes.values()
    ):
        raise ValueError("unknown reason class")
    if extra and any(
        k in REASON_CLASS and REASON_CLASS[k] != v for k, v in extra.items()
    ):
        raise ValueError("contribution changes an existing reason class")
    if code in classes:
        return classes[code]
    parts = code.split(":")
    if parts[0] in ("model_error", "tool_error", "run_error", "run"):
        operations = {
            "order",
            "shipment",
            "policy",
            "assess",
            "propose",
            "approval",
            "execute",
            "audit",
            "done",
            "order.lookup",
            "shipment.status",
            "policy.retrieve",
            "refund.propose",
            "approval.request",
            "refund.execute",
            "audit.append",
        }
        if (
            len(parts) == 2 or len(parts) == 3 and parts[1] in operations
        ) and parts[-1].isidentifier():
            return "system"
    # These are the core's parameterized codes, with a closed suffix vocabulary.
    if code.startswith("proposal_changed:"):
        fields = code.split(":", 1)[1].split(",")
        if fields and set(fields) <= {
            "request",
            "order",
            "amount",
            "reason",
            "policy_version",
            "evidence",
        }:
            return "system"
    tool, separator, detail = code.partition(":")
    if separator and tool in (
        "order.lookup",
        "shipment.status",
        "policy.retrieve",
        "refund.propose",
        "approval.request",
        "refund.execute",
        "audit.append",
    ):
        return reason_class(detail, extra=extra)
    raise ValueError(f"unclassified reason: {code}")
