"""Clause support for structured drafts; this is not a truth or authority check."""

from refund_agent.contracts import Outcome
from refund_agent.memory.assemble import draft_from_facts


def support(draft, register, request_facts):
    """Release-suite slot: evidence in force at the decision and known at the draft."""
    clauses = []
    report = {
        "request_id": draft.request_id,
        "decision": draft.outcome.value,
        "citation": draft.policy_citation,
        "clauses": clauses,
        "amount_cents": draft.amount_cents,
        "decision_at": request_facts["at"].isoformat(),
        "known_at": request_facts["known_at"].isoformat(),
        "status": "not_checked",
        "reason": draft.reason,
    }
    if draft.outcome == Outcome.REFER:
        return report
    sections = (
        ["eligibility", "cap"]
        if draft.outcome == Outcome.APPROVE
        else (["cap"] if draft.reason == "amount_over_policy" else ["eligibility"])
    )
    active, refusal = _support_context(draft, register, request_facts)
    values, refusal = _read_clauses(register, active, sections, clauses, refusal)
    if refusal is None:
        refusal = _entailment(draft, request_facts, values)
    report.update(status="refused" if refusal else "supported", reason=refusal)
    return report


def _support_context(draft, register, request_facts):
    known_at = request_facts["known_at"]
    known = [r for r in register.records if r.recorded_at <= known_at]
    cited = [r for r in known if r.citation == draft.policy_citation]
    active = register.as_of(request_facts["at"], known_at)
    refusal = None
    if not cited:
        refusal = "citation_not_in_register"
    elif draft.policy_citation not in {r.citation for r in active}:
        refusal = "version_not_in_force"
    return active, refusal


def _read_clauses(register, active, sections, clauses, refusal):
    values = {}
    for section in sections:
        record = next((r for r in active if r.section == section), None)
        if record is None:
            refusal = refusal or "support_incomplete"
            continue
        text = register.read_text(record)
        clauses.append(
            {
                "section": section,
                "citation": record.citation,
                "record_id": record.record_id,
                "text_hash": record.text_hash,
                "text": text.text,
                "value": record.value,
            }
        )
        if text.refusal:
            refusal = refusal or "support_incomplete"
        values[section] = record
    return values, refusal


def _entailment(draft, request_facts, values):
    refusal = None
    if draft.policy_citation not in {r.citation for r in values.values()}:
        refusal = "clause_does_not_entail"
    else:
        facts = dict(request_facts)
        for section, record in values.items():
            facts["return_days" if section == "eligibility" else "cap_cents"] = record.value
            facts["policy_version"] = record.version
        expected = draft_from_facts(facts)
        if (draft.request_id, draft.amount_cents, draft.tier, draft.outcome, draft.reason) != (
            expected.request_id,
            expected.amount_cents,
            expected.tier,
            expected.outcome,
            expected.reason,
        ):
            refusal = "clause_does_not_entail"
    return refusal
