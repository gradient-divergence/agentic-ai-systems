"""Matched shadow decisions; D6 needs a final window, not a coverage score."""

from collections import Counter
from datetime import datetime, timedelta

from refund_agent import contracts as c
from refund_agent.baseline import baseline
from refund_agent.data.generate import number
from refund_agent.offpolicy.policies import Policy
from refund_agent.release.candidate import build_tools, run_request
from refund_agent.release.graders import shadow_effects
from refund_agent.shadow.offers import OfferSelection


def _candidate_offer(candidate):
    fields = candidate.offer_policy["policy"]
    policy = Policy(
        fields["version"],
        tuple((cell, tuple(tuple(row) for row in table)) for cell, table in fields["tables"]),
        fields["source_fingerprint"],
        fields["frozen_at"],
    )
    return OfferSelection(
        candidate.offer_policy["name"],
        policy,
        candidate.offer_policy["candidates"],
        tuple(candidate.offer_policy["limits"]),
        candidate.offer_policy["evaluation_fingerprint"],
        {},
    )


def _validate_window(rows):
    ids = [r.get("request_id") for r in rows]
    if any(type(rid) is not str or not rid for rid in ids) or len(set(ids)) != len(ids):
        raise ValueError("distinct request identities are required for the shadow window")
    for row in rows:
        if row.get("person_saw_shadow") is not False:
            raise ValueError("the comparison requires a blind person decision")
        requested = c._utc_time(datetime.fromisoformat(row["request_at"]))
        paid_at = _payment_time(row, requested)
        if (
            row["person_paid"]
            and row["found_false_30d"] is not None
            and row.get("label_final_at") is None
        ):
            raise ValueError("a resolved paid outcome requires a final label time")
        if row["person_paid"] and row.get("label_final_at") is not None:
            final = c._utc_time(datetime.fromisoformat(row["label_final_at"]))
            if final < paid_at + timedelta(days=30):
                raise ValueError("paid labels must mature at least 30 days after payment")
        if row["found_false_30d"] is not None and (
            type(row["found_false_30d"]) is not int or row["found_false_30d"] not in (0, 1)
        ):
            raise ValueError("a paid outcome is an integer zero or one, or unresolved")


def _payment_time(row, requested):
    if not row["person_paid"]:
        if row.get("payment_at") is not None:
            raise ValueError("an unpaid decision cannot have a payment time")
        return None
    if row.get("payment_at") is None:
        raise ValueError("a paid decision requires a payment time")
    paid_at = c._utc_time(datetime.fromisoformat(row["payment_at"]))
    if paid_at < requested:
        raise ValueError("payment cannot precede the request")
    return paid_at


def shadow_to_canary(rows, *, at):
    _validate_window(rows)
    counts = _window_counts(rows, c._utc_time(datetime.fromisoformat(at)))
    passes = (
        counts["window_complete"]
        and counts["forbidden_payments"] == 0
        and counts["referrals"] <= counts["baseline_referrals"]
        and counts["wrongful_denials"] <= counts["baseline_wrongful_denials"]
    )
    return {
        "requirement": "shadow-to-canary",
        "version": "D6-2026-10-12",
        "provisional": False,
        "required_requests": 299,
        **counts,
        "measures_pass": passes,
        "signature": None,
        "wrongful_denial_threshold": "no more than baseline on the same 299 requests; person paid and payment not found false within 30 days",
        "authority": "chapter 32 owns the 299-request window and canary decision",
    }


def _comparison_rows(stream, candidate, tools, traces, baselines, now):
    expected = {d.request_id: d for d in stream.data.correct}
    offers, rows = _candidate_offer(candidate), []
    payments = {p.request.request_id for p in tools.shadow_payments}
    for i, (request, trace, base) in enumerate(zip(stream.data.requests, traces, baselines)):
        label = expected[request.request_id]
        person = _person_outcome(request, label, stream, i, now)
        would_pay = request.request_id in payments
        disagrees = trace.decision.outcome != label.outcome
        row = {
            "request_id": request.request_id,
            "request_at": request.at.isoformat(),
            "tier": int(label.tier),
            "agent": trace.decision.outcome.value,
            "agent_reason": trace.decision.reason,
            "baseline": base.outcome.value,
            **person,
            "would_pay": would_pay,
            "forbidden_payment": would_pay and label.outcome != c.Outcome.APPROVE,
            "disagreement": disagrees,
            "adjudication": "pending" if disagrees else "generated reference agrees",
        }
        row.update(
            _verifier_offer_fields(
                tools, request, offers, stream, i, would_pay, person["person_paid"]
            )
        )
        rows.append(row)
    return rows


def compare(stream, candidate, *, at=None):
    if not stream.intact():
        raise ValueError("frozen live stream differs from its fingerprint")
    now = (
        datetime.fromisoformat(at)
        if at
        else datetime.fromisoformat(stream.manifest["start"])
        + timedelta(days=stream.manifest["days"])
    )
    data = stream.data.model_copy(update={"correct": []})
    tools = build_tools(data, candidate, "shadow")
    ledger, reviews = tools.ledger, tools.reviews
    traces = [run_request(candidate, tools, r) for r in data.requests]
    base = baseline(data)
    rows = _comparison_rows(stream, candidate, tools, traces, base, now)
    if not stream.intact() or not shadow_effects(tools, ledger, reviews):
        raise ValueError("shadow altered a frozen stream or external store")
    return _shadow_record(
        stream, candidate, rows, tools, _comparison_summary(rows, tools, ledger, reviews), now
    )


def _window_counts(rows, now):
    tier1 = [r for r in rows if r["tier"] == 1]
    paid = [r for r in tier1 if r["person_paid"]]

    def mature(row):
        return (
            row["found_false_30d"] is not None
            and row["label_final_at"] is not None
            and datetime.fromisoformat(row["label_final_at"]) <= now
            and datetime.fromisoformat(row["payment_at"]) + timedelta(days=30) <= now
        )

    def wrongful(arm):
        return sum(r[arm] == "deny" and r["found_false_30d"] == 0 and mature(r) for r in paid)

    return {
        "requests": len(tier1),
        "window_complete": len(tier1) == 299 and all(mature(r) for r in paid),
        "forbidden_payments": sum(r["forbidden_payment"] for r in tier1),
        "referrals": sum(r["agent"] == "refer" for r in tier1),
        "baseline_referrals": sum(r["baseline"] == "refer" for r in tier1),
        "wrongful_denials": wrongful("agent"),
        "baseline_wrongful_denials": wrongful("baseline"),
        "wrongful_denials_pending": sum(not mature(r) for r in paid),
    }


def _person_outcome(request, label, stream, index, now):
    paid = label.outcome == c.Outcome.APPROVE
    final_at = request.at + timedelta(days=30) if paid else None
    found_false = None
    if final_at and final_at <= now:
        found_false = int(number(stream.manifest["seed"], index, "live-paid-outcome", 50) == 0)
    return {
        "person": label.outcome.value,
        "person_saw_shadow": False,
        "person_paid": paid,
        "payment_at": request.at.isoformat() if paid else None,
        "label_final_at": final_at.isoformat() if final_at else None,
        "found_false_30d": found_false,
    }


def _verifier_offer_fields(tools, request, offers, stream, index, would_pay, person_paid):
    log = getattr(getattr(tools, "verdict_log", None), "entries", ())
    verdicts = [v.model_dump(mode="json") for v in log if v.request_id == request.request_id]
    objection = any(v["answer"] == "object" for v in verdicts)
    return {
        "verifier": verdicts,
        "verifier_catch": objection and not person_paid,
        "verifier_false_objection": objection and person_paid,
        "offer": offers.notice(request.reason, seed=stream.manifest["seed"], index=index)
        if would_pay
        else None,
        "fixed_offer": "standard" if would_pay else None,
        "notice_sent": False,
        "offer_outcome": None,
    }


def _comparison_summary(rows, tools, ledger, reviews):
    return {
        "requests": len(rows),
        "agreement": sum(not r["disagreement"] for r in rows),
        "baseline_agreement": sum(r["baseline"] == r["person"] for r in rows),
        "disagreements": dict(
            Counter(f"{r['agent']} versus {r['person']}" for r in rows if r["disagreement"])
        ),
        "adjudication_pending": sum(r["disagreement"] for r in rows),
        "payments_recorded": len(tools.shadow_payments),
        "reviews_recorded": len(tools.shadow_reviews),
        "ledger_unchanged": tools.ledger == ledger,
        "review_queue_unchanged": tools.reviews == reviews,
        "gate_calls": tools.gate_calls,
        "verifier_catches": sum(r["verifier_catch"] for r in rows),
        "verifier_false_objections": sum(r["verifier_false_objection"] for r in rows),
        "notices_sent": 0,
        "offer_outcomes_observed": 0,
    }


def _shadow_record(stream, candidate, rows, tools, summary, now):
    record = {
        "version": "refund-shadow-record-2",
        "candidate_fingerprint": candidate.fingerprint,
        "suite_fingerprint": candidate.suite["fingerprint"],
        "stream_fingerprint": stream.fingerprint,
        "manifest": stream.manifest,
        "arms": [
            "candidate agent in shadow",
            "blind generated person reference",
            "rules-only baseline on identical requests",
        ],
        "rows": rows,
        "offer_policy": candidate.offer_policy["name"],
        "shadow_log": tools.shadow_log,
    }
    record.update(
        summary,
        shadow_to_canary=shadow_to_canary(rows, at=now.isoformat()),
        accesses=[
            {
                "purpose": "read generated blind reference after both decision arms; no future outcomes read",
                "stream_fingerprint": stream.fingerprint,
            }
        ],
        limits=[
            "generated labels stand in for people; agreement measures the label policy",
            "outcome of an unexecuted decision is unobserved",
            "first day is not the final 299-request canary window",
        ],
    )
    return record
