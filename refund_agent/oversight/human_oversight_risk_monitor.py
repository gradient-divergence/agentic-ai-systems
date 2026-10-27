"""Planted attention checks and blind agreement; no inference about real people."""

from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from fractions import Fraction
from hashlib import sha256
from math import ceil, comb, sqrt
from statistics import median

from refund_agent import contracts as c
from refund_agent.rules import draft, tier_for


def binom_cdf(k: int, n: int, p=Fraction(9, 10)) -> Fraction:
    if type(n) is not int or type(k) is not int or n < 0 or not 0 <= k <= n:
        raise ValueError("invalid binomial counts")
    p = Fraction(str(p))
    if not 0 <= p <= 1:
        raise ValueError("probability outside [0, 1]")
    return sum((Fraction(comb(n, j)) * p ** j * (1 - p) ** (n - j) for j in range(k + 1)), Fraction())


def alarm(caught, decided, target=Fraction(9, 10), alpha=Fraction(1, 20)):
    return decided > 0 and binom_cdf(caught, decided, target) < alpha


def alarm_power(n, actual=Fraction(1, 2)):
    rejected = [k for k in range(n + 1) if alarm(k, n)]
    return binom_cdf(max(rejected), n, actual) if rejected else Fraction()


def wilson_interval(k, n, z=1.96):
    if type(k) is not int or type(n) is not int or n <= 0 or not 0 <= k <= n:
        raise ValueError("agreement needs valid nonempty counts")
    rate, t = k / n, z * z / n
    center = (rate + t / 2) / (1 + t)
    half = z * sqrt(rate * (1 - rate) / n + t / (4 * n)) / (1 + t)
    return max(0.0, center - half), min(1.0, center + half)


def zero_failure_bound(n, chance=.05):
    if type(n) is not int or n < 1 or not 0 < chance < 1:
        raise ValueError("a clean-run bound needs nonempty independent cases")
    return 1 - chance ** (1 / n)


def load_band(queue_length):
    if type(queue_length) is not int or queue_length < 0:
        raise ValueError("queue length must be a nonnegative integer")
    return "quiet" if queue_length <= 5 else "busy" if queue_length <= 15 else "overloaded"


@dataclass(frozen=True)
class ReviewObservation:
    review_id: str
    at: datetime
    reviewer: str | None
    queue_length: int | None
    planted: bool
    action: str
    closed_at: datetime | None
    arrived_at: datetime | None = None


@dataclass(frozen=True)
class PlantedPacket:
    review_id: str
    corruption: str
    proposal: c.Proposal
    proposed: c.Decision


def planted_packet(proposal, review_id, corruption):
    """The three section-13.9 mistakes, in records kept apart from customers."""
    proposal = c._validated(c.Proposal, proposal)
    request, evidence = proposal.request, proposal.evidence
    if corruption == "shipment_contradiction":
        request = request.model_copy(update={"reason": "not_delivered"})
        evidence = evidence.model_copy(update={"shipment_events": [
            c.ShipmentEvent(at=proposal.created_at - timedelta(days=1), status="delivered")]})
        outcome, reason, citation = c.Outcome.APPROVE, "eligible", None
    elif corruption == "balance_plus_cent":
        balance = evidence.order.paid_cents - sum(r.amount_cents for r in evidence.order.prior_refunds)
        request = request.model_copy(update={"amount_cents": balance + 1})
        outcome, reason, citation = c.Outcome.APPROVE, "eligible", None
    elif corruption == "false_window_denial":
        request = request.model_copy(update={"reason": "damaged"})
        evidence = evidence.model_copy(update={"shipment_events": [
            c.ShipmentEvent(at=proposal.created_at - timedelta(days=1), status="delivered")]})
        outcome, reason = c.Outcome.DENY, "outside_return_window"
        citation = "refund_cap_v4@" + evidence.policy.version + "#eligibility"
    else:
        raise ValueError("unknown planted review-packet corruption")
    request = c._validated(c.Request, request)
    changed = c._validated(c.Proposal, proposal.model_copy(update={
        "proposal_id": review_id + "-proposal", "request_id": request.request_id,
        "request": request, "amount_cents": request.amount_cents,
        "reason": request.reason, "evidence": evidence,
    }))
    proposed = c.Decision(request_id=request.request_id, amount_cents=request.amount_cents,
                          tier=tier_for(request.amount_cents), outcome=outcome,
                          reason=reason, policy_citation=citation)
    state = c.State(request=request, trace_id=evidence.trace_id, order=evidence.order, policy=evidence.policy,
                    shipment_events=evidence.shipment_events, execution_at=proposal.created_at,
                    refunded_cents=sum(r.amount_cents for r in evidence.order.prior_refunds))
    if draft(state).outcome == proposed.outcome:
        raise ValueError("planted packet must contradict its policy reading")
    return PlantedPacket(review_id, corruption, changed, proposed)


def observed_by(rows, at):
    """Keep arrivals still waiting at the look; their eventual action is unknown."""
    visible = []
    for row in rows:
        if (row.arrived_at or row.at) > at:
            continue
        if row.at > at or row.queue_length is None:
            row = replace(row, reviewer=None, queue_length=None, action="open", closed_at=None)
        elif row.closed_at is None or row.closed_at > at:
            row = replace(row, action="open", closed_at=None)
        visible.append(row)
    return tuple(visible)


def load_report(rows):
    result = {}
    bands = ("quiet", "busy", "overloaded") + (("unopened",) if any(r.queue_length is None for r in rows) else ())
    for band in bands:
        group = [r for r in rows if (load_band(r.queue_length) if r.queue_length is not None else "unopened") == band]
        planted = [r for r in group if r.planted]
        decided = [r for r in planted if r.action in ("approved", "refused", "modified") and r.closed_at is not None]
        caught = sum(r.action in ("refused", "modified") for r in decided)
        all_decided = [r for r in group if r.action in ("approved", "refused", "modified") and r.closed_at is not None]
        changed = sum(r.action in ("refused", "modified") for r in all_decided)
        result[band] = {"planted": len(planted), "decided": len(decided), "caught": caught,
                        "unresolved": len(planted) - len(decided),
                        "catch_share": caught / len(decided) if decided else None,
                        "changed": changed, "all_decided": len(all_decided),
                        "change_or_refuse_share": changed / len(all_decided) if all_decided else None}
    return result


def trailing_alarm(rows, at, *, target=Fraction(9, 10)):
    """Busy means 6–15 waiting; undecided cases are pending, never catches."""
    selected = [r for r in rows if r.planted and r.queue_length is not None and 6 <= r.queue_length <= 15
                and r.closed_at is not None and at - timedelta(weeks=4) < r.closed_at <= at
                and r.action in ("approved", "refused", "modified")]
    caught = sum(r.action in ("refused", "modified") for r in selected)
    return {"caught": caught, "decided": len(selected), "alarm": alarm(caught, len(selected), target),
            "p": float(binom_cdf(caught, len(selected), target)) if selected else None,
            "power_at_one_half": float(alarm_power(len(selected)))}


def first_alarm(rows, start, days=30):
    for week in range(1, ceil(days / 7) + 1):
        at = min(start + timedelta(weeks=week), start + timedelta(days=days))
        state = trailing_alarm(rows, at)
        if state["alarm"]:
            return week, state["decided"]
    return None


def alarm_timing_summary(weeks):
    """Nearest-rank percentiles; no alarm is right-censored after the last look."""
    if not weeks or any(w is not None and (type(w) is not int or w < 1) for w in weeks):
        raise ValueError("first-alarm weeks require nonempty positive counts")
    ordered = sorted(float("inf") if w is None else w for w in weeks)

    def finite(value):
        return None if value == float("inf") else value

    return {"replications": len(weeks), "median": finite(median(ordered)),
            "p10": finite(ordered[ceil(.1 * len(ordered)) - 1]),
            "p90": finite(ordered[ceil(.9 * len(ordered)) - 1]),
            "alarmed_by_week2": sum(w is not None and w <= 2 for w in weeks) / len(weeks),
            "undetected": sum(w is None for w in weeks)}


def blind_sample(items, share=.1, seed=7):
    """Sample IDs before review; the caller supplies evidence only to the second person."""
    if not 0 <= share <= 1:
        raise ValueError("invalid sample share")
    threshold = int(share * 2 ** 256)
    return tuple(item for item in items if int.from_bytes(
        sha256(f"human-oversight-risk-blind:{seed}:{item}".encode()).digest(), "big") < threshold)


def blind_packet(proposal, *, first_reviewer, second_reviewer):
    if not second_reviewer or second_reviewer == first_reviewer:
        raise ValueError("blind reviewer must be independent")
    # No proposed decision or first action is passed to the second reviewer.
    return {"request": proposal.request.model_dump(mode="json"),
            "evidence": proposal.evidence.model_dump(mode="json"), "reviewer": second_reviewer}


def agreement(first, adjudicated):
    if set(first) != set(adjudicated) or not first:
        raise ValueError("blind disagreements need completed adjudication for every sampled ID")
    matched = sum(first[key] == adjudicated[key] for key in first)
    return {"matched": matched, "count": len(first), "agreement": matched / len(first),
            "wilson": wilson_interval(matched, len(first))}
