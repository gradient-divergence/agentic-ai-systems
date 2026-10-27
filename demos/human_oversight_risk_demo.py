"""Chapter 37's constructed queue evidence; partner-dependent fields stay pending.

Run with the repository's capture_outputs on this explicit file plus the
human_oversight_risk_queueing and human_oversight_risk_deadline modules. Those
two modules own their printed listing/output IDs; no ID is duplicated here.
"""

import csv
import json
from collections import Counter
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from fractions import Fraction
from functools import lru_cache
from math import ceil, log
from pathlib import Path
from random import Random
from textwrap import fill

from refund_agent import contracts as c
from refund_agent.data.generate import Dataset
from refund_agent.oversight.human_oversight_risk_demand import (
    Stage6Demand,
    generate_stage6,
    generator_mix_counterfactual,
)
from refund_agent.oversight.human_oversight_risk_design import (
    accept_months,
    check_design,
    default_design,
    poisson_tail,
)
from refund_agent.oversight.human_oversight_risk_monitor import (
    ReviewObservation,
    agreement,
    alarm_power,
    alarm_timing_summary,
    binom_cdf,
    blind_sample,
    first_alarm,
    load_report,
    observed_by,
    planted_packet,
    trailing_alarm,
    zero_failure_bound,
)
from refund_agent.oversight.human_oversight_risk_queueing import mm1, recovery
from refund_agent.oversight.human_oversight_risk_screen import screen
from refund_agent.oversight.human_oversight_risk_simulate import (
    desk_hours_between,
    first_empty_after,
    inattentive_catch,
    simulate,
    simulate_one,
    staffing,
)
from refund_agent.release.candidate import build_candidate, build_tools, run_request
from refund_agent.release.suite import build_suite
from refund_agent.rules import draft, tier_for

ROOT = Path(__file__).resolve().parents[1]
START = datetime(2026, 10, 1, tzinfo=timezone.utc)
LOG = ROOT / "exercises/human-oversight-risk/log-05.csv"
ALARM_SEEDS = range(7, 1007)


def say(*values):
    print(fill(" ".join(map(str, values)), 79, break_long_words=False, break_on_hyphens=False))


@lru_cache(maxsize=1)
def available_base_measurement():
    """Measure the available candidate, not an unmerged stage-6 candidate.

    Each independent order runs through build_tools and run_request. No label is
    read. The absent partner components are never replaced by a local stand-in.
    """
    data = generate_stage6(7, START, 30, Stage6Demand())
    candidate = build_candidate(build_suite())
    outcomes, payments, second_inputs = Counter(), [], []
    orders = {o.order_id: o for o in data.orders}
    shipments = {s.order_id: s for s in data.shipments}
    retrievals = {r.request_id: r for r in data.retrievals}
    for request in data.requests:
        if tier_for(request.amount_cents) == c.Tier.NAMED_APPROVER:
            continue
        order = orders[request.order_reference]
        one = Dataset(orders=[order], requests=[request], shipments=[shipments[order.order_id]],
                      policies=data.policies, retrievals=[retrievals[request.request_id]], correct=[],
                      reviewers=data.reviewers)
        tools = build_tools(one, candidate, "execute")
        trace = run_request(candidate, tools, request)
        if tier_for(request.amount_cents) == c.Tier.AGENT:
            outcomes[trace.decision.outcome.value] += 1
            if trace.receipt is not None:
                payments.append(trace.receipt)
        else:
            state = trace.steps[-1].state
            second_inputs.append((request.request_id, draft(state).outcome.value))
    n = sum(outcomes.values())
    demand = replace(Stage6Demand(), referral_share=outcomes["refer"] / n,
                     payment_share=outcomes["approve"] / n)
    return {"data": data, "candidate": candidate, "outcomes": outcomes,
            "n": n, "payments": tuple(payments), "second_inputs": tuple(second_inputs), "demand": demand}


@lru_cache(maxsize=1)
def queue_experiment():
    demand = available_base_measurement()["demand"]
    groups = {}
    for model in ("fixed", "exponential"):
        for name in ("D1", "D2", "D3", "D4", "D4-rework"):
            people = staffing("D4" if name == "D4-rework" else name)
            active_demand = replace(demand, modified_share=.1, refused_share=.05,
                                    blind_disagreement_share=.1) if name == "D4-rework" else demand
            design = default_design("D4" if name == "D4-rework" else name, START - timedelta(days=1))
            rows, first, mondays, monday_bad, worst, count_failures = [], None, 0, 0, 0, 0
            late_classes = Counter()
            for seed in range(7, 107):
                result = simulate(seed, START, demand=active_demand, people=people, service_model=model)
                if first is None:
                    first = result
                rows.append({"seed": seed, "blocking": result.blocking, "late": result.late_blocking,
                             "pending": result.pending_blocking})
                mondays += result.monday_count
                monday_bad += result.monday_breaches
                worst = max(worst, result.max_backlog)
                count_failures += sum(bool(r["failures"]) for r in check_design(design, active_demand, weekends=result.weekends)["observed_weekends"])
                late_classes.update({key: row["late"] for key, row in result.classes.items()})
            groups[(name, model)] = {"months": rows, "first": first, "mondays": mondays,
                                    "monday_bad": monday_bad, "max_backlog": worst,
                                    "late_classes": late_classes, "acceptance": accept_months(rows),
                                    "count_failures": count_failures}
    return groups


def print_arrivals():
    measured = available_base_measurement()
    say("Constructed seed 7, 30 days; Poisson arrivals around the clock.")
    say("Measured candidate: available as-built stage 4; stage-6 join PENDING.")
    say("Candidate fingerprint:", measured["candidate"].fingerprint)
    say("Selected partner components: none; this is a demand sensitivity run.")
    say("No calibration, security, explanation or canary is asserted here.")
    say("Generated customer requests by tier:", dict(Counter(int(tier_for(r.amount_cents)) for r in measured["data"].requests)))
    say("Tier 1 actual agent runs:", measured["n"], "outcomes:", dict(measured["outcomes"]))
    d = measured["demand"]
    say(f"Measured referral share={d.referral_share:.6%}; payment share={d.payment_share:.6%}")
    say("Brief's generated-label expectations: referral 14.3%; payment 36.6%.")
    for row in d.classes():
        if row.rate_per_day:
            say(f"{row.name}: {row.rate_per_day * 7:.3f}/week; {row.minutes:g}min;")
            say(f"  offered={row.rate_per_day * 7 * row.minutes / 60:.6f}h/week")
    checked = check_design(default_design("D3", START - timedelta(days=1)), d)
    say(f"Reviewers' queue offered={checked['reviewer_hours']:.6f}h/week")
    say(f"Tier 1 payments/day={d.tier1_monthly / d.month_days * d.payment_share:.6f}")
    say(f"Recheck EVERY tier 1 payment={d.tier1_monthly / d.month_days * d.payment_share * 7 / 12:.6f}h/week")
    say("Generator cycle counterfactual tier3/day:", generator_mix_counterfactual())
    say("Flags: 999 at 5min =", f"{999 * 5 / 60:.2f} reviewer-hours")


def audited_held_denials():
    """Count the canonical run's denials as chapter 36's audit-01 does."""
    record = json.loads((ROOT / "demos/data/audits_explanations_run.json").read_text())
    held = {(row["segment"], reason)
            for row in record["audit"]["segments"] if row["breach"]
            for reason in row["held_reasons"]}
    return sum(row["outcome"] == "deny" and (row["segment"], row["reason"]) in held
               for row in record["run"]["observations"])


def print_queue():
    demand = available_base_measurement()["demand"]
    say("Constructed queues; arrival and payment shares from available-base runs.")
    say("All 100 seeds 7..106 repeat the same 30-day calendar from 1 October.")
    say("Staffing comparisons keep sample rates fixed; the controller is tested separately.")
    say("The signed ceiling is checked separately on EVERY month, per model.")
    reference = check_design(default_design("D3", START - timedelta(days=1)), Stage6Demand())
    say("Brief-label arithmetic (not measured):")
    say(f"  reviewers={reference['reviewer_hours']:.6f}h; total={reference['total_hours']:.6f}h/week")
    say(f"  Monday={reference['monday_hours']:.6f}h; D2 margin={reference['monday_hours'] * -1 + 16:.6f}h")
    say(f"  D3 margin={reference['monday_margin']:.6f}h; tier3={reference['tier3_monday_hours']:.6f}h")
    say("Fixed-time tier3 count: >=17 in 55h; Poisson tail=", f"{poisson_tail(17, 4 * 55 / 24):.6f}")
    say("  one in", f"{1 / poisson_tail(17, 4 * 55 / 24):.3f}", "weekends; not a simulated breach frequency.")
    for (name, model), group in queue_experiment().items():
        design = default_design("D4" if name == "D4-rework" else name, START - timedelta(days=1))
        active_demand = replace(demand, modified_share=.1, refused_share=.05,
                                blind_disagreement_share=.1) if name == "D4-rework" else demand
        expected = check_design(design, active_demand, weekends=group["first"].weekends)
        accept = group["acceptance"]
        say(name, model, "capacity", expected["capacity_hours"], "h/week")
        say(f"  expected u={expected['utilization']:.6f}; reviewer-only u={expected['reviewer_only_utilization']:.6f}")
        say(f"  Monday due={expected['monday_hours']:.6f}h; margin={expected['monday_margin']:.6f}h")
        say("  necessary-count failures:", expected["failures"] or "none (not sufficient)")
        if name.startswith("D4"):
            say("  Revised staffing: morgan also Saturday 09:00..17:00 UTC;")
            say("  same named approver, 48h/week; decision-owner tasks explicit.")
        say(f"  Monday breaches={group['monday_bad']}/{group['mondays']} ({group['monday_bad'] / group['mondays']:.4%})")
        say(f"  pooled late={accept['pooled_share']:.6%}; breach months={accept['breach_months']}/100")
        say(f"  monthly breach frequency={accept['breach_frequency']:.4%}; accepted={accept['accepted']}")
        say("  late items by class:", dict(group["late_classes"]))
        say("  necessary-count failures across every simulated Monday:", group["count_failures"], "of", group["mondays"])
        say("  all simulated weekends' count failures:", sum(bool(r["failures"]) for r in expected["observed_weekends"]),
            "of", len(expected["observed_weekends"]), "in the first month")
        say("  seed late/blocking share pending")
        for row in group["months"]:
            print(f"  {row['seed']:3} {row['late']:4}/{row['blocking']:<4} {row['late'] / row['blocking']:10.6%} {row['pending']:3}")
        first = group["first"]
        say("  first month utilization:", {p: round(r["utilization"], 6) for p, r in first.people.items()})
        say("  first month max backlog:", first.max_backlog, "at", first.max_backlog_at.isoformat())
        for cls, row in first.classes.items():
            if not row["arrivals"]:
                continue
            say(f"  {cls}: arrivals={row['arrivals']} completed-at-close={row['completed_by_close']} late={row['late']}")
            say("    deadline breaches (blocking and nonblocking classes):", row["deadline_breaches"])
            say("    class largest waiting backlog:", row["max_backlog"], "at", row["max_backlog_at"].isoformat())
            say("    calendar mean/p95/max h:", row["mean_calendar_hours"], row["p95_calendar_hours"], row["max_calendar_hours"])
            say("    desk mean/p95 h:", row["mean_desk_hours"], row["p95_desk_hours"])
            say("    independent blind reviewer: alex excluded; unresolved samples:",
                row["arrivals"] - row["completed_after_drain"] if cls == "blind" else "n/a")
            row = first.conservation[cls]
            window_hours = (first.end - first.start).total_seconds() / 3600
            say(f"    L_window={row['mean_present']:.6f}; members/hour={row['members'] / window_hours:.6f}")
            say(f"    mean clipped stay={row['mean_clipped_stay']:.6f}h; unfinished durations censored at close")
            say(f"    occupancy={row['occupancy_hours']:.6f} item-h; clipped stays={row['clipped_stay_hours']:.6f} item-h")
            say(f"    finite-window relative gap={row['relative_gap']:.3g}; starting backlog={row['starting_backlog']}")
        say("  Boundary:", *first.notes)
    for label, changed in (("held-denial breach month", replace(demand, held_denials_monthly=audited_held_denials())),
                           ("tier1 reversal", replace(demand, reversal=True)),
                           ("rework stress (constructed 10% modified, 5% refused)",
                            replace(demand, modified_share=.1, refused_share=.05, blind_disagreement_share=.1))):
        checked = check_design(default_design("D3", START - timedelta(days=1)), changed)
        say(label)
        say(f"  week={checked['total_hours']:.6f}/88h u={checked['utilization']:.6f}")
        say(f"  Monday due={checked['monday_hours']:.6f}/24h; failures={checked['failures']}")
        for cls in ("held_denial", "modification", "refusal_followup", "adjudication"):
            say(f"  {cls} offered hours/week={checked['weekly_hours'][cls]:.6f}")
    pooled = accept_months([{"late": 0, "blocking": 100}] * 99 + [{"late": 10, "blocking": 100}])
    say("Regression: one 10% month plus 99 clean months:", pooled)
    say("Stage-6 staffing admission remains pending the assembled partner run.")
    say("A revised design changes staffing before any stage-6 state line prints.")


def print_absence():
    demand = available_base_measurement()["demand"]
    away_start = datetime(2026, 10, 5, tzinfo=timezone.utc)
    returned = datetime(2026, 10, 12, 9, tzinfo=timezone.utc)
    away = frozenset((away_start + timedelta(days=i)).date() for i in range(5))
    say("Constructed morgan away five working days; no deputy substituted.")
    say("Expected backlog=28; fluid recovery full-time=", f"{recovery(28, 16, 5.6):.6f} working days")
    say("Expected four-hours/day recovery=", f"{recovery(28, 8, 5.6):.6f} working days")
    checked = check_design(default_design("D3", START - timedelta(days=1)), demand)
    say(f"Reviewers alone during absence: {checked['reviewer_hours'] / 48:.6f} of 48h/week")
    for hours in (8, 4):
        people = tuple(replace(p, absent=away, daily_hours=hours) if p.name == "morgan" else p
                       for p in staffing("D3"))
        result = simulate(7, START, 45, demand=demand, people=people)
        held = [j for j in result.items if j.work.tier == 3 and away_start <= j.arrival.at < returned
                and (j.closed_at is None or j.closed_at > returned)]
        closed = [j for j in held if j.closed_at]
        recovered = max((j.closed_at for j in held if j.closed_at is not None), default=returned)
        say(f"morgan {hours}h/day: held cohort={len(held)}; original-deadline late={sum(j.closed_at is None or j.closed_at > j.due_at for j in held)}")
        say("  escalation events=", sum(len(j.ticket.events) for j in held if j.ticket))
        say("  recovery of absence cohort after return:", f"{desk_hours_between(returned, recovered) / 8:.6f} working days")
        empty_at = first_empty_after(result.items, "tier3", returned)
        say("  net backlog first empties, including new arrivals:",
            f"{desk_hours_between(returned, empty_at) / 8:.6f} working days" if empty_at else "not recovered in the run")
        say("  cohort pending after drain:", len(held) - len(closed))
        say("  tier3 peak waiting backlog:", result.classes["tier3"]["max_backlog"])
    say("Deputy during absence: policy version changes -> new proposal and review.")
    say("Test-constructed deputy payment and denial records: review-records-01.")
    say("Assembled deputy/credential trace pending chapters 33, 35 and 36.")


@lru_cache(maxsize=1)
def planted_prototype():
    data = available_base_measurement()["data"]
    request = next(r for r in data.requests if tier_for(r.amount_cents) == c.Tier.REVIEWER)
    request = request.model_copy(update={"amount_cents": 12000, "reason": "damaged"})
    order = next(o for o in data.orders if o.order_id == request.order_reference)
    order = order.model_copy(update={"paid_cents": 30000, "prior_refunds": []})
    return c.Proposal(
        proposal_id="human-oversight-risk-planted-prototype", request_id=request.request_id,
        request=request, amount_cents=request.amount_cents, reason=request.reason, created_at=request.at,
        evidence=c.Evidence(order=order, policy=data.policy_at(request.at),
                            shipment_events=[c.ShipmentEvent(at=request.at - timedelta(days=1), status="delivered")],
                            freshness={"order": 60, "shipment": 120, "policy": 3600},
                            risk_score=.1, trace_id="trace-" + request.request_id))


def planted_records(result):
    prototype, records = planted_prototype(), {}
    for index, job in enumerate(j for j in result.items if j.work.name == "planted"):
        # Six payment mistakes and four denial mistakes per ten records.
        corruption = ("shipment_contradiction" if index % 10 < 3 else
                      "balance_plus_cent" if index % 10 < 6 else "false_window_denial")
        request = prototype.request.model_copy(update={"request_id": job.arrival.review_id, "at": job.arrival.at})
        proposal = prototype.model_copy(update={"request": request, "request_id": request.request_id,
                                                "created_at": request.at})
        records[job.arrival.review_id] = planted_packet(proposal, job.arrival.review_id, corruption)
    return records


def attention_observations(result, records, seed):
    rng, rows = Random(seed), []
    for job in sorted(result.items, key=lambda j: j.opened_at or result.end):
        if job.work.name not in ("tier2", "planted") and job.arrival.review_id not in records:
            continue
        packet = records.get(job.arrival.review_id)
        if packet is not None and packet.review_id != job.arrival.review_id:
            raise ValueError("planted packet does not belong to this review")
        planted = packet is not None
        if job.opened_at is None:
            rows.append(ReviewObservation(job.arrival.review_id, job.arrival.at, None, None,
                                          planted, "open", None, job.arrival.at))
            continue
        probability = inattentive_catch(job.queue_length) if planted else .1
        action = ("refused" if rng.random() < probability else "approved") if job.closed_at else "open"
        rows.append(ReviewObservation(job.arrival.review_id, job.opened_at, job.reviewer,
                                      job.queue_length, planted, action, job.closed_at, job.arrival.at))
    return tuple(rows)


@lru_cache(maxsize=1)
def attention_rows():
    result = queue_experiment()[("D3", "exponential")]["first"]
    return attention_observations(result, planted_records(result), 7)


@lru_cache(maxsize=1)
def alarm_replications():
    demand = available_base_measurement()["demand"]
    weeks = []
    for seed in ALARM_SEEDS:
        result = simulate(seed, START, demand=demand, people=staffing("D3"), service_model="exponential")
        rows = attention_observations(result, planted_records(result), seed)
        found = first_alarm(rows, START)
        weeks.append(found[0] if found else None)
    return alarm_timing_summary(weeks)


def print_catches():
    say("PLANTED inattentive reviewer; constructed behavior, not human evidence.")
    say("Catch chances: waiting <=5: 0.9; 6..15: 0.7; >15: 0.4.")
    say("The alarm uses only busy-band planted cases, with 6..15 waiting.")
    say("Fixed-look tails: 6/10 at 0.9 =", f"{float(binom_cdf(6, 10)):.9f}")
    say("                 7/10 at 0.9 =", f"{float(binom_cdf(7, 10)):.9f}")
    say("Power at a true catch chance 0.5 after ten decided busy cases:", f"{float(alarm_power(10)):.6f}")
    seeded_alarm = first_alarm(attention_rows(), START)
    for week in range(1, 6):
        at = min(START + timedelta(weeks=week), START + timedelta(days=30))
        rows = observed_by(attention_rows(), at)
        state = trailing_alarm(rows, at)
        say("week", week, "trailing-four-week alarm:", state)
        for band, row in load_report(rows).items():
            say(" ", band, "catch", f"{row['caught']}/{row['decided']}", "unresolved", row["unresolved"],
                "change/refuse", f"{row['changed']}/{row['all_decided']}")
    say("Seed 7 first alarm (week, decided busy planted cases):", seeded_alarm or "undetected")
    packets = planted_records(queue_experiment()[("D3", "exponential")]["first"])
    say("Planted packet records:", dict(Counter(p.corruption for p in packets.values())))
    summary = alarm_replications()
    say("First-alarm study: D3, exponential service, same 30-day design;")
    say("seeds 7..1006, weekly looks, trailing four weeks, unchanged chances.")
    say("Replications:", summary["replications"])
    say("First-alarm week median / 10th / 90th percentile:",
        summary["median"], "/", summary["p10"], "/", summary["p90"])
    say(f"Share alarming by week 2: {summary['alarmed_by_week2']:.3%}")
    say("Undetected by day 30:", f"{summary['undetected']}/{summary['replications']}",
        f"({summary['undetected'] / summary['replications']:.3%})")
    say("Percentiles use nearest ranks; undetected runs are right-censored.")
    say("None means that percentile was not reached by day 30.")
    say("Small busy-band counts can take weeks to signal a fall; many runs")
    say("remain undetected after a month in this constructed design.")
    if seeded_alarm is None:
        say("Detection limitation: the planted fall was not detected in this run.")
    say("Power assumes independent cases with one catch probability; weekly")
    say("looks repeat the test, so 5% is not a lifetime false-alarm guarantee.")


def print_samples():
    measured = available_base_measurement()
    payments = measured["payments"]
    say("STAGE-4 SENSITIVITY RUN: available-base payments, seed 7, 30 days.")
    say("ASSEMBLED STAGE-6 NUMBERS PENDING: rerun after partner integration.")
    say("Stage-6 brief expectations: about 731 payments; 183/73 rechecks;")
    say("about 48 blind second reviews/month. Keep these separate from stage 4.")
    say("Month tier1 payments:", len(payments))
    ids = tuple(p.receipt_id for p in payments)
    for share in (.25, .10):
        sampled = blind_sample(ids, share)
        violations = 0
        by_id = {p.receipt_id: p for p in payments}
        for key in sampled:
            p = by_id[key]
            state = c.State(request=p.request, trace_id=p.evidence.trace_id, order=p.evidence.order,
                            policy=p.evidence.policy, shipment_events=p.evidence.shipment_events,
                            execution_at=p.at, refunded_cents=sum(r.amount_cents for r in p.evidence.order.prior_refunds))
            violations += draft(state).outcome != c.Outcome.APPROVE
        say(f"Recheck at {share:.0%}: {len(sampled)} sampled; policy-ineligible findings={violations}")
        if sampled and violations == 0:
            say(f"  independent clean-case 95% upper bound={zero_failure_bound(len(sampled)):.6%}")
    say("Brief's hand-count bounds: n=183 ->", f"{zero_failure_bound(183):.6%}; n=73 -> {zero_failure_bound(73):.6%}")
    selected = blind_sample(tuple(key for key, _ in measured["second_inputs"]))
    truth = dict(measured["second_inputs"])
    rng = Random(7)
    first, adjudicated, disagreements = {}, {}, 0
    for key in selected:
        unaided = truth[key]  # The independently read policy result, not the proposal.
        first[key] = "approve" if rng.random() < .1 else unaided
        adjudicated[key] = unaided
        disagreements += first[key] != unaided
    say("Constructed first/independent readings; no real reviewer behavior.")
    say("Blind second-review sample:", len(selected), "disagreements/adjudications:", disagreements)
    say("Agreement with adjudicated policy readings:", agreement(first, adjudicated))
    say("Follow-up work: adjudications", f"{disagreements * .5:.3f}h in the month")
    say("Authenticated, named adjudication records remain a chapter-35 integration point.")


def print_screen(ticket, proposal, proposed, explanation, credential):
    say("One tier2 denial ticket; TEST-CONSTRUCTED partner records.")
    say("Assembled screen PENDING chapter 35 authentication -> ReviewerContext")
    say("and chapter 36 explain/replay DenialExplanation -> ProposedExplanation.")
    for field in screen(ticket, proposal, proposed, explanation, credential):
        say(field)


def print_review_records(desk, result, at):
    say("Review action:", result.action, "request state:", result.request_state)
    say("Reason:", result.reason, "receipts:", len(desk.tools.payments))
    for ticket in desk.tickets.values():
        say("Ticket:", ticket.review_id, "assigned:", ticket.assignee)
        for event in ticket.events:
            say("  Event:", event.kind, "owner:", event.owner, "reason:", event.reason)
            say("  at:", event.at.isoformat(), "due:", event.due_at.isoformat() if event.due_at else None)
        for notice in ticket.notices:
            say("  Notice:", notice.requester_id, "owner:", notice.owner, "due:", notice.due_at.isoformat())
    rows = desk.waiting_expiry(at)
    say("Waiting eligibility expiry:", len(rows), "request(s); 48-hour horizon.")
    for row in rows:
        say(" ", row["request_id"], "eligible through:", row["eligible_through"].isoformat())
        say(f"  hours left={row['hours_left']:.6f}; expired={row['expired']}")


def exercise_log_rows():
    with LOG.open(newline="", encoding="utf-8") as source:
        records = list(csv.DictReader(source))
    rows = []
    for r in records:
        if r["class"] == "report":
            continue
        rows.append(ReviewObservation(r["review_id"], datetime.fromisoformat(r["opened_at"]), r["reviewer"],
                                      int(r["queue_length"]), r["planted"] == "yes", r["outcome"],
                                      datetime.fromisoformat(r["closed_at"]) if r["closed_at"] else None))
    return records, rows


def print_keys():
    say("Constructed exercise answers, checked from the given inputs.")
    say("1 Work accumulates: opening/due/close, backlog and escalations.")
    say("  Attention hazards: planted catches, change/refuse by load; bypass:")
    say("  unbound payments, deadline decisions and provider reconciliation.")
    say("2 Friday 18:20 -> Monday 17:00; then morgan, Tuesday 13:00.")
    say("  Notice; never a clock-written decision. Fresh gate can refuse expiry.")
    for lam in (10, 11):
        row = mm1(lam, 12)
        assert row.u is not None and row.L is not None and row.W is not None and row.W_wait is not None
        say(f"3 lambda={lam}: u={float(row.u):.6f}; L={float(row.L):.6f}; W={float(row.W) * 60:.6f}min")
        say(f"  wait={float(row.W_wait) * 60:.6f}min; P(T>30min)={row.tail(.5):.6f}")
    limit = 12 - 2 * log(10)
    say(f"  90% within 30min: lambda<={limit:.6f}/h; u<={limit / 12:.6f}")
    say("  Report the tail: a 30min mean still leaves 36.8% beyond 30min.")
    ordinary, senior = 30 * 7 * 20 / 60, 6 * 7 * 40 / 60
    adjusters = ceil(ordinary / (40 * .8))
    say(f"4 Offered ordinary/senior={ordinary:.1f}/{senior:.1f}h/week; ordinary adjusters={adjusters}")
    say(f"  ordinary u={ordinary / (adjusters * 40):.6f}; senior u={senior / 40:.6f}; L=60")
    say(f"  Friday 17:00..Sunday midnight: ordinary {30 * 55 / 24:.6f} claims = {30 * 55 / 24 / 3:.6f}h vs24h")
    say(f"  senior {6 * 55 / 24:.6f} claims = {6 * 55 / 24 * 2 / 3:.6f}h vs8h; fails")
    say(f"  Poisson Monday count failures: senior={poisson_tail(13, 6 * 55 / 24):.6f}; ordinary={poisson_tail(73, 30 * 55 / 24):.6f}")
    records, rows = exercise_log_rows()
    planted = [r for r in rows if r.planted]
    decided = [r for r in planted if r.action in ("approved", "refused", "modified")]
    caught = sum(r.action == "refused" for r in decided)
    say("5 Fixture's last report:", records[-1]["report"])
    say(f"  True catches={caught}/{len(decided)} decided ({caught / len(decided):.6%}); undecided={len(planted) - len(decided)}")
    say(f"  Exact P(X<=9 | n=14,p=.9)={float(binom_cdf(9, 14)):.9f}")
    for band, row in load_report(rows).items():
        say(" ", band, row)
    riley = [r for r in rows if r.reviewer == "riley"]
    say("  Riley refused", sum(r.action == "refused" for r in riley), "of", len(riley))
    say("  Escalations are not catches. Respond with sample cuts and staffing.")
    say("6 Starter fails: a deadline writes an approval; reference escalates.")


def print_wait_data():
    curve = []
    for u in range(0, 96, 5):
        row = mm1(Fraction(u, 50), 2)
        assert row.W is not None
        curve.append({"u": u / 100, "W": float(row.W), "tail4": row.tail(4)})
    points = []
    for u in (.5, .7, .8, .9):
        for desk in (False, True):
            points.append({"u": u, "desk": desk, **simulate_one(7, u, n=30000, desk=desk)})
    print(json.dumps({"constructed": True, "mu_per_hour": 2, "curve": curve, "points": points}, indent=2))


# region book:fnd.human-oversight-risk.arrivals-01 output
if __name__ == "__main__":
    print_arrivals()
# endregion book:fnd.human-oversight-risk.arrivals-01

# region book:fnd.human-oversight-risk.queue-01 output
if __name__ == "__main__":
    print_queue()
# endregion book:fnd.human-oversight-risk.queue-01

# region book:fnd.human-oversight-risk.absence-01 output
if __name__ == "__main__":
    print_absence()
# endregion book:fnd.human-oversight-risk.absence-01

# region book:fnd.human-oversight-risk.catch-01 output
if __name__ == "__main__":
    print_catches()
# endregion book:fnd.human-oversight-risk.catch-01

# region book:fnd.human-oversight-risk.sample-01 output
if __name__ == "__main__":
    print_samples()
# endregion book:fnd.human-oversight-risk.sample-01

# region book:fnd.human-oversight-risk.keys-01 output
if __name__ == "__main__":
    print_keys()
# endregion book:fnd.human-oversight-risk.keys-01

# region book:fnd.human-oversight-risk.wait-data-01 output
if __name__ == "__main__":
    print_wait_data()
# endregion book:fnd.human-oversight-risk.wait-data-01
