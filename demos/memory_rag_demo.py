"""Constructed retrieval and policy memory beside the unchanged refund loop.

Seeded, offline and training-only. Importing prints and writes nothing.
Run with --write-data for plots, the full support report and exercise fixtures.
"""

import argparse
import json
from collections import Counter
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from functools import lru_cache
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np

from demos.output_wrap import print_wrapped
from refund_agent import contracts as c
from refund_agent import rules
from refund_agent.data.generate import generate
from refund_agent.memory.assemble import assemble, draft_from_facts, fact_lines, required_facts
from refund_agent.memory.funnel import funnel, stage_flags
from refund_agent.memory.memory import MONTHLY_READ_DAY, NOTICE, PolicyMemory
from refund_agent.memory.register import PolicyRegister, clause_text, project
from refund_agent.memory.support import support
from refund_agent.retrieval.encoders import CountEncoder, words
from refund_agent.split import CUTS, time_split

PARAMETERS: dict[str, Any] = {
    "seed": 7,
    "services": 24,
    "procedures": 4,
    "base_questions": 400,
    "first_stage_depth": 12,
    "cutoff": 5,
    "word_budget": 180,
    "cutoffs": list(range(1, 11)),
    "passage_words": [45, 70, 95],
}
WINDOW, DRAFT, CARRIER_SHARE, OTHER_SHARES = 2048, 400, 250, 1140
DATA = Path(__file__).resolve().parent / "data"
EXERCISES = Path(__file__).resolve().parents[1] / "exercises/memory-rag"


def utc(month, day, hour=0):
    return datetime(2026, month, day, hour, tzinfo=timezone.utc)


def training_data(seed=7):
    data = generate(seed)
    requests = time_split(data.requests, CUTS, timedelta(0))["training"]
    return data, requests


def demonstration_register():
    register = PolicyRegister()
    for policy in generate(7).policies:
        register.announce(policy, policy.effective_at - NOTICE)
    return register


def request_state(data, request, policy=None):
    suffix = request.request_id.rsplit("-", 1)[1]
    order = next(o for o in data.orders if o.order_id == f"o-{suffix}")
    shipment = next(s for s in data.shipments if s.order_id == order.order_id)
    snapshot = next(s for s in data.retrievals if s.request_id == request.request_id)
    return c.State(
        request=request,
        trace_id=f"trace-{request.request_id}",
        order=order,
        shipment_events=shipment.events,
        policy=policy or data.policy_at(request.at),
        refunded_cents=sum(p.amount_cents for p in order.prior_refunds),
        freshness={"policy": (request.at - snapshot.observed_at).total_seconds()},
    )


def routed_draft(state):
    draft = rules.draft(state)
    if draft.tier == c.Tier.NAMED_APPROVER:
        return draft.model_copy(update={"outcome": c.Outcome.REFER})
    return draft


def monthly_read(at):
    start = at.replace(day=MONTHLY_READ_DAY, hour=0, minute=0, second=0, microsecond=0)
    if start > at:
        start = (start.replace(day=1) - timedelta(days=1)).replace(day=MONTHLY_READ_DAY)
    return start


def telecom_library(seed=7):
    rng = np.random.default_rng(seed)
    procedures = (
        (
            "routing",
            "recover packet route after route flaps",
            "bring traffic back when forwarding oscillates",
        ),
        (
            "radio",
            "reset radio cell after signal loss",
            "restore coverage when the antenna goes quiet",
        ),
        (
            "session",
            "restart session worker after session timeout",
            "revive connections when logins hang",
        ),
        ("power", "switch power feed after voltage drop", "keep equipment alive when supply sags"),
    )
    dates = (utc(1, 20), utc(3, 1), utc(5, 1))
    library = []
    for service in range(PARAMETERS["services"]):
        for p, (name, plain, paraphrase) in enumerate(procedures):
            count = int(rng.integers(1, 4))
            alarm = f"ALM-S{service:02d}-P{p}"
            for v in range(count):
                text = (
                    f"gateway-s{service:02d} {alarm} {name} procedure. {plain}. "
                    f"Verify the alarm before rollback; version {v + 1} requires the on-call acknowledgement."
                )
                size = int(rng.choice(PARAMETERS["passage_words"]))
                filler = "network operations log runbook monitor verify link service status safe rollback".split()
                text += " " + " ".join(
                    filler[i % len(filler)] for i in range(size - len(text.split()))
                )
                library.append(
                    {
                        "id": f"s{service:02d}-p{p}-v{v + 1}",
                        "procedure": f"s{service:02d}-p{p}",
                        "alarm": alarm,
                        "service": service,
                        "plain": plain,
                        "paraphrase": paraphrase,
                        "text": text,
                        "valid_from": dates[v],
                        "valid_to": dates[v + 1] if v + 1 < count else None,
                    }
                )
    questions = []
    for n in range(PARAMETERS["base_questions"]):
        service, p = (
            int(rng.integers(PARAMETERS["services"])),
            int(rng.integers(PARAMETERS["procedures"])),
        )
        at = utc(2, 1) + timedelta(days=int(rng.integers(210)))
        active = [
            r
            for r in library
            if r["procedure"] == f"s{service:02d}-p{p}"
            and r["valid_from"] <= at
            and (r["valid_to"] is None or at < r["valid_to"])
        ]
        needed = active[0]
        for wording, text in (
            ("identifier", f"{needed['alarm']} rollback procedure"),
            ("plain", f"gateway-s{service:02d} {needed['plain']}"),
            ("paraphrase", f"gateway-s{service:02d} {needed['paraphrase']}"),
        ):
            questions.append(
                {
                    "id": f"q{n:03d}-{wording}",
                    "wording": wording,
                    "text": text,
                    "needed": needed["id"],
                    "procedure": needed["procedure"],
                    "at": at,
                }
            )
    return library, questions


def keep_passages(ranked):
    kept, used = [], 0
    for record in ranked:
        size = len(record["text"].split())
        if used + size <= PARAMETERS["word_budget"]:
            kept.append(record)
            used += size
    return kept


def telecom_stages(library, k, date_filter):
    encoder = CountEncoder().fit([r["text"] for r in library])
    matrix = encoder.transform([r["text"] for r in library])
    norms = np.linalg.norm(matrix, axis=1)
    candidates_cache, ranks_cache = {}, {}

    def first_stage(question):
        if question["id"] not in candidates_cache:
            vector = encoder.transform([question["text"]])[0]
            norm = np.linalg.norm(vector)
            scores = matrix @ vector / np.maximum(norms * norm, 1e-12)
            eligible = [
                i
                for i, r in enumerate(library)
                if not date_filter
                or (
                    r["valid_from"] <= question["at"]
                    and (r["valid_to"] is None or question["at"] < r["valid_to"])
                )
            ]
            eligible.sort(key=lambda i: (-round(float(scores[i]), 12), library[i]["id"]))
            candidates_cache[question["id"]] = [
                library[i] for i in eligible[: PARAMETERS["first_stage_depth"]] if scores[i] > 0
            ]
        return candidates_cache[question["id"]]

    def rerank(question, candidates):
        key = (question["id"], tuple(r["id"] for r in candidates))
        if key not in ranks_cache:
            query = words(question["text"])
            pairs = set(zip(query, query[1:]))

            def score(record):
                passage = words(record["text"])
                exact = record["alarm"].casefold() in question["text"].casefold()
                return 10 * exact + len(pairs & set(zip(passage, passage[1:])))

            ranks_cache[key] = sorted(candidates, key=lambda r: (-score(r), r["id"]))
        return ranks_cache[key][:k]

    return first_stage, rerank, lambda question, ranked: keep_passages(ranked)


def telecom_run(seed=7):
    library, questions = telecom_library(seed)
    first, rerank, assembler = telecom_stages(library, PARAMETERS["cutoff"], True)
    flags = stage_flags(questions, first, rerank, assembler)
    unranked = stage_flags(
        questions, first, lambda q, rows: rows[: PARAMETERS["cutoff"]], assembler
    )
    return {
        "library": library,
        "questions": questions,
        "flags": flags,
        "unranked_flags": unranked,
        "stages": (first, rerank, assembler),
        "funnel": funnel(flags, "wording"),
        "unranked": funnel(unranked, "wording"),
    }


def retrieval_journey(experiment):
    """First identifier-ordered paraphrase kept only in the first stage's order."""
    first, rerank, assembler = experiment["stages"]
    original = {row["question"]: row for row in experiment["unranked_flags"]}
    reordered = {row["question"]: row for row in experiment["flags"]}
    question = next(
        question
        for question in sorted(experiment["questions"], key=lambda q: q["id"])
        if question["wording"] == "paraphrase"
        and original[question["id"]]["rank"]
        and reordered[question["id"]]["rank"]
        and original[question["id"]]["kept"]
        and not reordered[question["id"]]["kept"]
    )
    candidates = first(question)  # Reuse telecom_run's cached stages; no new random draws.
    orders = {}
    for name, ranked in (
        ("first stage", candidates[: PARAMETERS["cutoff"]]),
        ("reranked", rerank(question, candidates)),
    ):
        kept = {row["id"] for row in assembler(question, ranked)}
        passages, used, before = [], 0, None
        for row in ranked:
            size = len(row["text"].split())
            if row["id"] == question["needed"]:
                before = used
            passages.append({"id": row["id"], "words": size, "kept": row["id"] in kept})
            if row["id"] in kept:
                used += size
        orders[name] = {
            "passages": passages,
            "words_before_needed": before,
            "rank": any(row["id"] == question["needed"] for row in ranked),
            "kept": question["needed"] in kept,
        }
    needed = next(row for row in candidates if row["id"] == question["needed"])
    return {"question": question, "needed_words": len(needed["text"].split()), "orders": orders}


def print_journey(experiment):
    journey = retrieval_journey(experiment)
    question = journey["question"]
    print(f"seed 7; {question['id']}: {question['text']}")
    print(f"needed {question['needed']}: {journey['needed_words']} words; window 180 words")
    for name, order in journey["orders"].items():
        print(f"{name} order:")
        for passage in order["passages"]:
            print(
                f"  {passage['id']}: {passage['words']} words; "
                f"{'keep' if passage['kept'] else 'skip'}"
            )
        print(
            f"  words used before needed: {order['words_before_needed']}; "
            f"rank {order['rank']}; kept {order['kept']}"
        )


def sweep(library, questions):
    rows = []
    for date_filter in (True, False):
        first, rerank, _ = telecom_stages(library, 10, date_filter)
        all_candidates = {q["id"]: first(q) for q in questions}
        all_ranked = {q["id"]: rerank(q, all_candidates[q["id"]]) for q in questions}
        for k in PARAMETERS["cutoffs"]:
            flags = stage_flags(
                questions,
                lambda q: all_candidates[q["id"]],
                lambda q, candidates: all_ranked[q["id"]][:k],
                lambda q, ranked: keep_passages(ranked),
            )
            row = funnel([{**r, "group": "all"} for r in flags], "group")[0]
            rows.append(
                {
                    **row,
                    "k": k,
                    "date_filter": date_filter,
                    "superseded_share": sum(r["superseded"] for r in flags) / len(flags),
                }
            )
    return rows


def request_b():
    data, _ = training_data()
    request = c.Request(
        request_id="constructed-b",
        requester_id="customer-b",
        order_reference="o-b",
        at=utc(5, 12),
        amount_cents=2499,
        reason="changed_mind",
    )
    order = c.Order(
        order_id="o-b",
        reference="o-b",
        purchased_at=utc(4, 1),
        observed_at=utc(5, 12),
        items=[c.Item(name="physical desk lamp", quantity=1, unit_cents=4999)],
        paid_cents=4999,
        payment_id="payment-b",
    )
    events = [
        c.ShipmentEvent(at=utc(4, day), status=status)
        for day, status in ((2, "shipped"), (5, "lost"), (10, "resent"), (16, "delivered"))
    ]
    events += [c.ShipmentEvent(at=utc(5, day), status="return_scan") for day in range(2, 7)]
    state = c.State(
        request=request,
        trace_id="trace-b",
        order=order,
        shipment_events=events,
        policy=data.policy_at(request.at),
        freshness={"policy": 3600.0},
    )
    required_words = sum(len(line.split()) for line in fact_lines(required_facts(state)))
    fixed = "Application instructions request order policy and reserved context sections".split()
    size = OTHER_SHARES - required_words
    records = [
        {"id": "other-sections", "text": " ".join(fixed[i % len(fixed)] for i in range(size))}
    ]
    history = " ".join(f"carrier event {e.status} at {e.at.isoformat()}" for e in events).split()
    carrier = history + ["carrier log detail"] * 900
    carrier = " ".join(carrier).split()[:900]
    records += [
        {"id": f"carrier-{n:02d}", "text": " ".join(carrier[n * 50 : (n + 1) * 50])}
        for n in range(18)
    ]
    return state, records


def budget_report(data, requests):
    state, records = request_b()
    full = rules.draft(state)
    result = assemble(state, records, OTHER_SHARES + CARRIER_SHARE)
    compact = state.model_copy(
        update={"shipment_events": sorted(state.shipment_events, key=lambda e: e.at)[-5:]}
    )
    preserved = 0
    for request in requests:
        original = request_state(data, request)
        assembled = assemble(original, [], WINDOW - DRAFT)
        preserved += draft_from_facts(assembled.facts) == rules.draft(original)
    return {
        "raw_context": OTHER_SHARES + 900,
        "context": result.words,
        "draft": DRAFT,
        "window": WINDOW,
        "remaining": WINDOW - result.words,
        "kept": list(result.kept),
        "cut": list(result.cut),
        "cut_words": 900 - CARRIER_SHARE,
        "full": full.outcome.value,
        "assembled": draft_from_facts(result.facts).outcome.value,
        "summary": rules.draft(compact).reason,
        "preserved": preserved,
        "training": len(requests),
        "required_words": sum(len(line.split()) for line in fact_lines(result.facts)),
    }


def lifecycle_report(data, requests, register):
    ended, changed, ages = [], [], []
    refreshed_ended = refreshed_changed = check_at_use = 0
    for request in requests:
        read = monthly_read(request.at)
        old_policies = project(register.snapshot(read), read)
        old = max(
            (p for p in old_policies if p.effective_at <= request.at), key=lambda p: p.effective_at
        )
        current = data.policy_at(request.at)
        stale_state = request_state(data, request, old)
        right_state = request_state(data, request, current)
        stale, right = rules.draft(stale_state), rules.draft(right_state)
        if old.version != current.version:
            ended.append(request.request_id)
            ages.append((request.at - read).total_seconds() / 86400)
            observed = c.PolicySnapshot(
                request_id=request.request_id, version=old.version, observed_at=read
            )
            one = data.model_copy(update={"retrievals": [observed]})
            check_at_use += rules.evidence_problem(one, stale_state) == "non_current_policy"
        if (stale.outcome, stale.reason) != (right.outcome, right.reason):
            changed.append(
                {
                    "request": request.request_id,
                    "tier": int(stale.tier),
                    "right": routed_draft(right_state).outcome.value,
                    "copy": routed_draft(stale_state).outcome.value,
                    "reason": stale.reason,
                    "named_approver": old.named_approver if stale.tier == 3 else None,
                }
            )
        policies = project(register.snapshot(request.at - NOTICE), request.at - NOTICE)
        refreshed = max(
            (p for p in policies if p.effective_at <= request.at), key=lambda p: p.effective_at
        )
        refreshed_ended += refreshed.version != current.version
        new = rules.draft(request_state(data, request, refreshed))
        refreshed_changed += (new.outcome, new.reason) != (right.outcome, right.reason)
    return {
        "ended": ended,
        "changed": changed,
        "age_range": [min(ages), max(ages)],
        "training": len(requests),
        "within_30": sum(age <= 30 for age in ages),
        "check_at_use": check_at_use,
        "refreshed_ended": refreshed_ended,
        "refreshed_changed": refreshed_changed,
    }


def planted_support(register):
    state, _ = request_b()
    facts = required_facts(state)
    legitimate = rules.draft(state)
    eligibility = "refund_cap_v4@2026-05-01#eligibility"
    cap = "refund_cap_v4@2026-05-01#cap"
    cap_value = next(
        r.value for r in register.as_of(facts["at"], facts["known_at"]) if r.section == "cap"
    )
    over_cap = {**facts, "amount_cents": cap_value + 1, "balance_cents": cap_value + 50000}
    planted = [
        (
            legitimate.model_copy(
                update={"outcome": c.Outcome.DENY, "reason": "outside_return_window"}
            ),
            facts,
        ),
        (
            legitimate.model_copy(
                update={"policy_citation": "refund_cap_v4@2026-03-01#eligibility"}
            ),
            facts,
        ),
        (
            legitimate.model_copy(
                update={
                    "outcome": c.Outcome.DENY,
                    "reason": "outside_return_window",
                    "policy_citation": cap,
                }
            ),
            facts,
        ),
        (
            legitimate.model_copy(
                update={"policy_citation": "refund_cap_v4@2026-02-01#eligibility"}
            ),
            facts,
        ),
        (
            legitimate.model_copy(
                update={
                    "amount_cents": over_cap["amount_cents"],
                    "tier": c.Tier.NAMED_APPROVER,
                    "policy_citation": eligibility,
                }
            ),
            over_cap,
        ),
    ]
    return [support(draft, register, inputs) for draft, inputs in planted]


def support_report(data, requests, register):
    rows = []
    for request in requests:
        state = request_state(data, request)
        rows.append(support(routed_draft(state), register, required_facts(state)))
    wrong_register = PolicyRegister()
    for policy in data.policies:
        if policy.version == "2026-05-01":
            policy = policy.model_copy(
                update={
                    "return_days": 21,
                    "text": f"refund_cap_v4@{policy.version}: physical goods. "
                    f"#eligibility: {clause_text('eligibility', 21)} #cap: {clause_text('cap', policy.cap_cents)}",
                }
            )
        wrong_register.announce(policy, policy.effective_at - NOTICE)
    opening = next(r for r in requests if r.request_id == "r-7-0085")
    policy = next(
        p
        for p in project(wrong_register.snapshot(opening.at), opening.at)
        if p.version == "2026-05-01"
    )
    state = request_state(data, opening, policy)
    wrong = support(routed_draft(state), wrong_register, required_facts(state))
    return {"training": rows, "planted": planted_support(register), "wrong_register": wrong}


def bank_audit():
    versions = [
        (1, utc(1, 1), utc(1, 1), 4),
        (2, utc(3, 1), utc(2, 20), 5),
        (3, utc(5, 1), utc(5, 10), 6),
    ]
    rows = []
    for decision_at, read_at in (
        (utc(3, 2), utc(2, 25)),
        (utc(5, 5), utc(5, 3)),
        (utc(5, 12), utc(5, 12)),
    ):

        def select(known_at):
            return max(
                (v for v in versions if v[1] <= decision_at and v[2] <= known_at),
                key=lambda v: v[1],
            )

        copy, then, now = select(read_at), select(decision_at), select(utc(6, 1))
        age = (decision_at - read_at).days
        rows.append(
            {
                "at": decision_at.date().isoformat(),
                "copy": copy[0],
                "fee": copy[3],
                "age_days": age,
                "usable": 0 <= age <= 7,
                "right_then": copy == then,
                "right_now": copy == now,
            }
        )
    return rows


@lru_cache(maxsize=1)
def build_experiment(seed=7):
    data, requests = training_data(seed)
    register = demonstration_register()
    telecom = telecom_run(seed)
    return {
        "training_count": len(requests),
        "register": register,
        "telecom": telecom,
        "sweep": sweep(telecom["library"], telecom["questions"]),
        "budget": budget_report(data, requests),
        "lifecycle": lifecycle_report(data, requests, register),
        "support": support_report(data, requests, register),
        "accesses": [],
    }


def figure_data(report):
    register = report["register"]
    records = []
    for record in register.records:
        if record.section != "eligibility":
            continue
        end = register.superseded_at(record)
        records.append(
            {
                "record_id": record.record_id,
                "version": record.version,
                "valid_from": record.valid_from.isoformat(),
                "valid_to": record.valid_to.isoformat() if record.valid_to else None,
                "recorded_at": record.recorded_at.isoformat(),
                "superseded_at": end.isoformat() if end else None,
            }
        )
    return {
        "parameters": PARAMETERS,
        "register": records,
        "sweep": report["sweep"],
        "request_at": utc(5, 12).isoformat(),
        "copy_at": utc(4, 20).isoformat(),
    }


def print_budget_events():
    """Figure-only scan timeline, with facts computed from the complete state."""
    state, _ = request_b()
    facts = required_facts(state)
    for event in sorted(state.shipment_events, key=lambda event: event.at):
        latest = (
            " (latest delivery)"
            if event.status == "delivered" and event.at == facts["delivered_at"]
            else ""
        )
        print(f"{event.status}: {event.at.isoformat()}{latest}")
    lines = dict(zip(facts, fact_lines(facts)))
    groups = (
        ("request_id", "at", "known_at", "snapshot_at"),
        ("purchased_at", "delivered_at", "delivery_age_days", "future_evidence"),
        (
            "amount_cents", "reason", "balance_cents", "policy_version",
            "return_days", "cap_cents", "named_approver",
        ),
    )
    for group in groups:
        print("required facts: " + "; ".join(lines[key] for key in group))


def print_budget(report):
    row = report["budget"]
    print(f"Request B: raw {row['raw_context']} + draft {row['draft']} > window {row['window']}")
    print(
        f"required facts {row['required_words']} words; carrier 900 -> {CARRIER_SHARE}; cut {row['cut_words']}"
    )
    print(
        f"assembled {row['context']} + {row['draft']} = {row['context'] + row['draft']}; remaining before draft {row['remaining']}"
    )
    print(
        f"full {row['full']}; assembled {row['assembled']}; last five scans: refer {row['summary']}"
    )
    cut = row["cut"]
    cut_line = f"cut records: {', '.join(cut)}"
    if len(cut_line) > 80:
        prefix, _, start = cut[0].rpartition("-")
        if start.isdecimal() and cut == [
            f"{prefix}-{number:0{len(start)}d}"
            for number in range(int(start), int(start) + len(cut))
        ]:
            cut_line = f"cut records: {cut[0]} to {cut[-1]}, {len(cut)} records"
        if len(cut_line) > 80:
            from textwrap import wrap

            cut_line = "\n".join(
                wrap(
                    cut_line, width=80, subsequent_indent="  ",
                    break_long_words=False, break_on_hyphens=False,
                )
            )
    print(cut_line)
    print(f"constraint preservation: {row['preserved']} / {row['training']} training requests")
    print("required facts exceeding the budget: required_facts_exceed_budget -> refer")


def print_successor():
    data, requests = training_data()
    request = next(r for r in requests if r.request_id == "r-7-0085")
    state = request_state(data, request)
    delivered = max(e.at for e in state.shipment_events if e.status == "delivered")
    memory = PolicyMemory(demonstration_register(), utc(4, 20))
    age = (request.at - memory.observed_at).total_seconds() / 86400
    print(
        f"{request.request_id}: {request.at.isoformat()}, {request.amount_cents} cents, {request.reason}"
    )
    print(
        f"delivery age {(request.at - delivered).total_seconds() / 86400:.0f} days; May allows {state.policy.return_days}"
    )
    print(
        f"April copy age {age:.6f} days; within 30: {age <= 30}; notice 7: {memory.retrieve('damaged', request.at, 1).refusal}"
    )
    memory.refresh(request.at)
    policies = project(memory.snapshot, memory.snapshot.at, live=True)
    policy = max(
        (p for p in policies if p.effective_at <= request.at), key=lambda p: p.effective_at
    )
    chosen = routed_draft(request_state(data, request, policy))
    print(f"refreshed: {chosen.outcome.value} {chosen.reason}; {chosen.policy_citation}")
    boundary = PolicyMemory(memory.register, request.at - NOTICE)
    print(f"exactly 7 days: {boundary.retrieve('damaged', request.at, 1).refusal or 'accepted'}")
    print(
        f"7 days + 1 second: {boundary.retrieve('damaged', request.at + timedelta(seconds=1), 1).refusal}"
    )


def print_lifecycle(report):
    row = report["lifecycle"]
    print_wrapped(
        f"monthly copies: ended version {len(row['ended'])} / {row['training']}; within 30 days {row['within_30']}"
    )
    print_wrapped(f"ended-version ages: {row['age_range'][0]:.6f} .. {row['age_range'][1]:.6f} days")
    print_wrapped(
        f"policy conclusions differ: {len(row['changed'])}; tiers {dict(sorted(Counter(r['tier'] for r in row['changed']).items()))}"
    )
    for changed in row["changed"]:
        person = (
            f"; named approver {changed['named_approver']} decides"
            if changed["named_approver"]
            else ""
        )
        print_wrapped(
            f"{changed['request']}: tier {changed['tier']}, right {changed['right']}, copy {changed['copy']} {changed['reason']}{person}"
        )
    print_wrapped(f"unchanged check at use: {row['check_at_use']} -> refer non_current_policy")
    print_wrapped(
        f"copies no older than 7 days: ended {row['refreshed_ended']}, changed {row['refreshed_changed']}"
    )
    memory = PolicyMemory(report["register"], utc(4, 20))
    _, requests = training_data()
    cached = next(r for r in requests if r.request_id == "r-7-0001")
    print_wrapped(
        f"cached_policy {cached.request_id} at {cached.at.isoformat()}: {memory.retrieve('damaged', cached.at, 1).refusal}"
    )
    register = demonstration_register()
    citation = "refund_cap_v4@2026-05-01#eligibility"
    original = next(r for r in register.as_of(utc(5, 12), utc(5, 12)) if r.section == "eligibility")
    register.correct(citation, clause_text("eligibility", 25), 25, utc(10, 1))
    corrected = next(
        r for r in register.as_of(utc(5, 12), utc(10, 1)) if r.section == "eligibility"
    )
    print_wrapped(
        f"constructed correction: May as known 12 May {original.value} days; as known 1 October {corrected.value} days"
    )
    data, requests = training_data()
    opening = next(r for r in requests if r.request_id == "r-7-0085")

    def audit_decision(known_at):
        policies = project(register.snapshot(known_at), known_at)
        policy = max(
            (p for p in policies if p.effective_at <= opening.at), key=lambda p: p.effective_at
        )
        return routed_draft(request_state(data, opening, policy))

    then, now = audit_decision(opening.at), audit_decision(utc(10, 1))
    print_wrapped(
        f"audit of {opening.request_id}: {then.outcome.value} as known then; {now.outcome.value} {now.reason} as known now; correction broke notice"
    )
    before = register.records
    register.retain(utc(10, 2))
    read = register.read_text(original)
    print_wrapped(f"retention: records identical {before == register.records}; {read.refusal}")
    print_wrapped(
        f"retention events {len(register.log)}; expired texts {len(register.log[-1].text_hashes)}"
    )
    print_wrapped(f"expired text hash {read.text_hash}")


def print_support(report):
    rows = report["support"]
    statuses = Counter(r["status"] for r in rows["training"])
    print_wrapped(f"training drafts {len(rows['training'])}: {dict(sorted(statuses.items()))}")
    counts = Counter(tuple(r["section"] for r in row["clauses"]) for row in rows["training"])
    for sections, count in sorted(counts.items()):
        print_wrapped(f"needed {','.join(sections) or '(referral: no entailment check)'}: {count} drafts")
    print_wrapped("full clause records and text hashes: demos/data/memory_rag_support.json")
    for n, row in enumerate(rows["planted"], 1):
        print_wrapped(
            f"planted {n}: {row['status']} {row['reason']}; needed {','.join(r['section'] for r in row['clauses'])}; request -> refer"
        )
    last = rows["planted"][-1]
    cap = next(r["value"] for r in last["clauses"] if r["section"] == "cap")
    print_wrapped(
        f"planted approval: {last['amount_cents']} cents against cap {cap}; eligibility citation alone is insufficient"
    )
    wrong = rows["wrong_register"]
    print_wrapped(
        f"constructed on-time wrong register: {wrong['decision']} outside_return_window, {wrong['status']}; support is not truth"
    )


def print_sweep(report):
    print(
        f"constructed seed {PARAMETERS['seed']}; services {PARAMETERS['services']}; procedures/service {PARAMETERS['procedures']}; versions {len(report['telecom']['library'])}"
    )
    print(
        f"questions {len(report['telecom']['questions'])}; first depth {PARAMETERS['first_stage_depth']}; budget {PARAMETERS['word_budget']} words"
    )
    print("filter k n_ret n_rank n_kept p_ret p_rank p_kept product superseded")
    for row in report["sweep"]:
        _, ret, rank, kept = row["counts"]
        rates = " ".join(
            f"{rate:.6f}" if rate is not None else "undefined" for rate in row["rates"]
        )
        product = f"{row['product']:.6f}" if row["product"] is not None else "undefined"
        print(
            f"{'on' if row['date_filter'] else 'off'} {row['k']} {ret} {rank} {kept} {rates} {product} {row['superseded_share']:.6f}"
        )


def print_keys():
    counts = (1000, 900, 720, 684)
    rates = [b / a for a, b in zip(counts, counts[1:])]
    print(f"display counts {counts}; conditional {rates}; product {float(np.prod(rates)):.6f}")
    print(f"wrong unconditional-rank product {rates[0] * counts[2] / counts[0] * rates[2]:.6f}")
    for i, name in enumerate(("candidate", "rank", "window")):
        slope = float(np.prod([r for j, r in enumerate(rates) if i != j]))
        print(
            f"{name}: slope {slope:.6f}; capped ten-point gain {min(0.1, 1 - rates[i]) * slope:.6f}"
        )
    print(f"precision@5 {0.9 / 5:.6f}; precision@10 {0.95 / 10:.6f}")
    counts = (4000, 3800, 3420, 3249)
    rates = [b / a for a, b in zip(counts, counts[1:])]
    print(f"exercise 3 counts {counts}; conditional {rates}; product {float(np.prod(rates)):.6f}")
    # Decimal count ratios avoid rounding a binary approximation below the tie.
    wrong_product = (
        (Decimal(counts[1]) / counts[0])
        * (Decimal(counts[2]) / counts[0])
        * (Decimal(counts[3]) / counts[2])
    )
    print(f"exercise 3 wrong product {wrong_product:.6f}")
    for i, name in enumerate(("candidate", "rank", "window")):
        gain = 0.05 * float(np.prod([r for j, r in enumerate(rates) if i != j]))
        print(f"exercise 3 {name} five-point gain {gain:.6f}")
    print(
        f"exercise 3 precision@4 {0.9 / 4:.6f}; precision@6 {0.94 / 6:.6f}; window survival at both 0.900000"
    )
    for row in bank_audit():
        print(
            f"exercise 4 {row['at']}: version {row['copy']}, fee {row['fee']}, age {row['age_days']} days, usable {row['usable']}, right then {row['right_then']}, right now {row['right_now']}"
        )
    print(
        f"exercise 4 version 2 guarantee {(utc(3, 1) - utc(2, 20)).days} days; version 3 no safe read age"
    )
    print(
        "bank late-change counterexample: recorded 5 March, valid 1 March; at 4 March a 2-day copy passes and is wrong"
    )
    required = [{"p1", "p2"} for _ in range(10)]
    kept = [{"p1"} if i % 2 == 0 else {"p2"} for i in range(10)]
    recalled = sum(len(need & got) for need, got in zip(required, kept))
    complete = sum(need <= got for need, got in zip(required, kept))
    p1, p2 = (sum(name in got for got in kept) / len(kept) for name in ("p1", "p2"))
    print(
        f"constructed two-passage case: recall {recalled}/20 = {recalled / 20:.6f}; "
        f"complete evidence {complete}/10 = {complete / 10:.6f}; p1 = p2 = {p1:.6f}; "
        f"bounds [{max(0, p1 + p2 - 1):.6f}, {min(p1, p2):.6f}]; independent {p1 * p2:.6f}"
    )


def write_data(report):
    DATA.mkdir(exist_ok=True)
    for name, value in (("figures", figure_data(report)), ("support", report["support"])):
        (DATA / f"memory_rag_{name}.json").write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n"
        )
    EXERCISES.mkdir(parents=True, exist_ok=True)
    old, new = (
        "refer when early-warning score reaches 7",
        "refer when early-warning score reaches 6",
    )
    fixture = {
        "constructed": True,
        "not_clinical_guidance": True,
        "register": [
            {
                "citation": "ward-protocol@2026-02-01#threshold",
                "valid_from": "2026-02-01",
                "recorded_at": "2026-02-01",
                "text": new,
                "value": 6,
                "text_hash": sha256(new.encode()).hexdigest(),
            }
        ],
        "decision": {
            "at": "2026-03-10",
            "score": 6,
            "referred": False,
            "text_hash": sha256(old.encode()).hexdigest(),
        },
        "owner_correction": {"recorded_at": "2026-03-15", "text": new, "value": 6},
        "agent_audit": {
            "at": "2026-03-20",
            "verdict": "breach",
            "reason": "score 6 was not referred",
        },
    }
    (EXERCISES / "hospital_register_05.json").write_text(
        json.dumps(fixture, indent=2, sort_keys=True) + "\n"
    )


# region book:fnd.memory-rag.funnel-01 output
from demos import memory_rag_demo as telecom_demo
from refund_agent.memory.funnel import funnel as count_funnel

if __name__ == "__main__":
    experiment = telecom_demo.telecom_run(7)
    for row in count_funnel(experiment["flags"], "wording"):
        print(f"reranked {row['group']}: counts {row['counts']}")
        for name, rate, interval in zip(
            ("ret", "rank|ret", "kept|rank"), row["rates"], row["intervals"]
        ):
            value = (
                "undefined"
                if rate is None
                else f"{rate:.6f} [{interval[0]:.6f}, {interval[1]:.6f}]"
            )
            print(f"  {name}: {value}")
        product = (
            f"{row['product']:.6f}"
            if row["product"] is not None
            else "undefined"
        )
        print(f"  product {product}; end-to-end {row['end_to_end']:.6f}")
    for row in experiment["unranked"]:
        product = (
            f"{row['product']:.6f}"
            if row["product"] is not None
            else "undefined"
        )
        print(
            f"without reranking {row['group']}: "
            f"counts {row['counts']}; product {product}"
        )
# endregion book:fnd.memory-rag.funnel-01


# region book:fnd.memory-rag.budget-01 output
if __name__ == "__main__":
    print_budget(build_experiment())
# endregion book:fnd.memory-rag.budget-01


# region book:fnd.memory-rag.budget-02 output
from demos.memory_rag_demo import print_budget_events as print_carrier_events

if __name__ == "__main__":
    print_carrier_events()
# endregion book:fnd.memory-rag.budget-02


# region book:fnd.memory-rag.asof-01 output
from demos import memory_rag_demo as policy_demo
from refund_agent.memory.memory import PolicyMemory as MemoryCopy

if __name__ == "__main__":
    register = policy_demo.demonstration_register()
    before = register.as_of(
        policy_demo.utc(5, 12), known_at=policy_demo.utc(4, 20)
    )
    now = register.as_of(
        policy_demo.utc(5, 12), known_at=policy_demo.utc(5, 12)
    )
    for name, records in (("known 20 April", before), ("known 12 May", now)):
        clause = next(r for r in records if r.section == "eligibility")
        print(f"12 May, {name}: {clause.citation}, {clause.value} days")
    memory = MemoryCopy(register, policy_demo.utc(4, 20))
    print(
        f"April copy at 12 May: "
        f"{memory.retrieve('damaged', policy_demo.utc(5, 12), 1).refusal}"
    )
# endregion book:fnd.memory-rag.asof-01


# region book:fnd.memory-rag.successor-01 output
if __name__ == "__main__":
    print_successor()
# endregion book:fnd.memory-rag.successor-01


# region book:fnd.memory-rag.lifecycle-01 output
if __name__ == "__main__":
    print_lifecycle(build_experiment())
# endregion book:fnd.memory-rag.lifecycle-01


# region book:fnd.memory-rag.support-01 output
if __name__ == "__main__":
    print_support(build_experiment())
# endregion book:fnd.memory-rag.support-01


# region book:fnd.memory-rag.sweep-01 output
if __name__ == "__main__":
    print_sweep(build_experiment())
# endregion book:fnd.memory-rag.sweep-01


# region book:fnd.memory-rag.keys-01 output
if __name__ == "__main__":
    print_keys()
# endregion book:fnd.memory-rag.keys-01


# region book:fnd.memory-rag.journey-01 output
if __name__ == "__main__":
    print_journey(build_experiment()["telecom"])
# endregion book:fnd.memory-rag.journey-01


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-data", action="store_true")
    if parser.parse_args().write_data:
        write_data(build_experiment())
