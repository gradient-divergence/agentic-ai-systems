"""Chapter 33's constructed bundle, timing cost and exact promotion verdicts.

Capture uses the repository's capture_outputs function. No held-out labels are
read: the score rebuild uses development rows, and this fortnight's labels are
generated. Earlier modules and output files are never changed by the demo.
"""

import json
from datetime import datetime, timedelta, timezone
from fractions import Fraction
from functools import lru_cache
from math import exp, prod
from pathlib import Path
from textwrap import fill
from uuid import uuid4

from refund_agent.release.candidate import build_candidate, build_tools, run_request
from refund_agent.release.suite import build_suite, run_case

# Initialize the fixed core before cold component imports.
CORE = build_candidate(build_suite())

# isort: split

from refund_agent import contracts as c
from refund_agent.batch_stream.component import selector as event_selector
from refund_agent.batch_stream.generate import read_sources
from refund_agent.batch_stream.view import shipment_as_of
from refund_agent.canary.cloud_native_deployment_component import (
    JOURNAL_PATH as CANARY_JOURNAL,
)
from refund_agent.canary.cloud_native_deployment_component import (
    STORE_PATH,
    run_release,
)
from refund_agent.canary.cloud_native_deployment_component import (
    selector as canary_selector,
)
from refund_agent.canary.cloud_native_deployment_plan import CanaryPlan, sign_plan
from refund_agent.canary.cloud_native_deployment_rehearsal import run_rehearsal
from refund_agent.canary.cloud_native_deployment_start import start_canary
from refund_agent.canary.cloud_native_deployment_stop import CanarySession
from refund_agent.canary.cloud_native_deployment_store import AppendJournal, AppendStore
from refund_agent.canary.cloud_native_deployment_window import (
    live_event_log,
    next_run_at,
    run_window,
    sign_window,
)
from refund_agent.data.generate import _label, _policies, _retrieval
from refund_agent.data.live import generate_live
from refund_agent.memory.register import PolicyRegister, project
from refund_agent.memory.support import support
from refund_agent.promotion.mlops_mle_bundle import NAME, Component, from_candidate
from refund_agent.promotion.mlops_mle_component import REGISTRY_PATH, selector
from refund_agent.promotion.mlops_mle_record import decide, sign
from refund_agent.promotion.mlops_mle_registry import Registry
from refund_agent.promotion.mlops_mle_requirements import REQUIREMENTS
from refund_agent.release.extensions import resource_guard
from refund_agent.release.rule import sign as sign_release
from refund_agent.runtimes.swap import _project
from refund_agent.sources.data_systems_release import SELECTOR as DATA

ROOT = Path(__file__).resolve().parents[2]
START = datetime(2026, 10, 1, tzinfo=timezone.utc)
FORTNIGHT_PER_DAY = 4  # Constructed volume; the brief leaves counts to the run.
OUTPUT_NAMES = (
    "bundle-01",
    "requirements-01",
    "week-01",
    "records-01",
    "rollback-01",
    "collision-01",
    "keys-01",
)


def register_at(at, new=None):
    register = PolicyRegister()
    for policy in _policies():
        register.announce(policy, policy.effective_at)
    if new is not None:
        register.announce(new, at)
    return register


def candidates(snapshot, versions):
    components = [DATA, event_selector(), canary_selector(), selector(snapshot, versions)]
    suite = build_suite(components=components)
    candidate = build_candidate(suite)
    return suite, candidate


def mutate(bundle, part, **items):
    before = next(p for p in bundle.components if p.name == part)
    return bundle.model_copy(update={part: Component(name=part, items={**before.items, **items})})


def evidence_record(kind, bundle, passed, **payload):
    return {"kind": kind, "bundle_fingerprint": bundle.fingerprint, "passed": passed, **payload}


@resource_guard(0)
def release_evidence(candidate, suite, at):
    # A must-pass fixture that cannot be constructed blocks the full command.
    # Report its actual refusal instead of synthesizing a signed release record.
    for case in suite.cases:
        if case.handler not in ("data-systems", "batch-stream-absence"):
            continue
        try:
            trial = run_case(case, candidate)
        except ValueError as error:
            return None, [case.case_id + ": " + str(error).splitlines()[0]]
        if trial["failed"]:
            return None, [case.case_id + ": " + ", ".join(trial["failed"])]
    record = run_release(candidate, suite)
    if record["decision"] == "pass":
        record = sign_release(record, AppendJournal(CANARY_JOURNAL), "morgan", at.isoformat())
    return record, []


def policy_evidence(bundle, register, snapshot):
    versions = set(bundle.policy.items["versions"])
    policies = project(snapshot, snapshot.at)
    from refund_agent.adaptation.arms import dated_index, retrieval_arm
    from refund_agent.memory.assemble import required_facts
    from refund_agent.promotion.mlops_mle_cases import policy_fixture
    from refund_agent.release.suite import spent_regression
    from refund_agent.retrieval.clauses import policy_clauses
    from refund_agent.retrieval.encoders import CountEncoder
    from refund_agent.retrieval.evaluate import recall_at_k
    from refund_agent.rules import draft

    clauses = policy_clauses(policies, snapshot.at)
    encoder = CountEncoder().fit([r.text for r in clauses])
    index_at = dated_index(policies, encoder)
    # Spent Chapter 14 questions are regression evidence, not a fresh holdout.
    regression = spent_regression()
    _, baseline_questions = regression._retrieval_recall(retrieval_arm(regression.index))
    recalls, candidate_questions = regression._retrieval_recall(retrieval_arm(index_at))
    lost = baseline_questions - candidate_questions
    rankings = [index_at(q.as_of).retrieve(q.text, q.as_of, 3).hits for q in regression.questions]
    recalled = recall_at_k(rankings, [q.relevant for q in regression.questions], 3)
    prior_rankings = [
        regression.index(q.as_of).retrieve(q.text, q.as_of, 3).hits for q in regression.questions
    ]
    prior_recall = recall_at_k(prior_rankings, [q.relevant for q in regression.questions], 3)
    check = []
    for policy in policies:
        if policy.version not in versions:
            continue
        data, request = policy_fixture(policy, 0)
        state = c.State(
            request=request,
            trace_id="mlops-mle-support-" + policy.version,
            order=data.orders[0],
            policy=policy,
            shipment_events=data.shipments[0].events,
            refunded_cents=0,
        )
        check.append(support(draft(state), register, required_facts(state)))
    return {
        "support": evidence_record(
            "support", bundle, all(r["status"] == "supported" for r in check), results=check
        ),
        "policy_notice": evidence_record(
            "policy_notice", bundle, True, notice_days=7, owner="the policy owner"
        ),
        "frozen_questions": evidence_record(
            "frozen_questions", bundle, not lost, lost_clauses=len(lost)
        ),
        "dated_recall": evidence_record(
            "dated_recall",
            bundle,
            recalled >= prior_recall,
            recall_at_3=recalled,
            baseline_recall_at_3=prior_recall,
            recalls=recalls,
        ),
    }


def control_comparison(candidate, data, log, request, at):
    """Compare an actual shadow decision, including review tiers, to its label."""
    tools = build_tools(data, candidate, "shadow", event_log=log)
    trace = run_request(candidate, tools, request, at=at)
    label = _label(
        request.model_copy(update={"at": at}),
        data.orders[0],
        data.shipments[0],
        data.policy_at(at),
    )
    return {
        "control_candidate": candidate.fingerprint,
        "control_matches_label": trace.decision.outcome == label.outcome,
        "control_outcome": trace.decision.outcome.value,
        "control_reason": trace.decision.reason,
        "label_outcome": label.outcome.value,
        "label_reason": label.reason,
    }


def run_fortnight(stream, new, b0, b1, c0, c1, *, promote_day, rollback_day=None):
    log = live_event_log(stream)
    data = stream.data.model_copy(update={"policies": [*stream.data.policies, new], "correct": []})
    data = data.model_copy(
        update={
            "retrievals": [
                _retrieval(r, data.policy_at(r.at), data.policies) for r in data.requests
            ]
        }
    )
    first = datetime.fromisoformat(stream.manifest["start"])
    control = build_candidate(
        build_suite(
            components=[row for row in c0.tools_composition["components"] if row["name"] != NAME]
        )
    )
    rows = []
    for request in data.requests:
        # The whole fortnight is shadow. The rule applies to payments and denials
        # at every tier; the person still decides every shadow request.
        def select(at):
            if rollback_day is not None and at >= first + timedelta(days=rollback_day - 1):
                return c0, b0
            return (c1, b1) if at >= first + timedelta(days=promote_day - 1) else (c0, b0)

        chosen, bundle = select(request.at)
        observed = _project(data, request)
        at = request.at
        tools = build_tools(observed, chosen, "shadow", event_log=log)
        trace = run_request(chosen, tools, request, at=at)
        while trace.decision.reason == "shipment_incomplete":
            at = next_run_at(log, request.at, at)
            observed = read_sources(observed, at)
            observed = observed.model_copy(
                update={"shipments": [shipment_as_of(log, request.order_reference, at)]}
            )
            next_candidate, next_bundle = select(at)
            if next_candidate is not chosen:
                chosen, bundle = next_candidate, next_bundle
                tools = build_tools(observed, chosen, "shadow", event_log=log)
            else:
                tools.data = observed
            trace = run_request(chosen, tools, request, at=at)
        referred = trace.decision.reason == "policy_not_in_bundle"
        # A draft is only eligibility: the full control preserves review tiers.
        comparison = (
            control_comparison(control, observed, log, request, at)
            if referred
            else {
                "control_matches_label": None,
            }
        )
        rows.append(
            {
                "request_id": request.request_id,
                "request_at": request.at.isoformat(),
                "decision_at": at.isoformat(),
                "day": (at - first).days + 1,
                "bundle": bundle.fingerprint,
                "bundle_name": "B1" if bundle is b1 else "B0",
                "outcome": trace.decision.outcome.value,
                "reason": trace.decision.reason,
                "policy_not_in_bundle": referred,
                **comparison,
                "lineage_stamped": bool(tools.lineage)
                and all(lineage.bundle == bundle.fingerprint for lineage in tools.lineage),
            }
        )
    return rows


def rebuilt_risk(b0):
    from demos.delegated_model_demo import build, seal
    from demos.supervised_learning_demo import windows
    from refund_agent.data.history import generate_history

    selection_path = ROOT / "companion/book_outputs/_fnd.automl-and-selection.record-full-01.txt"
    stored = json.loads(selection_path.read_text())
    development = windows([row for row in generate_history(seed=7) if row.paid])
    built = build(
        window={name: development[name] for name in ("training", "validation")},
        budget=stored["commission"]["budget"],
        seed=stored["commission"]["seed"],
    )
    receipt = stored["receipt"]
    bundle = mutate(
        b0, "model", risk_score={"selection": stored["sha256"], "pipeline": stored["selected"]}
    )
    return bundle, {
        "selection": evidence_record(
            "selection", bundle, True, selection_fingerprint=stored["sha256"]
        ),
        "rebuild": evidence_record(
            "rebuild", bundle, seal(built) == receipt, receipt=receipt, rebuilt_receipt=seal(built)
        ),
        "calibration": evidence_record(
            "calibration", bundle, True, months=["2026-08"], required_months=["2026-10"]
        ),
    }


@resource_guard(0)
def rehearse(candidate, plan, stream, target):
    return run_rehearsal(plan, stream, 14, ["forbidden_payment"], "4h", target, candidate=candidate)


@lru_cache(maxsize=1)
def experiment():
    initial = register_at(START)
    snapshot0 = initial.snapshot(START)
    versions = [p.version for p in _policies()]
    suite0, c0 = candidates(snapshot0, versions)
    b0 = from_candidate(c0)
    window = run_window(c0, START, bundle_fingerprint=b0.fingerprint)
    if window.measures_pass:
        window = sign_window(window, "morgan", window.completed_at + timedelta(seconds=1))
    first = window.completed_at.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(
        days=1
    )
    effective = first + timedelta(days=7)
    new = c.Policy(
        version=effective.date().isoformat(),
        effective_at=effective,
        return_days=35,
        cap_cents=100000,
        named_approver=_policies()[-1].named_approver,
        text=f"refund_cap_v4@{effective.date().isoformat()}: physical goods. #eligibility: damaged or changed_mind "
        "within 35 days of delivery; not_delivered if no delivery. #cap: at most 100000 cents "
        "and the unrefunded payment balance.",
    )
    register1 = register_at(first, new)
    snapshot1 = register1.snapshot(first)
    suite1, c1 = candidates(snapshot1, [*versions, new.version])
    b1 = from_candidate(c1)
    plan = sign_plan(
        CanaryPlan(
            canary_id="mlops-mle-canary-01",
            candidate_fingerprint=c0.fingerprint,
            suite_fingerprint=suite0.fingerprint,
            rollback_target_fingerprint=c0.fingerprint,
            requirement_ref=window.window_id,
            rehearsal_ref="mlops-mle-rehearsal-01",
        ),
        "morgan",
        first,
    )
    stream = generate_live(7, first, 14, FORTNIGHT_PER_DAY)
    rehearsal = rehearse(c0, plan, stream, c0)
    release0, blockers0 = release_evidence(c0, suite0, first)
    release1, blockers1 = release_evidence(c1, suite1, first + timedelta(days=10))
    e0 = {"release": release0}
    e1 = {"release": release1, **policy_evidence(b1, register1, snapshot1)}
    initial_record = decide(b0, evidence=e0, journal=AppendJournal(CANARY_JOURNAL))
    replacement = decide(
        b1, b0, from_stage="shadow", evidence=e1, journal=AppendJournal(CANARY_JOURNAL)
    )
    ecanary = {
        **e0,
        "shadow_window": window.model_dump(mode="json"),
        "canary_plan": plan.model_dump(mode="json"),
        "rehearsal": rehearsal.model_dump(mode="json"),
    }
    promotion = decide(
        b0,
        b0,
        from_stage="shadow",
        to_stage="canary",
        evidence=ecanary,
        journal=AppendJournal(CANARY_JOURNAL),
    )
    risk, risk_evidence = rebuilt_risk(b0)
    held = decide(risk, b0, from_stage="shadow", evidence=risk_evidence)
    records = [initial_record, replacement, promotion, held]
    records = [
        sign(r, "morgan", first + timedelta(days=10) if i == 1 else first)
        if r.status == "promoted"
        else r
        for i, r in enumerate(records)
    ]
    namespace = "mlops-mle-demo-" + uuid4().hex
    registry = Registry(REGISTRY_PATH, namespace)
    for bundle in (b0, b1, risk):
        registry.append(bundle)
    for record in records:
        registry.append(record)
    session = CanarySession(plan, AppendStore(STORE_PATH, namespace))
    started = start_canary(
        plan,
        release0,
        window,
        rehearsal.model_dump(mode="json"),
        candidate=c0,
        suite=suite0,
        journal=AppendJournal(CANARY_JOURNAL),
        session=session,
        owner="morgan",
        at=first,
        records={
            "mlops-mle": {"bundle": b0, "namespace": namespace, "current_policy": _policies()[-1]}
        },
    )
    late = run_fortnight(stream, new, b0, b1, c0, c1, promote_day=11)
    early = run_fortnight(stream, new, b0, b1, c0, c1, promote_day=5)
    rolled = run_fortnight(stream, new, b0, b1, c0, c1, promote_day=11, rollback_day=12)
    rollback = None
    if records[1].status == "promoted":
        rollback = registry.rollback(records[1], b0, owner="morgan", at=first + timedelta(days=11))
    mixed = mutate(b1, "policy", register_snapshot=b0.policy.items["register_snapshot"])
    return {
        "b0": b0,
        "b1": b1,
        "c0": c0,
        "c1": c1,
        "suite0": suite0,
        "suite1": suite1,
        "window": window,
        "plan": plan,
        "rehearsal": rehearsal,
        "release0": release0,
        "release1": release1,
        "release_blockers": {"release0": blockers0, "release1": blockers1},
        "records": records,
        "late": late,
        "early": early,
        "rolled": rolled,
        "first": first,
        "new_policy": new,
        "stream": stream,
        "risk": risk,
        "risk_evidence": risk_evidence,
        "start": started,
        "registry": registry,
        "mixed": mixed,
        "rollback": rollback,
        "evidence1": e1,
        "snapshot0": snapshot0,
        "snapshot1": snapshot1,
    }


def print_bundle(bundle):
    print("Bundle:", bundle.fingerprint)
    facts = (
        ("model", "scripted + text risk", "22/29", "yes"),
        ("prompt", "structured draft", "14", "no"),
        ("tools", "core, selected, lock", "23/27-33", "yes"),
        ("policy", "versions/register/offer", "21/26/33", "yes"),
        ("index", "encoder + clauses", "11/26", "no"),
    )
    print("component | holds | chapters | prefix | loop reads")
    for component, holds, chapters, read in facts:
        part = next(p for p in bundle.components if p.name == component)
        print(" | ".join((component, holds, chapters, part.fingerprint[:16], read)))
    print("Text risk threshold:", bundle.model.items["risk_threshold"])
    print("Fingerprinted files:", len(bundle.tools.items["agent_code_files"]))
    print("Declared coverage:", ", ".join(bundle.policy.items["versions"]))


def print_requirements():
    print("Every promotion: release, passing declared-version cases, release owner.")
    for row in REQUIREMENTS:
        print(fill(row.component + "." + "/".join(row.items) + ": " + row.description, 79))
    print("Threshold experiment: run or decline signed by the decision owner")
    print("with the cost-derived threshold's assumption and signature time.")


def daily(rows):
    return [
        {
            "day": day,
            "B0": sum(
                r["day"] == day and r["bundle_name"] == "B0" and not r["policy_not_in_bundle"]
                for r in rows
            ),
            "referred": sum(r["day"] == day and r["policy_not_in_bundle"] for r in rows),
            "B1": sum(
                r["day"] == day and r["bundle_name"] == "B1" and not r["policy_not_in_bundle"]
                for r in rows
            ),
        }
        for day in range(1, max(14, max(r["day"] for r in rows)) + 1)
    ]


def print_week(result):
    print("Constructed policy and requests; generated labels; all runs in shadow.")
    print("Day 1:", result["first"].isoformat())
    print("New policy:", result["new_policy"].version, "35 days; announced day 1, effective day 8")
    for name in ("late", "early"):
        print("Run:", name, "promotion day", 11 if name == "late" else 5)
        print("decision day | B0 | policy_not_in_bundle | B1")
        for row in daily(result[name]):
            print(f"{row['day']:2} | {row['B0']:2} | {row['referred']:2} | {row['B1']:2}")
        referred = [r for r in result[name] if r["policy_not_in_bundle"]]
        print(
            "Referrals:",
            len(referred),
            "; control matches generated label:",
            sum(r["control_matches_label"] for r in referred),
        )
    print(
        "Every lineage stamped:",
        all(r["lineage_stamped"] for name in ("late", "early", "rolled") for r in result[name]),
    )


def print_records(result):
    for label, record in zip(("initial B0", "(a) B1", "(b) B0", "(c) B_risk"), result["records"]):
        print(label, record.from_stage + " -> " + record.to_stage, record.status)
        print("  changed:", ", ".join(sorted({d["component"] for d in record.changes})) or "none")
        print(fill("  unmet: " + (", ".join(record.unmet) or "none"), 79))
        print("  met:", len([k for k, v in record.requirements.items() if v is not None]))
    print("Rebuild receipt matches:", result["risk_evidence"]["rebuild"]["passed"])
    print("Window candidate:")
    print(result["window"].candidate_fingerprint)
    print("Window frozen bundle:")
    print(result["window"].bundle_fingerprint)
    print("Frozen requests:", result["window"].required_requests)
    print("Canary id:", result["plan"].canary_id)
    print("Rehearsal id:", result["rehearsal"].rehearsal_id)
    print("D6: forbidden, referrals/baseline, wrongful denials/baseline:")
    window = result["window"]
    print(
        window.forbidden_payments,
        window.referrals,
        window.baseline_referrals,
        window.wrongful_denials,
        window.baseline_wrongful_denials,
    )
    print("Deployed labels mature at:", window.labels_mature_at.isoformat())
    print(
        "Canary start accepted:", result["start"].accepted, "; checks:", len(result["start"].checks)
    )
    for check in result["start"].checks:
        print(fill(check.name + ": " + ("pass" if check.passed else check.detail), 79))
    for name in ("release0", "release1"):
        record = result[name]
        print(
            name + ":", record["decision"] if record is not None else "blocked before full release"
        )
        failures = (
            record["failed_cases"] if record is not None else result["release_blockers"][name]
        )
        print(fill("failed cases: " + ", ".join(failures), 79))


def print_rollback(result):
    registry, mixed = result["registry"], result["mixed"]
    print("Mixed state: B1 with B0's register snapshot")
    print("Shadow admission:", registry.admitted(mixed.fingerprint, "shadow"))
    print("Canary admission:", registry.admitted(mixed.fingerprint, "canary"))
    print("Signed rollback produced:", result["rollback"] is not None)
    for row in daily(result["rolled"]):
        if row["day"] >= 12:
            print("Decision day", row["day"], "policy_not_in_bundle:", row["referred"])
    print(
        "From day 12 referrals:",
        sum(r["day"] >= 12 and r["policy_not_in_bundle"] for r in result["rolled"]),
    )
    print("Restoring software does not renew spent, ended or expired authority.")


def collision_bound(n, hexadecimal):
    if type(n) is not int or n < 0 or type(hexadecimal) is not int or hexadecimal < 1:
        raise ValueError("nonnegative integer count and positive integer prefix required")
    return min(Fraction(1), Fraction(n * (n - 1), 2 * 16**hexadecimal))


def print_collision():
    for n, width in ((1000000, 16), (10000, 7), (50000, 12), (50000, 13)):
        bound = collision_bound(n, width)
        print(f"{n} bundles, {width} hex: {n * (n - 1) // 2}/{16**width}")
        print(f"  bound approximately {float(bound):.12g}")
    print(f"Birthday approximation, 10000 and 7: {1 - exp(-float(collision_bound(10000, 7))):.12g}")
    print("Constructed hospital: versions", sum((3, 4, 2, 6, 2)), "bundles", prod((3, 4, 2, 6, 2)))


def print_keys():
    pairs = 50000 * 49999 // 2
    shortest = next(
        width for width in range(1, 65) if collision_bound(50000, width) < Fraction(1, 1000000)
    )
    print("Exercise 3: pairs", pairs, "; shortest prefix:", shortest)
    print("Exercise 4: June 1-3:", 3 * 400, "; signed inside notice:", 0)
    print("Exercise 5: unreported index change lacks frozen-question evidence.")
    print("Exercise 6: runtime/adapter change requires the runtime swap record.")


def write_data(result):
    """Persist constructed figure data and explicitly stipulated exercises."""
    from dataclasses import asdict

    from refund_agent.promotion.mlops_mle_record import PromotionRecord
    from refund_agent.retrieval.clauses import policy_clauses
    from refund_agent.retrieval.encoders import CountEncoder
    from refund_agent.retrieval.index import PolicyIndex

    data_root = ROOT / "companion/demos/data"
    exercise_root = ROOT / "companion/exercises/mlops-mle"
    exercise_root.mkdir(parents=True, exist_ok=True)

    def write(path, value):
        path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")

    write(
        data_root / "mlops_mle_window.json",
        {
            "constructed": True,
            "candidate": result["c0"].record(),
            "bundle": result["b0"].record(),
            "window": result["window"].model_dump(mode="json"),
        },
    )
    write(
        data_root / "mlops_mle_figures.json",
        {
            "constructed": True,
            "clock": "decision day (late terminal decisions retained)",
            "control_method": "full shadow pipeline without Chapter 33; same terminal inputs/time",
            "late": daily(result["late"]),
            "early": daily(result["early"]),
            "rolled": daily(result["rolled"]),
            "requests": {name: result[name] for name in ("late", "early", "rolled")},
        },
    )
    write(
        data_root / "mlops_mle_data_facts.json",
        {
            "constructed": True,
            "control_method": "full shadow pipeline without Chapter 33; same terminal inputs/time",
            "candidate0": result["c0"].record(),
            "bundle0": result["b0"].record(),
            "bundle1": result["b1"].record(),
            "risk_bundle": result["risk"].record(),
            "mixed_bundle": result["mixed"].record(),
            "risk_evidence": result["risk_evidence"],
            "seed": 7,
            "requests_per_day": FORTNIGHT_PER_DAY,
            "source_stream_fingerprint": result["stream"].fingerprint,
            "source_manifest": result["stream"].manifest,
            "policy": result["new_policy"].model_dump(mode="json"),
            "announcement": result["first"].isoformat(),
            "notice_days": 7,
            "promotion_days": [11, 5],
            "rollback_day": 12,
            "policy_change_cases": [
                case.record()
                for case in result["suite1"].cases
                if case.case_id.startswith("mlops-mle-policy-change-")
            ],
            "window": result["window"].model_dump(mode="json"),
            "plan": result["plan"].model_dump(mode="json"),
            "rehearsal": result["rehearsal"].model_dump(mode="json"),
            "records": [record.record() for record in result["records"]],
            "start": result["start"].model_dump(mode="json"),
            "scheduled_counterfactuals": True,
            "actual_signed_rollback": result["rollback"].record() if result["rollback"] else None,
            "release_blockers": result["release_blockers"],
        },
    )
    b0, b1 = result["b0"], result["b1"]
    runtime = {**b0.tools.items["adapter"], "version": "mlops-mle-exercise-adapter-2"}
    candidate06 = mutate(b0, "tools", adapter=runtime)
    policies = project(result["snapshot0"], result["snapshot0"].at)
    clauses = policy_clauses(policies, result["snapshot0"].at)
    rarity_index = PolicyIndex.build(
        policies,
        result["snapshot0"].at,
        CountEncoder(rarity=True).fit([clause.text for clause in clauses]),
    )
    candidate05 = mutate(candidate06, "index", **rarity_index.manifest)
    write(exercise_root / "incumbent-05.json", b0.record())
    write(exercise_root / "candidate-05.json", candidate05.record())
    write(exercise_root / "candidate_06.json", candidate06.record())
    write(exercise_root / "b1_06.json", b1.record())
    # A deliberately wrong paper record for diagnosis, never appended to Registry.
    planted = sign(
        PromotionRecord(
            candidate_fingerprint=candidate05.fingerprint,
            incumbent_fingerprint=b0.fingerprint,
            from_stage="shadow",
            to_stage="shadow",
            changes=({"component": "tools", "item": "adapter"},),
            requirements={
                "release": "stipulated changed suite",
                "policy_coverage": "stipulated passing policy coverage",
                "runtime_swap": "stipulated passing swap",
            },
            status="promoted",
            unmet=(),
            rollback_target=b0.fingerprint,
            refs={
                "candidate": "constructed future runtime",
                "suite": "constructed changed evaluation manifest",
            },
        ),
        "morgan",
        result["first"],
    )
    write(exercise_root / "record-05.json", planted.record())
    scenario = (
        "constructed exercise; signed global prerequisites are stipulated, no runtime authority"
    )
    write(
        exercise_root / "mlops_mle_evidence_06.json",
        {
            "scenario": scenario,
            "by_bundle": {
                candidate06.fingerprint: {
                    name: "stipulated signed prerequisite"
                    for name in ("release", "policy_coverage", "release_owner")
                },
                b1.fingerprint: {
                    name: "stipulated signed prerequisite"
                    for name in (
                        "release",
                        "policy_coverage",
                        "release_owner",
                        "policy_change_cases",
                        "support",
                        "policy_notice",
                        "frozen_questions",
                        "dated_recall",
                    )
                },
            },
            "rows": [asdict(row) for row in REQUIREMENTS],
        },
    )
