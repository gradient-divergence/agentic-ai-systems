"""Constructed renewal evidence. Missing stage-6 records hold; nothing pays."""

import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from demos.output_wrap import print_wrapped
from refund_agent.data.generate import _policies
from refund_agent.data.history import END_AT
from refund_agent.future_directions_risk import selector
from refund_agent.oversight.human_oversight_risk_demand import Stage6Demand
from refund_agent.oversight.human_oversight_risk_design import QueueDesign
from refund_agent.release.candidate import build_candidate
from refund_agent.release.suite import Suite, build_suite
from refund_agent.renewal.future_directions_adopt import AdoptionPorts, adopt
from refund_agent.renewal.future_directions_candidate import recalibrate, selected_score
from refund_agent.renewal.future_directions_decide import Decision, decide
from refund_agent.renewal.future_directions_evidence import Evidence, evaluate
from refund_agent.renewal.future_directions_math import (
    calibrated,
    calibration_power,
    false_reversals,
    wilson_interval,
)
from refund_agent.renewal.future_directions_requirement import (
    BundleIdentity,
    ReplacementRequirement,
    at_day,
)
from refund_agent.renewal.future_directions_watch import BudgetStatus, WatchStore, watch
from refund_agent.renewal.future_directions_windows import SPACING, Window, generate_forward
from refund_agent.selection import fingerprint

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / ".tmp-sol/future-directions-final-run"
EXERCISES = ROOT / "companion/exercises/future-directions"


def write_record(directory, name, value):
    path = directory / name
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if path.exists() and path.read_text() != text:
        raise ValueError(f"stored run differs at {name}; use a fresh run directory")
    if not path.exists():
        path.write_text(text)


@dataclass(frozen=True)
class RenewalInputs:
    """Spine assembly supplies actual records/clients; no producer is emulated.

    capacity_demand is chapter 37's signed Stage6Demand; demand(window) supplies
    all pre-existing review arrivals for that period. check_design is chapter
    37's check_design(design, demand, weekends=...).
    release binds chapter 29's evaluate_release/sign/journal to this suite.
    AdoptionPorts binds chapter 32 run_rehearsal and chapter 33's commands;
    its on_promotion consumes chapter 39's held state and the new promotion.
    budget_status/withdraw/rollback bind chapters 34/38/33 respectively.
    """

    incumbent_suite: Suite
    bundle_identity: Callable
    admitted_targets: tuple[dict, ...]
    launch_state: dict
    check_design: Callable
    capacity_design: QueueDesign
    capacity_demand: Stage6Demand
    demand: Callable
    release: Callable
    adoption_ports: AdoptionPorts
    registry: object
    budget_status: Callable
    withdraw: Callable
    rollback: Callable


def store_window(directory, window):
    write_record(
        directory,
        window.window_id + ".json",
        {
            "window_id": window.window_id,
            "start": window.start.isoformat(),
            "seed": window.seed,
            "paid_ids": window.paid_ids,
            "rows": [r.model_dump(mode="json") for r in window.rows],
            "fingerprint": window.fingerprint,
        },
    )


def adoption_records(req, selected, ev):
    selection = json.loads(
        (ROOT / "companion/book_outputs/_fnd.automl-and-selection.record-full-01.txt").read_text()
    )
    if (
        selection["sha256"] != selected.selection_fingerprint
        or fingerprint({k: v for k, v in selection.items() if k != "sha256"})
        != selected.selection_fingerprint
    ):
        raise ValueError("selection record changed since its sealed rebuild")
    common = {"requirement_fingerprint": req.fingerprint, "passed": True}
    return {
        "selection": {**common, "fingerprint": selected.selection_fingerprint, "record": selection},
        "calibration": {
            **common,
            "evidence_fingerprint": ev.fingerprint,
            "window_fingerprint": ev.window_fingerprint,
            "interval": ev.interval,
            "score_fingerprint": ev.score_fingerprint,
            "first_day": ev.first_day,
            "last_day": ev.last_day,
            "mature_day": ev.closed_day,
            "prediction": ev.prediction,
            "threshold_band": ev.threshold_band,
        },
        "signed_cost": {
            **common,
            "fingerprint": req.cost_fingerprint,
            "signer": req.cost_signer,
            "matrix": req.cost_matrix,
            "review_cents": req.review_cents,
            "signed_day": req.signed_day,
        },
        "signed_decline": {
            **common,
            "signer": req.experiment_signer,
            "decision": req.experiment,
            "signed_day": req.signed_day,
            "assumption": req.experiment_assumption,
        },
        "rebuild_check": {
            **common,
            "sealed_receipt": selected.sealed_receipt,
            "selected_score_fingerprint": selected.rebuild_fingerprint,
            "fit_window_fingerprint": ev.fit_window_fingerprint,
        },
    }


def watched_month(directory, component, adoption, req, inputs):
    start = at_day(adoption.day)
    generated = generate_forward(req.seed, start, req.paid_per_window)
    month = [r for r in generated if r.amount_cents < 5000 and r.at < start + timedelta(days=30)]
    predictions = component.predict(month)
    selected = [
        (r, float(p), r.paid and p < component.threshold) for r, p in zip(month, predictions)
    ]
    # Inspect all payment times before reading any paid outcome.
    final = max(r.label_final_at for r, _, paying in selected if paying)
    day = (final - END_AT).total_seconds() / 86400
    rows = [
        SimpleNamespace(
            ordinal=i,
            request_id=r.request_id,
            paid=bool(paying),
            prediction=p,
            paid_day=(r.at - END_AT).total_seconds() / 86400 if paying else None,
            label_final_day=(r.at - END_AT).total_seconds() / 86400 + 30 if paying else None,
            found_false_30d=r.found_false_30d if paying else None,
        )
        for i, (r, p, paying) in enumerate(selected, 1)
    ]
    bound = {**adoption.model_dump(mode="json"), "fingerprint": adoption.fingerprint}
    store = WatchStore(
        directory / "future-directions-watch.jsonl", adoption.fingerprint, req.fingerprint
    )
    budget = BudgetStatus.model_validate(inputs.budget_status(bound))
    result = watch(
        rows,
        bound,
        req,
        day,
        store,
        period="month-01",
        budget=budget,
        withdraw=inputs.withdraw,
        rollback=inputs.rollback,
    )
    paid = [r for r in rows if r.paid]
    wrong = sum(r.found_false_30d for r in paid)
    record = {
        "period": "month-01",
        "day": day,
        "scored": len(rows),
        "paid": len(paid),
        "wrong": wrong,
        "prediction": sum(r.prediction for r in paid) / len(paid),
        "interval": wilson_interval(wrong, len(paid)),
        "requests": [vars(r) for r in rows],
    }
    write_record(directory, "future-directions-watched-month.json", record)
    return result.model_dump(mode="json"), record


def run(directory, *, inputs: RenewalInputs | None = None):
    directory = Path(directory)
    if inputs is not None and (directory / "future-directions-requirement.json").exists():
        raise ValueError(
            "partner integration requires a fresh pre-evidence requirement/run directory"
        )
    power = calibration_power()
    annual = false_reversals(power["correct_score_failure"])
    write_record(directory, "future-directions-power.json", {"power": power, "annual": annual})
    selected = selected_score()
    base_suite = build_suite() if inputs is None else inputs.incumbent_suite
    incumbent = build_candidate(base_suite)
    policy = _policies()[-1].version
    missing = (
        {}
        if inputs is not None
        else {
            "incumbent-admission": "Capstone Blueprint B6 launch record and MLOps and ML Engineering stage-6 registry absent; core text-rule comparator only",
            "capacity": "Human Oversight and Risk Management check_design and its demand/capacity design absent",
            "release": "Future Directions selected-component signed release record absent; fixture authority integration pending",
            "promotion": "MLOps and ML Engineering requirements/registry command absent; signed decline acceptance unverified",
            "withdrawal": "Organizational Adoption stage-6 withdrawal client absent",
            "launch-record": "Capstone Blueprint held-launch record and on_promotion absent",
            "budget": "Observability and Reliability reloadable status and registered budget_spent callback absent",
        }
    )

    def candidate_suite(component):
        return build_suite(
            trials=base_suite.trials,
            requirement=base_suite.requirement,
            components=(*base_suite.components, selector(component, policy_version=policy)),
        )

    def bundle_identity(candidate):
        if inputs is None:
            return None
        identity = BundleIdentity.model_validate(inputs.bundle_identity(candidate))
        if identity.candidate_fingerprint != candidate.fingerprint:
            raise ValueError("version bundle references another chapter 29 candidate")
        return identity.bundle_fingerprint

    incumbent_bundle = bundle_identity(incumbent)
    try:
        suite = candidate_suite(selected.component)
        candidate = build_candidate(suite)
        candidate_fingerprint = candidate.fingerprint
        candidate_bundle = bundle_identity(candidate)
    except ValueError as error:
        suite, candidate_fingerprint, candidate_bundle = None, None, None
        missing["component-selection"] = str(error)
    req = ReplacementRequirement(
        incumbent_bundle_fingerprint=incumbent_bundle,
        incumbent_candidate_fingerprint=incumbent.fingerprint,
        candidate_bundle_fingerprint=candidate_bundle,
        candidate_fingerprint=candidate_fingerprint,
        policy_version=policy,
        selection_fingerprint=selected.selection_fingerprint,
        sealed_receipt=selected.sealed_receipt,
        score_fingerprint=selected.component.fingerprint,
        reversal_target=incumbent_bundle,
        first_request_day=SPACING.total_seconds() / 86400,
    )
    # A fresh requirement is persisted before either forward window exists. A
    # previous missing-input run is never upgraded after its evidence was read.
    write_record(directory, "future-directions-requirement.json", req.model_dump(mode="json"))
    admitted = () if inputs is None else inputs.admitted_targets
    if inputs is not None and (
        inputs.launch_state["bundle_fingerprint"] != req.incumbent_bundle_fingerprint
        or inputs.launch_state["candidate_fingerprint"] != req.incumbent_candidate_fingerprint
    ):
        admitted = ()
        missing["launch-record"] = "held launch names another incumbent"

    def evaluate_window(window, component, suite, **rebuild):
        release_record, journal = None, None
        candidate_bundle = None if suite is None else bundle_identity(build_candidate(suite))
        if inputs is not None and suite is not None:
            try:
                release_record, journal = inputs.release(
                    suite=suite, component=component, day=window.mature_at, directory=directory
                )
            except (ValueError, FileNotFoundError) as error:
                missing["release-" + window.window_id] = str(error)
        return evaluate(
            [r for r in window.rows if r.paid],
            component,
            incumbent.fingerprint,
            req,
            window.mature_at,
            window=window,
            expected_start=window.start,
            admitted_targets=admitted,
            check_design=None if inputs is None else inputs.check_design,
            capacity_design=None if inputs is None else inputs.capacity_design,
            capacity_demand=None if inputs is None else inputs.capacity_demand,
            capacity_arrivals=None if inputs is None else inputs.demand(window),
            release_record=release_record,
            suite=suite,
            journal=journal,
            candidate_bundle_fingerprint=candidate_bundle,
            **rebuild,
        )

    window = Window.generate("future-directions-window-01", req.seed, END_AT, req.paid_per_window)
    store_window(directory, window)
    ev = evaluate_window(window, selected.component, suite)
    decision = decide(req, ev, 1)
    write_record(directory, "future-directions-evidence-01.json", ev.model_dump(mode="json"))
    write_record(directory, "future-directions-decision-01.json", decision.model_dump(mode="json"))
    ev2, decision2, rebuilt = None, None, None
    if decision.verdict == "rebuild":
        rebuilt = recalibrate(
            selected.component, [r for r in window.rows if r.paid], evaluated_at=window.mature_at
        )
        try:
            suite2 = candidate_suite(rebuilt)
        except ValueError as error:
            missing["rebuilt-component-selection"] = str(error)
        else:
            window2 = Window.generate(
                "future-directions-window-02", req.seed, at_day(decision.day), req.paid_per_window
            )
            store_window(directory, window2)
            ev2 = evaluate_window(
                window2,
                rebuilt,
                suite2,
                first_decision=decision,
                first_evidence=ev,
                fit_window=window,
                original_score=selected.component,
            )
            decision2 = decide(req, ev2, 2, decision)
            write_record(
                directory, "future-directions-evidence-02.json", ev2.model_dump(mode="json")
            )
            write_record(
                directory, "future-directions-decision-02.json", decision2.model_dump(mode="json")
            )
    final_decision, final_evidence = (decision2, ev2) if decision2 is not None else (decision, ev)
    records = (
        adoption_records(req, selected, final_evidence)
        if final_decision.verdict == "replace"
        else {}
    )
    write_record(directory, "future-directions-adoption-records.json", records)
    adoption = adopt(
        final_decision,
        req,
        None if inputs is None else inputs.registry,
        "morgan",
        final_decision.day + 1,
        records=records,
        ports=None if inputs is None else inputs.adoption_ports,
        launch_state=None if inputs is None else inputs.launch_state,
    )
    write_record(directory, "future-directions-adoption.json", adoption.model_dump(mode="json"))
    watched, month = None, None
    if adoption.verdict in {"adopted", "adopted-launch-pending"}:
        watched, month = watched_month(
            directory, rebuilt or selected.component, adoption, req, inputs
        )
        write_record(directory, "future-directions-watch-result.json", watched)
    for name, reason in missing.items():
        write_record(directory, "missing/future-directions-" + name + ".json", {"reason": reason})
    result = {
        "power": power,
        "annual": annual,
        "requirement": req.model_dump(mode="json"),
        "evidence": ev.model_dump(mode="json"),
        "decision": decision.model_dump(mode="json"),
        "evidence2": None if ev2 is None else ev2.model_dump(mode="json"),
        "decision2": None if decision2 is None else decision2.model_dump(mode="json"),
        "rebuilt_score_fingerprint": None if rebuilt is None else rebuilt.fingerprint,
        "adoption": adoption.model_dump(mode="json"),
        "watch": watched,
        "watched_month": month,
        "timing": {
            "adoption_day": adoption.day if watched is not None else None,
            "signature_day": req.signed_day,
        },
        "missing": missing,
    }
    # Use JSON values in memory too, so read-run is byte-identical in both paths.
    result = json.loads(json.dumps(result))
    write_record(directory, "future-directions-result.json", result)
    return result


def load(directory):
    result = json.loads((Path(directory) / "future-directions-result.json").read_text())
    req = ReplacementRequirement.model_validate(result["requirement"])
    ev = Evidence.model_validate(result["evidence"])
    decision = Decision.model_validate(result["decision"])
    if (
        ev.requirement_fingerprint != req.fingerprint
        or decision.evidence_fingerprint != ev.fingerprint
    ):
        raise ValueError("stored run identity differs")
    if decide(req, ev, 1) != decision:
        raise ValueError("stored decision differs from recomputation")
    if result["decision2"] is not None:
        ev2 = Evidence.model_validate(result["evidence2"])
        decision2 = Decision.model_validate(result["decision2"])
        if decide(req, ev2, 2, decision) != decision2:
            raise ValueError("stored second decision differs from recomputation")
    return result


def write_fixtures(directory=EXERCISES):
    """A hypothetical admitted replacement; exactly one invalid reversal signal."""
    fixture: dict = {
        "constructed": True,
        "verdict": "replace",
        "stage": 6,
        "signed_day": 0,
        "first_request_day": 1,
        "rebuild": {"allowed": 1, "after": "calibration alone", "judged_on": "window 2"},
        "evidence": {
            "calibration": "pass",
            "cost_mean": -0.20,
            "cost_upper_95": -0.05,
            "capacity": "pass",
            "wrongful_denials": [0, 0],
            "release": "signed pass",
        },
        "reversal": {
            "signal": "monthly paired cost against incumbent",
            "owner": "decision_owner",
            "target": "B6 admitted at stage 6",
            "mechanism": "watch and rehearsed rollback",
            "confirmation": "next 2,000 requests, persisted boundaries and paid-only outcomes",
            "budget": "immediate withdrawal to shadow/person, independent of rollback",
        },
    }
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "decision_05.json").write_text(
        json.dumps(fixture, sort_keys=True, indent=2) + "\n"
    )
    fixture["reversal"]["signal"] = "dated paid-only calibration with persisted confirmation"
    (directory / "future_directions_reference_05.json").write_text(
        json.dumps(fixture, sort_keys=True, indent=2) + "\n"
    )


def print_evidence(result, attempt=1):
    ev = result["evidence" if attempt == 1 else "evidence2"]
    decision = result["decision" if attempt == 1 else "decision2"]
    print_wrapped(
        f"Window {attempt}: {ev['paid']:,} paid requests; {ev['wrong']} wrong payments; no payment made by this comparison"
    )
    print_wrapped(
        f"Observed {ev['wrong'] / ev['paid']:.6f}; Wilson 95% [{ev['interval'][0]:.6f}, {ev['interval'][1]:.6f}]"
    )
    print_wrapped(f"Mean prediction {ev['prediction']:.6f}")
    mean, low, high = ev["difference"]
    print_wrapped(f"Candidate minus incumbent ${mean:.6f}/request; 95% [${low:.6f}, ${high:.6f}]")
    print_wrapped(
        f"Referrals {ev['referrals']}; added blocking review minutes {ev['added_minutes']}; wrongful denials 0 versus 0"
    )
    for check in ev["checks"]:
        print_wrapped(f"{check['name']}: {'pass' if check['passed'] else 'fail'}; {check['detail']}")
    print_wrapped(
        f"Days: first {ev['first_day']:.6f}; last {ev['last_day']:.6f}; duration {ev['last_day'] - ev['first_day']:.6f}; last outcome closed {ev['closed_day']:.6f}"
    )
    print_wrapped(
        f"Decision {attempt}: {decision['verdict']}; reasons {', '.join(decision['reasons'])}; day {decision['day']:.6f}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path(os.environ.get("FUTURE_DIRECTIONS_RUN_DIRECTORY", str(RUN))),
    )
    parser.add_argument("--read-run", action="store_true")
    parser.add_argument("--write-fixtures", action="store_true")
    args = parser.parse_args()
    result = load(args.run_dir) if args.read_run else run(args.run_dir)
    if args.write_fixtures:
        write_fixtures()


# region book:fnd.future-directions.requirement-01 output
if __name__ == "__main__":
    print(ReplacementRequirement.model_validate(result["requirement"]).as_table())
# endregion book:fnd.future-directions.requirement-01

# region book:fnd.future-directions.decision-01 output
if __name__ == "__main__":
    print_evidence(result)
# endregion book:fnd.future-directions.decision-01

# region book:fnd.future-directions.decision-02 output
if __name__ == "__main__":
    if result["decision2"] is None:
        print("Window 2: not generated; decision 1 " + result["decision"]["verdict"] + ".")
        print("No second candidate was judged and no replace verdict was issued.")
    else:
        print("First failure retained: " + ", ".join(result["decision"]["reasons"]))
        print_evidence(result, 2)
# endregion book:fnd.future-directions.decision-02

# region book:fnd.future-directions.adopt-01 output
if __name__ == "__main__":
    from demos.output_wrap import print_wrapped

    print_wrapped("Adoption: " + result["adoption"]["verdict"] + "; " + result["adoption"]["reason"])
    a = result["adoption"]
    if a["verdict"] in {"adopted", "adopted-launch-pending"}:
        print_wrapped(
            f"Stage {a['stage']}; release {a['release_record']['decision']}; rehearsal {a['rehearsal_record']['verdict']}; day {a['day']:.6f}"
        )
        print_wrapped("Records: " + ", ".join(a["records"]))
        print_wrapped("Launch gate: " + str(a["launch_verdict"]))
        if a["launch_record"] is not None:
            failures = a["launch_record"]["failures"]
            print_wrapped("Held/reopened rows: " + json.dumps(sorted({f["row"] for f in failures})))
            for failure in failures:
                print_wrapped(
                    f"Row {failure['row']}: {failure['kind']}/{failure['check']}; "
                    + failure["reason"]
                )
        print_wrapped(
            f"Watch {a['watch_id']}; owner {result['requirement']['reversal_owner']}; target bundle {a['incumbent_bundle_fingerprint'][:16]}; candidate {a['incumbent_candidate_fingerprint'][:16]}"
        )
        print_wrapped(
            f"Days from signature to adoption: {a['day'] - result['requirement']['signed_day']:.6f}"
        )
    else:
        print_wrapped("No stage-6 promotion, launch-gate rerun or watch registration occurred.")
        print_wrapped("Days from signature to adoption: unavailable; adoption did not occur.")
    for name, reason in sorted(result["missing"].items()):
        print_wrapped(name + ": " + reason)
# endregion book:fnd.future-directions.adopt-01

# region book:fnd.future-directions.watch-01 output
if __name__ == "__main__":
    if result["watch"] is None:
        print("First month after adoption: unavailable; adoption did not occur.")
        print("No post-adoption outcome or no-reversal verdict is claimed.")
    else:
        w = result["watched_month"]
        print(
            f"First month: {w['scored']} scored; {w['paid']} paid; {w['wrong']} wrong; matured day {w['day']:.6f}"
        )
        print(
            f"Mean prediction {w['prediction']:.6f}; Wilson 95% [{w['interval'][0]:.6f}, {w['interval'][1]:.6f}]"
        )
        print("Watch: " + result["watch"]["verdict"] + "; " + result["watch"]["detail"])
# endregion book:fnd.future-directions.watch-01

# region book:fnd.future-directions.power-01 output
if __name__ == "__main__":
    p = result["power"]
    print(f"2,000 paid requests; prediction {p['prediction']:.3f}; actual {p['actual']:.3f}")
    print(
        f"First high rejection {p['first_high_rejection']}; Wilson lower {p['wilson_at_cut'][0]:.9f}"
    )
    print(f"Power {p['power']:.9f}; correct-score failure {p['correct_score_failure']:.9f}")
# endregion book:fnd.future-directions.power-01

# region book:fnd.future-directions.confirmation-01 output
if __name__ == "__main__":
    a = result["annual"]
    print("Constructed: independent homogeneous all-paid monthly batches of 2,000.")
    print("Confirmation consumes the next month's separate batch; twelve batches, start clear.")
    print(f"Yearly false reversal: single {a['single']:.9f}; confirmation {a['confirmation']:.9f}")
    print(
        f"Pending at year end {a['pending']:.9f}; heterogeneous paid subsets require other probabilities."
    )
# endregion book:fnd.future-directions.confirmation-01

# region book:fnd.future-directions.threshold-01 output
if __name__ == "__main__":
    print("Constructed counterexample: two equal bands of 1,000 paid requests.")
    print("Low band: predicted .020, observed .060; threshold band: predicted .120, observed .080.")
    print(
        f"Aggregate .070: {'pass' if calibrated(0.070, 140, 2000) else 'fail'}; threshold band: {'pass' if calibrated(0.120, 80, 1000) else 'fail'}"
    )
    b = result["evidence"]["threshold_band"]
    print(
        f"Window 1 risk band [.060, .100]: {b['paid']} paid; {b['wrong']} wrong; mean prediction {b['mean_prediction']:.6f}"
    )
    print(
        f"Wilson 95% [{b['interval'][0]:.6f}, {b['interval'][1]:.6f}]; aggregate check alone cannot validate the operating threshold."
    )
# endregion book:fnd.future-directions.threshold-01

# region book:fnd.future-directions.keys-01 output
if __name__ == "__main__":
    low, high = wilson_interval(128, 2000)
    print(f"Exercise 3a: 128/2,000 = .064; Wilson [{low:.6f}, {high:.6f}]; .051 outside: fail.")
    print("Exercise 3b/c: cost upper +$0.09; keep at either attempt; two checks failed.")
    print(
        f"Exercise 3d: 0/40 Wilson [0, {wilson_interval(0, 40)[1]:.6f}]; .05 inside: pass, low power."
    )
    print(
        f"Exercise 4: {380 - 250} fewer late minutes ({(380 - 250) / 380:.6%}); 40 added minutes on six routes."
    )
    print(
        "Exercise 5: paired post-adoption cost is unobservable; adoption held until the signal is repaired."
    )
    if result["watch"] is None:
        print(
            "Exercise 6: no adoption-month rerun available; constructed reference tests cover all four confirmation transitions."
        )
    else:
        print(
            "Exercise 6: adopted-month reference check "
            + result["watch"]["verdict"]
            + "; persisted confirmation state."
        )
# endregion book:fnd.future-directions.keys-01
