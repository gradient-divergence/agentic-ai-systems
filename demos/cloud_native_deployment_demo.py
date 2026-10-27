"""Chapter 32's constructed windows, canary and rollback evidence."""

import json
from datetime import datetime, timedelta, timezone
from fractions import Fraction
from functools import lru_cache
from pathlib import Path
from textwrap import fill
from uuid import uuid4

from refund_agent.release.candidate import build_candidate
from refund_agent.release.rule import sign
from refund_agent.release.suite import build_suite, run_case

# isort: split
# Initialize the core before importing extension readers.
CORE = build_candidate(build_suite())

from refund_agent.batch_stream.component import selector as event_selector
from refund_agent.canary.cloud_native_deployment_component import (
    JOURNAL_PATH,
    STORE_PATH,
    run_release,
    selector,
)
from refund_agent.canary.cloud_native_deployment_exposure import (
    clean_hours,
    exposure,
    further_requests,
    loss_cents,
)
from refund_agent.canary.cloud_native_deployment_plan import CanaryPlan, in_cohort, sign_plan
from refund_agent.canary.cloud_native_deployment_rehearsal import run_rehearsal
from refund_agent.canary.cloud_native_deployment_start import start_canary
from refund_agent.canary.cloud_native_deployment_stop import CanarySession
from refund_agent.canary.cloud_native_deployment_store import AppendJournal, AppendStore
from refund_agent.canary.cloud_native_deployment_window import run_window, sign_window
from refund_agent.data.live import generate_live
from refund_agent.sources.data_systems_release import SELECTOR as DATA

ROOT = Path(__file__).resolve().parents[1]
START = datetime(2026, 10, 1, tzinfo=timezone.utc)


def candidates(*, planted_field=False, reader="closed"):
    selections = [DATA, event_selector(), selector(
        record_schema="canary-id" if planted_field else "closed",
        reader="canary-id" if planted_field else reader,
    )]
    suite = build_suite(components=selections)
    return suite, build_candidate(suite)


def plan_for(candidate, target, window_ref, signed_at, *, identifier="cloud-native-deployment-01"):
    return sign_plan(CanaryPlan(
        canary_id=identifier, candidate_fingerprint=candidate.fingerprint,
        suite_fingerprint=candidate.suite["fingerprint"], rollback_target_fingerprint=target.fingerprint,
        requirement_ref=window_ref, rehearsal_ref=identifier + "-rehearsal",
    ), "morgan", signed_at)


@lru_cache(maxsize=1)
def experiment():
    suite, candidate = candidates()
    target = build_candidate(suite, verifier="independent")
    window = run_window(candidate, START)
    if window.measures_pass:
        window = sign_window(window, "morgan", window.completed_at + timedelta(seconds=1))
    next_day = window.completed_at.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
    plan = plan_for(candidate, target, window.window_id, next_day - timedelta(seconds=1))
    stream = generate_live(7, next_day, 14, 233)
    rehearsals = [run_rehearsal(plan, stream, 14, ["forbidden_payment"], cadence, target,
                               candidate=candidate) for cadence in ("4h", "daily")]
    failures = []
    for case in suite.cases:
        if case.handler not in ("data-systems", "batch-stream-absence"):
            continue
        try:
            result = run_case(case, candidate)
            if result["failed"]:
                failures.append(case.case_id + ": " + ", ".join(result["failed"]))
        except ValueError as error:
            failures.append(case.case_id + ": " + str(error).splitlines()[0])
    release = None
    if not failures:
        release = run_release(candidate, suite)
        if release["decision"] == "pass":
            release = sign(release, AppendJournal(JOURNAL_PATH), "morgan", next_day.isoformat())
    session = CanarySession(plan, AppendStore(STORE_PATH, "cloud-native-deployment-start-" + uuid4().hex))
    started = start_canary(plan, release, window, rehearsals[0].model_dump(mode="json"),
                           candidate=candidate, suite=suite, journal=AppendJournal(JOURNAL_PATH),
                           session=session, owner="morgan", at=next_day)
    planted_suite, planted = candidates(planted_field=True)
    planted_target = build_candidate(suite)
    planted_plan = plan_for(planted, planted_target, "rehearsal-only", next_day,
                            identifier="cloud-native-deployment-planted-field")
    incompatible = run_rehearsal(planted_plan, stream, 14, ["forbidden_payment"], "4h",
                                 planted_target, candidate=planted)
    cohort = sum(in_cohort(plan.canary_id, r.requester_id) for r in stream.data.requests)
    return {"suite": suite, "candidate": candidate, "target": target, "window": window,
            "plan": plan, "start": started, "rehearsals": rehearsals, "incompatible": incompatible,
            "cohort": cohort, "requesters": len(stream.data.requests),
            "release_blockers": failures, "release": release,
            "planted_suite": planted_suite}


def measured_settle():
    records = json.loads((ROOT / "demos/data/batch-stream-records.json").read_text())
    final = next(row for row in records["claims"]["rows"] if row["rule"] == "final")
    return Fraction(str(final["evidence_final"]["mean_hours"]))


def _print(*values):
    text = " ".join(str(value) for value in values)
    print(fill(text, width=79, break_long_words=False, break_on_hyphens=False))


def print_window():
    row = experiment()["window"]
    _print("Constructed join: live request times; generated carrier delays/copies.")
    _print("Generated person/30-day labels stand in immediately; blind to agent.")
    _print("Frozen eligible request identifiers in arrival order:", len(row.request_ids))
    _print("Terminal decisions awaited:", len(row.completion_order))
    _print("Forbidden payments:", row.forbidden_payments)
    _print(f"Referral share: agent {row.referrals}/299; baseline {row.baseline_referrals}/299")
    _print(f"Wrongful denials: agent {row.wrongful_denials}; baseline {row.baseline_wrongful_denials}")
    for name, verdict in row.measure_verdicts.items():
        _print(name + ":", "pass" if verdict else "fail")
    _print("Measures:", "pass" if row.measures_pass else "fail")
    _print("Decision-owner signature:", row.signed_by, row.signed_at)
    _print("Deployed labels mature at:", row.labels_mature_at.isoformat())
    _print("Suite:", len(experiment()["suite"].cases), "cases; sha256:")
    _print(experiment()["suite"].fingerprint)
    _print("Candidate sha256:")
    _print(row.candidate_fingerprint)


def print_plan():
    run = experiment()
    row = run["plan"]
    _print("Constructed signed canary plan:", row.canary_id)
    _print("Cohort:", run["cohort"], "/", run["requesters"], "requesters")
    _print("Realized share:", f"{run['cohort'] / run['requesters']:.6f}")
    _print("Hash share:", row.share)
    _print("Maximum per payment:", row.max_cents_per_payment, "cents")
    _print("Named worst day:", row.worst_day_cents, "cents")
    _print("Derived daily caps:", row.daily_cap_count, "payments;", row.daily_cap_cents, "cents")
    _print("Recheck:", row.recheck_hours, "hours; time box:", row.time_box_days, "days")
    _print("Comparator: tier 1 outside cohort, same days, blind person.")
    _print("Over $500: named approver decides every request, denials included.")
    _print("Clean count:", row.clean_count_rule)
    _print("Stops:", ", ".join(row.stop_triggers))
    _print("Budget reference:", row.error_budget_ref or "not named")
    _print("Offer rule:", row.offer_rule)
    _print("Signed:", row.signed_by, row.signed_at.isoformat())


def print_start():
    run = experiment()
    for check in run["start"].checks:
        _print(check.name + ":", "pass" if check.passed else "refused")
        if not check.passed:
            _print("  " + check.detail)
    _print("Canary:", "started" if run["start"].accepted else "refused; binding remains shadow")
    for blocker in run["release_blockers"]:
        _print("Predecessor integration blocker:", blocker)
    _print("Promotion/scope checks join through selected start_checks hooks.")


def print_exposure():
    row = exposure(measured_settle())
    _print("Constructed constant-rate model; expected counts, not observations.")
    _print(f"Tier 1 rate: {row['rate']} per hour = {float(row['rate']):.6f}")
    _print(f"Normal cohort day: {row['normal_day']} = {float(row['normal_day']):.6f}")
    for arm in row["rows"]:
        _print(f"{arm['hours']}h: further {arm['further']} = {float(arm['further']):.6f}")
        _print(f"  loss <= {arm['loss_cents']} cents; outward <= {arm['outward_cents']} cents")
    _print("Daily 24h arithmetic is a counterfactual; overdue pauses at 4h.")
    _print("Any single day's hard cap: 15 payments; 74985 cents.")
    _print("25% / 24h expectation:", f"{float(row['quarter_daily_loss']):.6f} cents")
    _print(f"Measured mean final-evidence wait: {float(row['settle_hours']):.9f} hours")
    _print("Source: book_outputs/_fnd.batch-stream.claims-full-01.txt")
    for share, hours in row["clean"].items():
        _print(f"share {share}: 299 clean cases in {float(hours):.6f}h = {float(hours / 24):.6f} days")
    _print(f"28-day projected settled cases: {float(row['projection']):.9f}")
    _print(f"If 88 observed cases have zero failures: 95% bound {row['bound_88']:.9%}")
    _print("Assumptions: independent, one failure rate, target-representative cases.")
    _print(f"Rule of three: {float(row['rule_of_three']):.9%}")
    for n, probability in row["fault_seen"].items():
        _print(f"Fault at 1/1000 seen among {n}: {float(probability):.9f}")


def print_rehearsals():
    for row in experiment()["rehearsals"]:
        _print("Constructed version rollback; cadence:", row.cadence)
        _print("First normal cohort payment:", row.first_payment_cents, "cents")
        _print("Planted forbidden payment, counted apart:", row.triggering_payment_cents, "cents")
        _print("Further payments before stop (trigger excluded):", row.payments_after_trigger,
               "; cents:", row.cents_after_trigger)
        _print("Exposure stops on:", row.exposure_stop_trigger, row.exposure_stop_at.isoformat())
        _print("Fault detected at:", row.detected_at.isoformat())
        _print("4h modeled further requests:", row.expected_further)
        if row.cadence == "daily":
            _print("24h counterfactual further requests: 10/3; never paid by this arm.")
        _print("Overdue-recheck pauses:", row.recheck_overdue_pauses)
        for name, verdict in row.checks.items():
            _print(name + ":", "pass" if verdict else "fail")
        for name, verdict in row.components.items():
            _print("component " + name + ":", verdict)
        _print("Policy in force:", row.policy_version)
        _print("Requests paid twice:", row.paid_twice)
        _print("Fresh restart:", row.restart_fresh_process)
        _print("Executed corrections:", row.corrections)
        for blocker in row.blockers:
            _print("Core fixture blocker:", blocker)
        _print("Verdict:", row.verdict)


def print_incompatible():
    row = experiment()["incompatible"]
    _print("Constructed later writer: canary_id on each receipt and proposal.")
    _print("Rollback target: closed receipt/proposal contracts.")
    for name, verdict in row.checks.items():
        _print(name + ":", "pass" if verdict else "fail")
    _print("Rollback record read failed:", bool(row.unreadable_records))
    _print("paid_twice:", row.paid_twice)
    _print("verdict:", row.verdict)


def print_keys():
    share, rate = Fraction(1, 20), Fraction(3600, 720)
    for hours in (6, 24):
        _print(f"Exercise 3 / {hours}h: further {further_requests(share, rate, Fraction(hours))}")
        _print("  loss cents:", loss_cents(share, rate, Fraction(hours)))
    _print("Exercise 3 / clean hours:", clean_hours(share, rate, 299, Fraction(36)))
    _print("Exercise 3 / shorten stop delay; clean time unchanged.")
    for hours in (24, 2):
        _print(f"Exercise 4 / {hours}h: {further_requests(share, Fraction(400, 24), Fraction(hours))}")
    _print("Exercise 4 / rollback restores version; bookings/texts remain.")


# region book:fnd.cloud-native-deployment.window-01 output
if __name__ == "__main__":
    print_window()
# endregion book:fnd.cloud-native-deployment.window-01

# region book:fnd.cloud-native-deployment.plan-01 output
if __name__ == "__main__":
    print_plan()
# endregion book:fnd.cloud-native-deployment.plan-01

# region book:fnd.cloud-native-deployment.start-01 output
if __name__ == "__main__":
    print_start()
# endregion book:fnd.cloud-native-deployment.start-01

# region book:fnd.cloud-native-deployment.exposure-01 output
if __name__ == "__main__":
    print_exposure()
# endregion book:fnd.cloud-native-deployment.exposure-01

# region book:fnd.cloud-native-deployment.rehearsal-01 output
if __name__ == "__main__":
    print_rehearsals()
# endregion book:fnd.cloud-native-deployment.rehearsal-01

# region book:fnd.cloud-native-deployment.rehearsal-02 output
if __name__ == "__main__":
    print_incompatible()
# endregion book:fnd.cloud-native-deployment.rehearsal-02

# region book:fnd.cloud-native-deployment.keys-01 output
if __name__ == "__main__":
    print_keys()
# endregion book:fnd.cloud-native-deployment.keys-01
