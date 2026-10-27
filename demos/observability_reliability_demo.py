"""Chapter 34's captured arithmetic, shadow control and cohort rehearsal.

Import is silent and does not write. Decisions use chapter 33's bundle stamps;
deployment history reads its registry, separately from rehearsal authority.
"""

import argparse
import json
import math
from dataclasses import asdict
from datetime import timedelta
from fractions import Fraction
from functools import lru_cache
from pathlib import Path
from textwrap import fill

from refund_agent.release.candidate import build_candidate
from refund_agent.release.suite import build_suite

CORE = build_candidate(build_suite())

# isort: split
from refund_agent.batch_stream.component import selector as event_selector
from refund_agent.canary.cloud_native_deployment_component import (
    selector as canary_selector,
)
from refund_agent.data.generate import _policies
from refund_agent.memory.register import PolicyRegister
from refund_agent.observability_reliability import budget as b
from refund_agent.observability_reliability.component import selector
from refund_agent.observability_reliability.fortnight import (
    CANARY_ID,
    DAYS,
    SEED,
    START,
    make_stream,
    run_canary_rehearsal,
    run_shadow_fortnight,
)
from refund_agent.observability_reliability.incident import reconstruct
from refund_agent.observability_reliability.objectives import (
    nearest_rank,
    share_within,
)
from refund_agent.observability_reliability.silence import threshold
from refund_agent.observability_reliability.spans import (
    DURATIONS,
    audit_coverage,
)
from refund_agent.promotion.mlops_mle_bundle import from_candidate
from refund_agent.promotion.mlops_mle_component import REGISTRY_PATH
from refund_agent.promotion.mlops_mle_component import selector as bundle_component
from refund_agent.promotion.mlops_mle_registry import Registry
from refund_agent.release.extensions import RuntimeStore
from refund_agent.sources.data_systems_release import SELECTOR

ROOT = Path(__file__).resolve().parents[1]
PREFIX = "fnd.observability-reliability."


def say(text):
    print(fill(str(text), width=79, break_long_words=False, break_on_hyphens=False))


def bundle_selector():
    register = PolicyRegister()
    policies = _policies()
    for policy in policies:
        register.announce(policy, policy.effective_at)
    return bundle_component(register.snapshot(START), [p.version for p in policies])


@lru_cache(maxsize=1)
def experiment():
    stream = make_stream()
    runs = []
    canary = canary_selector(
        store=RuntimeStore(
            "../../.tmp-sol/observability-reliability-canary/store.jsonl",
            "morgan",
            "append",
        ),
        journal=RuntimeStore(
            "../../.tmp-sol/observability-reliability-canary/journal.jsonl",
            "morgan",
            "append",
        ),
    )
    for fix in (False, True):
        suite = build_suite(
            trials=1,
            components=[
                SELECTOR,
                event_selector(),
                bundle_selector(),
                selector(region_finality=fix),
            ],
        )
        run = run_shadow_fortnight(build_candidate(suite), stream, fix=fix)
        bundle = from_candidate(run["candidate"])
        registry = Registry(REGISTRY_PATH, "observability-reliability-" + bundle.fingerprint)
        if not registry.read():
            registry.append(bundle)
        run["registry"] = registry
        run["suite"] = suite
        cohort_suite = build_suite(
            trials=1,
            components=[
                SELECTOR,
                event_selector(),
                canary,
                bundle_selector(),
                selector(region_finality=fix),
            ],
        )
        run["rehearsal"] = run_canary_rehearsal(
            build_candidate(cohort_suite),
            build_candidate(cohort_suite, verifier="independent"),
            stream,
            fix=fix,
        )
        runs.append(run)
    built, fixed = runs
    incident = reconstruct(
        built["store"].entries,
        built["audits"],
        built["lineage"],
        None,
        stream.log,
        built["corrections"],
        built["rehearsal"]["stops"],
        payments=built["payments"],
        snapshots=built["inputs"],
        window_start=START,
        window_end=START + timedelta(days=DAYS),
        budget_status=built["monitor"].status(),
        cohort_payments=built["rehearsal"]["payments"],
        cohort_corrections=built["rehearsal"]["corrections"],
        cohort_spans=built["rehearsal"]["store"].entries,
        registry=built["registry"],
    )
    return built, fixed, incident


def print_budget():
    requests, payments = Fraction(5600, 3), 1120
    timely, corrected = b.budget(0.02, requests), b.budget(0.01, payments)
    print("Constructed forecast: all tier 1 requests, 2000/30 days; window 28 days")
    print(f"requests {float(requests):.6f}; payments {payments}; paid share 0.600000")
    print(f"allowances timeliness {float(timely):.6f}; corrections {float(corrected):.6f}")
    print(f"cohort requests {float(requests * 0.05):.6f}; payments 56; correction allowance .56")
    print(f"600 requests: 9 system referrals + 3 late = 12/600; SLI {1 - 12 / 600:.6f}")
    print(f"cohort half-rate overspend {float(1 - b.within_budget(56, 0.56, 0.005)):.6f}")
    print(f"cohort double-rate within {float(b.within_budget(56, 0.56, 0.02)):.6f}")
    print(f"all half-rate within {float(b.within_budget(1120, corrected, 0.005)):.6f}")
    print(f"all double-rate within {float(b.within_budget(1120, corrected, 0.02)):.6f}")
    print(f"target overspend timely {float(1 - b.within_budget(1867, timely, 0.02)):.6f}")
    print(f"target overspend corrections {float(1 - b.within_budget(1120, corrected, 0.01)):.6f}")
    print(f"quarter-late burn {float(b.burn_rate(1, 4, 0.02)):.6f}")
    for intensity in (Fraction(25, 9), Fraction(50, 9)):
        print(
            f"actual eligible/hour {float(intensity):.6f}; untouched lifetime "
            f"{float(b.remaining_lifetime(timely, intensity, 0.25)):.2f} hours"
        )
    print("Lifetime assumes fixed actual traffic and bad share; forecast match only")
    for hours in (720, 672):
        print(
            f"{hours}-hour burn thresholds "
            + ", ".join(
                f"{float(b.alert_threshold(s, hours, w)):.6f}"
                for s, w in ((0.02, 1), (0.05, 6), (0.1, 72))
            )
        )
    print(f"hourly bad-answer trigger {float(0.02 * timely):.6f}")
    print(
        f"hourly false-page chance {float(b.binom_tail(1, 3, 0.02)):.6f}; "
        f"daily pages {float(24 * b.binom_tail(1, 3, 0.02)):.6f}"
    )
    for n, p, windows in ((17, 0.02, 112), (40, 0.01, 28)):
        k = b.min_count(n, p, 0.01)
        print(
            f"n={n}, rho={p}: k*={k}; tails K>=2 {float(b.binom_tail(2, n, p)):.6f}, "
            f"K>=3 {float(b.binom_tail(3, n, p)):.6f}"
        )
        print(
            f"{windows} disjoint windows: expected crossings "
            f"{float(windows * b.binom_tail(k, n, p)):.6f}"
        )
    print(
        f"40 timely requests: K>=3 tail {float(b.binom_tail(3, 40, 0.02)):.6f}; "
        f"k* {b.min_count(40, 0.02, 0.01)}"
    )
    print(
        f"40 payments: every-event page {100 * float(b.every_event_false_page(40, 0.01, 0.01)):.6f}%"
    )
    print(f"40 payments: closed-window page {100 * float(b.binom_tail(3, 40, 0.01)):.6f}%")
    print(
        f"fault detection: quarter-late {float(b.binom_tail(3, 17, 0.25)):.6f}; "
        f"tenth-corrected {float(b.binom_tail(3, 40, 0.1)):.6f}"
    )
    print(
        f"slow ticket (4/200): target {float(b.binom_tail(4, 200, 0.02)):.6f}; "
        f"half-target {float(b.binom_tail(4, 200, 0.01)):.6f}"
    )
    durations = (2,) * 99 + (2100,)
    print(
        f"100-run latency: mean {sum(durations) / 100:.2f}s; "
        f"p99 {nearest_rank(durations, 0.99)}s; over 30s {1 - share_within(durations, 30):.2%}"
    )
    print(f"two hangs p99 {nearest_rank((2,) * 98 + (2100,) * 2, 0.99)}s")
    print(f"illustrative review: 400 cents * 2/15 = {400 * 2 / 15:.6f} cents/request")
    print("Signed forecast spend and final realized ratio are separate checks")


def print_trace():
    built, _, _ = experiment()
    stream = built["stream"]
    request = next(r for r in stream.requests if r.request_id == stream.silence.request_id)
    event = next(e for e in stream.log.entries if e.event_id == stream.silence.event_id)
    cohort = built["rehearsal"]
    spans = cohort["store"].trace(f"trace-{request.request_id}")
    roots = sorted((s for s in spans if s.name == "run"), key=lambda s: s.start)
    print("Constructed cohort rehearsal; payments enter the simulated ledger")
    print(
        f"request {request.request_id}; {request.amount_cents} cents; claim {request.at.isoformat()}"
    )
    print(
        f"delivery scan {event.at.isoformat()}; "
        f"claim after scan {(request.at - event.at).total_seconds() / 3600:.6f} hours"
    )
    print(f"second run wait {(roots[1].start - roots[0].start).total_seconds() / 3600:.6f} hours")
    payment = next(
        p for p in cohort["payments"] if p.receipt.request.request_id == request.request_id
    )
    print(
        f"executed cohort payment {payment.receipt.amount_cents} cents; "
        f"shipment age {payment.receipt.evidence.freshness['shipment']:.6f} seconds"
    )
    say(f"Virtual duration table (seconds): {DURATIONS}; no tokenizer, tokens null")
    for span in sorted(spans, key=lambda s: (s.start, s.end, s.span_id)):
        if span.name in (
            "run",
            "watermark_advance",
            "correction",
            "gate",
            "store:ledger",
        ):
            say(
                f"{span.start.isoformat()} {span.name} attempt {span.attempt} "
                f"status {span.status}; links {span.links}"
            )


def arrival_summary(incident):
    """Read complete hourly bins inside the observed no-arrival interval."""
    from collections import defaultdict
    from datetime import datetime

    bins = defaultdict(list)
    for row in incident.arrivals_by_region_hour:
        hour = datetime.fromisoformat(row["hour"])
        if incident.last_arrival < hour and hour + timedelta(hours=1) <= incident.batch_at:
            bins[row["region"]].append(row["scans"])
    silent = sum(bool(counts) and not any(counts) for counts in bins.values())
    sending = sum(any(counts) for counts in bins.values())
    if silent == 1 and sending == 2:
        return "Hourly regional arrivals: two regions keep sending while one falls silent"
    return f"Hourly regional arrivals: {sending} regions sending; {silent} silent"


def budget_summary(runs):
    """Check recorded admission fields, without issuing a start."""
    from refund_agent.canary.cloud_native_deployment_start import _budget

    passed = True
    for run in runs:
        rehearsal = run["rehearsal"]
        try:
            _budget(rehearsal["budget_admission"], rehearsal["plan"], START)
        except (ValueError, KeyError, TypeError):
            passed = False
    status = "pass" if passed else "fail"
    return f"Budget admission fields {status} chapter 32's start reader; no start issued"


def print_incident():
    built, fixed, row = experiment()
    print("Constructed tier 1 shadow control and actual cohort rehearsal")
    print(f"region {row.region}; last arrival {row.last_arrival.isoformat()}")
    print(f"batch {row.batch_at.isoformat()}; {row.batch_scans} distinct scans")
    print(f"observed no-arrival gap {row.observed_gap_hours:.6f} hours")
    print(
        f"shadow-control corrections {row.corrections}; "
        f"rehearsed cohort corrections {row.cohort_corrections}"
    )
    print(
        f"affected requests {len(row.affected_decisions)}; "
        f"held past batch {row.held_after_batch}; policy referrals {row.policy_referrals}"
    )
    print(
        f"watermark lag range {row.watermark_lag_hours[0]:.6f}..{row.watermark_lag_hours[1]:.6f} hours"
    )
    print(f"late scans share {row.late_share:.6%}")
    say(f"corrected receipts' shipment ages seconds: {row.shipment_ages_seconds}")
    print(
        f"cash loss {row.cash_loss_cents} cents; "
        f"hypothetical shadow exposure {row.shadow_exposure_cents} cents"
    )
    print(f"illustrative review cost {row.illustrative_review_cost_cents} cents")
    for name, row_status in row.budget_status["objectives"].items():
        print(
            f"{name}: {row_status['bad']} bad / {row_status['eligible']} eligible; "
            f"allowance {row_status['signed_allowance']}; pending {row_status['pending']}"
        )
    print(f"bundle count {len(row.bundles)}; unstamped lineage {row.unstamped_lineage}")
    print(f"registry promotions/rollbacks during the fortnight: {row.releases}")
    d6 = row.budget_status["d6"]
    say(f"D6 unavailable: {d6['reason']}")
    print(
        f"cohort exposure {row.cohort_exposure} payments; "
        f"{row.cohort_exposure_cents} cents, including the trigger"
    )
    for stop in row.cohort_stops:
        print(
            f"cohort stop {stop['at']}: {stop['trigger']} {stop['outcome']}; "
            f"further payments {stop['payments_after_trigger']}"
        )
    print(arrival_summary(row))
    print(budget_summary((built, fixed)))
    print("Unknown: batch completeness and cause of silence")
    for action in row.actions:
        say(f"Action: {action['owner']}: {action['action']}")


def print_objectives():
    built, fixed, _ = experiment()
    print("Constructed shadow decisions by received day; payments by payment day")
    for name, run in (("as-built", built), ("fixed", fixed)):
        for day in range(DAYS):
            start, end = (
                START + timedelta(days=day),
                START + timedelta(days=day + 1),
            )
            rows = [o for o in run["observations"] if start <= o.received_at < end]
            late = sum(o.decided_at > o.deadline for o in rows)
            system = sum(
                o.reason
                not in (
                    "eligible",
                    "already_delivered",
                    "outside_return_window",
                    "delivery_unknown",
                    "amount_over_policy",
                )
                for o in rows
            )
            paid = [p for p in run["payments"] if start <= p.receipt.at < end]
            ids = {(r.mode, r.payment_id) for r in run["corrections"]}
            bad = sum((p.mode, p.receipt.receipt_id) in ids for p in paid)
            print(
                f"{name:8} day {day + 1:2}: requests {len(rows):2}; late {late}; "
                f"system {system}; payments {len(paid):2}; corrected {bad}"
            )


def print_detector():
    built, _, incident = experiment()
    rate = 400 / 72
    print(f"Constructed: three regions, {rate:.6f} scans/hour each; window 672 hours")
    print(f"tolerance .1/region/window: threshold {threshold(rate, 672, 0.1):.6f} hours")
    for quiet in (1, 2):
        print(
            f"quiet {quiet} hours: expected false pages {rate * 672 * math.exp(-rate * quiet):.6f}"
        )
    print(f"last-arrival page {built['page'].window_end.isoformat()}")
    for region in range(3):
        count = sum(row["region"] == region for row in built["pages"])
        print(f"region {region}: replayed pages {count}")
    print(f"observed gap {incident.observed_gap_hours:.6f} hours; one page per gap")
    print(f"regional stale-age alternative: {2000 / 30 / 3 * 48 / 24:.6f} referrals (forecast)")


def print_fix():
    built, fixed, _ = experiment()
    wrong = [
        p
        for p in built["payments"]
        if any(r.payment_id == p.receipt.receipt_id for r in built["corrections"])
    ]
    page = built["page"].window_end
    lead = (min(p.receipt.at for p in wrong) - page).total_seconds() / 3600 if wrong else None
    print("Constructed regional finality: shadow control and cohort rehearsal")
    print(f"detector page {page.isoformat()}; lead to first shadow payment {lead:.6f} hours")
    print(
        f"corrections {len(fixed['corrections'])}; late answers "
        f"{sum(o.decided_at > o.deadline for o in fixed['observations'])}"
    )
    print(
        f"fixed cohort corrections {fixed['rehearsal']['record'].corrections}; "
        f"state {fixed['rehearsal']['record'].final_outcome}; "
        f"stops {len(fixed['rehearsal']['stops'])}"
    )
    for run in (built, fixed):
        print(
            f"fix={run['fix']}: requests {len(run['observations'])}; "
            f"shadow payments {len(run['payments'])}; "
            f"timeliness bad {run['monitor'].status()['objectives']['timeliness']['bad']}"
        )


def print_loss():
    built, _, _ = experiment()
    spans = built["store"].entries
    kept = tuple(s for i, s in enumerate(spans, 1) if i % 50)
    coverage = audit_coverage(kept, built["audits"])
    print(f"Constructed sink drops one in 50: {len(spans) - len(kept)} / {len(spans)} spans")
    print(
        f"audit coverage {coverage['covered']}/{coverage['audits']}; "
        f"missing {len(coverage['missing'])}; pass {coverage['pass']}"
    )


def print_sampling():
    print("Independent retention of distinct traces; not jointly selected traces")
    for m in (3, 10):
        print(f"p_keep=0.1, traces={m}: chance at least one {float(1 - Fraction(9, 10) ** m):.6f}")


def print_keys():
    print("3: 3000/hour * 720 hours * .001 = 2160 bad decisions allowed")
    print(
        f"3: burn {float(b.burn_rate(2, 100, 0.001)):.1f}; hourly page at "
        f"{float(Fraction(2160) * Fraction(2, 100)):.1f} minutes"
    )
    print("3: 90 bad decisions; six-hour burn 5 < 6; 72-hour burn 5/12 < 1")
    print(f"3: spend {float(Fraction(90, 2160)):.6%}; no 6-hour page or ticket")
    print(
        f"4: threshold {b.min_count(40, 0.01, 0.01)}; fault detection "
        f"{float(b.binom_tail(3, 40, 0.1)):.6f}"
    )
    print(
        f"4: cohort 2/day threshold {b.min_count(2, 0.01, 0.01)}; "
        f"fault detection {float(b.binom_tail(2, 2, 0.1)):.6f}"
    )
    print("5: read-to-payment gap 2100s; age at payment 2250s > 1800s")
    print(f"6: open venue threshold {3600 * threshold(3600, 325, 1):.6f} seconds")
    print(f"6: daily average threshold {3600 * threshold(975, 1200, 1):.6f} seconds")


def figure_data():
    built, fixed, incident = experiment()
    trace_id = "trace-" + built["stream"].silence.request_id
    return {
        "scope": "constructed tier 1 shadow control and cohort rehearsal",
        "incident": incident.model_dump(mode="json"),
        "spans": [s.model_dump(mode="json") for s in built["rehearsal"]["store"].trace(trace_id)],
        "audit": [
            a.model_dump(mode="json")
            for a in built["rehearsal"]["audits"]
            if a.trace_id == trace_id
        ],
        "fixed_status": fixed["monitor"].status(),
        "cohort_budget_status": {
            name: run["rehearsal"]["monitor"].status()
            for name, run in (("as-built", built), ("fixed", fixed))
        },
        "detector_pages": [
            {"region": row["region"], **row["alert"].model_dump(mode="json")}
            for row in built["pages"]
        ],
        "rehearsals": {
            name: run["rehearsal"]["record"].model_dump(mode="json")
            for name, run in (("as-built", built), ("fixed", fixed))
        },
        "budget_admission": {
            name: run["rehearsal"]["budget_admission"]
            for name, run in (("as-built", built), ("fixed", fixed))
        },
        "canary_plans": {
            name: run["rehearsal"]["plan"].model_dump(mode="json")
            for name, run in (("as-built", built), ("fixed", fixed))
        },
        "plant": {
            **asdict(built["stream"].silence),
            "start": built["stream"].silence.start.isoformat(),
            "end": built["stream"].silence.end.isoformat(),
        },
        "constants": {
            "seed": SEED,
            "days": DAYS,
            "canary_id": CANARY_ID,
            "durations": DURATIONS,
        },
    }


# region book:fnd.observability-reliability.budget-01 output
if __name__ == "__main__":
    print_budget()
# endregion book:fnd.observability-reliability.budget-01
# region book:fnd.observability-reliability.trace-01 output
if __name__ == "__main__":
    print_trace()
# endregion book:fnd.observability-reliability.trace-01
# region book:fnd.observability-reliability.objectives-01 output
if __name__ == "__main__":
    print_objectives()
# endregion book:fnd.observability-reliability.objectives-01
# region book:fnd.observability-reliability.incident-01 output
if __name__ == "__main__":
    print_incident()
# endregion book:fnd.observability-reliability.incident-01
# region book:fnd.observability-reliability.detector-01 output
if __name__ == "__main__":
    print_detector()
# endregion book:fnd.observability-reliability.detector-01
# region book:fnd.observability-reliability.fix-01 output
if __name__ == "__main__":
    print_fix()
# endregion book:fnd.observability-reliability.fix-01
# region book:fnd.observability-reliability.loss-01 output
if __name__ == "__main__":
    print_loss()
# endregion book:fnd.observability-reliability.loss-01
# region book:fnd.observability-reliability.sampling-01 output
if __name__ == "__main__":
    print_sampling()
# endregion book:fnd.observability-reliability.sampling-01
# region book:fnd.observability-reliability.keys-01 output
if __name__ == "__main__":
    print_keys()
# endregion book:fnd.observability-reliability.keys-01
# region book:fnd.observability-reliability.figure-01 output
if __name__ == "__main__":
    print(json.dumps(figure_data(), sort_keys=True, indent=2))
# endregion book:fnd.observability-reliability.figure-01


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-exercises", action="store_true")
    args = parser.parse_args(argv)
    if args.write_exercises:
        from demos.observability_reliability_exercises import write_exercises

        write_exercises(experiment(), ROOT / "exercises/observability-reliability")


if __name__ == "__main__":
    main()
