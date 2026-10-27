"""Chapter 31's constructed event-time evidence; imports print and write nothing."""

import argparse
import json
import os

from demos.batch_stream_capture import key_run, write_data
from refund_agent.batch_stream.experiment import (
    claim_run,
    count_run,
    figure_run,
    json_default,
    opening_run,
    stream,
)
from refund_agent.batch_stream.generate import SETTINGS
from refund_agent.batch_stream.replay import print_replay
from refund_agent.batch_stream.view import (
    first_complete,
    late_events,
    shipment_as_of,
    watermark,
)


def identity():
    print("Constructed stream sha256:")
    print(stream().fingerprint)


def print_opening():
    run = opening_run()
    request, log, arrival = run["request"], run["log"], run["arrival"]
    identity()
    for event in log.entries:
        if event.order_id == request.order_reference:
            print(
                f"{event.status}: scan {event.at:%d %H:%M}; arrival {event.arrived_at:%d %H:%M}"
            )
    shipment = shipment_as_of(log, request.order_reference, request.at)
    print(
        f"request {request.at:%d %H:%M}; amount {request.amount_cents} cents"
    )
    print(
        f"observed {shipment.observed_at:%d %H:%M}; age {(request.at - shipment.observed_at).total_seconds():g} seconds"
    )
    print(
        f"delivery delay {(arrival.arrived_at - arrival.at).total_seconds():g} seconds"
    )
    print(f"watermark at request {watermark(log, request.at):%d %H:%M}")
    print(
        f"watermark at delivery arrival {watermark(log, arrival.arrived_at):%d %H:%M}"
    )
    print(
        f"watermark passes delivery {first_complete(log, arrival.at):%d %H:%M}"
    )
    for name in ("fast_trace", "held", "final"):
        decision = run[name].decision
        print(
            f"{name}: {decision.outcome.value}, {decision.reason}; executed {decision.executed}"
        )
    print(f"watermark passes request {run['complete_at']:%d %H:%M}")
    print(f"evidence final {run['final_at']:%d %H:%M}")
    print(
        f"delivery late relative to watermark: {arrival.event_id in {e.event_id for e in late_events(log)}}"
    )
    for record in run["corrections"]:
        print(f"correction {record.mode}: {record.payment_id}")
        print(f"  scan: {record.event_id}")
    print(
        f"corrections {len(run['corrections'])}; paid cents still {sum(p.amount_cents for p in run['fast'].payments)}"
    )


def print_counts():
    identity()
    print(
        "UTC event day | fast distinct n_day n_final copies misses arrival_day pane"
    )
    result = count_run()
    for row in result["rows"]:
        values = " ".join(
            str(row[k])
            for k in (
                "fast",
                "distinct",
                "n_day",
                "n_final",
                "copies",
                "misses",
                "arrival_day",
                "pane",
            )
        )
        print(f"{row['at']:%d %H:%M:%S} | {values}")
    lo, hi = result["interval"]
    print(
        f"late unique scans {result['late']}/{result['scans']}: {result['late'] / result['scans']:.6f}"
    )
    print(f"Wilson 95% [{lo:.6f}, {hi:.6f}]")


def _print_claim_result(row):
    print(
        f"{row['rule']}: {row['claims']} claims, {row['payments']} paid, {row['contradicted']} contradicted"
    )
    lo, hi = row["interval"]
    print(f"rho_late {row['rho_late']:.6f}; Wilson 95% [{lo:.6f}, {hi:.6f}]")


def print_claims():
    result = claim_run()
    denied = sum(
        d["initial"]["reason"] == "already_delivered"
        for d in result["populations"]["fast"]["decisions"]
    )
    print("Constructed claims (excerpt)")
    print(f"Already-delivered claims denied at once: {denied}")
    for row in result["rows"]:
        _print_claim_result(row)
        waiting = row["awaiting_absence"]["requests"]
        if waiting:
            print(f"awaiting_absence: {waiting} requests")
        honest = row["honest"]
        print(
            f"honest: {honest['requests']} requests; mean {honest['mean_hours']:.2f} hours"
        )
        print(f"corrections by mode: {row['corrections']}")
    final = result["rows"][0]["evidence_final"]
    print(
        f"evidence_final: mean {final['mean_hours']:.2f}; "
        f"longest {final['longest_hours']:.2f} hours"
    )
    print(
        f"longest original scan delay {result['longest_delay_hours']:.2f} hours"
    )


def print_claims_full():
    identity()
    print("Settings:")
    for key, value in SETTINGS.items():
        print(f"{key}: {value}")
    result = claim_run()
    for row in result["rows"]:
        _print_claim_result(row)
        for group in ("awaiting_absence", "honest", "evidence_final"):
            waits = row[group]
            print(f"{group}: {waits['requests']} requests")
            print(
                f"  mean {waits['mean_hours']}; longest {waits['longest_hours']} hours"
            )
        print(f"corrections by mode: {row['corrections']}")
    print(
        f"longest original scan delay {result['longest_delay_hours']:.6f} hours"
    )
    print(
        f"delay at most 36 hours: {result['longest_delay_at_most_36_hours']}"
    )
    print(
        "Already-delivered claims are denied at once; only absent evidence waits."
    )
    print(
        "Shadow counts replay the same recorded payment population, moving no money."
    )


def arrival_counts():
    """Count original arrivals by their generated delay, within the event window."""
    from collections import Counter
    from datetime import timedelta

    from refund_agent.batch_stream.generate import DAYS, START

    first = {}
    for event in stream().log.entries:
        if START <= event.at < START + timedelta(days=DAYS):
            first.setdefault(event.event_id, event)
    counts = Counter()
    for event in first.values():
        hours = (event.arrived_at - event.at).total_seconds() / 3600
        kind = "within minutes" if hours < 1 else "depot sync" if hours < 12 else "daily file"
        counts[kind] += 1
    return dict(counts)


# region book:fnd.batch-stream.arrivals-01 output
if __name__ == "__main__":
    counts = arrival_counts()
    print(f"14 days; {sum(counts.values()):,} distinct scans by event time")
    for category in ("within minutes", "depot sync", "daily file"):
        print(f"{category}: {counts[category]:,}")
# endregion book:fnd.batch-stream.arrivals-01


# region book:fnd.batch-stream.opening-01 output
if __name__ == "__main__":
    print_opening()
# endregion book:fnd.batch-stream.opening-01

# region book:fnd.batch-stream.counts-01 output
if __name__ == "__main__":
    print_counts()
# endregion book:fnd.batch-stream.counts-01

# region book:fnd.batch-stream.claims-01 output
if __name__ == "__main__":
    print_claims()
# endregion book:fnd.batch-stream.claims-01

# region book:fnd.batch-stream.claims-full-01 output
if __name__ == "__main__":
    print_claims_full()
# endregion book:fnd.batch-stream.claims-full-01

if __name__ == "__main__":
    print_replay()

# region book:fnd.batch-stream.figure-01 output
if __name__ == "__main__":
    print(json.dumps(figure_run(), default=json_default, indent=2))
# endregion book:fnd.batch-stream.figure-01

# region book:fnd.batch-stream.keys-01 output
if __name__ == "__main__":
    print(json.dumps(key_run(), default=json_default, indent=2))
    if os.environ.get("BATCH_STREAM_CAPTURE_DIR"):
        write_data(destination=os.environ["BATCH_STREAM_CAPTURE_DIR"])
# endregion book:fnd.batch-stream.keys-01


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-data", action="store_true")
    if parser.parse_args(argv).write_data:
        write_data()


if __name__ == "__main__":
    main()
