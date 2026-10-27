"""Captured chapter 39 results. Missing operating evidence is stated explicitly."""

import json
import os
import runpy
from functools import lru_cache
from pathlib import Path
from tempfile import TemporaryDirectory
from textwrap import fill

from refund_agent.launch.capstone_blueprint_exercises import (
    decision_05,
    packet_03,
    write_exercises,
)
from refund_agent.launch.capstone_blueprint_gate import check
from refund_agent.launch.capstone_blueprint_requirement import (
    Period,
    refund_rows,
    reversals,
)
from refund_agent.launch.capstone_blueprint_timeline import Demand, operate
from refund_agent.schedule.inputs import REVIEW_MINUTES

ROOT = Path(__file__).resolve().parents[1]


@lru_cache(maxsize=1)
def run_state():
    path = Path(
        os.environ.get(
            "CAPSTONE_BLUEPRINT_RUN",
            str(ROOT / ".tmp-sol/capstone-blueprint-run"),
        )
    )
    summary = path / "capstone-blueprint-run.json"
    if not summary.exists():
        operate(7, demand=Demand(), out=path)
    return json.loads(summary.read_text())


def run_path():
    run_state()
    return Path(
        os.environ.get(
            "CAPSTONE_BLUEPRINT_RUN",
            str(ROOT / ".tmp-sol/capstone-blueprint-run"),
        )
    )


def saved(name):
    path = run_path() / ("capstone-blueprint-" + name + ".json")
    return json.loads(path.read_text()) if path.exists() else None


def reference():
    return runpy.run_path(
        str(
            ROOT
            / "exercises/capstone-blueprint/capstone_blueprint_reference.py"
        )
    )


def print_packet():
    state = run_state()
    print(
        "Prepared refund requirement rows; operating packet: "
        + state["status"]
    )
    requirement = saved("requirement")
    if requirement is not None:
        print(
            f"Bundle {requirement['launched_bundle'][:16]}; signed by {requirement['signed_by']} on day {requirement['signed_day']:g}"
        )
    else:
        print("No launched stage 6 bundle or signature is fabricated.")
    print("row | question | records (producer; bundle rule; signer)")
    if requirement is not None:
        from refund_agent.launch import LaunchRequirement

        rows = LaunchRequirement.model_validate(requirement).rows
    else:
        rows = refund_rows(Period(start=50, end=64))
    for row in rows:
        print(f"{row.row_id} | {row.question}")
        for need in row.records:
            print(
                f"  {need.kind} | {need.producer} | {need.bundle_rule} | {need.signer_role}"
            )
    print("Comparison rerunnable: false")


def print_gate():
    state = run_state()
    verdict = saved("gate-01")
    if verdict is not None:
        print(
            f"Operating gate-01: {verdict['verdict']}; assembly day {verdict['day']:g}"
        )
        for row in verdict["rows"]:
            print(
                f"row {row['row']}: "
                + "; ".join(
                    f"{name}={'pass' if passed else 'hold'}"
                    for name, passed in row["checks"].items()
                )
            )
        for failure in verdict["failures"]:
            print(
                f"row {failure['row']}/{failure['kind']}/{failure['check']}: {failure['reason']}"
            )
        return
    print("Operating gate-01: unavailable; " + state["status"])
    print(
        "Assembly day: unavailable; no shared-clock operating run completed."
    )
    for row in refund_rows(Period(start=50, end=64)):
        kinds = [
            n.kind for n in row.records if n.kind in state["missing_kinds"]
        ]
        print(
            f"row {row.row_id}: required records missing: {', '.join(kinds)}"
        )
    print("No launch admission or decision-owner signature.")


def print_refresh():
    verdict = saved("gate-02")
    if verdict is not None:
        refreshed = saved("refresh")
        print("Refreshed: " + ", ".join(refreshed["refreshed"]))
        print("Earlier results retained: " + ", ".join(refreshed["retained"]))
        for unavailable in refreshed["unavailable"]:
            print(unavailable)
        print(
            f"Operating gate-02: {verdict['verdict']}; day {verdict['day']:g}"
        )
        for failure in verdict["failures"]:
            print(
                f"row {failure['row']}/{failure['kind']}/{failure['check']}: {failure['reason']}"
            )
        return
    print(
        "Operating gate-02: unavailable; launched bundle and first verdict absent."
    )
    print("No rehearsal rerun; no earlier result replaced.")
    print("A comparison held on its verdict is never refreshed.")
    print(
        "Constructed-record tests verify the earlier-bundle hold and target-bound refresh."
    )


def print_refused():
    from refund_agent.launch.__main__ import owner_roles
    from refund_agent.launch.capstone_blueprint_scan import LaunchLogs

    owners = owner_roles()
    with TemporaryDirectory(
        prefix="capstone-blueprint-refused-", dir=run_path()
    ) as path:
        fixtures = runpy.run_path(str(ROOT / "tests/refund_agent/capstone_blueprint_fixtures.py"))
        owners["layers"] = fixtures["OWNERS"]["layers"]
        req, packet, registry = fixtures["rehearsal_packet"](Path(path))
        verdict = check(
            packet, req, registry, LaunchLogs(), owners, req.gate_day
        )
    print("Constructed earlier-bundle rehearsal packet:")
    print(verdict.model_dump_json(indent=2))


def print_compare():
    comparison = saved("comparison")
    if comparison is not None:
        period = comparison["period"]
        print(
            f"Comparison period: days {period['start']:g} to {period['end']:g}; {(period['end'] - period['start']):g} days"
        )
        for arm in comparison["arms"]:
            print(
                f"{arm['arm']}: {arm['requests']} requests; {arm['hours_per_week']:.6f} reviewer-hours/week; cost {arm['cost_cents']} cents"
            )
            print(
                f"  wrongful denials {arm['wrongful_denials']}; money paid against label {arm['wrong_payment_cents']} cents; payments later false {arm['payments_later_false']}"
            )
            print(
                f"  tier 1 within promise: {arm['tier1_answered_in_time']}/{arm['tier1_requests']}"
            )
            for kind, minutes in arm["review_minutes"].items():
                print(
                    f"  {kind}: {minutes / 60 * 7 / (period['end'] - period['start']):.6f} hours/week"
                )
        print("Manual assumption: " + comparison["manual_assumption"])
        print(
            "Against manual: "
            + comparison["against_manual"]
            + "; against rules: "
            + comparison["against_rules"]
        )
        print(
            "Rule: "
            + comparison["rule"]
            + "; verdict: "
            + comparison["verdict"]
        )
        return
    print(
        "Operating comparison: unavailable; stage 6 decision/review/queue records absent."
    )
    print("Period and request counts: unavailable.")
    print("Agent, manual and measured RulesClient arms: no values fabricated.")
    print(
        "Required accounting: tier 2, tier 3, tier 1 referrals, rechecks, blind second reviews, planted cases."
    )
    print(
        "Constructed review minutes by tier: "
        + ", ".join(
            f"{tier}={minutes}" for tier, minutes in REVIEW_MINUTES.items()
        )
    )
    print("Manual assumption: no review of its reviewers' work.")
    print(
        "Required rule: better than each alternative on at least one business measure; worse on none."
    )
    print(
        "Expected manual saving and rules tie remain unverified in the operating run."
    )


def print_tier3():
    record = saved("tier3")
    if record is not None:
        print(
            f"Tier 3 period: days {record['period']['start']:g} to {record['period']['end']:g}"
        )
        print(
            f"Requests {record['requests']}; verdict {record['verdict']}; decided otherwise {len(record['decided_otherwise'])}; pending {len(record['pending'])}"
        )
        for outcome in record["outcomes"]:
            print(
                f"{outcome['request_id']}: {outcome['outcome']} by {outcome['approver']}"
            )
        return
    print(
        "Operating tier 3 path: unavailable; shared-clock month and review outcomes absent."
    )
    print(
        "Configured report period: first complete 30 days of stage 6; no actual period established."
    )
    print(
        "Actual requests, named-approver outcomes and decisions otherwise: unavailable."
    )
    print(
        "Every request over $500, including denials, requires its named approver."
    )


def print_decision():
    record = saved("decision")
    if record is None:
        print(
            "Operating held-decision record: unavailable; launched bundle and packet absent."
        )
        print(
            "Stage 6 authority: unavailable; no grants fabricated. Signed by: nobody."
        )
        conditions = reversals()
    else:
        print(
            f"Decision {record['verdict']}; bundle {record['bundle'][:16]}; packet {record['packet_fingerprint']}; signed by {record['signed_by']}"
        )
        for row, reasons in record["held_rows"].items():
            print(f"Held row {row}: " + "; ".join(reasons))
        print("Passed rows: " + ", ".join(record["passed_rows"]))
        print(
            "Stage 6 authority records: "
            + json.dumps(record["stage6_authority_records"], sort_keys=True)
        )
        from refund_agent.launch.capstone_blueprint_requirement import (
            ReversalCondition,
        )

        conditions = tuple(
            ReversalCondition.model_validate(c) for c in record["reversals"]
        )
    print("condition | withdraws | returns to | owner roles | mechanism")
    for condition in conditions:
        row = (
            f"{condition.condition_id} | {condition.withdraws} | {condition.returns_to} | "
            f"{', '.join(condition.owner_roles)} | {condition.mechanism}"
        )
        if condition.condition_id == "tier3_without_named_approver":
            row = fill(row, width=79, subsequent_indent="  ")
        print(row)
    print("Payments already made stand under every reversal.")
    print(
        "The first four conditions govern stage 6; the fifth governs a signed launch."
    )
    if record is None:
        print(
            "Open joins: monthly audit, daily reconciliation and budget monitor with their actual owners."
        )
        print(
            "A replacement model shown better than rules would require a new decision and gate."
        )
    else:
        for control in record["controls"]:
            print(
                f"Control {control['name']}: {control['period']} by {control['owner']}"
            )
        for item in record["open_items"]:
            print(f"Open item {item['item']}: {item['owner']}")
        for row, clears in record["clears"].items():
            print(f"Clear row {row}: {clears}")


def print_keys():
    answers = reference()
    print("Constructed exercise keys, not operating evidence.")
    print("1: " + ", ".join(answers["exercise_01"]()))
    second = answers["exercise_02"]()
    print(
        f"2: stage {second['stage']} stands; budget exhaustion withdraws "
        f"{second['withdraws']} authority; {second['decider_after']} decides; "
        f"{second['mechanism']}; paid refunds stand."
    )
    third = answers["exercise_03"](packet_03())
    print(
        "3: "
        + third["verdict"]
        + "; records not counted: "
        + ", ".join(third["not_counted"])
    )
    fourth = answers["exercise_04"]()
    print(
        f"4: manual {float(fourth['manual_hours']):.6f} hours/week; agent {float(fourth['agent_hours']):.6f} hours/week"
    )
    print("4: reopened 2 versus 3; " + fourth["verdict"])
    if os.environ.get("CAPSTONE_BLUEPRINT_USE_REFERENCE") == "1":
        answers["print_exercise_05_key"](decision_05())
    print(
        "6: constructed last payment day 54; deployed gate day 64 must hold until day 84."
    )
    print("6: operating rerun unavailable; no assembled launch packet.")


def write_data():
    state = run_state()
    path = ROOT / "demos/data/capstone_blueprint_records.json"
    records = []
    data = {
        "status": state["status"],
        "records": records,
        "gate_day": state["gate_day"],
        "missing_kinds": state["missing_kinds"],
        "operating_figure_available": False,
    }
    requirement = saved("requirement")
    if requirement is not None:
        from refund_agent.launch import LaunchRequirement
        from refund_agent.launch.capstone_blueprint_packet import assemble

        req = LaunchRequirement.model_validate(requirement)
        packet = assemble(run_path(), None, req)
        gates = [saved("gate-01"), saved("gate-02")]
        for entry in packet.records:
            claim = entry.claims
            rows = [
                r.row_id
                for r in req.rows
                if any(n.kind == claim.kind for n in r.records)
            ]
            counted = [
                bool(
                    g
                    and any(claim.record_id in r["counted"] for r in g["rows"])
                )
                for g in gates
            ]
            records.append(
                {
                    "record_id": claim.record_id,
                    "kind": claim.kind,
                    "day": claim.day,
                    "bundle": claim.bundle,
                    "rows": rows,
                    "counted_gate_01": counted[0],
                    "counted_gate_02": counted[1],
                    "verdict": claim.verdict,
                    "period": claim.period.model_dump()
                    if claim.period
                    else None,
                }
            )
        data["operating_figure_available"] = bool(packet.records)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    write_exercises(ROOT / "exercises/capstone-blueprint")


# region book:fnd.capstone-blueprint.packet-01 output
if __name__ == "__main__":
    print_packet()
# endregion book:fnd.capstone-blueprint.packet-01

# region book:fnd.capstone-blueprint.gate-01 output
if __name__ == "__main__":
    print_gate()
# endregion book:fnd.capstone-blueprint.gate-01

# region book:fnd.capstone-blueprint.gate-02 output
if __name__ == "__main__":
    print_refresh()
# endregion book:fnd.capstone-blueprint.gate-02

# region book:fnd.capstone-blueprint.refused-01 output
if __name__ == "__main__":
    print_refused()
# endregion book:fnd.capstone-blueprint.refused-01

# region book:fnd.capstone-blueprint.compare-01 output
if __name__ == "__main__":
    print_compare()
# endregion book:fnd.capstone-blueprint.compare-01

# region book:fnd.capstone-blueprint.tier3-01 output
if __name__ == "__main__":
    print_tier3()
# endregion book:fnd.capstone-blueprint.tier3-01

# region book:fnd.capstone-blueprint.decision-01 output
if __name__ == "__main__":
    print_decision()
# endregion book:fnd.capstone-blueprint.decision-01

# region book:fnd.capstone-blueprint.keys-01 output
if __name__ == "__main__":
    print_keys()
# endregion book:fnd.capstone-blueprint.keys-01


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-data", action="store_true")
    if parser.parse_args().write_data:
        write_data()
