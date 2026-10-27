"""Captured arithmetic and packet table; complete verdicts require chapter 39."""

import json
import os
import sys
from fractions import Fraction as F
from pathlib import Path
from textwrap import fill

from pydantic import BaseModel

from refund_agent.oversight.human_oversight_risk_queueing import mm1
from refund_agent.sources.data_systems_budget import budget_hours, stale_chance
from refund_agent.transfer.case_studies_domain_agnostic_check import (
    check,
    expected_cost,
    optimal_rule,
)
from refund_agent.transfer.case_studies_domain_agnostic_data import (
    DATA,
    DOMAINS,
    EXERCISES,
    SLUG,
    cells,
    constructed_day,
    domain_facts,
    packets,
    refund_rule,
    validate_packet_inventory,
    write_data,
)
from refund_agent.transfer.case_studies_domain_agnostic_domains import (
    alarm_numbers,
    bank_costs,
    holds_per_fraud,
    hospital_day,
    hospital_orders,
    order_balance,
)
from refund_agent.transfer.case_studies_domain_agnostic_facts import (
    CostMatrix,
    load_facts,
    require_inventory,
)
from refund_agent.transfer.case_studies_domain_agnostic_packet import Cell, Interval, Rule


def _print(*values):
    print(
        fill(
            " ".join(str(v) for v in values),
            width=79,
            break_long_words=False,
            break_on_hyphens=False,
        )
    )


def print_packets():
    files = tuple(
        path.name for kind in ("facts", "cells") for path in DATA.glob(f"{SLUG}-*.{kind}.json")
    )
    required = tuple(
        f"{SLUG}-{domain}.{kind}.json" for domain in DOMAINS for kind in ("facts", "cells")
    )
    require_inventory(files, required, "table file")
    columns = []
    for domain in DOMAINS:
        facts = load_facts(DATA / f"{SLUG}-{domain}.facts.json")
        raw = json.loads((DATA / f"{SLUG}-{domain}.cells.json").read_text())
        column = tuple(Cell.model_validate(c) for c in raw)
        if facts.domain != domain or column != cells(facts):
            raise ValueError(f"{domain} table answers contradict domain facts")
        columns.append(column)
    print("| Question | Refund | Incident desk | Hospital | Bank |")
    print("|---|---|---|---|---|")
    for index in range(7):
        question = columns[0][index].question
        answers = [
            column[index].answer + (f" ({column[index].kind})" if n else "")
            for n, column in enumerate(columns)
        ]
        print("| " + " | ".join([question, *answers]) + " |")


def print_triage():
    facts = domain_facts("triage")
    require_inventory((facts.domain,), ("triage",), "triage facts")
    rate = facts.sources[0].rate_per_hour
    if rate is None:
        raise ValueError("alarm demonstration requires its given clearing rate")
    numbers = alarm_numbers(rate, facts.tolerance)
    require_inventory(
        numbers,
        ("rate_per_hour", "chance_20_minutes", "chance_30_minutes", "budget_seconds"),
        "triage result",
    )
    _print("Constructed constant clearing rate; tolerance named before measurement.")
    _print(f"alarm rate: {numbers['rate_per_hour']:.6f} per hour")
    _print(f"alarm chance at 20 minutes: {numbers['chance_20_minutes']:.6f}")
    _print(f"alarm chance at 30 minutes: {numbers['chance_30_minutes']:.6f}")
    _print(f"alarm budget at 1%: {numbers['budget_seconds']:.6f} seconds")
    _print(
        f"redesign: alarm.probe at restart; 8-second limit; chance {stale_chance(numbers['rate_per_hour'], 8 / 3600):.6f}"
    )
    _print(f"session count at 30 minutes: {stale_chance(0.01, 0.5):.6f}")
    _print(f"shipment chance at 30 minutes: {stale_chance(0.02, 0.5):.9f}")
    _print(f"shipment budget at 1%: {budget_hours(0.02, 0.01) * 60:.6f} minutes")
    _print(f"shipment/alarm budget ratio: {numbers['rate_per_hour'] / 0.02:.6f}")


def print_hospital():
    day = constructed_day()
    request, batch = hospital_day(day, "request"), hospital_day(day, "batch")
    for name, policy in (("request", request), ("batch", batch)):
        require_inventory(
            (row.patient_id for row in policy),
            (patient.patient_id for patient in day.patients),
            f"hospital {name} patient",
        )
    batch_by_patient = {row.patient_id: row for row in batch}
    _print("Constructed: 18 discharges, 6 urgent, 12 routine; 21 slots, 6 near.")
    _print(
        "Urgent discharge positions:",
        ", ".join(str(n) for n, p in enumerate(day.patients, 1) if p.urgency == "urgent"),
    )
    print("patient       urgency  request       batch        notification hours")
    for first in request:
        second = batch_by_patient[first.patient_id]
        a = f"d{first.day}/s{first.slot}" if first.day else "refer"
        b = f"d{second.day}/s{second.slot}" if second.day else "refer"
        print(
            f"{first.patient_id:13} {first.urgency:8} {a:13} {b:13} {first.notification_hour:02d}/{second.notification_hour:02d}"
        )
    for name, policy in (("request", request), ("batch", batch)):
        for urgency in ("urgent", "routine"):
            selected = [r for r in policy if r.urgency == urgency]
            _print(
                name,
                urgency,
                f"in time {sum(r.day is not None for r in selected)}/{len(selected)};",
                f"referred {sum(r.day is None for r in selected)}",
            )
    routine_days = sorted({r.day for r in batch if r.urgency == "routine" and r.day is not None})
    _print("batch routine days:", ", ".join(map(str, routine_days)))
    _print(
        "first routine notification wait:",
        batch[0].notification_hour - day.patients[0].discharged_hour,
        "hours",
    )
    trials = hospital_orders(day)
    _print("random orders:", trials["n"], "; seed:", trials["seed"])
    _print(f"mean urgent booked in time: {trials['mean']:.4f}; exact: {trials['exact_mean']}")
    _print("exact all-six probability:", trials["exact_all_probability"])
    _print("observed all-six orders:", trials["all_in_time_observed"])
    _print(f"slot chance at 30 minutes: {stale_chance(0.01, 0.5):.6f}")


def bank_rules():
    facts = domain_facts("bank")
    require_inventory((facts.domain,), ("bank",), "bank facts")
    if facts.cost_matrix is None:
        raise ValueError("bank demonstration requires card-level costs")
    require_inventory(facts.cost_matrix.entries, ("clear", "refer", "hold"), "bank cost action")
    require_inventory(facts.actions_on_score, ("clear", "refer", "hold"), "bank score action")
    copied = refund_rule()
    mapped = Rule(
        intervals=tuple(
            Interval.model_validate(
                {**i.model_dump(), "action": {"pay": "clear", "refer": "refer"}[i.action]}
            )
            for i in copied.intervals
        ),
        actions_alone=("clear",),
        referral="refer",
        status="designed-held",
    )
    band = optimal_rule(facts.cost_matrix, facts.actions_on_score, "refer", ("clear", "hold"))
    return facts, mapped, band


def print_bank():
    facts, copied, band = bank_rules()
    matrix = facts.cost_matrix
    if matrix is None:
        raise ValueError("bank demonstration requires card-level costs")
    _print("Constructed score: card-level probability of fraudulent use within 60 days.")
    _print("4/37 payment-flag precision transfers only under that card-level definition.")
    review = mm1(12, 15)
    if review.W is None:
        raise ValueError("bank review queue is unstable")
    _print("M/M/1 review-open time:", review.W * 60, "minutes")
    _print("Fraud spending during review:", F(150) * review.W, "dollars")
    print("action  legitimate  fraudulent  dollars per card")
    for action, costs in matrix.entries.items():
        print(f"{action:7} {float(costs[0]):10.2f} {float(costs[1]):11.2f}")
    _print(f"clear/refer edge: {band.intervals[0].upper} = {float(band.intervals[0].upper):.4f}")
    _print(
        f"refer/hold edge: {band.intervals[-1].lower} = {float(band.intervals[-1].lower):.4f}; tie refers"
    )
    _print(
        f"copied refund edge: {copied.intervals[0].upper} = {float(copied.intervals[0].upper):.4f}; designed and held"
    )
    risks = (F(1, 20), F(1, 2), F(4, 37))
    costs = bank_costs(matrix, {"copy": copied, "band": band}, risks)
    for risk in risks:
        row = costs[str(risk)]
        _print(
            f"risk {risk} = {float(risk):.6f}: copy {row['copy']['action']} ${float(row['copy']['cost']):.2f};",
            f"band {row['band']['action']} ${float(row['band']['cost']):.2f}",
        )
        _print(
            "  clear/refer/hold:",
            "/".join(
                f"${float(expected_cost(matrix, a, risk)):.2f}" for a in facts.actions_on_score
            ),
        )
    _print("legitimate holds per fraudulent card stopped:", float(holds_per_fraud(F(4, 37))))
    _print("account closure: level 2, named approver; no score authority to close")


def print_balance():
    row = order_balance()
    require_inventory(
        row,
        (
            "first_payment_cents",
            "second_payment_cents",
            "second_outcome",
            "second_reason",
            "second_assessed",
            "second_assessment_reason",
            "second_execution_attempts",
            "second_execution_refused",
            "paid_cents",
            "remaining_cents",
        ),
        "balance result",
    )
    _print("Constructed: two 3000-cent requests against one 4000-cent order.")
    _print("First payment:", row["first_payment_cents"], "cents")
    _print(
        "Second assessment:",
        row["second_assessed"],
        "; execution attempts:",
        row["second_execution_attempts"],
        "; refused for balance:",
        row["second_execution_refused"],
    )
    _print(
        "Second payment:",
        row["second_payment_cents"],
        "; outcome:",
        row["second_outcome"],
        "; reason:",
        row["second_reason"],
    )
    _print("Total paid:", row["paid_cents"], "cents; remaining:", row["remaining_cents"], "cents")
    _print("The existing refund gate rechecks the order balance at execution.")


def print_keys():
    fixture = json.loads((EXERCISES / "insurer_05.json").read_text())
    require_inventory(
        fixture,
        (
            "constructed",
            "domain",
            "label_window_days",
            "label_window_basis",
            "facts",
            "other_answers",
            "integration",
        ),
        "exercise fixture",
    )
    require_inventory(
        fixture["facts"],
        ("fraud_findings_after_30_days", "all_findings_mature_by_days", "basis"),
        "exercise facts",
    )
    _print("Exercise 1: packet a,c,e,h; refund answers b,d,f,g.")
    _print(
        "Exercise 2: clock -> seconds/probe; coupling -> clinic allocation; rule -> bank actions."
    )
    import math

    rate = math.log(2) / 1.5
    _print(
        f"Exercise 3: rate {rate:.6f}/hour; budget {budget_hours(rate, 0.02) * 60:.6f} minutes;",
        f"30-minute chance {stale_chance(rate, 0.5):.6f}",
    )
    matrix = CostMatrix(
        outcomes=("legitimate", "fraudulent"),
        entries={"clear": (F(0), F(600)), "refer": (F(5), F(95)), "hold": (F(20), F(5))},
        unit="dollars",
        basis="exercise 4 given",
    )
    band = optimal_rule(matrix, ("clear", "refer", "hold"), "refer", ("clear", "hold"))
    _print(
        f"Exercise 4: clear/refer {band.intervals[0].upper} = {float(band.intervals[0].upper):.6f};",
        f"refer/hold {band.intervals[-1].lower} = {float(band.intervals[-1].lower):.6f}",
    )
    for risk in (F(3, 100), F(1, 4)):
        action = band.action_at(risk)
        _print(
            f"risk {float(risk):.2f}: {action}; clear/refer/hold",
            "/".join(
                f"${float(expected_cost(matrix, a, risk)):.2f}" for a in ("clear", "refer", "hold")
            ),
        )
    long_hold = CostMatrix.model_validate(
        {**matrix.model_dump(), "entries": {**matrix.entries, "hold": (F(365), F(5))}}
    )
    late = optimal_rule(long_hold, ("clear", "refer", "hold"), "refer", ("clear", "hold"))
    _print("24-hour wrongful hold: $360; hold edge:", late.intervals[-1].lower)
    _print(
        "Exercise 5: copied label window",
        fixture["label_window_days"],
        "days; required",
        fixture["facts"]["all_findings_mature_by_days"],
        "days; hold pending mature labels.",
    )
    _print(
        "Findings lost after 30 days:",
        fixture["facts"]["fraud_findings_after_30_days"],
        "; four-check integration pending chapter 39.",
    )
    _print("Exercise 6: cents/sessions and cents/follow-ups hold; money/money passes.")
    _print("The bank copy still holds on rule; units cannot validate a numeric cap.")


def print_verdicts(requirement: BaseModel):
    rows = tuple(packets(requirement))
    validate_packet_inventory(rows)
    verdicts = tuple((packet, check(packet, facts)) for packet, facts in rows)
    for packet, verdict in verdicts:
        _print(packet.domain, packet.form, verdict.verdict)
        for failure in verdict.failures:
            _print(" ", failure.check, failure.item, failure.reason)
            if failure.check == "rule":
                n = failure.numbers
                left, right = (
                    ("[" if n["left_closed"] == "True" else "("),
                    ("]" if n["right_closed"] == "True" else ")"),
                )
                _print(
                    f"  {left}{float(F(n['lower'])):.4f}, {float(F(n['upper'])):.4f}{right}",
                    f"exact {n['lower']}..{n['upper']}; {n['copied']} vs {n['cheapest']}",
                )
                _print(
                    f"  midpoint costs ${float(F(n['copied_cost'])):.2f} vs ${float(F(n['cheapest_cost'])):.2f}"
                )
            else:
                _print(" ", json.dumps(failure.numbers, sort_keys=True))


if __name__ == "__main__" and sys.argv[1:] == ["--write-data"]:
    write_data(DATA)
    raise SystemExit(0)


# region book:fnd.case-studies-domain-agnostic.packets-01 output
if __name__ == "__main__":
    print_packets()
# endregion book:fnd.case-studies-domain-agnostic.packets-01

# region book:fnd.case-studies-domain-agnostic.triage-01 output
if __name__ == "__main__":
    print_triage()
# endregion book:fnd.case-studies-domain-agnostic.triage-01

# region book:fnd.case-studies-domain-agnostic.hospital-01 output
if __name__ == "__main__":
    print_hospital()
# endregion book:fnd.case-studies-domain-agnostic.hospital-01

# region book:fnd.case-studies-domain-agnostic.bank-01 output
if __name__ == "__main__":
    print_bank()
# endregion book:fnd.case-studies-domain-agnostic.bank-01

# region book:fnd.case-studies-domain-agnostic.balance-01 output
if __name__ == "__main__":
    print_balance()
# endregion book:fnd.case-studies-domain-agnostic.balance-01

# region book:fnd.case-studies-domain-agnostic.keys-01 output
if __name__ == "__main__":
    print_keys()
# endregion book:fnd.case-studies-domain-agnostic.keys-01

# region book:fnd.case-studies-domain-agnostic.verdicts-01 output
if __name__ == "__main__":
    from refund_agent.launch import LaunchRequirement

    path = Path(os.environ["CASE_STUDIES_DOMAIN_AGNOSTIC_REQUIREMENT"])
    print_verdicts(LaunchRequirement.model_validate_json(path.read_text()))
# endregion book:fnd.case-studies-domain-agnostic.verdicts-01
