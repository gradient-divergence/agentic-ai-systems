"""Run chapter 30's contract, release, lineage and arithmetic observations offline."""

import json

from demos.data_systems_records import (
    EXERCISES,
    chapter_2_denial,
    experiment,
    generator_probe,
    lineage_example,
    reference_answers,
    release_evidence,
)
from refund_agent import contracts as c
from refund_agent.data.generate import Dataset
from refund_agent.sources.data_systems_budget import (
    budget_hours,
    rate_estimate,
    rate_upper,
    stale_chance,
)
from refund_agent.sources.data_systems_lineage import rebuild
from refund_agent.sources.data_systems_releases import dollars
from refund_agent.tools import Tools


def print_budget():
    from scipy.stats import chi2

    n, hours, age, tolerance = 48, 2400, 0.5, 0.01
    estimate, upper = rate_estimate(n, hours), rate_upper(n, hours)
    print(f"constructed carrier log: {n} scans; {hours} transit parcel-hours")
    print(f"chi-square(98) 95% quantile: {chi2.ppf(0.95, 98):.6f}")
    print(f"estimate {estimate:.6f}/hour; upper {upper:.6f}/hour")
    print(f"age {age * 60:g} minutes; spec limit 1800 seconds")
    print(f"chance at estimate {stale_chance(estimate, age):.6f}")
    print(f"chance at upper {stale_chance(upper, age):.6f}")
    print(
        f"tolerance {tolerance:.2f}: estimate budget {budget_hours(estimate, tolerance):.6f} hours"
    )
    print(
        f"estimate budget {60 * budget_hours(estimate, tolerance):.2f} minutes"
    )
    print(f"upper budget {60 * budget_hours(upper, tolerance):.2f} minutes")
    day, day_upper = rate_estimate(n, 1200), rate_upper(n, 1200)
    print(
        f"daytime: {n} scans in 1200 parcel-hours; rate {day:.6f}; upper {day_upper:.6f}"
    )
    print(
        f"daytime chance {stale_chance(day, age):.6f}; upper {stale_chance(day_upper, age):.6f}"
    )
    for source in ("order", "policy", "ledger"):
        budget = default_registry().contract(source).budget
        print(f"{source} limit: {budget.limit_seconds:g} seconds")
        print(fill(f"{source} basis: {budget.basis}", width=79))
        print("  rate evidence:", budget.count, budget.exposure_hours)
        print(
            "  accepted chance:", budget.chance_estimate, budget.chance_upper
        )
        print("  approver:", budget.approver_role)
    print(f"6000 quiet hours: upper {rate_upper(0, 6000):.9f}/hour")
    print(
        f"reciprocal quiet bound: {1 / rate_upper(0, 6000):.2f} hours per change"
    )


def print_run_table():
    _, _, runs = experiment()
    names = (
        "generated-contracts",
        "rename-bare",
        "rename-contracts",
        "harmless-contracts",
    )
    print("| Outcome / reason | Generated | Rename | Contracts | Harmless |")
    print("|---|---:|---:|---:|---:|")
    categories = sorted(
        {key for name in names for key in runs[name]["counts"]}
    )
    for outcome, reason in categories:
        counts = [
            str(runs[name]["counts"][(outcome, reason)]) for name in names
        ]
        print(f"| {outcome} {reason} | " + " | ".join(counts) + " |")
    print("| Total decisions | " + " | ".join("40" for _ in names) + " |")
    print(
        "| Cents paid | "
        + " | ".join(str(runs[name]["paid_cents"]) for name in names)
        + " |"
    )


def print_run():
    data, _, runs = experiment()
    print_run_table()
    for label, first, second in (
        ("generated", "generated-bare", "generated-contracts"),
        ("harmless without contracts", "generated-bare", "harmless-bare"),
    ):
        changed = sum(
            a.decision != b.decision
            for a, b in zip(runs[first]["traces"], runs[second]["traces"])
        )
        print(f"{label}: changed decisions {changed}")
    wrongful = {"r-7-0014", "r-7-0029", "r-7-0035"}
    tools = runs["rename-bare"]["tools"]
    paid = [p for p in tools.payments if p.request.request_id in wrongful]
    print(
        f"renamed delivered parcels: {len(paid)} wrongful payments; {sum(p.amount_cents for p in paid)} cents"
    )
    for payment in tools.payments:
        request = payment.request
        shipment = next(
            s for s in data.shipments if s.order_id == payment.order_id
        )
        delivered = [e.at for e in shipment.events if e.status == "delivered"]
        age = (
            (request.at - max(delivered)).total_seconds() / 86400
            if delivered
            else None
        )
        print(
            f"  {request.request_id}: {payment.amount_cents} cents; delivery age days {age}"
        )
    print(
        "Legitimate payment unaffected by the rename: r-7-0008."
    )
    print(
        f"  {runs['rename-contracts']['paid_cents']} cents; no delivery scan to rename"
    )
    delivered_ids = {
        s.order_id
        for s in data.shipments
        if any(e.status == "delivered" for e in s.events)
    }
    delivered_requests = {
        r.request_id
        for r in data.requests
        if r.order_reference in delivered_ids
    }
    other = [
        t.decision.request_id
        for t in runs["rename-bare"]["traces"]
        if t.decision.tier == 1
        and t.decision.reason == "delivery_unknown"
        and t.decision.request_id in delivered_requests
    ]
    print("other tier 1 referrals:", ", ".join(other))
    higher = [
        r
        for r in data.requests
        if r.reason == "not_delivered"
        and r.amount_cents >= 5000
        and any(
            e.status == "delivered"
            for s in data.shipments
            if s.order_id == r.order_reference
            for e in s.events
        )
    ]
    print(
        "higher-tier not-delivered drafts on delivered parcels:", len(higher)
    )
    tools = Tools(rename_delivery(data))
    approves = 0
    for request in higher:
        state = c.State(
            request=request, trace_id=f"trace-{request.request_id}"
        )
        for name in ("order.lookup", "shipment.status", "policy.retrieve"):
            tools.call(c.ToolCall(name=name), state)
        proposal = tools.assess(state)
        approves += proposal.outcome == c.Outcome.APPROVE
        print(
            f"  {request.request_id}: tier {proposal.tier.value}, "
            f"draft {proposal.outcome.value}, {proposal.reason}"
        )
    print("higher-tier approving drafts:", approves)


def print_releases():
    data = generate().model_copy(update={"correct": []})
    for announced in (True, False):
        release = dollars(data, announced)
        Dataset.model_validate_json(release.data.model_dump_json())
        problems = [
            contract_problem("order", o, release.served_versions["order"], 60)
            for o in release.data.orders
        ]
        print(
            f"{'announced' if announced else 'silent'} dollars: served {release.served_versions['order']}"
        )
        print(
            f"  {len(problems)} typed orders; checks: {sorted(set(map(str, problems)))}"
        )
        print(
            "  units/invariants remain plausible after every amount moves together"
        )
    bare, contracted = chapter_2_denial()
    print(
        f"chapter 2 single-field release: {bare.outcome.value}, {bare.reason}"
    )
    print(f"with contracts: {contracted.outcome.value}, {contracted.reason}")


def print_generator():
    row = generator_probe()
    print(f"generated shipments {row['records']}; requests {row['requests']}")
    print(f"scan after observed_at: {row['bad_records']} records")
    print("offsets seconds:", row["offsets_seconds"])
    print(
        f"readers {len(row['readers'])}; public readers {row['public_readers']}"
    )
    for request, date in zip(row["readers"], row["reader_dates"]):
        print(f"  {request}: {date}")
    print("reason: contract_invariant:shipment.not_after_observation")
    print(
        "training/validation only for decision comparisons; generator left unchanged"
    )


def print_lineage():
    row, _, result = lineage_example()
    print(
        f"request {row.request.request_id}: {row.request.amount_cents} cents; {row.request.reason}"
    )
    print("requester:", row.request.requester_id)
    print("trace:", row.trace_id)
    print("decision time:", row.decision_at.isoformat())
    print(
        "decision:",
        row.decision.outcome.value,
        row.decision.reason,
        row.decision.executed,
    )
    print("code version (release candidate fingerprint):")
    print(row.code_version)
    print("registry:", row.registry_version, "bundle:", row.bundle)
    for source in ("order", "shipment", "policy", "ledger"):
        ref = next(r for r in row.inputs if r.source == source and r.used)
        print(
            f"{source} {ref.contract_version}: age {ref.age_seconds:g} seconds"
        )
        print("  fingerprint:", ref.fingerprint)
        print("  observed:", ref.observed_at.isoformat())
        print("  captured:", ref.captured_at.isoformat())
    print("balance:")
    for field, value in row.derivations["balance"].items():
        print(f"  {field}: {value}")
    print("runtime:", row.candidate["runtime"]["identity"])
    print("effective step budget:", row.max_steps)
    for component in row.candidate["tools_composition"]["components"]:
        print("components:", component["name"], component["version"])
    print("delivery age seconds:", row.derivations["delivery_age_seconds"])
    print(
        "recorded model proposal:",
        row.model_proposal.outcome.value,
        row.model_proposal.reason,
    )
    print("approval evidence:", len(row.approvals), "(tier 1 needs none)")
    print("rebuild equal:", result["equal"])
    print("deterministic replay:", result["replay"])
    print("model limit:", result["fresh_model_limit"])
    _, _, runs = experiment()
    for name in ("generated-contracts", "rename-contracts"):
        tools = runs[name]["tools"]
        equal = sum(
            rebuild(r, tools.record_store)["equal"] for r in tools.lineage
        )
        print(f"{name}: equal rebuilds {equal}/40")


def print_keys():
    rows = reference_answers()
    hospital, quiet, card = (
        rows["hospital"],
        rows["quiet_and_day"],
        rows["card"],
    )
    print("exercise 1: refer; contract_vocabulary:shipment.status")
    print("exercise 2: no per-record check catches the silent locker meaning")
    print(
        f"exercise 3: estimate {hospital['estimate']:.6f}; upper {hospital['upper']:.6f}"
    )
    print(
        f"  chance {hospital['chance']:.6f}; budget {hospital['minutes']:.2f} minutes"
    )
    print(
        f"exercise 4: quiet upper {quiet['quiet_upper']:.9f}; 3/6000 {3 / 6000:.9f}"
    )
    print(
        f"  quiet chance {quiet['quiet_chance']:.9f}; exponent {quiet['quiet_upper'] * 0.5:.9f}"
    )
    print(
        f"  daytime estimate {quiet['daytime_rate']:.6f}; upper {quiet['daytime_upper']:.6f}"
    )
    print(f"  estimate chance {stale_chance(quiet['daytime_rate'], 0.5):.6f}")
    print(
        f"  upper budget {60 * budget_hours(quiet['daytime_upper'], 0.01):.2f} minutes"
    )
    print("exercise 5:", rows["diagnosis"]["evidence"], "; condition lineage")
    print(
        "  offset 40 minutes; fingerprint matches the later record; original unavailable"
    )
    print("exercise 6: 30 refunds / 600 hours; tolerance 0.01")
    print(f"  estimate {card['ESTIMATE']:.6f}; upper {card['UPPER']:.6f}")
    print(
        f"  budget {card['BUDGET_SECONDS'] / 3600:.6f} hours; {card['BUDGET_SECONDS'] / 60:.2f} minutes"
    )
    feed = json.loads((EXERCISES / "feed-06.json").read_text())["records"]
    for record in feed:
        print(" ", record["id"], card["problem"](record))


# region book:fnd.data-systems.contract-01 output
from textwrap import fill

from refund_agent.data.generate import generate
from refund_agent.sources.data_systems_check import contract_problem
from refund_agent.sources.data_systems_registry import default_registry
from refund_agent.sources.data_systems_releases import rename_delivery

shipment_contract = default_registry().contract("shipment")
assert shipment_contract.vocabularies["status"] == ("shipped", "delivered")
assert "not_after_observation" in shipment_contract.invariants


def check_shipment(record, age_seconds):
    return contract_problem("shipment", record, "shipment@1", age_seconds)


if __name__ == "__main__":
    registry = default_registry()
    for source in registry.pins:
        contract = registry.contract(source)
        notice = contract.notice.days
        print(f"{source}: {contract.version}, notice {notice} days")
        print(fill(f"  notice basis: {contract.notice.basis}", width=79))
    print("shipment vocabulary:", shipment_contract.vocabularies["status"])
    print("shipment owner:", shipment_contract.owner_role)
    for field in shipment_contract.fields:
        print(f"{field.field}: {field.unit}; {field.meaning}")
    for invariant in shipment_contract.invariants:
        print("shipment invariant:", invariant)
    changed = rename_delivery(generate(7, orders=40, requests=40))
    print("renamed status check:", check_shipment(changed.shipments[29], 120))
# endregion book:fnd.data-systems.contract-01

# region book:fnd.data-systems.budget-01 output
if __name__ == "__main__":
    print_budget()
# endregion book:fnd.data-systems.budget-01

# region book:fnd.data-systems.run-01 output
if __name__ == "__main__":
    print_run()
# endregion book:fnd.data-systems.run-01

# region book:fnd.data-systems.run-table-01 output
if __name__ == "__main__":
    print_run_table()
# endregion book:fnd.data-systems.run-table-01

# region book:fnd.data-systems.releases-01 output
if __name__ == "__main__":
    print_releases()
# endregion book:fnd.data-systems.releases-01

# region book:fnd.data-systems.generator-01 output
if __name__ == "__main__":
    print_generator()
# endregion book:fnd.data-systems.generator-01

# region book:fnd.data-systems.lineage-01 output
if __name__ == "__main__":
    print_lineage()
# endregion book:fnd.data-systems.lineage-01

# region book:fnd.data-systems.keys-01 output
if __name__ == "__main__":
    print_keys()
# endregion book:fnd.data-systems.keys-01

# region book:fnd.data-systems.inputs-access-01 output
if __name__ == "__main__":
    print(
        json.dumps(
            {
                "reader": "brief writer, chapter 30",
                "purpose": (
                    "planning probe of inputs (requests, orders, shipments), "
                    "no labels, 11 October 2026"
                ),
                "access_kind": (
                    "inputs only; planning probes included "
                    "golden-window inputs"
                ),
                "journal_design": (
                    "GoldenSet.consult journals label reads; this is a note "
                    "beside the stored manifest output, "
                    "not a label consultation"
                ),
                "recorded_at": "2026-10-12T00:00:00+00:00",
            },
            indent=2,
            sort_keys=True,
        )
    )
# endregion book:fnd.data-systems.inputs-access-01

# region book:fnd.data-systems.release-01 output
if __name__ == "__main__":
    release = release_evidence()
    print("stage 5 contracts candidate:", release["decision"])
    print("failed cases:", release["failed_cases"])
    print("cases/trials:", release["cases_run"], release["trials_run"])
    print("chapter 3 floor:", release["floor"]["pass"])
    print("must-pay cases:", release["must_pay_cases"])
    print("must-pay cents per trial:", release["must_pay_cents_per_trial"])
# endregion book:fnd.data-systems.release-01

# region book:fnd.data-systems.release-record-01 output
if __name__ == "__main__":
    print(json.dumps(release_evidence(), indent=2, sort_keys=True))
# endregion book:fnd.data-systems.release-record-01
