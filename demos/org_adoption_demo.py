"""Chapter 38: capture the actual gate, plan validation and accounting exercises.

The assembled demand/queue window remains blocked until its predecessors merge.
No synthetic audit, budget, promotion or credential is used to admit stage 6.
"""

import argparse
import json
import sys
from pathlib import Path
from textwrap import fill

from demos import regions as _regions
from demos.org_adoption_fixtures import (
    accounting_fixture,
    constructed_plan,
    serialize_fixture,
)
from refund_agent.release.candidate import build_candidate
from refund_agent.release.suite import build_suite
from refund_agent.rollout.org_adoption_desk import (
    EXERCISE_BALANCE,
    TEXT_BALANCE,
    balance_case,
)
from refund_agent.rollout.org_adoption_plan import validate, validate_structure
from refund_agent.rollout.org_adoption_reconcile import reconcile
from refund_agent.rollout.org_adoption_records import ProviderPayment
from refund_agent.rollout.org_adoption_rule import count_reasons

ROOT = Path(__file__).resolve().parents[1]
PREFIX = "fnd.org-adoption."


def say(text):
    print(fill(text, width=79, subsequent_indent="  "))


def print_plan():
    plan = constructed_plan(build_candidate(build_suite(trials=1)).fingerprint)
    say("Constructed rollout settings; no deployed grant is issued.")
    say(
        f"Plan: {plan.plan_id}; signer: {plan.signed_by}; signed: {plan.signed_at.isoformat()}"
    )
    say(
        f"Hash namespace (constructed): {plan.canary_id}; unchanged across steps."
    )
    for step in plan.steps:
        say(
            f"{step.step_id} (constructed): requester share {step.requester_share:.0%}; tiers {step.tiers}"
        )
        say(
            f"Window (constructed): {step.window.start.isoformat()} to {step.window.end.isoformat()}"
        )
        say(
            f"Closing snapshot (constructed): {step.window.closed_at.isoformat()}; requests by received time"
        )
        say(
            "Threshold (constructed): governed share >= 99%; paid around path = 0"
        )
        for ref in step.required:
            say(
                f"Required (constructed reference): {ref.record_type} {ref.record_id}"
            )
            if ref.record_type == "audit":
                say(
                    f"Audit period: {ref.period_start.isoformat()} to {ref.period_end.isoformat()}; labels mature 30 days later"
                )
        for owner in step.owners:
            say(
                f"Owner (constructed): {owner.role}={owner.person}; work: {owner.work}"
            )
        say("Manual route (constructed): " + step.manual_route)
    for name in ("widen_rule", "hold_rule", "reverse_rule"):
        say(f"Rule (constructed): {name} = " + getattr(plan, name))
    validate_structure(plan)
    print("Structural fields: accepted")
    try:
        validate(plan, ())
    except ValueError as error:
        say("Admission validation: hold; " + str(error))
    else:
        raise AssertionError(
            "absent predecessor evidence must not admit a step"
        )


def print_balance_result(row):
    say(f"Fixture: {row['fixture']}; order: {row['order']}")
    say(
        f"Console (constructed): {row['console_cents']} cents at 10:00; history posting {row['settles_at']}"
    )
    print(f"Balance read: {row['balance_read_cents']} cents")
    print(
        f"Agent paid: {row['agent_paid_cents']} cents; tool payments: {row['agent_payments']}"
    )
    print(
        f"Total refunded: {row['total_refunded_cents']} cents against {row['paid_cents']} paid"
    )
    print(f"Reconciled excess: {row['excess_cents']} cents")
    print(
        f"Paid around path: {len(row['count'].around)}; imported receipts: {row['imported_receipts']}"
    )
    print(f"Agent outcome: {row['agent_outcome']}; reason: {row['reason']}")


def print_balance():
    print("Text case; constructed settlement, actual unchanged Tools gate")
    for early in (False, True):
        print("Early settlement control" if early else "Delayed settlement")
        print_balance_result(balance_case(TEXT_BALANCE, early=early))


def print_keys():
    print("Exercise 1: governed a,c,d; workarounds b,e")
    say(
        "A human refusal is authenticated through review and audit; an approved payment additionally needs refund.execute."
    )
    say(
        "Exercise 2: decision owner signs; release owner stops; reconciler records causes daily; desk reviews tier 2, decides referrals and keeps planted cases and second checks."
    )
    for number, n, console in ((3, 20, True), (5, 80, False)):
        count = reconcile(**accounting_fixture(n, console=console))
        print(
            f"Exercise {number}: received {count.received}; governed {count.governed}; paid around {len(count.around)}"
        )
        print(
            f"Missing decisions {len(count.missing)}; governed share {count.governed_percent:.2f}%"
        )
        # Both count gates fail on these fixtures, irrespective of admission evidence.
        assert count.governed_percent < 99
        print("Count-rule verdict: hold")
        if number == 5:
            print(
                "Planted denominator: decision records; falsely reports 100.00%"
            )
    print("Exercise 4: its own 9000/6000/4500-cent fixture (B8)")
    for early in (False, True):
        print("Early settlement control" if early else "Delayed settlement")
        print_balance_result(balance_case(EXERCISE_BALANCE, early=early))
    fixture = accounting_fixture(20, console=True)
    item = fixture["intake"][2]
    bank = ProviderPayment(
        payment_id="org-adoption-bank-1",
        order_id=item.order_id,
        amount_cents=2500,
        at=item.received_at,
        request_id=item.request.request_id,
        channel="bank",
    )
    result = reconcile(
        **{**fixture, "provider": fixture["provider"] + (bank,)}
    )
    reasons = count_reasons(
        constructed_plan("accounting-exercise-only").steps[0], result
    )
    assert "paid_around_path" in reasons
    say(
        f"Exercise 6 accounting rerun: bank {bank.order_id}, {bank.amount_cents} cents; "
        f"governed {result.governed}/{result.received}; paid around {len(result.around)}; hold."
    )
    say(
        "Duplicate money-out feeds count once; reference acceptance tests check the join."
    )
    unpaid = reconcile(**accounting_fixture(100, console=False))
    assert not count_reasons(
        constructed_plan("accounting-exercise-only").steps[0], unpaid
    )
    say(
        f"Unpaid gap count-rule verdict: widen; {unpaid.governed}/{unpaid.received} = "
        f"{unpaid.governed_percent:.2f}%. Conditional on complete admission evidence and recorded disposition."
    )


# region book:fnd.org-adoption.plan-01 output
if __name__ == "__main__":
    print_plan()
# endregion book:fnd.org-adoption.plan-01

# region book:fnd.org-adoption.balance-01 output
if __name__ == "__main__":
    print_balance()
# endregion book:fnd.org-adoption.balance-01

# region book:fnd.org-adoption.keys-01 output
if __name__ == "__main__":
    print_keys()
# endregion book:fnd.org-adoption.keys-01


def capture_org_adoption(destination):
    extractor = vars(_regions)
    source = Path(__file__).resolve()
    snippets = {
        s.snippet_id: s
        for s in extractor["extract_snippets_from_text"](
            source.read_text(), source
        )
    }
    outputs = extractor["capture_outputs"](
        snippets, sys.executable, timeout=120
    )
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    for name, text in outputs.items():
        (destination / f"_{name}.txt").write_text(text)
    return outputs


def write_data(destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    fixtures = {
        "org_adoption_window_03.json": serialize_fixture(
            accounting_fixture(20, console=True)
        ),
        "org_adoption_record_05.json": {
            **serialize_fixture(accounting_fixture(80, console=False)),
            "team_record": {
                "decision": "widen to everyone",
                "governed_percent": 100,
                "paid_around": 0,
                "denominator": "decision records",
            },
        },
        "org_adoption_balance_04.json": EXERCISE_BALANCE.model_dump(
            mode="json"
        ),
    }
    for name, value in fixtures.items():
        (destination / name).write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-outputs", action="store_true")
    parser.add_argument("--write-data", action="store_true")
    args = parser.parse_args()
    if args.write_outputs:
        outputs = capture_org_adoption(ROOT / "book_outputs")
        print(
            f"Captured {len(outputs)} outputs through the repository capture function"
        )
    if args.write_data:
        write_data(ROOT / "demos/data")


if __name__ == "__main__":
    main()
