"""Reference solution. Given exercise records are not operating evidence."""

from fractions import Fraction


def exercise_01():
    return ("consistent", "reversible", "signed_in_order", "complete")


def exercise_02():
    return {
        "stage": 6,
        "launch_signed": False,
        "withdraws": "tier1",
        "mechanism": "chapter 34 monitor -> chapter 38 stage authority",
        "decider_after": "person",
        "paid_refunds_undone": False,
    }


def exercise_03(packet):
    failed = []
    for record in packet["records"]:
        bundle = record["bundle"]
        if record["rule"] == "any" or bundle == packet["launched_bundle"]:
            continue
        if record["rule"] == "strict":
            failed.append(record["kind"])
            continue
        needed = set()
        seen = set()
        while bundle != packet["launched_bundle"]:
            if bundle in seen:
                raise ValueError("promotion cycle")
            seen.add(bundle)
            link = next(
                p for p in packet["promotions"] if p["incumbent"] == bundle
            )
            needed.update(
                kind
                for item in link["changes"]
                for kind in packet["requirements"][item]
            )
            bundle = link["candidate"]
        if record["kind"] in needed:
            failed.append(record["kind"])
    return {
        "verdict": "hold" if failed else "launch",
        "not_counted": tuple(failed),
    }


def exercise_04():
    manual = Fraction(200 * 6 + 80 * 20 + 20 * 45, 60)
    agent = Fraction(30 * 6 + 34 * 6 + 80 * 20 + 20 * 45, 60)
    return {
        "manual_hours": manual,
        "agent_hours": agent,
        "manual_reopened": 2,
        "agent_reopened": 3,
        "verdict": "hold" if agent < manual and 3 > 2 else "launch",
    }


def exercise_05(decision):
    rehearsal, promotion = decision["rehearsal"], decision["promotion"]
    return {
        "verdict": "launch"
        if (
            rehearsal["candidate"] == decision["bundle"]
            and rehearsal["rollback_target"] == promotion["rollback_target"]
        )
        else "hold",
        "target": promotion["rollback_target"],
    }


def print_exercise_05_key(decision):
    answer = exercise_05(decision)
    print(
        f"5: {answer['verdict']}; rehearse {decision['bundle']} with "
        f"{answer['target']}, not {decision['rehearsal']['rollback_target']}"
    )


def check_maturity(record, gate_day, mode="companion"):
    if mode not in {"companion", "deployed"}:
        raise ValueError("mode must be companion or deployed")
    if mode == "companion" or not record.label_dependent:
        return ()
    if record.last_payment_day is None:
        return (f"{record.record_id}: last payment day missing",)
    mature = record.last_payment_day + 30
    return (
        (f"{record.record_id}: labels mature on day {mature:g}",)
        if gate_day < mature
        else ()
    )
