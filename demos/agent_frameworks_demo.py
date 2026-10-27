"""The refund agent on two offline runtimes, with chosen failures and cross-runtime resumes."""

# region book:fnd.agent-frameworks.adapter-01
from refund_agent.runtimes.adapter import AdapterSettings, RefundAdapter


def mount(tools, model=None, max_steps=12):
    """Mount these tools.

    The runtime supplies mechanism and the adapter the rules.
    """
    settings = AdapterSettings(
        end_at_audit=True,  # No-tool drafts continue; keep one step for audit.
        # before_call checks the phase before dispatch.
        one_tool_per_phase=True,
        # after_call and on_error end refusals as referrals.
        refer_on_failure=True,
        no_retry=True,  # A caller retries a run; the runtime retries no call.
        agent_budget=True,
        max_steps=max_steps,
    )
    # Every registered handler reaches this instance's call,
    # including its gate.
    return RefundAdapter(tools, model, settings=settings)


# endregion book:fnd.agent-frameworks.adapter-01


import argparse
import json
from dataclasses import replace
from functools import lru_cache
from pathlib import Path
from typing import cast

from refund_agent.runtimes.swap import (
    FIELDS,
    build_cases,
    build_record,
    compare,
    write_record,
)

HERE = Path(__file__).resolve().parent
BOOK = HERE.parent / "book_outputs"


@lru_cache(maxsize=1)
def results():
    return build_record(seed=7)


def paid(record):
    return sum(r["amount_cents"] for r in record["ledger_receipts"])


def fixture_03(record):
    config = next(row for row in record["configurations"] if row["name"] == "referral ending")
    names = (
        "public-r-7-0007",
        "refusal-4-ambiguous_order",
        "execute-before",
        "execute-after",
        "shipment-timeout",
        "budget-four",
    )
    rows = {row["case_id"]: row for row in config["cases"]}
    return {
        "configuration": config["name"],
        "settings": config["settings"],
        "fields_compared": list(FIELDS),
        "cases": [rows[name] for name in names],
    }


def fixture_05(record):
    config = next(row for row in record["configurations"] if row["name"] == "referral ending")
    rows = [row for row in config["cases"] if row["kind"] == "public"]
    return {
        "configuration": config["name"],
        "settings": config["settings"],
        "claim": "unchanged, ready for shadow",
        "fields_compared": ["decision", "paid_cents"],
        "cases": [
            {
                "case_id": row["case_id"],
                "own_decision": row["own"]["decision"],
                "second_decision": row["second"]["decision"],
                "own_paid_cents": paid(row["own"]),
                "second_paid_cents": paid(row["second"]),
            }
            for row in rows
        ],
    }


def print_swap(record):
    counts = record["case_counts"]
    total = sum(counts.values())
    print(
        f"seed {record['seed']}; {total} chosen cases: "
        + ", ".join(f"{value} {kind}" for kind, value in counts.items())
    )
    print("| Configuration | Cases that differ | Money differs | Cents against policy |")
    print("|---|---:|---:|---:|")
    for row in record["configurations"]:
        print(
            f"| {row['name']} | {row['different_cases']}/{total} | "
            f"{row['money_differences']} | {row['unauthorized_cents']} |"
        )
    full = record["configurations"][-1]
    print(f"full adapter: {full['decision']}")
    print("Runtime defaults and injected faults are this companion's own.")
    print("The full adapter compares: " + ", ".join(record["fields_compared"]) + ".")
    print(
        "Own loop: refund_agent/agent/__init__.py; second runtime: refund_agent/runtimes/runner.py."
    )
    print(
        "Adapter: refund_agent/runtimes/adapter.py; contracts and gate remain in the agent's code."
    )
    print(
        f"adapter version {record['adapter']['version']}; agent budget 12, one step reserved for audit."
    )
    print("Session and spans: in-process runtime; approvals, ledger and audit: supplied Tools.")
    print("Frozen reads: 0; held-out reads: 0.")
    for limit in record["limits"]:
        print(limit)


def print_timeout(record):
    config = next(row for row in record["configurations"] if row["name"] == "referral ending")
    row = next(row for row in config["cases"] if row["case_id"] == "execute-before")
    print(
        "refund.execute times out before acting; audit ending and refusal handling carried, retry left on."
    )
    for side in ("own", "second"):
        observed = row[side]
        print(
            f"{side}: {observed['outcome']} {observed['reason']}; paid {paid(observed)} cents; "
            f"execute calls {sum(call['name'] == 'refund.execute' for call in observed['tool_calls'])}; "
            f"receipts {len(observed['ledger_receipts'])}"
        )
        print(f"  failure detail: {observed['failure_detail'] or 'none'}")


def print_resume(record):
    rows = [r for r in record["configurations"][-1]["cases"] if r["kind"] == "resume"]
    for row in rows:
        seen = row["second"]
        print(
            f"{row['case_id']}: tier {seen['tier']}; approved; paid {paid(seen)} cents;\n  "
            f"receipts {len(seen['ledger_receipts'])}; differences {len(row['different_fields'])}"
        )
    print("Approval records stay in supplied Tools; no hosted session was exported.")


def exercise_settings():
    full = AdapterSettings()
    return (
        ("failure back to model", replace(full, refer_on_failure=False), "refer_on_failure=True"),
        ("two retries", replace(full, no_retry=False, retries=2), "no_retry=True"),
        ("no-tool reply ends run", replace(full, end_at_audit=False), "end_at_audit=True"),
        (
            "six turns including audit",
            replace(full, max_steps=6),
            "agent budget 12, reserve 1 for audit",
        ),
        (
            "all four",
            replace(
                full,
                refer_on_failure=False,
                no_retry=False,
                retries=2,
                end_at_audit=False,
                max_steps=6,
            ),
            "full adapter",
        ),
    )


def print_keys(record):
    fixture = fixture_03(record)
    rows = fixture["cases"]
    print(f"Exercise 3: {len(rows)} cases; configuration {fixture['configuration']}.")
    print("| Field | Cases that differ |")
    print("|---|---:|")
    for field in FIELDS:
        print(f"| {field} | {sum(row['own'][field] != row['second'][field] for row in rows)} |")
    print("Decision: held. Retry and the runtime's turn limit must carry the agent's rules.")
    print("Exercise 4: each setting alone over the full adapter, then all four.")
    cases = build_cases()
    for label, settings, remedy in exercise_settings():
        compared = compare(cases, settings)
        differing = [row for row in compared["cases"] if row["different_fields"]]
        catching = differing[0]["case_id"] if differing else "none"
        print(
            f"{label}: {len(differing)}/{len(cases)} differ; catching case {catching}; remove with {remedy}."
        )
        timeout = next(row for row in compared["cases"] if row["case_id"] == "execute-before")
        seen = timeout["second"]
        print(
            f"  execute-before: {seen['outcome']} {seen['reason']}; {paid(seen)} cents; "
            f"execute calls {sum(call['name'] == 'refund.execute' for call in seen['tool_calls'])}; "
            f"audits {len(seen['audits'])}."
        )
    timeout = next(
        row for row in record["configurations"][-1]["cases"] if row["case_id"] == "execute-before"
    )
    own = timeout["own"]
    print(f"own execute-before: {own['outcome']}; {paid(own)} cents; {own['failure_detail']}.")
    from refund_agent.agent import Agent
    from refund_agent.tools import Tools

    public = [case for case in cases if case.kind == "public"]
    traces = [Agent(Tools(case.data)).run(case.request) for case in public]
    longest = max(len(trace.steps) for trace in traces)
    print(
        f"Public loop: 9 phase names; longest run {longest} executed steps; "
        f"{sum(len(t.steps) == longest for t in traces)} runs at that length; "
        f"{sum(t.decision.executed for t in traces)} payments."
    )
    bad = fixture_05(record)
    disagreement = sum(
        row["own_decision"] != row["second_decision"]
        or row["own_paid_cents"] != row["second_paid_cents"]
        for row in bad["cases"]
    )
    print(
        f"Exercise 5: {len(bad['cases'])} public cases; {disagreement} differences in its two fields."
    )
    print(
        "The retry remains on. Missing faults and fields invalidate 'unchanged, ready for shadow'."
    )
    retry = next(row for row in record["configurations"] if row["name"] == "referral ending")
    caught = next(row for row in retry["cases"] if row["case_id"] == "execute-before")
    print(
        f"Catching case execute-before: own {paid(caught['own'])} cents, "
        f"second {paid(caught['second'])} cents; fields {', '.join(caught['different_fields'])}. Held."
    )
    from demos.harness_demo import print_exercise_key

    print_exercise_key()


def cumulative_replay():
    """Training-only memory experiment, then the merged loop on three tool/runtime compositions."""
    from demos.memory_rag_demo import (
        DRAFT,
        WINDOW,
        demonstration_register,
        request_state,
        routed_draft,
        training_data,
        utc,
    )
    from refund_agent.agent import Agent
    from refund_agent.memory.assemble import assemble, draft_from_facts
    from refund_agent.memory.memory import NOTICE, PolicyMemory
    from refund_agent.memory.register import project
    from refund_agent.memory.support import support
    from refund_agent.runtimes.faults import FaultTools
    from refund_agent.runtimes.swap import observe
    from refund_agent.tools import Tools
    from refund_agent.verifier.experiment import scoped_data
    from refund_agent.verifier.independent import IndependentVerifier, VerifiedTools

    data, requests = training_data(7)
    request = next(r for r in requests if r.request_id == "r-7-0085")
    register = demonstration_register()
    old = PolicyMemory(register, utc(4, 20))
    fresh = PolicyMemory(register, request.at - NOTICE)
    old_read = old.retrieve(request.reason, request.at, 1)
    fresh_read = fresh.retrieve(request.reason, request.at, 1)
    old_policy = max(project(old.snapshot, old.snapshot.at), key=lambda p: p.effective_at)
    fresh_policy = next(
        p
        for p in project(fresh.snapshot, fresh.snapshot.at)
        if p.version == fresh_read.hits[0].clause.version
    )
    fresh_state = request_state(data, request, fresh_policy)
    fresh_state.freshness["policy"] = (request.at - fresh.observed_at).total_seconds()
    assembly = assemble(fresh_state, [], WINDOW - DRAFT)
    fresh_draft = draft_from_facts(assembly.facts)
    memory = {
        "old_refusal": old_read.refusal,
        "old_draft": routed_draft(request_state(data, request, old_policy)),
        "fresh_refusal": fresh_read.refusal,
        "fresh_age_days": (request.at - fresh.observed_at).total_seconds() / 86400,
        "fresh_draft": fresh_draft,
        "assembly": assembly,
        "support": support(fresh_draft, register, assembly.facts),
    }
    runs = {}
    for name in ("own ordinary", "own verified", "second verified"):
        inputs = scoped_data(data, request)  # Generated snapshot; no PolicyMemory enters the loop.
        owner = (
            Tools(inputs)
            if name == "own ordinary"
            else VerifiedTools(inputs, IndependentVerifier())
        )
        tools = FaultTools(owner)
        runtime = mount(tools) if name == "second verified" else Agent(cast(Tools, tools))
        trace = runtime.run(request)
        runs[name] = {
            "runtime": runtime,
            "tools": owner,
            "record": observe(trace, tools),
            "verdicts": owner.verdict_log.entries if isinstance(owner, VerifiedTools) else (),
        }
    own, second = runs["own verified"]["record"], runs["second verified"]["record"]
    return {
        "request": request,
        "memory": memory,
        "runs": runs,
        "different_fields": [field for field in FIELDS if own[field] != second[field]],
    }


def print_replay(replay):
    request, memory = replay["request"], replay["memory"]
    print(
        f"{request.request_id}: {request.at.isoformat()}; {request.amount_cents} cents; {request.reason}"
    )
    print("separate memory experiment beside the loop:")
    old, fresh = memory["old_draft"], memory["fresh_draft"]
    print(
        f"April copy refused at use: {memory['old_refusal']};\n  "
        f"counterfactual draft {old.outcome.value} {old.reason}"
    )
    print(
        f"copy age {memory['fresh_age_days']:.0f} days: {memory['fresh_refusal'] or 'accepted'};\n  "
        f"draft {fresh.outcome.value} {fresh.reason}; {fresh.policy_citation}"
    )
    assembly = memory["assembly"]
    print(f"assembler required facts: {assembly.words} words")
    for line in assembly.lines:
        print(f"  {line}")
    print(
        f"approval support: {memory['support']['status']}; {memory['support']['reason'] or 'no refusal'}"
    )
    print("loop as merged: generated policy snapshot; memory copy is not wired in")
    print("composition: VerifiedTools under RefundAdapter (full adapter);\n  FaultTools records calls")
    for name, run in replay["runs"].items():
        record = run["record"]
        print(
            f"{name}: {record['outcome']} {record['reason']}; tier {record['tier']};\n  "
            f"paid {paid(record)} cents; receipts {len(record['ledger_receipts'])}; audit records {len(record['audits'])}"
        )
        print(
            "  receipts: "
            + ", ".join(
                f"{r['receipt_id']} ({r['amount_cents']} cents)" for r in record["ledger_receipts"]
            )
        )
        print(
            "  audits: "
            + ",\n    ".join(
                f"{a['event']} (money_moved {a['money_moved']}, {a['paid_amount_cents']} cents)"
                for a in record["audits"]
            )
        )
        for verdict in run["verdicts"]:
            print(f"  verifier: {verdict.answer}; {verdict.objection_reason or 'no objection'}")
    print(f"runtime fields differing: {len(replay['different_fields'])}/{len(FIELDS)}")


# region book:fnd.agent-frameworks.swap-01 output
if __name__ == "__main__":
    print_swap(results())
# endregion book:fnd.agent-frameworks.swap-01


# region book:fnd.agent-frameworks.swap-02 output
if __name__ == "__main__":
    print_timeout(results())
# endregion book:fnd.agent-frameworks.swap-02


# region book:fnd.agent-frameworks.resume-01 output
if __name__ == "__main__":
    print_resume(results())
# endregion book:fnd.agent-frameworks.resume-01


# region book:fnd.agent-frameworks.keys-01 output
if __name__ == "__main__":
    print_keys(results())
# endregion book:fnd.agent-frameworks.keys-01


# region book:fnd.agent-frameworks.replay-01 output
if __name__ == "__main__":
    print_replay(cumulative_replay())
# endregion book:fnd.agent-frameworks.replay-01


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-data", action="store_true")
    args = parser.parse_args(argv)
    if args.write_data:
        record = results()
        write_record(record, BOOK / "_fnd.agent-frameworks.swap-record.json")
        for name, fixture in (("03", fixture_03(record)), ("05", fixture_05(record))):
            path = HERE / "data" / f"agent_frameworks_swap_{name}.json"
            path.write_text(json.dumps(fixture, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
