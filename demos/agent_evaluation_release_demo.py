"""Constructed release evidence, effect trials and an admission-controlled shadow switch.

The full suite decides the result; an expected pass is never a stored assertion.
Importing prints and writes nothing. --write-data writes chapter-owned artifacts.
"""

import argparse
import contextlib
import io
import json
import os
from collections import Counter
from datetime import datetime, timezone
from fractions import Fraction
from functools import lru_cache
from pathlib import Path
from tempfile import TemporaryDirectory
from textwrap import fill

from demos.ml_data_evaluation_demo import wilson_interval
from refund_agent import contracts as c
from refund_agent.baseline import score_decisions
from refund_agent.data.generate import generate
from refund_agent.data.live import generate_live
from refund_agent.evaluation import _positive_data
from refund_agent.release.candidate import build_candidate, build_tools, run_request
from refund_agent.release.journal import Journal
from refund_agent.release.judges import agreement
from refund_agent.release.rule import evaluate_release, sign
from refund_agent.release.suite import build_suite
from refund_agent.release.trials import (
    constructed_curves,
    pass_at_k,
    pass_hat_k,
    plug_at_k,
    plug_hat_k,
)
from refund_agent.shadow.__main__ import start

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "demos/data"
EXERCISES = ROOT / "exercises/agent-evaluation"


def binding_test():
    data = _positive_data()
    candidate = build_candidate(build_suite(), verifier="independent")
    tools = build_tools(data, candidate, "shadow")
    before = (tools.ledger, tools.reviews)
    for request in data.requests:
        run_request(candidate, tools, request)
    return {
        "requests": len(data.requests),
        "payments_recorded": len(tools.shadow_payments),
        "reviews_recorded": len(tools.shadow_reviews),
        "gate_calls": tools.gate_calls,
        "ledger_unchanged": tools.ledger == before[0],
        "reviews_unchanged": tools.reviews == before[1],
        "money_moved": sum(p.amount_cents for p in tools.payments),
        "limit": "binding unit test; admission requires a signed passing release record",
    }


@lru_cache(maxsize=1)
def build_experiment(seed=7):
    if seed != 7:
        raise ValueError("the chosen release cases use seed 7")
    suite = build_suite()
    records = []
    with TemporaryDirectory(prefix="agent-evaluation-") as directory:
        journal = Journal(Path(directory) / "journal.jsonl")
        for name in ("as-built", "resume-saved", "deny-all", "gate-removed"):
            candidate = build_candidate(suite, name=name, verifier="independent")
            records.append(evaluate_release(candidate, suite, journal))
        if records[0]["decision"] == "pass":
            # Constructed owner's sign-off for this offline example, after its run.
            records[0] = sign(records[0], journal, "morgan", "2026-10-12T12:01:00+00:00")
        path = Path(directory) / "release.json"
        path.write_text(json.dumps(records[0]))
        try:
            shadow = start(path)
            shadow.update(
                status="started",
                reason="journaled pass and constructed owner sign-off match candidate and suite",
                first_day="2026-10-01",
                prepared_requests=shadow["requests"],
                binding_test=binding_test(),
            )
        except ValueError as error:
            stream = generate_live(seed, datetime(2026, 10, 1, tzinfo=timezone.utc), 1, 120)
            shadow = {
                "status": "unstarted",
                "reason": str(error),
                "first_day": None,
                "stream_fingerprint": stream.fingerprint,
                "prepared_requests": len(stream.data.requests),
                "person_reference": "generated labels; blind",
                "binding_test": binding_test(),
                "D6": {
                    "provisional": False,
                    "required_requests": 299,
                    "wrongful_denial_threshold": "no more than baseline on the same 299 requests; person paid and payment not found false within 30 days",
                    "signature": None,
                },
                "limits": [
                    "The full candidate is blocked; no shadow day was run.",
                    "A binding unit test does not grant release authority.",
                ],
            }
    return {"release": {"records": records}, "shadow": shadow}


def figure_data(seed=7):
    return constructed_curves(seed)


def print_wrapped(text):
    print(fill(text, width=79, break_long_words=False, break_on_hyphens=False))


def print_suite(suite):
    print_wrapped(
        "fixed must-pass suite; seed 7; 5 trials; hard pass conditions read ledger, audit, review queue and decision fields"
    )
    print_wrapped("| Part | Cases | Must-pass | Pass condition |")
    print_wrapped("|---|---:|---|---|")
    descriptions = {
        "public": "correct effects; tier 3 named review and preserved reason",
        "refusal": "no payment; audited refusal",
        "positive": "one payment of the expected amount after required approval",
        "fault": "correct persisted effects after timeout or malformed reply",
        "resume": "one approved payment across runtimes",
        "listed": "policy effects on the listed set; named review at tier 3",
        "state-continuation": "changed evidence reread; no stale payment",
        "contracts": "strict contracts and nonempty idempotency key",
        "approval": "changed proposal cannot reuse approval",
        "execution-time": "freshness limits checked at use",
        "memory-support": "support on candidate drafts; register beside loop",
        "regression": "spent change guard, not quality",
        "runtime-swap": "same observations with supplied tools",
        "offer-record": "fixed notice or eligible recorded policy",
        "verifier": "matched record and objection/timeout/void composition",
    }
    for part, count in Counter(case.part for case in suite.cases).items():
        print_wrapped(f"| {part} | {count} | yes; verifier when selected | {descriptions[part]} |")
    print_wrapped(f"manifest cases {len(suite.cases)}; fingerprint {suite.fingerprint}")
    print_wrapped(
        "comparison arms: candidate agent bundle; rules-only baseline on identical requests"
    )
    regression = __import__(
        "refund_agent.release.suite", fromlist=["spent_regression"]
    ).spent_regression()
    print_wrapped(
        f"regression: {len(regression.requests)} drafting cases; {len(regression.questions)} retrieval questions; logged spent artifact, no new golden consultation"
    )
    for line in (ROOT / "book_outputs/_fnd.memory-rag.funnel-01.txt").read_text().splitlines():
        if line.startswith("reranked "):
            print_wrapped("memory funnel reported apart: " + line)

    for generated in suite.generated:
        print_wrapped(
            f"generated apart: tau {generated.manifest['tau']}; {len(generated.scenarios)} cases; human review of {generated.manifest['human_review']['sample_size']} pending; no release authority alone"
        )
    print_wrapped(
        "budget: 2048 window; shares 300+600+150+250+200+400=1900; margin 148; step limit 12"
    )


def print_release(report):
    records = report["release"]["records"]
    print_wrapped("| Candidate | Decision | Failed cases | Added failures by part | Floor | Exit |")
    print_wrapped("|---|---|---:|---|---|---:|")
    baseline = set(records[0]["failed_cases"])
    for row in records:
        added = Counter(f["part"] for f in row["failures"] if f["case_id"] not in baseline)
        parts = ", ".join(f"{k}:{v}" for k, v in added.items()) or "none"
        print_wrapped(
            f"| {row['candidate']['name']} | {row['decision']} | {len(row['failed_cases'])} | {parts} | {row['floor']['pass']} | {int(row['decision'] != 'pass')} |"
        )
    first = records[0]
    print_wrapped("as-built failures: " + (", ".join(first["failed_cases"]) or "none"))
    print_wrapped(
        "every unexecuted tier 3 outcome opens named-approver review, including delivery_unknown; decision 1 requires it above 50000 cents"
    )
    print_wrapped(
        f"floor: {first['floor']['cases']} tier 1 public requests; forbidden {first['floor']['forbidden_payments']}; referrals/trial {first['floor']['referrals_per_trial']:g}; baseline {first['floor']['baseline_referrals']}"
    )
    for row in records:
        signature = "recorded" if row["signature"] else "absent"
        print_wrapped(
            f"{row['candidate']['name']}: {row['cases_run']} cases x {row['trials_run']} trials; earlier failed verdicts {len(row['earlier_failures'])}; owner sign-off {signature}"
        )
    resume = records[1]
    for case in resume["case_results"]:
        if case["part"] == "state-continuation":
            trial = case["trials"][0]
            control = next(r for r in first["case_results"] if r["case_id"] == case["case_id"])[
                "trials"
            ][0]
            print_wrapped(
                f"{case['case_id']}: saved-source mutation pays {trial['paid_cents']} cents; unchanged Tools pays {control['paid_cents']} cents; reason {control['reason']}"
            )
    print_wrapped(
        "planted refund.execute: gate(saved, ...), substituting the saved Dataset for live sources; unchanged Tools: gate(self._data, ...); skipping the reread alone is insufficient"
    )
    for generated in first["generated_apart"]:
        print_wrapped(
            f"as-built generated apart tau {generated['tau']}: {generated['passing']}/{generated['cases']} effects pass; human review pending"
        )
    removed = next(r for r in records if r["candidate"]["name"] == "gate-removed")
    for case in removed["case_results"]:
        if case["part"] == "refusal" and case["case_id"] in removed["failed_cases"]:
            trial = case["trials"][0]
            decision = trial["decision"]
            print_wrapped(
                f"gate-removed {case['case_id']}: "
                f"executed {decision['amount_cents'] if decision['executed'] else 0} cents"
            )
    print_wrapped(
        "release record: demos/data/agent_evaluation_release.json; sign-off is a recorded owner and time, not cryptographic"
    )


def print_trials():
    print_wrapped(f"scenario mu 0.5: pass@3 {1 - 0.5**3:.6f}; mu 0.95: pass^5 {0.95**5:.6f}")
    print_wrapped(
        f"box n=10, pass=3, k=5: 1-21/252={pass_at_k(10, 3, 5):.6f}; plug-in {plug_at_k(10, 3, 5):.6f}"
    )
    print_wrapped(
        f"box n=10, pass=8, k=3: 56/120={pass_hat_k(10, 8, 3):.6f}; plug-in {plug_hat_k(10, 8, 3):.6f}"
    )
    print_wrapped(
        "| Suite | k | Exact pass@k | Estimate pass@k | Exact pass^k | Estimate pass^k | Pooled pass^k |"
    )
    print_wrapped("|---|---:|---:|---:|---:|---:|---:|")
    curves = figure_data()
    for name, suite in curves["suites"].items():
        for row in suite["rows"]:
            if row["k"] in (1, 3, 5, 10):
                print_wrapped(
                    f"| {name} | {row['k']} | {row['exact_at']:.6f} | {row['estimate_at']:.6f} | {row['exact_hat']:.6f} | {row['estimate_hat']:.6f} | {row['pooled_hat']:.6f} |"
                )
    print_wrapped(
        "curves constructed; seed 7; 40 scenarios per suite; 20 trials per scenario; k 1 through 10"
    )
    first = build_experiment()["release"]["records"][0]
    public = [r for r in first["case_results"] if r["part"] == "public"]
    passing = sum(not r["trials"][0]["failed"] for r in public)
    identical = all(all(t == r["trials"][0] for t in r["trials"]) for r in public)
    decisions = [
        c.Decision.model_validate_json(json.dumps(r["trials"][0]["decision"])) for r in public
    ]
    raw = score_decisions(decisions, generate(7, orders=40, requests=40).correct)
    print_wrapped(
        f"agent public effects: {passing}/40 on each of 5 trials; identical {identical}; pass@5=pass^5={passing / 40:.6f}"
    )
    print_wrapped(
        f"raw generated-label score {raw.correct}/{raw.total}; 15 tier 3 referrals, including 14 deny-labelled cases; label comparison is separate from effect grading"
    )
    print_wrapped(
        "200 repeated deterministic public runs are 40 scenarios, not 200 independent new requests; this coverage suite bounds no live rate"
    )


def print_judge():
    print_wrapped("constructed bank: adjudicated reference, not measured bank data")
    print_wrapped("| Reference / judge | Fails | Passes |")
    print_wrapped("|---|---:|---:|")
    print_wrapped("| Failing | 14 | 6 |")
    print_wrapped("| Good | 5 | 175 |")
    row = agreement(14, 6, 5, 175)
    print_wrapped(
        f"200 letters; 20 failing; agreement {row['agreement']:.6f}; chance {row['chance']:.6f}; kappa_C {row['kappa']:.6f}; miss rate {row['miss_rate']:.6f}=6/20"
    )
    low, high = wilson_interval(6, 20)
    print_wrapped(f"miss rate 95% Wilson interval: [{low:.4f}, {high:.4f}]")
    all_pass = agreement(0, 20, 0, 180)
    print_wrapped(
        f"judge passes all: agreement {all_pass['agreement']:.6f}; kappa_C {all_pass['kappa']:.6f}"
    )
    print_wrapped(
        "a soft grader may hold for a person; its pass alone cannot pass a must-pass case; kappa is undefined when chance agreement is 1"
    )


def print_shadow(report):
    row = report["shadow"]
    print_wrapped(f"main shadow entry point: {row['status']}; {row['reason']}")
    if row["status"] == "started":
        print_wrapped(
            f"first day: {row['requests']} requests; agent agreement {row['agreement']}; baseline agreement {row['baseline_agreement']}; adjudication pending {row['adjudication_pending']}"
        )
        print_wrapped(
            "disagreements: "
            + ", ".join(f"{kind} {n}" for kind, n in sorted(row["disagreements"].items()))
        )
        print_wrapped(
            f"shadow day: would-pay {row['payments_recorded']}; would-review {row['reviews_recorded']}; gate calls {row['gate_calls']}; real ledger unchanged {row['ledger_unchanged']}; real review queue unchanged {row['review_queue_unchanged']}; moved 0 cents"
        )
        disagreements = [r for r in row["rows"] if r["disagreement"]]
        denial_referrals = sum(
            r["tier"] == 3 and r["agent"] == "refer" and r["person"] == "deny"
            for r in disagreements
        )
        stale_referrals = sum(
            r["tier"] in (1, 2) and r["agent"] == "refer" and r["agent_reason"].startswith("stale_")
            for r in disagreements
        )
        print_wrapped(
            f"tier 3 denial referrals {denial_referrals}; "
            f"stale-evidence referrals {stale_referrals}"
        )
        print_wrapped(
            f"verifier catches {row['verifier_catches']}; "
            f"false objections {row['verifier_false_objections']}"
        )
        window = row["shadow_to_canary"]
        print_wrapped(
            f"tier 1: {window['requests']} requests; forbidden {window['forbidden_payments']}; referrals {window['referrals']}; baseline referrals {window['baseline_referrals']}; paid labels pending {window['wrongful_denials_pending']}"
        )
    else:
        print_wrapped("no first-day agent comparison: shadow was not admitted")
    print_wrapped(
        f"prepared stream: {row['prepared_requests']} generated requests; start 2026-10-01; seed 7; generated labels stand in for a blind person's decisions"
    )
    test = row["binding_test"]
    print_wrapped(
        f"binding unit test: {test['requests']} payable/approval requests; would-pay {test['payments_recorded']}; would-review {test['reviews_recorded']}; gate calls {test['gate_calls']}; moved {test['money_moved']} cents"
    )
    print_wrapped(
        f"real ledger unchanged {test['ledger_unchanged']}; real review queue unchanged {test['reviews_unchanged']}; both refund.execute and approval.request recorded"
    )
    print_wrapped(
        "entry point refusal tests: missing, failed, unsigned, candidate mismatch, suite mismatch, prior unchanged failure, forged sign-off; each leaves shadow unstarted"
    )
    print_wrapped(
        "chapter 21 record: A held; B held; fixed notice retained; notices sent 0; offer outcomes observed 0"
    )
    print_wrapped(
        "D6: binding, not provisional; same 299 tier 1 requests; wrongful denials no more than baseline; paid person decision not found false within 30 days"
    )
    print_wrapped(
        "299-request window and paid-only 30-day labels incomplete here; no canary sign-off"
    )
    print_wrapped(
        "companion: simulated ledger; constructed owner sign-off for the offline example; deployment: a person carries out every refund through Part V"
    )


def print_keys():
    print("exercise 3, constructed telecom: n=12, pass=9, k=3")
    print(
        f"pass@3 219/220={pass_at_k(12, 9, 3):.8f}; plug-in {plug_at_k(12, 9, 3):.8f}; pass^3 84/220={pass_hat_k(12, 9, 3):.8f}; plug-in {plug_hat_k(12, 9, 3):.8f}"
    )
    row = agreement(30, 10, 18, 342)
    print(
        "exercise 4, constructed insurance: cells 30,10,18,342; 400 letters, 40 failing, 360 good"
    )
    print(
        f"agreement {row['agreement']:.6f}; chance {row['chance']:.6f}; kappa_C {row['kappa']:.6f}; miss rate {row['miss_rate']:.6f}; false holds 18/360={row['false_hold_rate']:.6f}; all-pass agreement 0.900000, kappa_C 0"
    )
    fixture = json.loads((EXERCISES / "record-05.json").read_text())
    trials = [p for r in fixture["runs"] for p in r["trial_results"]]
    print(
        f"exercise 5: same candidate and suite in 2 runs; {sum(trials)}/{len(trials)} trials pass; first run fails trial 3 of 5; unchanged rerun cannot erase it"
    )
    print(
        "exercise 6: fixed-scenario wiring of the tier 2 approval-reuse cell; stale-approval candidate must exit 1, as-built must exit 0; submitted assertions are rejected"
    )
    print(
        "practice pooled suite: 9 always-pass, 1 with chance 0.5; k=4; scenario pass@4 0.99375000, pass^4 0.90625000; pooled pass@4 0.99999375, pass^4 0.81450625"
    )
    print(
        f"practice shadow: shown 39/40={float(Fraction(39, 40)):.6f}; blind 68/80={float(Fraction(68, 80)):.6f}; pooled 107/120={float(Fraction(107, 120)):.6f}; 40 contaminated requests kept apart"
    )


def print_cli(report):
    from refund_agent.release.__main__ import main as release_main
    from refund_agent.release.__main__ import write_record
    from refund_agent.shadow.__main__ import main as shadow_main

    print_wrapped("cached release runs from build_experiment; sign and shadow commands run here")
    with TemporaryDirectory(prefix="agent-evaluation-cli-") as directory:
        for row in report["release"]["records"][:2]:
            name = row["candidate"]["name"]
            run_dir = Path(directory) / name
            path = run_dir / "release.json"
            unsigned = {**row, "signature": None}
            Journal(run_dir / "journal.jsonl").append(unsigned)
            write_record(path, unsigned)
            print_wrapped(
                f"$ python -m refund_agent.release --candidate {name} "
                f"--verifier independent --run-dir RUN/{name}"
            )
            print_wrapped(
                f"{name}: {row['decision']}; failed cases: "
                + (", ".join(row["failed_cases"]) or "none")
            )
            print(f"exit {int(row['decision'] != 'pass')}")
            commands = (
                (
                    f"python -m refund_agent.release sign --owner morgan --run-dir RUN/{name}",
                    release_main,
                    [
                        "sign",
                        "--owner",
                        "morgan",
                        "--run-dir",
                        str(run_dir),
                        "--at",
                        "2026-10-12T12:01:00+00:00",
                    ],
                ),
                (
                    f"python -m refund_agent.shadow --release-record RUN/{name}/release.json",
                    shadow_main,
                    ["--release-record", str(path), "--output", str(run_dir / "shadow.json")],
                ),
            )
            for command, main, argv in commands:
                print_wrapped("$ " + command)
                with contextlib.redirect_stdout(io.StringIO()) as response:
                    status = main(argv)
                print_wrapped(response.getvalue().splitlines()[0])
                print(f"exit {status}")
            assert (run_dir / "shadow.json").exists() == (row["decision"] == "pass")
    destination = os.environ.get("AGENT_EVALUATION_CAPTURE_DIR")
    if destination:
        data = Path(destination) / "demos/data"
        data.mkdir(parents=True, exist_ok=True)
        for name, value in (
            ("release", report["release"]),
            ("shadow", report["shadow"]),
            ("figures", figure_data()),
        ):
            (data / f"agent_evaluation_{name}.json").write_text(
                json.dumps(value, indent=2, sort_keys=True) + "\n"
            )


# region book:fnd.agent-evaluation.suite-01 output
if __name__ == "__main__":
    from demos.agent_evaluation_release_demo import print_suite
    from refund_agent.release.suite import build_suite

    # Cases and effect predicates are fixed before any trial.
    suite = build_suite()
    print_suite(suite)
# endregion book:fnd.agent-evaluation.suite-01

# region book:fnd.agent-evaluation.release-01 output
if __name__ == "__main__":
    from demos.agent_evaluation_release_demo import (
        build_experiment,
        print_release,
    )

    print_release(build_experiment())
# endregion book:fnd.agent-evaluation.release-01

# region book:fnd.agent-evaluation.judge-01 output
if __name__ == "__main__":
    from demos.agent_evaluation_release_demo import print_judge

    print_judge()
# endregion book:fnd.agent-evaluation.judge-01

# region book:fnd.agent-evaluation.shadow-01 output
if __name__ == "__main__":
    from demos.agent_evaluation_release_demo import (
        build_experiment,
        print_shadow,
    )

    # start() validates the matching signed record and journal before a run.
    print_shadow(build_experiment())
# endregion book:fnd.agent-evaluation.shadow-01

# region book:fnd.agent-evaluation.keys-01 output
if __name__ == "__main__":
    from demos.agent_evaluation_release_demo import print_keys

    print_keys()
# endregion book:fnd.agent-evaluation.keys-01


# region book:fnd.agent-evaluation.cli-01 output
if __name__ == "__main__":
    from demos.agent_evaluation_release_demo import build_experiment, print_cli

    print_cli(build_experiment())
# endregion book:fnd.agent-evaluation.cli-01


def write_data(report, *, data=DATA, exercises=EXERCISES):
    data.mkdir(parents=True, exist_ok=True)
    exercises.mkdir(parents=True, exist_ok=True)
    for name, value in (
        ("figures", figure_data()),
        ("release", report["release"]),
        ("shadow", report["shadow"]),
    ):
        (data / f"agent_evaluation_{name}.json").write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n"
        )
    coverage = build_suite().coverage()
    coverage["approval-reuse"]["2"] = []
    (exercises / "coverage-06.json").write_text(
        json.dumps(coverage, indent=2, sort_keys=True) + "\n"
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-data", action="store_true")
    args = parser.parse_args(argv)
    if args.write_data:
        write_data(build_experiment())


if __name__ == "__main__":
    main()
