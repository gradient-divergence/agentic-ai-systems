"""Chapter 32's local start and rehearsal commands; no remote effects."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from sys import stdin

from refund_agent.release.candidate import build_candidate
from refund_agent.release.suite import build_suite


def main():
    # Initialize the fixed core at the host boundary before selecting extensions.
    build_candidate(build_suite())
    from refund_agent.canary.cloud_native_deployment_plan import CanaryPlan, sign_plan
    from refund_agent.canary.cloud_native_deployment_rehearsal import restart_payload, run_rehearsal
    from refund_agent.canary.cloud_native_deployment_start import start_canary
    from refund_agent.canary.cloud_native_deployment_stop import CanarySession
    from refund_agent.canary.cloud_native_deployment_store import AppendJournal, AppendStore
    from refund_agent.canary.cloud_native_deployment_window import ShadowWindowResult

    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("restart")
    restart_store = commands.add_parser("restart-store")
    restart_store.add_argument("--namespace", required=True)
    restart_store.add_argument("--store", type=Path)
    rehearse = commands.add_parser("rehearse")
    rehearse.add_argument("--candidate", choices=("as-built", "planted-field"), required=True)
    rehearse.add_argument("--cadence", choices=("4h", "daily"), required=True)
    rehearse.add_argument("--out", type=Path, required=True)
    start = commands.add_parser("start")
    for name in ("plan", "release-record", "window", "rehearsal"):
        start.add_argument("--" + name, type=Path, required=True)
    for name in ("error-budget", "promotion-record", "scope-record", "output"):
        start.add_argument("--" + name, type=Path)
    start.add_argument("--owner", required=True)
    start.add_argument("--at", required=True)
    start.add_argument("--namespace", default="cloud-native-deployment-start")
    args = parser.parse_args()
    if args.command == "restart":
        print(json.dumps(restart_payload(json.load(stdin)), sort_keys=True))
        return 0
    if args.command == "restart-store":
        from refund_agent.canary.cloud_native_deployment_component import (
            STORE_PATH,
            restore_selection,
            runtime_paths,
        )

        store = AppendStore(args.store if args.store is not None else STORE_PATH, args.namespace)
        payload = store.latest("restart-payload", "rollback")
        if payload is None:
            raise ValueError("restart payload missing")
        target = payload["target"]
        restore_selection(target)
        suite = build_suite(components=target["tools_composition"]["components"])
        current = build_candidate(suite, target["name"], target["runtime"]["identity"],
                                  target["tools_composition"]["verifier"])
        if current.record() != target:
            raise ValueError("restart target differs from its recorded implementation")
        if store.path != runtime_paths(current)["store"]:
            raise ValueError("restart transport differs from the selected owner declaration")
        store.append("restart-result", "rollback", restart_payload(payload))
        return 0
    if args.command == "rehearse":
        from refund_agent.batch_stream.component import selector as event_selector
        from refund_agent.canary.cloud_native_deployment_component import selector
        from refund_agent.data.live import generate_live
        from refund_agent.sources.data_systems_release import SELECTOR as DATA

        planted = args.candidate == "planted-field"
        source_suite = build_suite(components=[DATA, event_selector(), selector(
            record_schema="canary-id" if planted else "closed",
            reader="canary-id" if planted else "closed")])
        target_suite = build_suite(components=[DATA, event_selector(), selector()])
        source = build_candidate(source_suite)
        target = build_candidate(target_suite, verifier="off" if planted else "independent")
        at = datetime(2026, 10, 15, tzinfo=timezone.utc)
        plan = sign_plan(CanaryPlan(canary_id="cloud-native-deployment-cli-rehearsal",
                                    candidate_fingerprint=source.fingerprint,
                                    suite_fingerprint=source_suite.fingerprint,
                                    rollback_target_fingerprint=target.fingerprint,
                                    requirement_ref="rehearsal-only", rehearsal_ref="cloud-native-deployment-cli"),
                         "morgan", at)
        result = run_rehearsal(plan, generate_live(7, at, 14, 233), 14, ["forbidden_payment"],
                               args.cadence, target, candidate=source)
        args.out.mkdir(parents=True, exist_ok=True)
        text = result.model_dump_json(indent=2)
        (args.out / "cloud-native-deployment-rehearsal.json").write_text(text + "\n")
        print(text)
        return 0 if result.verdict == "pass" else 1
    # Input documents are read at the host boundary, before candidate selection.
    plan = CanaryPlan.model_validate_json(args.plan.read_text())
    window_text = args.window.read_text()
    release = json.loads(args.release_record.read_text())
    rehearsal = json.loads(args.rehearsal.read_text())
    budget = json.loads(args.error_budget.read_text()) if args.error_budget else None
    promotion = json.loads(args.promotion_record.read_text()) if args.promotion_record else None
    scope = json.loads(args.scope_record.read_text()) if args.scope_record else None
    from refund_agent.canary.cloud_native_deployment_component import (
        restore_selection,
        runtime_paths,
    )

    restore_selection(release["candidate"])
    suite = build_suite(components=release["candidate"]["tools_composition"].get("components", ()))
    recorded = release["candidate"]
    candidate = build_candidate(suite, recorded["name"], recorded["runtime"]["identity"],
                                recorded["tools_composition"]["verifier"])
    window = ShadowWindowResult.model_validate_json(window_text)
    paths = runtime_paths(candidate)
    session = CanarySession(plan, AppendStore(paths["store"], args.namespace))
    result = start_canary(plan, release, window, rehearsal, candidate=candidate, suite=suite,
                          journal=AppendJournal(paths["journal"]),
                          session=session, owner=args.owner, at=datetime.fromisoformat(args.at),
                          error_budget=budget, records={"promotion": promotion, "scope": scope})
    text = result.model_dump_json(indent=2)
    if args.output:
        args.output.write_text(text + "\n")
    print(text)
    return 0 if result.accepted else 1


if __name__ == "__main__":
    raise SystemExit(main())
