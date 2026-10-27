"""Launch commands; real registry/producer adapters are supplied by the merge."""

import argparse
import json
from collections.abc import Callable, Mapping
from importlib import import_module
from pathlib import Path
from typing import Protocol, TypedDict

from refund_agent.canary.cloud_native_deployment_plan import OWNERS
from refund_agent.launch import LaunchRequirement
from refund_agent.launch.capstone_blueprint_decision import (
    Admission,
    AdmissionPort,
    OpenItem,
    StandingControl,
    admit,
    sign,
)
from refund_agent.launch.capstone_blueprint_gate import (
    LaunchVerdict,
    RegistryPort,
    check,
)
from refund_agent.launch.capstone_blueprint_packet import (
    EvidenceFile,
    assemble,
)
from refund_agent.launch.capstone_blueprint_refresh import refresh
from refund_agent.launch.capstone_blueprint_scan import LaunchLogs, gate_events


class CommandRegistry(RegistryPort, AdmissionPort, Protocol):
    """The merge's real registry adapter implements both reader and admission."""


class CommandContext(TypedDict):
    registry: CommandRegistry | None
    owners: dict
    logs: LaunchLogs
    producers: Mapping[str, Callable[..., EvidenceFile]]
    controls: tuple[StandingControl, ...]
    open_items: tuple[OpenItem, ...]


def owner_roles():
    rows = json.loads(OWNERS.read_text())["owners"]
    return {
        "roles": {
            role: tuple(r["id"] for r in rows if role in r["roles"])
            for role in (
                "decision_owner",
                "release_owner",
                "reviewer",
                "named_approver",
            )
        },
        "layers": {},
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("gate", "refresh", "admit", "sign")
    )
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--requirement", required=True, type=Path)
    parser.add_argument(
        "--context",
        help="merged refund_agent module:function returning real registry/producer adapters",
    )
    parser.add_argument("--owner", default="morgan")
    parser.add_argument("--day", type=float)
    args = parser.parse_args(argv)
    try:
        req = LaunchRequirement.model_validate_json(
            args.requirement.read_text()
        )
        context: CommandContext = {
            "registry": None,
            "owners": owner_roles(),
            "logs": LaunchLogs(),
            "producers": {},
            "controls": (),
            "open_items": (),
        }
        if args.context:
            module, function = args.context.split(":", 1)
            if not module.startswith("refund_agent."):
                raise ValueError("context must be a refund_agent module")
            context = getattr(import_module(module), function)(
                run_dir=args.run, requirement=req
            )
        packet = assemble(args.run, context["registry"], req)
        day = req.gate_day if args.day is None else args.day
        if args.command == "gate":
            result = check(
                packet,
                req,
                context["registry"],
                context["logs"],
                context["owners"],
                day,
            )
            (args.run / "capstone-blueprint-gate-cli.json").write_text(
                result.model_dump_json(indent=2) + "\n"
            )
        else:
            verdict = LaunchVerdict.model_validate_json(
                (args.run / "capstone-blueprint-gate-cli.json").read_text()
            )
            if args.command in {"admit", "sign"}:
                current = check(
                    packet,
                    req,
                    context["registry"],
                    context["logs"],
                    context["owners"],
                    req.gate_day,
                )
                if current.verdict != "launch" or current != verdict:
                    raise ValueError(
                        "current gate holds or saved verdict differs from current packet/requirement"
                    )
                conditions = {
                    condition.condition_id
                    for row in req.rows
                    for condition in row.reversals
                }
                if any(
                    verdict.day < event.day <= day
                    and event.condition_id in conditions
                    and not event.planned_time_box
                    for event in gate_events(
                        context["logs"], context["registry"], 0
                    )
                ):
                    raise ValueError(
                        "current reversal fired after the saved verdict"
                    )
                verdict = current
            if args.command == "refresh":
                if context["registry"] is None:
                    raise ValueError(
                        "refresh requires the real promotion registry and producers"
                    )
                result = refresh(
                    verdict,
                    req,
                    args.run,
                    packet=packet,
                    registry=context["registry"],
                    producers=context["producers"],
                )
            elif args.command == "admit":
                result = admit(verdict, args.owner, day, context["registry"])
                (args.run / "capstone-blueprint-admission.json").write_text(
                    result.model_dump_json(indent=2) + "\n"
                )
            else:
                path = args.run / "capstone-blueprint-admission.json"
                admission = (
                    Admission.model_validate_json(path.read_text())
                    if path.exists()
                    else None
                )
                result = sign(
                    verdict,
                    admission,
                    args.owner,
                    day,
                    requirement=req,
                    controls=context["controls"],
                    open_items=context["open_items"],
                )
                (args.run / "capstone-blueprint-decision.json").write_text(
                    result.model_dump_json(indent=2) + "\n"
                )
        print(result.model_dump_json(indent=2))
        return int(
            isinstance(result, LaunchVerdict) and result.verdict == "hold"
        )
    except (ValueError, KeyError, OSError) as error:
        print("launch command refused: " + str(error))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
