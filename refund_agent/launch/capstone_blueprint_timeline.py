"""One-clock orchestration. Unavailable producers leave explicit missing records."""

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

from pydantic import Field

from refund_agent.data.generate import Dataset
from refund_agent.launch.capstone_blueprint_compare import (
    ScriptedModelDisagreement,
    compare,
)
from refund_agent.launch.capstone_blueprint_decision import hold_record
from refund_agent.launch.capstone_blueprint_gate import (
    RegistryPort,
    check,
    complete,
    consistent,
    select,
    signed_in_order,
)
from refund_agent.launch.capstone_blueprint_packet import (
    EvidenceClaims,
    EvidenceFile,
    assemble,
    missing,
    write_record,
)
from refund_agent.launch.capstone_blueprint_refresh import refresh
from refund_agent.launch.capstone_blueprint_requirement import (
    Day,
    LaunchRecord,
    LaunchRequirement,
    Period,
    refund_requirement,
    refund_rows,
)
from refund_agent.launch.capstone_blueprint_tier3 import tier3_path
from refund_agent.selection import fingerprint

START = datetime(2026, 10, 1, tzinfo=timezone.utc)


class Demand(LaunchRecord):
    tier1_monthly: float = Field(default=2000, gt=0)
    month_days: int = Field(default=30, gt=0, strict=True)
    tier2_daily: float = Field(default=16, ge=0)
    tier3_daily: float = Field(default=4, ge=0)


class StageResult(LaunchRecord):
    end_day: Day
    records: tuple[EvidenceFile, ...]


class OperatingDriver(Protocol):
    """Adapters call real predecessor functions. No default producer is supplied.

    generate_stage6 -> 37's generator (29 has no tier_mix); stage4 -> 29's
    release/shadow and 32's window; stage5 -> the Part VI integration driver,
    with 34's monitor and the regional fix; audit_period -> 36; stage6_bundle
    -> 29/33 with 36/37/38's composition inside 35's scope; rollout_step ->
    38's reconcile/decide/authority; month_controls -> 34/36/37 and the daily
    launch scan, its events wired to decision.reverse. The driver supplies
    real model replay/manual queue inputs for compare(), never a fixed tie.
    """

    def generate_stage6(
        self, seed: int, start: datetime, days: int, demand: Demand
    ) -> Dataset: ...
    def stage4(self, stream: Dataset, out: Path) -> StageResult: ...
    def stage5(
        self, stream: Dataset, out: Path, after: float
    ) -> StageResult: ...
    def audit_period(
        self, stream: Dataset, out: Path, period: Period
    ) -> StageResult: ...
    def stage6_bundle(
        self, stream: Dataset, out: Path, after: float
    ) -> str: ...
    def rollout_step(
        self, stream: Dataset, out: Path, step: int, period: Period
    ) -> StageResult: ...
    def sign_requirement(
        self, requirement: LaunchRequirement, out: Path
    ) -> LaunchRequirement: ...
    def month_controls(
        self,
        stream: Dataset,
        out: Path,
        period: Period,
        requirement: LaunchRequirement,
    ) -> StageResult: ...

    registry: RegistryPort
    owners: dict
    authority: object
    refresh_producers: dict
    controls: tuple
    open_items: tuple

    def launch_logs(self): ...


@dataclass(frozen=True)
class RunDir:
    path: Path
    status: str
    missing_kinds: tuple[str, ...]
    requirement: LaunchRequirement | None
    gate_day: float | None


def tier3_governing_record(packet, requirement, registry, owners, start):
    need = next(
        n
        for row in requirement.rows
        for n in row.records
        if n.kind == "routing"
    )
    record = select(packet, need, requirement.gate_day)
    if record is None:
        raise ValueError("tier 3 needs a signed stage 6 routing record")
    reasons = (
        complete(packet, need, record, registry)
        + consistent(packet, requirement, need, record, registry)
        + signed_in_order(requirement, need, record, owners)
    )
    if reasons or record.claims.signed_day >= start:
        raise ValueError(
            "tier 3 needs a valid signed stage 6 routing record before its period: "
            + "; ".join(reasons)
        )
    return record


def operate(
    seed: int,
    *,
    demand: Demand,
    out: Path,
    driver: OperatingDriver | None = None,
) -> RunDir:
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    demand = Demand.model_validate(demand.model_dump(mode="json"))
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    configuration = {
        "seed": seed,
        "demand": demand.model_dump(mode="json"),
        "clock": "days from the first day; calendar date never printed",
        "generated_label_shortcut": True,
    }
    previous = out / "capstone-blueprint-configuration.json"
    if previous.exists() and json.loads(previous.read_text()) != configuration:
        raise ValueError(
            "run directory already belongs to another seed or demand"
        )
    previous.write_text(
        json.dumps(configuration, indent=2, sort_keys=True) + "\n"
    )
    summary_path = out / "capstone-blueprint-run.json"
    if summary_path.exists():
        if driver is not None:
            raise ValueError(
                "operating run already exists; load it instead of rerunning"
            )
        saved = json.loads(summary_path.read_text())
        req_path = out / "capstone-blueprint-requirement.json"
        req = (
            LaunchRequirement.model_validate_json(req_path.read_text())
            if req_path.exists()
            else None
        )
        return RunDir(
            out,
            saved["status"],
            tuple(saved["missing_kinds"]),
            req,
            saved["gate_day"],
        )
    kinds = tuple(
        dict.fromkeys(
            n.kind
            for row in refund_rows(Period(start=50, end=64))
            for n in row.records
        )
    )
    if driver is None:
        for kind in kinds:
            missing(
                out,
                kind,
                "assembled producer absent on spine 2; no record constructed in its place",
                "33/34/35/36/37/38 public interfaces and the Part VI integration driver",
            )
        missing(
            out,
            "stream",
            "29's generate_live has no tier_mix; 37's generate_stage6 is not on this base",
            "oversight/human_oversight_risk_demand.py:generate_stage6",
        )
        result = RunDir(
            out, "integration_missing", ("stream",) + kinds, None, None
        )
    else:
        if (out / "capstone-blueprint-stream.json").exists():
            raise ValueError(
                "operating run already exists; load it instead of rerunning"
            )
        # The two prerequisite audits and the monthly audit await payment labels.
        stream = driver.generate_stage6(seed, START, 180, demand)
        stream = Dataset.model_validate_json(stream.model_dump_json())
        stream_fp = fingerprint(stream.model_dump(mode="json"))
        (out / "capstone-blueprint-stream.json").write_text(
            stream.model_dump_json() + "\n"
        )

        def save(result: StageResult, after: float):
            if result.end_day < after or any(
                r.claims.day > result.end_day for r in result.records
            ):
                raise ValueError(
                    "producer moved the shared clock backwards or wrote future evidence"
                )
            if fingerprint(stream.model_dump(mode="json")) != stream_fp:
                raise ValueError(
                    "producer changed the frozen operating stream"
                )
            for record in result.records:
                write_record(out, record)
            return result.end_day

        shadow_end = save(driver.stage4(stream, out), 1)
        canary_end = save(driver.stage5(stream, out, shadow_end), shadow_end)
        audit_end = save(
            driver.audit_period(stream, out, Period(start=canary_end - 28, end=canary_end)),
            canary_end + 30,
        )
        bundle = driver.stage6_bundle(stream, out, audit_end)
        first_period = Period(start=audit_end + 0.001, end=audit_end + 14.001)
        first_end = save(
            driver.rollout_step(stream, out, 1, first_period), first_period.end
        )
        second_audit = save(
            driver.audit_period(stream, out, first_period), max(first_end, first_period.end + 30)
        )
        signed_day = second_audit + 0.001
        period = Period(start=signed_day + 0.001, end=signed_day + 14.001)
        gate_day = max(first_period.start + 60, period.end) + 0.001
        req = refund_requirement(
            bundle,
            signed_day=signed_day,
            gate_day=gate_day,
            comparison_period=period,
        )
        signed = driver.sign_requirement(req, out)
        if signed != req:
            raise ValueError(
                "producer changed the requirement during sign-off"
            )
        (out / "capstone-blueprint-requirement.json").write_text(
            req.model_dump_json(indent=2) + "\n"
        )
        second_end = save(
            driver.rollout_step(stream, out, 2, period), period.end
        )
        control_end = save(
            driver.month_controls(
                stream,
                out,
                Period(start=first_period.start, end=first_period.start + 30),
                req,
            ),
            max(second_end, first_period.start + 60),
        )
        if control_end > gate_day:
            missing(
                out,
                "gate_clock",
                "evidence not ready on the predeclared gate day",
                "new signed launch requirement",
            )
            result = RunDir(
                out, "gate_clock_held", ("gate_clock",), req, gate_day
            )
        else:
            unavailable = []
            for kind, producer in (
                (
                    "comparison",
                    lambda: compare(out, req, roles=driver.owners["roles"]),
                ),
                (
                    "tier3",
                    lambda: tier3_path(
                        out,
                        stream.policy_at(START),
                        Period(
                            start=first_period.start,
                            end=first_period.start + 30,
                        ),
                    ),
                ),
            ):
                governing = None
                try:
                    record = producer()
                    if kind == "tier3":
                        governing = tier3_governing_record(
                            assemble(out, driver.registry, req),
                            req,
                            driver.registry,
                            driver.owners,
                            first_period.start,
                        )
                except ScriptedModelDisagreement:
                    raise
                except (OSError, ValueError, KeyError) as error:
                    unavailable.append(kind)
                    missing(
                        out,
                        kind,
                        str(error),
                        "actual stage 6 replay, desk, intake and policy records",
                    )
                    continue
                record_period = record.period
                claims = EvidenceClaims(
                    record_id="capstone-blueprint-"
                    + kind
                    + "-"
                    + record.fingerprint
                    + ("-under-" + governing.sha256 if governing else ""),
                    kind=kind,
                    day=gate_day,
                    bundle=bundle,
                    signed_by=governing.claims.signed_by
                    if governing
                    else req.signed_by,
                    signer_role=governing.claims.signer_role
                    if governing
                    else "decision_owner",
                    signed_day=governing.claims.signed_day
                    if governing
                    else req.signed_day,
                    run_started_day=period.start
                    if kind == "comparison"
                    else first_period.start,
                    verdict=record.verdict,
                    period=record_period,
                    label_dependent=kind == "comparison",
                    last_payment_day=record.last_payment_day
                    if kind == "comparison"
                    else None,
                )
                if governing is not None:
                    (
                        out / "capstone-blueprint-tier3-governing-record.json"
                    ).write_text(governing.model_dump_json(indent=2) + "\n")
                write_record(
                    out,
                    EvidenceFile(
                        claims=claims, payload=record.model_dump(mode="json")
                    ),
                )
                (out / f"capstone-blueprint-{kind}.json").write_text(
                    record.model_dump_json(indent=2) + "\n"
                )
            packet = assemble(out, driver.registry, req)
            logs = driver.launch_logs()
            first = check(
                packet, req, driver.registry, logs, driver.owners, gate_day
            )
            (out / "capstone-blueprint-gate-01.json").write_text(
                first.model_dump_json(indent=2) + "\n"
            )
            refreshed = refresh(
                first,
                req,
                out,
                packet=packet,
                registry=driver.registry,
                producers=driver.refresh_producers,
            )
            (out / "capstone-blueprint-refresh.json").write_text(
                refreshed.model_dump_json(indent=2) + "\n"
            )
            second = check(
                assemble(out, driver.registry, req),
                req,
                driver.registry,
                logs,
                driver.owners,
                gate_day,
            )
            (out / "capstone-blueprint-gate-02.json").write_text(
                second.model_dump_json(indent=2) + "\n"
            )
            if second.verdict == "hold":
                held = hold_record(
                    second,
                    req,
                    driver.registry,
                    driver.authority,
                    gate_day,
                    controls=driver.controls,
                    open_items=driver.open_items,
                )
                (out / "capstone-blueprint-decision.json").write_text(
                    held.model_dump_json(indent=2) + "\n"
                )
            result = RunDir(out, "operated", tuple(unavailable), req, gate_day)
    summary = {
        "status": result.status,
        "missing_kinds": result.missing_kinds,
        "gate_day": result.gate_day,
        "requirement_fingerprint": result.requirement.fingerprint
        if result.requirement
        else None,
    }
    (out / "capstone-blueprint-run.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    return result
