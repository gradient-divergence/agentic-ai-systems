"""Append-only bundles and transitions; rollback restores software, never grants."""

import fcntl
import json
from contextlib import contextmanager
from pathlib import Path

from refund_agent import contracts as c
from refund_agent.canary.cloud_native_deployment_plan import has_role
from refund_agent.canary.cloud_native_deployment_stop import StopRecord
from refund_agent.promotion.mlops_mle_bundle import VersionBundle
from refund_agent.promotion.mlops_mle_record import PromotionRecord, sign


class Registry:
    def __init__(self, path, namespace="mlops-mle"):
        self.path = Path(path)
        self.namespace = namespace

    def read(self, *, all_namespaces=False):
        if not self.path.exists():
            return []
        rows = [json.loads(line) for line in self.path.read_text().splitlines()]
        if any(row["entry"] != i for i, row in enumerate(rows, 1)):
            raise ValueError("registry sequence differs")
        for row in rows:
            (VersionBundle if row["kind"] == "bundle" else PromotionRecord).from_record(
                row["value"]
            )
        return rows if all_namespaces else [r for r in rows if r["namespace"] == self.namespace]

    @contextmanager
    def transaction(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as writer:
            fcntl.flock(writer.fileno(), fcntl.LOCK_EX)
            yield self.read(all_namespaces=True), writer

    def append(self, value):
        kind = "bundle" if isinstance(value, VersionBundle) else "transition"
        raw = value.record()
        with self.transaction() as (rows, writer):
            own = [r for r in rows if r["namespace"] == self.namespace]
            if any(row["value"]["fingerprint"] == value.fingerprint for row in own):
                raise ValueError("registry refuses an overwrite or duplicate")
            if kind == "transition":
                known = {r["value"]["fingerprint"] for r in own if r["kind"] == "bundle"}
                if value.candidate_fingerprint not in known:
                    raise ValueError("register the complete bundle before its transition")
                if (
                    value.kind != "demotion"
                    and value.status == "promoted"
                    and (not has_role(value.signed_by, "release_owner") or value.signed_at is None)
                ):
                    raise ValueError("admission requires the release owner's dated sign-off")
            writer.write(
                json.dumps(
                    {
                        "entry": len(rows) + 1,
                        "kind": kind,
                        "value": raw,
                        "namespace": self.namespace,
                    },
                    sort_keys=True,
                )
                + "\n"
            )

    def transitions(self):
        return tuple(
            PromotionRecord.from_record(r["value"])
            for r in self.read()
            if r["kind"] == "transition"
        )

    def admitted(self, bundle_fingerprint, stage, *, at=None):
        active = {}
        for record in self.transitions():
            if record.status == "held" or record.unmet:
                continue
            effective = record.signed_at
            if effective is None or (at is not None and effective > c._utc_time(at)):
                continue
            if record.kind == "demotion":
                current = active.get(record.from_stage)
                if current is None or (
                    current.candidate_fingerprint,
                    current.refs.get("canary_id"),
                ) != (record.candidate_fingerprint, record.refs.get("canary_id")):
                    continue
                del active[record.from_stage]
                target = active.get(record.to_stage)
                if target is not None and target.signed_at > effective:
                    continue
            active[record.to_stage] = record
        current = active.get(stage)
        return current is not None and current.candidate_fingerprint == bundle_fingerprint

    def rollback(self, reversed_record, target, *, owner, at):
        transitions = self.transitions()
        if (
            reversed_record not in transitions
            or reversed_record.kind != "promotion"
            or (target.fingerprint != reversed_record.rollback_target)
        ):
            raise ValueError("rollback must restore the reversed record's rollback target")
        if not self.admitted(
            reversed_record.candidate_fingerprint, reversed_record.to_stage, at=at
        ):
            raise ValueError("rollback must reverse the current admission")
        target_admission = next(
            (
                r
                for r in reversed(transitions)
                if r.candidate_fingerprint == target.fingerprint
                and r.signed_by
                and r.signed_at is not None
                and r.signed_at <= c._utc_time(at)
                and r.to_stage == reversed_record.to_stage
                and r.status == "promoted"
                and not r.unmet
            ),
            None,
        )
        if target_admission is None:
            raise ValueError("rollback target was never admitted to that stage")
        record = sign(
            PromotionRecord(
                kind="rollback",
                candidate_fingerprint=target.fingerprint,
                incumbent_fingerprint=reversed_record.candidate_fingerprint,
                from_stage=reversed_record.to_stage,
                to_stage=reversed_record.to_stage,
                changes=(),
                requirements={},
                status="promoted",
                unmet=(),
                rollback_target=reversed_record.candidate_fingerprint,
                refs={**target_admission.refs, "reverses": reversed_record.fingerprint},
            ),
            owner,
            at,
        )
        self.append(record)
        return record

    def demote(self, stop: StopRecord, bundle, *, canary_id):
        stop = c._validated(StopRecord, stop)
        if stop.canary_id != canary_id:
            raise ValueError("stop names another canary")
        if stop.outcome != "ended":
            return None
        latest = next(
            (
                r
                for r in reversed(self.transitions())
                if r.to_stage == "canary"
                and r.candidate_fingerprint == bundle.fingerprint
                and r.refs.get("canary_id") == canary_id
                and r.signed_by
            ),
            None,
        )
        if latest is None:
            raise ValueError("end requires this canary's signed admission")
        existing = next(
            (
                r
                for r in self.transitions()
                if r.kind == "demotion"
                and (r.refs.get("canary_id"), r.refs.get("stop")) == (canary_id, stop.stop_id)
            ),
            None,
        )
        if existing is not None:
            return existing
        record = PromotionRecord(
            kind="demotion",
            candidate_fingerprint=bundle.fingerprint,
            incumbent_fingerprint=bundle.fingerprint,
            from_stage="canary",
            to_stage="shadow",
            changes=(),
            requirements={},
            status="promoted",
            unmet=(),
            signed_at=stop.at,
            rollback_target=latest.rollback_target,
            refs={"stop": stop.stop_id, "canary_id": canary_id},
        )
        self.append(record)
        return record
