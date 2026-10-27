"""An owner-selected append store stays local across the real restart worker."""

import subprocess
import sys
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from os import getpid
from types import ModuleType

import pytest

from refund_agent.release.candidate import build_candidate
from refund_agent.release.extensions import RuntimeStore, resource_guard
from refund_agent.release.suite import build_suite
from refund_agent.sources.data_systems_release import replay_fingerprint

# isort: split
from refund_agent.canary.cloud_native_deployment_component import (
    JOURNAL_PATH,
    STORE_PATH,
    selector,
)
from refund_agent.canary.cloud_native_deployment_plan import CanaryPlan, sign_plan
from refund_agent.canary.cloud_native_deployment_rehearsal import fixture, run_rehearsal
from refund_agent.canary.cloud_native_deployment_store import AppendStore
from refund_agent.data.generate import Dataset
from refund_agent.data.live import LiveStream


def digest(path):
    return sha256(path.read_bytes()).hexdigest() if path.exists() else None


@pytest.mark.parametrize("owned", [False, True])
def test_canary_replay_reads_scenario_resources_under_guard(tmp_path, owned):
    stores = {
        "store": RuntimeStore(str(tmp_path / "store.jsonl"), "morgan", "append"),
        "journal": RuntimeStore(str(tmp_path / "journal.jsonl"), "morgan", "append"),
    } if owned else {}
    candidate = build_candidate(build_suite(trials=1, components=[selector(**stores)]))

    @resource_guard(0)
    def guarded_replay(candidate):
        return replay_fingerprint()

    assert guarded_replay(candidate) == replay_fingerprint()


def test_owned_store_survives_fresh_process_restart_without_default_writes(tmp_path, monkeypatch):
    before = {path: digest(path) for path in (STORE_PATH, JOURNAL_PATH)}
    path = tmp_path / "operating-run" / "store.jsonl"
    journal = path.with_name("journal.jsonl")
    choice = selector(store=RuntimeStore(str(path), "morgan", "append"),
                      journal=RuntimeStore(str(journal), "morgan", "append"))
    suite = build_suite(components=[choice])
    candidate = build_candidate(suite)
    target = build_candidate(suite, verifier="independent")
    start = datetime(2026, 10, 15, tzinfo=timezone.utc)
    cid = "cloud-native-deployment-owned-store"
    pieces = [fixture(cid, f"owned-{i}", start + timedelta(days=i))[0] for i in range(3)]
    data = Dataset(orders=sum((p.orders for p in pieces), []),
                   requests=sum((p.requests for p in pieces), []),
                   shipments=sum((p.shipments for p in pieces), []), policies=pieces[0].policies,
                   retrievals=sum((p.retrievals for p in pieces), []), correct=[])
    stream = LiveStream(data, {"seed": 7, "start": start.isoformat(), "days": 3, "per_day": 1})
    plan = sign_plan(CanaryPlan(
        canary_id=cid, candidate_fingerprint=candidate.fingerprint,
        suite_fingerprint=suite.fingerprint, rollback_target_fingerprint=target.fingerprint,
        requirement_ref="rehearsal-only", rehearsal_ref=cid,
    ), "morgan", start)
    original = subprocess.run
    component = ModuleType("refund_agent._nested_restart_component")
    component.__file__ = str(tmp_path / "nested.py")
    component.launch, component.executable = original, sys.executable
    source = "def spawn():\n    return launch([executable, '-c', 'pass'])\n"
    exec(compile(source, component.__file__, "exec"), component.__dict__)

    def restart(command, **kwargs):
        with pytest.raises(ValueError, match="component capability refused: process spawning"):
            component.spawn()
        return original(command, **kwargs)

    monkeypatch.setattr(subprocess, "run", restart)
    result = run_rehearsal(plan, stream, 3, ["forbidden_payment"], "4h", target,
                           candidate=candidate)
    assert result.restart_fresh_process and result.paid_twice == 0
    assert result.checks["receipts_kept"] and result.checks["receipts_confirmed_at_restart"]
    assert result.checks["intents_closed_unpaid"] and result.checks["reconcile_idempotent"]
    assert result.final_outcome == "ended"
    rows = AppendStore(path, "unused").rows(all_namespaces=True)
    payload = next(row["value"] for row in rows if row["kind"] == "restart-payload")
    assert payload["store_path"] == str(path)
    assert any(row["kind"] == "restart-result" for row in rows)
    restarted = next(row["value"] for row in rows if row["kind"] == "restart-result")
    assert restarted["process_id"] != getpid()
    assert {path: digest(path) for path in before} == before
