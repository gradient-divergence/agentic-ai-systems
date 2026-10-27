"""Chapter 32 owns its registration, full closure, and append-store authority."""

from dataclasses import asdict
from pathlib import Path

from refund_agent.canary.cloud_native_deployment_binding import CanaryTools, before_run
from refund_agent.canary.cloud_native_deployment_stop import CanarySession
from refund_agent.canary.cloud_native_deployment_store import (
    AppendJournal,
    CanaryProposal,
    CanaryReceipt,
    ExpandedProposal,
    ExpandedReceipt,
    ReadContracts,
)
from refund_agent.release.agent_evaluation_resources import fixture_active
from refund_agent.release.extensions import Component, RuntimeStore, register, resource_guard
from refund_agent.selection import fingerprint
from refund_agent.shadow.tools import ShadowTools
from refund_agent.sources.data_systems_release import RESOURCE_FILES

STORE = RuntimeStore("../../.tmp-sol/cloud-native-deployment-run/store.jsonl", "morgan", "append")
JOURNAL = RuntimeStore("../../.tmp-sol/cloud-native-deployment-run/journal.jsonl", "morgan", "append")
STORE_PATH = (Path(__file__).resolve().parents[1] / STORE.path).resolve()
JOURNAL_PATH = (Path(__file__).resolve().parents[1] / JOURNAL.path).resolve()
NAME = "cloud-native-deployment.canary"


def selector(*, record_schema="closed", reader="closed", store=None, journal=None):
    """Select owner-declared append paths before building the suite/candidate."""
    if record_schema not in ("closed", "canary-id") or reader not in ("closed", "canary-id", "expanded"):
        raise ValueError("unknown canary record contract")
    name = NAME
    settings = {"record_schema": record_schema, "reader": reader}
    if store is not None or journal is not None:
        stores = {"store": STORE if store is None else store,
                  "journal": JOURNAL if journal is None else journal}
        if any(not isinstance(s, RuntimeStore) or s.mode != "append" for s in stores.values()):
            raise ValueError("canary stores require owner-declared append RuntimeStore records")
        settings["stores"] = {key: asdict(value) for key, value in stores.items()}
        name += "." + fingerprint(settings["stores"])[:16]
        _register(name, tuple(stores.values()))
    return {"name": name, "module": __name__, "settings": settings}


def _record(candidate):
    rows = [r for r in candidate.tools_composition["components"] if r["module"] == __name__]
    if len(rows) != 1:
        raise ValueError("candidate must select exactly one canary binding")
    return rows[0]


def runtime_paths(candidate):
    stores = _record(candidate)["settings"].get("stores", {
        "store": asdict(STORE), "journal": asdict(JOURNAL),
    })
    root = Path(__file__).resolve().parents[1]
    return {key: (root / value["path"]).resolve() for key, value in stores.items()}


def restore_selection(candidate_record):
    """Re-register the recorded owner paths at a fresh process's host boundary."""
    for row in candidate_record["tools_composition"].get("components", ()):
        if row["module"] != __name__:
            continue
        settings = row["settings"]
        stores = {key: RuntimeStore(**value) for key, value in settings.get("stores", {}).items()}
        selected = selector(record_schema=settings["record_schema"], reader=settings["reader"], **stores)
        if selected != {key: row[key] for key in ("name", "module", "settings")}:
            raise ValueError("recorded canary store selection differs")


def read_contracts(candidate):
    settings = _record(candidate)["settings"]
    return {"closed": ReadContracts(), "canary-id": ReadContracts(CanaryReceipt, CanaryProposal),
            "expanded": ReadContracts(ExpandedReceipt, ExpandedProposal)}[settings["reader"]]


@resource_guard(0)
def run_release(candidate, suite):
    from refund_agent.release.rule import evaluate_release

    return evaluate_release(candidate, suite, AppendJournal(runtime_paths(candidate)["journal"]))


def compose(*, base, kwargs, candidate, settings, mode, context):
    if mode == "shadow" or fixture_active(context.get("release_fixture"), candidate, mode):
        return base, kwargs
    session = context.get("canary_session")
    if not isinstance(session, CanarySession):
        raise ValueError("canary_session_required; payment binding remains closed")
    if session.store.path != runtime_paths(candidate)["store"]:
        raise ValueError("canary session store differs from the selected owner declaration")
    rollback = session.plan.candidate_fingerprint != candidate.fingerprint
    if rollback and (session.plan.rollback_target_fingerprint != candidate.fingerprint
                     or session.outcome not in ("paused", "ended")):
        raise ValueError("canary plan names another candidate")
    shadow = type("CloudNativeDeploymentShadow", (ShadowTools, base), {})
    return type("CloudNativeDeploymentCandidate", (CanaryTools, base), {}), {
        **kwargs, "canary_session": session, "shadow_type": shadow,
        "shadow_kwargs": {k: "shadow" if k in ("lineage_mode", "batch_stream_mode") else v
                          for k, v in kwargs.items()}, "canary_rollback_only": rollback,
        "read_contracts": read_contracts(candidate),
        "write_canary_id": session.plan.canary_id if settings["record_schema"] == "canary-id" else None,
    }


def _register(name, stores):
    register(Component(
        name=name, version=CanaryTools.version, module=__name__, order=32,
        modules=tuple(f"refund_agent.canary.cloud_native_deployment_{name}" for name in (
            "plan", "window", "store", "binding", "start", "stop", "restart", "desk", "rehearsal", "exposure",
        )) + ("refund_agent.canary.__main__",),
        files=RESOURCE_FILES, stores=stores,
        hooks={"compose": compose, "before_run": before_run},
        slots={"32": "canary binding, persisted authority and rollback rehearsal"},
    ))


_register(NAME, (STORE, JOURNAL))
SELECTOR = selector()
