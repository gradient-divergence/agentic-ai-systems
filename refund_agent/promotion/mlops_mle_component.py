"""Chapter 33's hooks: cover policy, stamp lineage and require current admission."""

from datetime import datetime, timezone
from pathlib import Path

from refund_agent import contracts as c
from refund_agent.canary.cloud_native_deployment_component import STORE as CANARY_STORE
from refund_agent.promotion.mlops_mle_bundle import NAME, from_candidate, release_settings
from refund_agent.promotion.mlops_mle_coverage import PolicyCoverage
from refund_agent.promotion.mlops_mle_registry import Registry
from refund_agent.promotion.mlops_mle_stamp import BundledRecordStore
from refund_agent.release.extensions import Component, RuntimeStore, register

REGISTRY_STORE = RuntimeStore("../../.tmp-sol/mlops-mle-run/registry.jsonl", "morgan", "append")
RELEASE_STORE = RuntimeStore("../../.tmp-sol/mlops-mle-run/journal.jsonl", "morgan", "append")
ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = (ROOT / REGISTRY_STORE.path).resolve()
JOURNAL_PATH = (ROOT / RELEASE_STORE.path).resolve()


def selector(snapshot, versions):
    return {"name": NAME, "module": __name__, "settings": release_settings(snapshot, versions)}


def compose(*, base, kwargs, candidate, settings, mode, context):
    bundle = from_candidate(candidate)
    if "bundle" in context and context["bundle"].fingerprint != bundle.fingerprint:
        raise ValueError("loaded bundle differs from the selected candidate")
    if "shadow_type" in kwargs:
        kwargs = {
            **kwargs,
            "shadow_type": type(
                "MlopsMleCanaryShadow", (PolicyCoverage, kwargs["shadow_type"]), {}
            ),
            "shadow_kwargs": {**kwargs["shadow_kwargs"], "bundle": bundle},
        }
    wrapped = type("MlopsMleCandidateTools", (PolicyCoverage, base), {})
    return wrapped, {**kwargs, "bundle": bundle}


def before_run(*, tools, candidate, settings, **context):
    # Called after all constructors, before the first lineage can be written.
    from refund_agent.runtimes.faults import FaultTools

    mounted = tools.inner if isinstance(tools, FaultTools) else tools
    if not isinstance(mounted.record_store, BundledRecordStore):
        if mounted.lineage:
            raise ValueError("cannot stamp after an unstamped decision")
        mounted.record_store = BundledRecordStore(mounted.bundle)


def admit_shadow(*, candidate, settings, record, journal, path):
    bundle = from_candidate(candidate)
    registry = Registry(REGISTRY_PATH, path.parent.name)
    at = datetime.now(timezone.utc)
    if not registry.admitted(bundle.fingerprint, "shadow", at=at):
        raise ValueError("mlops-mle bundle not admitted to shadow")


def start_checks(*, candidate, settings, plan, at, records):
    try:
        bundle = from_candidate(candidate)
        supplied = records["mlops-mle"]
        if supplied["bundle"].fingerprint != bundle.fingerprint:
            raise ValueError("loaded bundle or register snapshot differs")
        registry = Registry(REGISTRY_PATH, supplied["namespace"])
        if not registry.admitted(bundle.fingerprint, "canary", at=at):
            raise ValueError("bundle not admitted to canary")
        latest = next(
            r for r in reversed(registry.transitions()) if r.to_stage == "canary" and r.signed_by
        )
        if (
            latest.refs.get("canary_id"),
            latest.refs.get("window"),
            latest.refs.get("candidate"),
        ) != (plan.canary_id, plan.requirement_ref, candidate.fingerprint):
            raise ValueError("promotion names another plan, window or candidate")
        # The released register snapshot cannot report a later external change.
        # The caller supplies the current policy read, not the bundle's snapshot.
        policy = c._validated(c.Policy, supplied["current_policy"])
        if policy.effective_at > at or not bundle.covers(policy.version):
            raise ValueError("policy in force is outside the bundle's evaluated coverage")
    except (KeyError, ValueError, TypeError, FileNotFoundError) as error:
        return [{"name": "mlops-mle.promotion", "passed": False, "detail": str(error)}]
    return [{"name": "mlops-mle.promotion", "passed": True, "detail": "pass"}]


def cases(*, settings):
    from refund_agent.promotion.mlops_mle_cases import policy_cases

    return policy_cases(settings)


def run_case(*, case, candidate, settings, execution):
    if not case.handler.startswith("mlops-mle-"):
        return None
    execution.run()
    return True


register(
    Component(
        name=NAME,
        version="mlops-mle.bundle@1",
        module=__name__,
        order=33,
        modules=tuple(
            "refund_agent.promotion.mlops_mle_" + name
            for name in (
                "bundle",
                "diff",
                "requirements",
                "record",
                "registry",
                "coverage",
                "stamp",
                "cases",
            )
        ),
        files=(
            "../uv.lock",
            "../pyproject.toml",
            "release/owners.json",
            "../demos/data/generative_models_scenarios_listed.json",
            "../demos/data/generative_models_scenarios_tau1.json",
            "../demos/data/generative_models_scenarios_tau2.json",
        ),
        stores=(REGISTRY_STORE, RELEASE_STORE, CANARY_STORE),
        hooks={
            "compose": compose,
            "before_run": before_run,
            "admit_shadow": admit_shadow,
            "start_checks": start_checks,
            "cases": cases,
            "run_case": run_case,
        },
        slots={"33": "policy coverage and bundle stamp; promotion on every start"},
    )
)
