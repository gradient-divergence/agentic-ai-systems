"""Chapter 38 owns this registration; chapter 29's shared files stay unchanged."""

from dataclasses import asdict
from pathlib import Path

from refund_agent.release.extensions import Component, RuntimeStore, register
from refund_agent.rollout.org_adoption_authority import (
    AuthorityTools,
    StageAuthority,
)
from refund_agent.selection import fingerprint

NAME = "org-adoption.authority"


def selector(store):
    """Declare the run's owner-provided append store before selection."""
    if not isinstance(store, RuntimeStore) or store.mode != "append":
        raise ValueError("authority requires an owner-declared append RuntimeStore")
    settings = {"store": asdict(store)}
    name = NAME + "." + fingerprint(settings)[:16]
    _register(name, store)
    return {"name": name, "module": __name__, "settings": settings}


def restore_selection(candidate_record):
    """Restore only the store declaration recorded in the selected candidate."""
    for row in candidate_record["tools_composition"].get("components", ()):
        if row["module"] != __name__:
            continue
        selected = selector(RuntimeStore(**row["settings"]["store"]))
        if selected != {key: row[key] for key in ("name", "module", "settings")}:
            raise ValueError("recorded authority store selection differs")


def compose(*, base, kwargs, candidate, settings, mode, context):
    if mode == "shadow":
        return base, kwargs
    authority = context.get("org_adoption_authority")
    if not isinstance(authority, StageAuthority):
        raise ValueError("org_adoption_authority_required")
    path = (Path(__file__).resolve().parents[1] / settings["store"]["path"]).resolve()
    if authority.path.resolve() != path:
        raise ValueError("authority store differs from the selected owner declaration")
    if authority.plan.candidate_fingerprint != candidate.fingerprint:
        raise ValueError("rollout plan covers another candidate")
    return type("OrgAdoptionAuthorityTools", (AuthorityTools, base), {}), {
        **kwargs,
        "stage_authority": authority,
        "org_adoption_candidate": candidate,
        "org_adoption_context": context,
    }


def promotion_requirements(*, candidate, settings, **context):
    component = next(
        row["name"]
        for row in candidate.tools_composition["components"]
        if row["module"] == __name__
    )
    return (
        {
            "component": component,
            "version": StageAuthority.version,
            "required_tests": ("tests/refund_agent/test_org_adoption_rollout.py",),
        },
    )


def reasons(*, candidate, settings, **context):
    return {
        "org_adoption_no_signed_step": "system",
        "org_adoption_step_not_started": "system",
        "org_adoption_withdrawn": "system",
        "org_adoption_outside_step": "policy",
        "org_adoption_person_required": "policy",
    }


def _register(name, store):
    register(
        Component(
            name=name,
            version=StageAuthority.version,
            module=__name__,
            order=38,
            modules=tuple(
                "refund_agent.rollout.org_adoption_" + name
                for name in (
                    "authority",
                    "plan",
                    "records",
                    "matching",
                    "reconcile",
                    "rule",
                )
            ),
            files=("release/owners.json",),
            stores=(store,),
            hooks={
                "compose": compose,
                "promotion_requirements": promotion_requirements,
                "reasons": reasons,
            },
            slots={"38": "signed rollout authority and withdrawal"},
        )
    )
