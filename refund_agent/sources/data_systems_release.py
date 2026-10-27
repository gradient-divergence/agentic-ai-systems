"""Chapter 30's versioned component and its absolute release-suite check."""

from refund_agent.data.generate import generate
from refund_agent.release.extensions import Component, register
from refund_agent.selection import fingerprint
from refund_agent.sources.data_systems_registry import default_registry

RESOURCE_FILES = (
    "release/owners.json",
    "../demos/data/generative_models_scenarios_listed.json",
    "../demos/data/generative_models_scenarios_tau1.json",
    "../demos/data/generative_models_scenarios_tau2.json",
)

FILES = tuple(
    f"sources/data_systems_{name}.py"
    for name in (
        "budget",
        "contract",
        "registry",
        "check",
        "tools",
        "lineage",
        "releases",
        "release",
    )
) + ("sources/__init__.py", *RESOURCE_FILES)


SELECTOR = {
    "name": "data-systems.contracts",
    "module": __name__,
    "settings": {
        "registry_version": default_registry().version,
        "registry_fingerprint": fingerprint(default_registry().record()),
    },
}


def replay_fingerprint() -> str:
    from refund_agent.release.candidate import source_fingerprint
    from refund_agent.release.suite import IMPLEMENTATION_FILES

    return source_fingerprint(*dict.fromkeys((*FILES, *IMPLEMENTATION_FILES)))


def compose(*, base, kwargs, candidate, settings, mode, context):
    from refund_agent.sources import ContractedTools

    return type("DataSystemsCandidateTools", (ContractedTools, base), {}), {
        **kwargs,
        "code_version": candidate.fingerprint,
        "candidate_record": candidate.record(),
        "lineage_mode": "shadow" if mode == "shadow" else "executed",
    }


def wrap_model(*, value, tools, runtime_settings, candidate, settings):
    return tools.recording_model(value, max_steps=runtime_settings.max_steps)


def cases(*, settings):
    from refund_agent.release.suite import Case
    from refund_agent.runtimes.swap import SwapCase

    data = generate(7, orders=40, requests=40).model_copy(update={"correct": []})
    request = next(r for r in data.requests if r.request_id == "r-7-0029")
    # Exercise a contract refusal without violating the event log's ingress schema.
    data = data.model_copy(update={"orders": [
        o.model_copy(update={"paid_cents": o.paid_cents - 1})
        if o.reference == request.order_reference else o for o in data.orders
    ]})
    return (
        Case(
            "slot-data-systems",
            "data-systems",
            "source-contracts",
            1,
            fixture=SwapCase("slot-data-systems", "data-systems", data, request, 0),
            handler="data-systems",
            expected={
                "paid_cents": 0,
                "outcome": "refer",
                "reason": "contract_invariant:order.paid_equals_items",
            },
        ),
    )


def run_case(*, case, candidate, settings, execution):
    if case.handler != "data-systems":
        return None
    execution.run()
    return True


COMPONENT = register(
    Component(
        name=SELECTOR["name"],
        version="data-systems.contracts@1",
        module=__name__,
        modules=("refund_agent.sources",)
        + tuple(
            "refund_agent.sources.data_systems_" + name
            for name in ("budget", "contract", "registry", "check", "tools", "lineage", "releases")
        ),
        files=RESOURCE_FILES,
        hooks={"compose": compose, "wrap_model": wrap_model, "cases": cases, "run_case": run_case},
        slots={"30": "data-systems source contracts"},
    )
)
