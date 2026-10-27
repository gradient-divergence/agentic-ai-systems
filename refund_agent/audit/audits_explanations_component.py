"""Select chapter 36 without changing any chapter-29 file or earlier identity."""

from pathlib import Path

from refund_agent.audit.audits_explanations_cases import cases, run_case
from refund_agent.audit.audits_explanations_explain import GENERATOR_VERSION
from refund_agent.audit.audits_explanations_holds import AuditHolds
from refund_agent.audit.audits_explanations_rates import AuditRule
from refund_agent.audit.audits_explanations_records import (
    EvaluationRecord,
    ReviewRoute,
    SegmentAssignments,
)
from refund_agent.audit.audits_explanations_stage6 import (
    AuditInputs,
    AuditsExplanationsEvaluatedTools,
    AuditsExplanationsTools,
    after_run,
)
from refund_agent.memory.register import Snapshot
from refund_agent.release.extensions import Component, RuntimeStore, register
from refund_agent.selection import fingerprint
from refund_agent.tools import Tools

STORE = RuntimeStore(
    "../../.tmp-sol/audits-explanations-run/holds.jsonl", "morgan", "append"
)
STORE_PATH = (Path(__file__).resolve().parents[1] / STORE.path).resolve()


def selector(rule: AuditRule):
    return {
        "name": "audits-explanations.stage6",
        "module": __name__,
        "settings": {
            "rule": rule.model_dump(mode="json"),
            "generator": GENERATOR_VERSION,
        },
    }


def compose(*, base, kwargs, candidate, settings, mode, context):
    inputs = context.get("audits_explanations")
    if not isinstance(inputs, AuditInputs):
        raise ValueError(
            "audits-explanations source, disclosure, review and hold inputs required"
        )
    if (
        inputs.rule.model_dump(mode="json") != settings["rule"]
        or settings["generator"] != GENERATOR_VERSION
    ):
        raise ValueError(
            "audit rule or explanation generator differs from released candidate"
        )
    if inputs.holds.path != STORE_PATH:
        raise ValueError(
            "audit holds must use the component's declared append store"
        )
    evaluated = (
        AuditsExplanationsEvaluatedTools
        if base is Tools
        else type("AuditsExplanationsSourceView", (base, AuditsExplanationsEvaluatedTools), {})
    )
    return type("AuditsExplanationsCandidateTools", (AuditsExplanationsTools, evaluated), {}), {
        **kwargs,
        "audit_inputs": inputs,
    }


def fixture_context(*, value, data, candidate, settings, mode, purpose, **fixture):
    """Consume partner records contributed before chapter 36's release hook."""
    segments = value.get("audits_explanations_segments")
    if not isinstance(segments, SegmentAssignments):
        raise ValueError("chapter 34 region records are required")
    disclosure = value.get("audits_explanations_disclosure")
    if not callable(disclosure):
        raise ValueError("chapter 35 disclosure authority is required")
    routes = value.get("audits_explanations_review_routes")
    if not isinstance(routes, dict):
        raise ValueError("chapter 37 review tickets and deadlines are required")
    register = value.get("audits_explanations_register")
    if not isinstance(register, Snapshot):
        raise ValueError("the source policy register snapshot is required")
    history = value.get("audits_explanations_history")
    if not isinstance(history, dict):
        raise ValueError("historical source records are required")
    for order in data.orders:
        segments.region_of(order.order_id)
    for request in data.requests:
        route = routes.get(request.request_id)
        if not isinstance(route, ReviewRoute) or route.request_id != request.request_id:
            raise ValueError("chapter 37 review ticket must name its request")
    histories = {
        rid: tuple(EvaluationRecord.model_validate(r) for r in records)
        for rid, records in history.items()
    }
    if any(r.request.request_id != rid for rid, rows in histories.items() for r in rows):
        raise ValueError("historical source record names another request")
    return {
        **value,
        "audits_explanations": AuditInputs(
            AuditRule.model_validate(settings["rule"]),
            segments,
            register,
            histories,
            routes,
            disclosure,
            AuditHolds(STORE_PATH),
        ),
    }


def bundle_items(*, candidate, settings):
    return {
        "tools": {
            "audits-explanations.stage6": AuditsExplanationsTools.version,
            "audits-explanations.generator": settings["generator"],
            "audits-explanations.rule": fingerprint(settings["rule"]),
        }
    }


def promotion_requirements(*, candidate, settings):
    return (
        {
            "name": "audits-explanations.stage6",
            "changes": tuple(
                bundle_items(candidate=candidate, settings=settings)["tools"]
            ),
            "tests": (
                "test_audits_explanations_explain.py",
                "test_audits_explanations_stage6.py",
            ),
            "must_pass": True,
        },
    )


def reasons(*, candidate, settings):
    return {"explanation_failed": "referral"}


register(
    Component(
        name="audits-explanations.stage6",
        version=AuditsExplanationsTools.version,
        module=__name__,
        order=36,
        modules=tuple(
            "refund_agent.audit.audits_explanations_" + n
            for n in (
                "records",
                "rates",
                "explain",
                "holds",
                "stage6",
                "cases",
            )
        ),
        files=(
            "release/owners.json",
            "../exercises/audits-explanations/two_clause_06.json",
        ),
        stores=(STORE,),
        hooks={
            "compose": compose,
            "fixture_context": fixture_context,
            "after_run": after_run,
            "bundle_items": bundle_items,
            "promotion_requirements": promotion_requirements,
            "reasons": reasons,
            "cases": cases,
            "run_case": run_case,
        },
        slots={"audits-explanations": "stage-6 denial completeness and replay; must-pass"},
    )
)
