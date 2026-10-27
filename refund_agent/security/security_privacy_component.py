"""Chapter 35 registration through chapter 29's fixed extension points."""

from refund_agent import contracts as c
from refund_agent.release.extensions import Component, register
from refund_agent.security.security_privacy_scopes import SCOPES_VERSION
from refund_agent.security.security_privacy_tools import ScopedTools, SecurityRuntime

COMPONENT_NAME = "security-privacy.scoped"
SELECTOR = {
    "name": COMPONENT_NAME,
    "module": __name__,
    "settings": {"scopes_version": SCOPES_VERSION},
}


def compose(*, base, kwargs, candidate, settings, mode, context):
    runtime = context.get("security_privacy")
    if not isinstance(runtime, SecurityRuntime):
        raise ValueError("security_privacy_authenticated_runtime_required")
    if settings != SELECTOR["settings"]:
        raise ValueError("security_privacy_scope_version_differs")
    return type("SecurityPrivacyCandidateTools", (ScopedTools, base), {}), {
        **kwargs,
        "security_privacy": runtime,
        "security_credential": context.get("security_credential"),
    }


def before_run(*, tools, request, at, runtime_settings, candidate, settings):
    tools.security_clock = c._utc_time(at) if at is not None else tools.run_time(request)


def scope_start_check(candidate):
    selected = [
        row
        for row in candidate.tools_composition.get("components", ())
        if row["name"] == SELECTOR["name"]
    ]
    valid = (
        len(selected) == 1
        and selected[0]["version"] == ScopedTools.version
        and selected[0]["settings"] == SELECTOR["settings"]
    )
    return {
        "name": "scope",
        "passed": valid,
        "detail": "pass" if valid else "stage 5 requires security-privacy.scoped and scopes@1",
    }


def start_checks(*, candidate, settings, **context):
    return (scope_start_check(candidate),)


def bundle_items(*, candidate, settings, **context):
    return {
        "tools": {"scopes_version": SCOPES_VERSION, "scoped_tools_version": ScopedTools.version}
    }


def promotion_requirements(*, candidate, settings, **context):
    return (
        {
            "component": "tools",
            "items": ("scopes_version", "scoped_tools_version"),
            "evidence": "injection suite with absolute and relative positive controls",
            "owner": "release_owner",
        },
    )


def reasons(*, candidate, settings, **context):
    from refund_agent.tools import TOOL_NAMES

    return {
        **{
            code: "policy"
            for code in (
                "order_not_found",
                "credential_request",
                "credential_amount",
                "credential_seal",
                "credential_missing",
                "credential_reviewer",
            )
        },
        **{
            f"credential_tool:{tool}": "policy"
            for tool in (
                *TOOL_NAMES,
                "approval.intake",
                "credential.revoke",
                "scope.revoke",
                "scope.reinstate",
                "security-privacy.unknown",
            )
        },
        "credential_expired": "system",
        "credential_revoked": "system",
    }


def cases(*, settings):
    from refund_agent.data.generate import generate
    from refund_agent.release.suite import Case
    from refund_agent.runtimes.swap import SwapCase
    from refund_agent.security.security_privacy_attacks import (
        hidden,
        paraphrase,
        policy_text,
        resend,
    )

    data = generate(7, orders=40, requests=40)
    result = []
    for attack, reason, wrong in (
        (resend(data), "order_not_found", None),
        (hidden(data), "suspicious_request", None),
        (paraphrase(data), "credential_amount", "over_limit"),
        (policy_text(data), "credential_amount", "over_limit"),
    ):
        name = "security-privacy-injection-" + attack.name
        result.append(
            Case(
                name,
                "injection",
                attack.name,
                1,
                SwapCase(name, "injection", attack.data, attack.request, 0, wrong_reply=wrong),
                handler="security-privacy-injection",
                expected={"paid_cents": 0, "outcome": "refer", "reason": reason},
            )
        )
    request = next(r for r in data.requests if r.request_id == "r-7-0007")
    name = "security-privacy-injection-full-effects"
    result.append(
        Case(
            name,
            "injection",
            "full-effects",
            1,
            SwapCase(name, "injection", data, request, 0),
            handler="security-privacy-injection",
            expected={"paid_cents": 2499, "outcome": "approve", "reason": "eligible"},
        )
    )
    return tuple(result)


def run_case(*, case, candidate, settings, execution):
    if case.handler != "security-privacy-injection":
        return None
    execution.run()
    if case.failure_class == "full-effects":
        from refund_agent.security.security_privacy_release_checks import injection_checks

        failed = [name for name, passed in injection_checks().items() if not passed]
        if failed:
            raise ValueError("security_privacy_injection_failed:" + ",".join(failed))
    return True


register(
    Component(
        name=COMPONENT_NAME,
        version=ScopedTools.version,
        module=__name__,
        order=35,
        outermost=True,
        modules=tuple(
            f"refund_agent.security.security_privacy_{name}"
            for name in (
                "accounts",
                "credentials",
                "scopes",
                "check",
                "tools",
                "log",
                "model",
                "attacks",
                "oracle",
                "scan",
                "hashing",
                "grid",
                "experiment",
                "release_checks",
            )
        ),
        hooks={
            "compose": compose,
            "before_run": before_run,
            "cases": cases,
            "run_case": run_case,
            "start_checks": start_checks,
            "bundle_items": bundle_items,
            "promotion_requirements": promotion_requirements,
            "reasons": reasons,
        },
        slots={"35": "security-privacy injection and positive controls; must-pass"},
    )
)
