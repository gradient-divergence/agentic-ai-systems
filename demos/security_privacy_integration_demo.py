"""An honest integration preflight; it supplies no missing partner records.

The assembled run follows after chapters 33/34 and core fixture issuance land.
Run this preflight first so missing authority is visible in captured evidence.
"""

from refund_agent.batch_stream.component import selector as events
from refund_agent.canary.cloud_native_deployment_component import SELECTOR as CANARY
from refund_agent.release.candidate import build_candidate
from refund_agent.release.suite import build_suite, run_case
from refund_agent.security.security_privacy_component import SELECTOR, scope_start_check
from refund_agent.sources.data_systems_release import SELECTOR as CONTRACTS


def preflight():
    scope = "not run; selected closure unavailable"
    try:
        suite = build_suite(trials=1, components=(CONTRACTS, events(), CANARY, SELECTOR))
        candidate = build_candidate(suite)
        scope = "pass" if scope_start_check(candidate)["passed"] else "fail"
        case = next(case for case in suite.cases if case.part == "injection")
        run_case(case, candidate)
    except ValueError as error:
        refusal = str(error)
    else:
        refusal = "authenticated fixture execution succeeded"
    return (
        "Constructed stage 5 integration preflight: BLOCKED.\n"
        f"Scope check: {scope}\n"
        f"Release fixture: {refusal}\n"
        "Release record: unavailable for the assembled candidate.\n"
        "Window: not run; authenticated issuance required on all 299 requests.\n"
        "Plan: not signed without the actual window and partner records.\n"
        "Error budget: chapter 34 record not merged.\n"
        "Rehearsal: cannot certify the assembled candidate on this base.\n"
        "Promotion: chapter 33 record not merged.\n"
        "Seven-check start, correction pause/resume and time-box end: not run.\n"
        "No cohort payments or simulated successful start are claimed.\n"
        "Core start still needs an absence check outside selected hooks.\n"
    )


# region book:fnd.security-privacy.integration-01 output
if __name__ == "__main__":
    print(preflight(), end="")
# endregion book:fnd.security-privacy.integration-01
