"""Three matched arms, fixed authority and agreement rules, one post-run truth read."""

import math
from dataclasses import dataclass
from statistics import NormalDist
from time import perf_counter

from refund_agent import contracts as c
from refund_agent.adaptation.experiment import agreement, decide, paired
from refund_agent.agent import Agent
from refund_agent.data.generate import generate
from refund_agent.rules import tier_for
from refund_agent.selection import fingerprint
from refund_agent.tools import Tools
from refund_agent.verifier.agreeing import AgreeingVerifier
from refund_agent.verifier.fixture import build_fixture
from refund_agent.verifier.independent import IndependentVerifier, VerifiedTools, independent_check
from refund_agent.verifier.record import ReadOnlyTools
from refund_agent.verifier.shared_error import error_table

ALPHA = 0.025
REVIEW_CENTS = 400
COMMISSION = {
    "baseline": "single",
    "challengers": ("agreeing", "independent"),
    "alpha": ALPHA,
    "rule": "no additional payments; paired interval above zero and exact chance at most .025; tie keeps single",
    "measure": "agreement with outcome, amount and tier; denial citation",
    "costs": "wrong payment loses its amount; every referral costs 400 cents; call price excluded",
    "grants": "eligible for chapter 29's suite and shadow; no deployment authority",
}


@dataclass(frozen=True)
class CaseRun:
    trace: c.Trace
    verdicts: tuple
    audits: tuple
    reviews: int
    verification_calls: int


def scoped_data(data, request):
    """Only this request's runtime records; no labels, frozen sets or other ledgers."""
    orders = [o for o in data.orders if request.order_reference in (o.order_id, o.reference)]
    ids = {o.order_id for o in orders}
    return data.model_copy(
        update={
            "orders": orders,
            "requests": [request],
            "shipments": [s for s in data.shipments if s.order_id in ids],
            "retrievals": [s for s in data.retrievals if s.request_id == request.request_id],
            "correct": [],
            "public_scenarios": [],
            "instructor_scenarios": [],
        }
    )


def run_case(data, request, verifier=None, *, tools_class=VerifiedTools, deadline_seconds=1):
    data = scoped_data(data, request)
    if verifier is None:
        tools = Tools(data)
        trace = Agent(tools).run(request)
        return CaseRun(trace, (), tuple(tools.audits.values()), len(tools.reviews), 0)
    tools = tools_class(data, verifier, deadline_seconds)
    trace = Agent(tools).run(request)
    return CaseRun(
        trace,
        tools.verdict_log.entries,
        tuple(tools.audits.values()),
        len(tools.reviews),
        tools.verification_calls,
    )


def run_arm(data, verifier=None):
    return [run_case(data, request, verifier) for request in data.requests]


def _matched_cases(baseline, challenger, truth=None):
    request_ids = [[row.trace.decision.request_id for row in arm] for arm in (baseline, challenger)]
    if truth is not None:
        request_ids.append([row.request_id for row in truth])
    ids = request_ids[0]
    if not ids or len(set(ids)) != len(ids) or any(other != ids for other in request_ids[1:]):
        raise ValueError("nonempty, equal-length, unique request IDs in the same order required")
    return zip(baseline, challenger) if truth is None else zip(baseline, challenger, truth)


def authority_count(baseline, challenger):
    return sum(
        b.trace.receipt is not None
        and (
            a.trace.receipt is None or b.trace.receipt.amount_cents != a.trace.receipt.amount_cents
        )
        for a, b in _matched_cases(baseline, challenger)
    )


def changed_decisions(baseline, challenger):
    return sum(
        (a.trace.decision.outcome, a.trace.decision.amount_cents, a.trace.decision.tier)
        != (b.trace.decision.outcome, b.trace.decision.amount_cents, b.trace.decision.tier)
        for a, b in _matched_cases(baseline, challenger)
    )


def case_cost(run, truth):
    if run.trace.receipt is not None and not truth.owed:
        return run.trace.receipt.amount_cents
    return REVIEW_CENTS if run.trace.decision.outcome == c.Outcome.REFER else 0


def saving(baseline, challenger, truth, alpha=ALPHA):
    cases = list(_matched_cases(baseline, challenger, truth))
    values = [case_cost(a, t) - case_cost(b, t) for a, b, t in cases]
    n = len(values)
    mean = sum(values) / n
    # Chapter 14's variance convention divides the sum of squared deviations by n.
    se = math.sqrt(sum((v - mean) ** 2 for v in values)) / n
    half = NormalDist().inv_cdf(1 - alpha / 2) * se
    return {
        "mean_dollars": mean / 100,
        "standard_error_dollars": se / 100,
        "interval_dollars": ((mean - half) / 100, (mean + half) / 100),
        "stopped_wrong_payments": sum(
            a.trace.receipt is not None and b.trace.receipt is None and not t.owed
            for a, b, t in cases
        ),
        "false_objections": sum(
            b.trace.decision.outcome == c.Outcome.REFER and t.owed for _, b, t in cases
        ),
    }


def _arm_record(rows, baseline, labels, truth):
    matches = agreement([row.trace.decision for row in rows], labels)
    single = agreement([row.trace.decision for row in baseline], labels)
    card = paired(single, matches, alpha=ALPHA)
    authority = authority_count(baseline, rows)
    verdicts = [v for row in rows for v in row.verdicts]
    return {
        "cases": len(rows),
        "correct": sum(matches),
        "wrong_payments": sum(
            row.trace.receipt is not None and not t.owed for row, t in zip(rows, truth)
        ),
        "payments": sum(row.trace.receipt is not None for row in rows),
        "referrals": sum(row.trace.decision.outcome == c.Outcome.REFER for row in rows),
        "changed_decisions": changed_decisions(baseline, rows),
        "additional_payments": authority,
        "paired": card,
        "decision": decide(authority == 0, card, alpha=ALPHA),
        "saving": saving(baseline, rows, truth),
        "errors": error_table([not x for x in single], [not x for x in matches]),
        "verification_calls": sum(row.verification_calls for row in rows),
        "words_per_verification": sum(v.words_read for v in verdicts) / len(verdicts)
        if verdicts
        else 0,
    }


def public_comparison():
    data = generate(7)
    ids = set(data.public_scenarios)
    labels = [d for d in data.correct if d.request_id in ids]
    data = data.model_copy(
        update={"requests": [r for r in data.requests if r.request_id in ids], "correct": []}
    )
    runs = {
        "single": run_arm(data),
        "agreeing": run_arm(data, AgreeingVerifier()),
        "independent": run_arm(data, IndependentVerifier()),
    }
    result = {}
    for name, rows in runs.items():
        result[name] = {
            "cases": len(rows),
            "correct": sum(agreement([r.trace.decision for r in rows], labels)),
            "additional_payments": authority_count(runs["single"], rows),
            "changed_decisions": changed_decisions(runs["single"], rows),
            "payments": sum(row.trace.receipt is not None for row in rows),
            "verification_calls": sum(row.verification_calls for row in rows),
            "tier3_referrals": sum(
                row.trace.decision.tier == c.Tier.NAMED_APPROVER
                and row.trace.decision.outcome == c.Outcome.REFER
                for row in rows
            ),
        }
    return result


def _commission(fixture, independent):
    commission = {
        **COMMISSION,
        "seed": fixture.manifest["seed"],
        "fixture_sha256": fixture.manifest["sha256"],
        "verifiers": {"agreeing": AgreeingVerifier.version, "independent": independent.version},
        "deadline_seconds": 1,
    }
    commission["sha256"] = fingerprint(commission)
    return commission


def run_experiment(seed=7, *, independent=None, include_public=True):
    fixture = build_fixture(seed)
    independent = independent if independent is not None else IndependentVerifier()
    commission = _commission(fixture, independent)
    events, runs = [], {}
    for name, verifier in (
        ("single", None),
        ("agreeing", AgreeingVerifier()),
        ("independent", independent),
    ):
        runs[name] = run_arm(fixture.data, verifier)
        events.append(f"{name} finished")
    if not fixture.intact() or fixture.truth.log:
        raise ValueError("fixture changed or truth was consulted before all arms finished")
    truth = fixture.truth.consult(
        "matched experiment", "score all completed arms; no fitting or tuning"
    )
    events.append("truth consulted")
    labels = [row.label for row in truth]
    arms = {name: _arm_record(rows, runs["single"], labels, truth) for name, rows in runs.items()}
    return {
        "fixture": fixture,
        "commission": commission,
        "events": events,
        "runs": runs,
        "arms": arms,
        "public": public_comparison() if include_public else {},
    }


class InAgentTools(Tools):
    """Diagnostic: the same scripted evidence check inside the existing payment gate."""

    def __init__(self, data):
        super().__init__(data)
        self._checks: dict[str, str | None] = {}
        self.words_read = 0
        self.computation_seconds = 0.0

    @property
    def checks(self):
        return len(self._checks)

    @property
    def verification_calls(self):
        return 0

    def _authorize(self, state: c.State, amount: int, approval: c.Approval | None) -> str | None:
        problem = super()._authorize(state, amount, approval)
        if problem or tier_for(amount) != c.Tier.AGENT:
            return problem
        assert state.proposal_id is not None and state.execution_at is not None
        proposal = self.proposals[state.proposal_id]
        if proposal.proposal_id not in self._checks:
            reads = ReadOnlyTools(self._data, state.execution_at)
            started = perf_counter()
            try:
                answer, reason = independent_check(proposal, reads)
                problem = f"verifier_objection:{reason}" if answer == "object" else None
            except TimeoutError:
                problem = "verifier_timeout"
            except (TypeError, ValueError):
                problem = "verifier_verdict_void"
            except Exception:
                problem = "verifier_error"
            finally:
                self.computation_seconds += perf_counter() - started
                self.words_read += reads.words + len(proposal.request.text.split())
            self._checks[proposal.proposal_id] = problem
        problem = self._checks[proposal.proposal_id]
        if problem:
            self._review(c.ToolCall(name="approval.request"), state)
        return problem


def run_diagnostic(seed=7):
    """Fresh arms and one post-run truth read; outside the commission and its rule."""
    fixture = build_fixture(seed)
    runs = {
        "single": run_arm(fixture.data),
        "independent": run_arm(fixture.data, IndependentVerifier()),
    }
    profiles, inline = [], []
    for request in fixture.data.requests:
        tools = InAgentTools(scoped_data(fixture.data, request))
        trace = Agent(tools).run(request)
        inline.append(CaseRun(trace, (), tuple(tools.audits.values()), len(tools.reviews), 0))
        profiles.append(tools)
    runs["in_agent"] = inline
    if not fixture.intact() or fixture.truth.log:
        raise ValueError("fixture changed or truth consulted before all diagnostic arms finished")
    truth = fixture.truth.consult(
        "in-agent diagnostic", "score completed diagnostic arms; outside the pre-run commission"
    )
    return {
        "fixture": fixture,
        "runs": runs,
        "events": [*(f"{name} finished" for name in runs), "truth consulted"],
        "arm": _diagnostic_arm(runs, profiles, inline, truth),
    }


def _diagnostic_arm(runs, profiles, inline, truth):
    checks = sum(tools.checks for tools in profiles)
    return {
        "cases": len(inline),
        "correct": sum(
            agreement([row.trace.decision for row in inline], [t.label for t in truth])
        ),
        "payments": sum(row.trace.receipt is not None for row in inline),
        "wrong_payments": sum(
            row.trace.receipt is not None and not t.owed for row, t in zip(inline, truth)
        ),
        "referrals": sum(row.trace.decision.outcome == c.Outcome.REFER for row in inline),
        "additional_payments": authority_count(runs["single"], inline),
        "changed_decisions": changed_decisions(runs["single"], inline),
        "independent_differences": sum(
            a.trace.decision != b.trace.decision for a, b in zip(inline, runs["independent"])
        ),
        "verification_calls": 0,
        "verdict_records": 0,
        "review_items": sum(row.reviews for row in inline),
        "checks": checks,
        "words_per_check": sum(tools.words_read for tools in profiles) / checks
        if checks
        else 0,
        "seconds_per_request": sum(tools.computation_seconds for tools in profiles)
        / len(inline),
    }
