"""A fixed coverage suite with explicit agent and rules-only comparison arms.

Existing fixtures supply cases, not quality estimates. Generated scenario sets and
memory funnels are reported separately. A spent Chapter 14 artifact is logged and
fed through RegressionSuite; no GoldenSet is constructed or consulted here.
"""

import json
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timedelta, timezone
from functools import lru_cache, wraps
from pathlib import Path
from types import MappingProxyType

from refund_agent import contracts as c
from refund_agent.adaptation.arms import dated_index, prompt_arm
from refund_agent.adaptation.experiment import read_evidence
from refund_agent.adaptation.suite import RegressionSuite
from refund_agent.baseline import baseline
from refund_agent.data.generate import Dataset, generate
from refund_agent.evaluation import _positive_data
from refund_agent.labels import LABEL_POLICY
from refund_agent.memory.assemble import required_facts
from refund_agent.memory.memory import NOTICE
from refund_agent.memory.register import PolicyRegister
from refund_agent.memory.support import support
from refund_agent.model import ScriptedModel, Transcript
from refund_agent.release import extensions
from refund_agent.retrieval.encoders import CountEncoder
from refund_agent.retrieval.index import policy_clauses
from refund_agent.rules import evidence_problem
from refund_agent.runtimes import swap
from refund_agent.runtimes.faults import FaultTools
from refund_agent.scenarios.suite import ScenarioSet
from refund_agent.selection import fingerprint
from refund_agent.split import CUTS, time_split
from refund_agent.split import fingerprint as data_fingerprint
from refund_agent.tools import Tools

IMPLEMENTATION_FILES = extensions.resources.IMPLEMENTATION_FILES

DATA = Path(__file__).resolve().parents[2] / "demos/data"
AT = datetime(2026, 10, 1, tzinfo=timezone.utc)


def _signed_requirement():
    from refund_agent.release.candidate import source_fingerprint

    requirement = json.loads(Path(__file__).with_name("requirement.json").read_text())
    data = generate(7, orders=40, requests=40)
    tier1 = [r for r in data.requests if r.amount_cents < 5000]
    share = sum(d.outcome == c.Outcome.REFER for d in baseline(data, tier1)) / len(tier1)
    identity = fingerprint(
        {
            "code": source_fingerprint("baseline.py"),
            "requests": [r.model_dump(mode="json") for r in tier1],
        }
    )
    if share != requirement["max_referral_share"] or identity != requirement["baseline_identity"]:
        raise ValueError("baseline differs from the frozen signed referral requirement")
    return requirement


@dataclass(frozen=True)
class Case:
    case_id: str
    part: str
    failure_class: str
    tier: int
    fixture: swap.SwapCase | None = None
    handler: str = "loop"
    expected: dict | None = None
    must_pass: bool = True

    def record(self):
        row = {k: _plain(v) for k, v in vars(self).items() if k != "fixture"}
        if self.fixture is not None:
            row["fixture"] = asdict(self.fixture)
            row["fixture"]["data"] = self.fixture.data.model_dump(mode="json")
            row["fixture"]["request"] = self.fixture.request.model_dump(mode="json")
        return row


def _freeze(value):
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (tuple, list)):
        return tuple(_freeze(item) for item in value)
    return value


def _plain(value):
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


# region chapter29:evaluation-snapshots
def detached_case(case):
    """Copy fixtures and freeze detached predicates before exposing a case."""
    return replace(case, fixture=deepcopy(case.fixture), expected=_freeze(_plain(case.expected)))


def manifest_guard(function):
    @wraps(function)
    def guarded(candidate, suite, *args, **kwargs):
        manifest, software = suite.fingerprint, candidate.fingerprint
        try:
            return function(candidate, suite, *args, **kwargs)
        finally:
            if suite.fingerprint != manifest or candidate.fingerprint != software:
                raise ValueError("evaluation manifest or software changed during execution")

    return guarded


# endregion chapter29:evaluation-snapshots


# region chapter29:execution-artifacts
@dataclass(frozen=True)
class ExecutionArtifacts:
    """Internal observations captured from the core-owned execution context."""

    trace: c.Trace
    tools: Tools
    tool_calls: int = 0


# endregion chapter29:execution-artifacts


# region chapter29:case-execution
@dataclass(frozen=True, slots=True)
class CaseExecution:
    """Read-only scheduling surface; execution stores remain with the core."""

    _schedule: Callable[..., None]
    _approval: Callable[..., bool]
    _count: Callable[[], int]

    @property
    def runs(self):
        return self._count()

    def run(self, *, at=None, max_steps=None):
        self._schedule(at=at, max_steps=max_steps)

    def approve(self, *, role=None):
        return self._approval(role=role)


class _CaseRun:
    """Construct, execute and retain observations from the selected candidate."""

    def __init__(self, case, candidate, mode):
        if case.fixture is None or case.expected is None:
            raise ValueError("scheduled execution requires a fixed fixture and expectations")
        self._case, self._candidate = detached_case(case), candidate
        self._manifest = _freeze(self._case.record())
        fixture = self._case.fixture
        self._tools = _fixture_tools(
            fixture.data.model_copy(deep=True, update={"correct": []}),
            candidate,
            mode,
            purpose="loop-case",
            case=case,
        )
        self._before = (self._tools.ledger, self._tools.reviews)
        self._wrapped = FaultTools(self._tools, fixture.fault_tool, fixture.after)
        self._model = fixture.model() if candidate.name != "deny-all" else None
        self._trace = None
        self.runs = 0
        self.context = CaseExecution(self.run, self.approve, lambda: self.runs)

    def run(self, *, at=None, max_steps=None):
        from refund_agent.release.candidate import run_request

        fixture = self._case.fixture
        budget = fixture.max_steps if max_steps is None else min(fixture.max_steps, max_steps)
        self._trace = run_request(
            self._candidate,
            self._wrapped,
            fixture.request,
            model=self._model,
            at=at,
            max_steps=budget,
        )
        self.runs += 1

    def approve(self, *, role=None):
        fixture = self._case.fixture
        role = role or fixture.approval_role
        if role is None:
            raise ValueError("scheduled approval requires a role")
        return approve(self._tools, fixture.request, role)

    def _artifacts(self):
        if self._trace is None:
            raise ValueError("case handler must schedule execution through the core context")
        return ExecutionArtifacts(self._trace, self._tools, self._wrapped.calls)


# endregion chapter29:case-execution


@dataclass(frozen=True)
class Suite:
    cases: tuple[Case, ...]
    accesses: tuple[dict, ...]
    generated: tuple[ScenarioSet, ...]
    version: str = "refund-release-suite-2"
    trials: int = 5
    requirement: dict = field(default_factory=_signed_requirement)
    components: tuple = ()

    def __post_init__(self):
        if type(self.trials) is not int or self.trials < 1:
            raise ValueError("positive integer declared trials required")
        object.__setattr__(self, "requirement", _freeze(self.requirement))
        object.__setattr__(self, "cases", tuple(detached_case(case) for case in self.cases))
        for name in ("accesses", "generated", "components"):
            object.__setattr__(self, name, deepcopy(getattr(self, name)))
        extensions.validate_suite(self)

    def manifest(self):
        from refund_agent.release.candidate import source_fingerprint

        budget = json.loads((DATA / "adaptation_model_choice_record.json").read_text())["budget"]
        manifest = dict(
            version=self.version,
            seed=7,
            trials=self.trials,
            arms=["candidate agent bundle", "rules-only baseline on identical requests"],
            cases=[case.record() for case in self.cases],
            budget={"max_steps": 12, "window": budget["window"], "shares": budget["shares"]},
            rule="every selected must-pass case on every trial; drafts-to-shadow floor; no unchanged rerun after failure",
            slots={
                "26": "candidate draft support; funnel apart",
                "27": "current matched verifier record and composed checks when selected",
                "28": "swap record on supplied candidate tools, both runtimes",
                "21": "evaluation record for selected notice; no outcomes read",
            },
            implementation=source_fingerprint(*IMPLEMENTATION_FILES),
            generated_fingerprints=[s.fingerprint for s in self.generated],
            human_review={"status": "pending", "sample_size": 5},
            requirement=_plain(self.requirement),
        )
        slots = extensions.suite_slots(self.components)
        if manifest["slots"].keys() & slots.keys():
            raise ValueError("component replaces a stage-4 manifest slot")
        manifest["slots"].update(slots)
        if self.components:
            manifest["components"] = extensions.records(self.components)
        return manifest

    @property
    def fingerprint(self):
        return fingerprint(self.manifest())

    def without(self, case_id):
        return replace(self, cases=tuple(c for c in self.cases if c.case_id != case_id))

    def coverage(self):
        matrix = defaultdict(lambda: {str(t): [] for t in (1, 2, 3)})
        for case in self.cases:
            if case.tier:
                matrix[case.failure_class][str(case.tier)].append(case.case_id)
        return dict(sorted(matrix.items()))


def _scenario_cases(scenario_set, part):
    cases = []
    for row in scenario_set.scenarios:
        request = row.records.request
        paid = request.amount_cents if row.label.outcome == c.Outcome.APPROVE else 0
        fixture = swap.SwapCase(f"{part}-{row.draw}", part, _scenario_data(row), request, paid)
        expected = (
            _named_expectation(row.label.reason)
            if row.label.tier == 3
            else {
                "paid_cents": paid,
                "outcome": row.label.outcome.value,
                "reason": row.label.reason,
            }
        )
        cases.append(
            Case(
                fixture.case_id,
                part,
                "eligibility-and-tier",
                int(row.label.tier),
                fixture,
                expected=expected,
            )
        )
    return cases


def _loop_cases():
    data = generate(7, orders=40, requests=40)
    labels = {d.request_id: d for d in data.correct}
    cases = []
    for fixture in swap.build_cases():
        request = fixture.request
        tier = int(request.amount_cents < 5000) or (2 if request.amount_cents <= 50000 else 3)
        expected = {"paid_cents": fixture.allowed_cents}
        failure_class = {
            "public": "eligibility-and-tier",
            "refusal": "unsafe-or-stale",
            "positive": "must-pay",
            "fault": "tool-failure",
            "resume": "state-continuation",
        }[fixture.kind]
        if fixture.kind == "public":
            label = labels[request.request_id]
            if tier == 3:
                expected = _named_expectation(label.reason)
            else:
                chosen = baseline(fixture.data)[0]
                expected.update(outcome=chosen.outcome.value, reason=chosen.reason)
        if fixture.kind == "refusal":
            reason = fixture.case_id.split("-", 2)[2]
            expected = {"paid_cents": 0, "reason": reason, "outcome": "refer"}
        if fixture.kind == "fault":
            expected["paid_cents"] = 2499 if fixture.case_id == "execute-after" else 0
        cases.append(
            Case(fixture.case_id, fixture.kind, failure_class, tier, fixture, expected=expected)
        )
    return cases


def _named_expectation(reason):
    return {
        "paid_cents": 0,
        "named_reason": reason,
        "policy_reason": "eligible" if reason == "approval_required" else reason,
    }


def build_suite(trials=5, *, requirement=None, components=()):
    # Check the baseline even when the fixed fixtures are cached.
    frozen = json.loads(Path(__file__).with_name("requirement.json").read_text())
    if requirement is None or requirement == frozen:
        requirement = _signed_requirement()
    from refund_agent.release.rule import _requirement

    _requirement(requirement)
    return extensions.extend_suite(
        replace(_base_suite(trials), requirement=requirement), components
    )


@lru_cache(maxsize=4)
def _base_suite(trials):
    listed = ScenarioSet.read(DATA / "generative_models_scenarios_listed.json")
    generated = tuple(
        ScenarioSet.read(DATA / f"generative_models_scenarios_tau{t}.json") for t in (1, 2)
    )
    cases = _loop_cases() + _scenario_cases(listed, "listed") + _control_cases()
    accesses = (
        {
            "path": "demos/data/adaptation_report_05.json",
            "purpose": "reuse spent chapter 14 regression artifact; no golden consultation",
        },
        {
            "path": "demos/data/practical_rl_evaluation_record.json",
            "purpose": "read chapter 21 evaluation record; no outcomes read",
        },
        {
            "path": "book_outputs/_fnd.memory-rag.funnel-01.txt",
            "purpose": "read chapter 26 funnel record; counts reported apart",
        },
        {
            "path": "book_outputs/_fnd.multi-agent-systems.arms-01.txt",
            "purpose": "read chapter 27 matched record; current component judged again",
        },
    )
    return Suite(tuple(cases), accesses, generated, trials=trials, requirement={})


build_suite.cache_clear = _base_suite.cache_clear


def selected(case, candidate):
    chosen = extensions.handle(candidate, "select_case", case=case)
    if chosen is not None and not case.must_pass:
        return chosen
    return case.handler != "verifier" or candidate.tools_composition["verifier"] != "off"


def approve(tools, request, role="reviewer"):
    proposal = next(
        (p for p in tools.proposals.values() if p.request_id == request.request_id), None
    )
    if proposal is None:
        return False
    approval = c.Approval(
        approval_id=f"approval-{proposal.proposal_id}",
        request_id=request.request_id,
        order_id=proposal.evidence.order.order_id,
        proposal_id=proposal.proposal_id,
        amount_cents=proposal.amount_cents,
        reason=proposal.reason,
        policy_version=proposal.evidence.policy.version,
        reviewer="morgan" if role == "named_approver" else "alex",
        role=role,
        approved_at=proposal.created_at + timedelta(seconds=1),
    )
    candidate = getattr(tools, "_release_candidate", None)
    context = (
        {}
        if candidate is None
        else extensions.apply(
            candidate, "approval_context", value={}, tools=tools, approval=approval
        )
    )
    return tools.record_approval(approval, **context).ok


def run_loop_case(case, candidate, *, mode="execute"):
    return _run_loop_case(case, candidate, mode, _CaseRun(case, candidate, mode))


def _run_loop_case(case, candidate, mode, execution):
    result = extensions.handle(
        candidate, "run_loop_case", case=case, mode=mode, execution=execution.context
    )
    if result is None:
        execution.run()
        if case.fixture.approval_role and execution.approve():
            execution.run()
    return _grade_artifacts(case, candidate, execution, mode=mode)


# region chapter29:effect-grading
def _grade_artifacts(case, candidate, execution, *, mode="execute"):
    from refund_agent.release.candidate import validate_final_decision

    if not isinstance(execution, _CaseRun) or case.fixture is None or case.expected is None:
        raise ValueError("case handler must schedule execution through the core context")
    case = execution._case  # Grade the detached core fixture, never the hook's view.
    if _plain(execution._manifest) != case.record():
        raise ValueError("evaluation manifest fixture changed during execution")
    artifacts = execution._artifacts()
    trace, tools = artifacts.trace, artifacts.tools
    if not isinstance(trace, c.Trace) or not isinstance(tools, Tools):
        raise ValueError("execution artifacts require a trace and actual tool stores")
    validate_final_decision(trace, tools, case.fixture.request)
    failed = _loop_effects(case, candidate, trace, tools) if mode == "execute" else []
    if mode == "shadow":
        from refund_agent.release.graders import shadow_effects

        if not hasattr(tools, "shadow_log") or not shadow_effects(tools, *execution._before):
            failed.append("shadow_boundary_preserved")
    return {
        "failed": failed,
        "decision": trace.decision.model_dump(mode="json"),
        "trace_id": trace.trace_id,
        "tool_calls": artifacts.tool_calls,
        "approvals": [a.approval_id for a in trace.steps[-1].state.approvals],
        "payments": len(tools.payments),
        "review_items": len(tools.reviews),
        "transcript": Transcript.from_trace(trace).model_dump(mode="json"),
    }


# endregion chapter29:effect-grading


def _loop_effects(case, candidate, trace, tools):
    from refund_agent.release.graders import effects

    budget = min(case.fixture.max_steps, candidate.adapter["settings"]["max_steps"])
    return effects(trace, tools, max_steps=budget, **case.expected)


def _continued(case, candidate):
    prepared = _approved_state(candidate)
    if prepared is None:
        return {"failed": [], "reason": "no continuation; control refused", "payments": 0}
    data, tools, state, approval = prepared
    tools.data = _changed_evidence(data, case.handler)
    extensions.notify(candidate, "changed_source", tools=tools, case=case)
    state.execution_at = state.request.at + timedelta(seconds=2, microseconds=1)
    arguments = c.ToolArguments(
        amount_cents=state.request.amount_cents,
        idempotency_key=f"continue-{case.handler}",
        approval_id=approval.approval_id,
    )
    result = tools.call(c.ToolCall(name="refund.execute", arguments=arguments), state)
    return {
        "failed": ["no_payment_on_changed_evidence"] if tools.payments else [],
        "reason": result.reason,
        "payments": len(tools.payments),
        "paid_cents": sum(p.amount_cents for p in tools.payments),
        "audit_events": len(tools.audits),
        "review_items": len(tools.reviews),
    }


def _approval_reuse(candidate):
    from refund_agent.release.candidate import run_request

    data = _positive_data()
    request = next(r for r in data.requests if r.request_id == "positive-5000")
    tools = _fixture_tools(
        swap._project(data, request), candidate, "execute", purpose="approval-reuse"
    )
    trace = run_request(candidate, tools, request)
    if not approve(tools, request):
        return {"failed": [], "reason": "no approval to reuse"}
    state = trace.steps[-1].state.model_copy(deep=True)
    approval = next(iter(tools._approvals.values()))
    state.execution_at = approval.approved_at + timedelta(microseconds=1)
    state.request = request.model_copy(update={"amount_cents": 4999})
    call = c.ToolCall(
        name="refund.execute",
        arguments=c.ToolArguments(
            amount_cents=4999, idempotency_key="stale-approval", approval_id=approval.approval_id
        ),
    )
    result = tools.call(call, state)
    return {
        "failed": ["proposal_bound_approval_no_payment"] if tools.payments else [],
        "reason": result.reason,
        "payments": len(tools.payments),
        "review_items": len(tools.reviews),
    }


def _contract(case, candidate):
    if case.handler == "bad-key":
        return _empty_key_case(candidate)
    fields = (
        {"name": "refund.execute", "arguments": {"amount_cents": -1}}
        if case.handler == "negative-amount"
        else {"name": "order.lookup", "unexpected": True}
    )
    try:
        c.ToolCall.model_validate(fields)
    except ValueError:
        return {"failed": []}
    return {"failed": ["strict_contract_rejected"]}


def _freshness(case, candidate):
    from refund_agent.release.candidate import run_request
    from refund_agent.release.graders import effects
    from refund_agent.rules import _FRESHNESS_LIMITS

    source, edge = case.handler.split("-")
    data = _positive_data()
    request, age = data.requests[0], _FRESHNESS_LIMITS[source] + int(edge == "past")
    key = {"order": "orders", "shipment": "shipments", "policy": "retrievals"}[source]
    records = [
        row.model_copy(update={"observed_at": request.at - timedelta(seconds=age)})
        if (
            getattr(row, "order_id", None) == request.order_reference
            or getattr(row, "request_id", None) == request.request_id
        )
        else row
        for row in getattr(data, key)
    ]
    data = data.model_copy(update={key: records})
    tools = _fixture_tools(data, candidate, "execute", purpose="freshness", case=case)
    trace = run_request(candidate, tools, request)
    failed = (
        effects(trace, tools, paid_cents=request.amount_cents)
        if edge == "at"
        else (["no_stale_payment"] if tools.payments else [])
    )
    return {
        "failed": failed,
        "executed": trace.decision.executed,
        "reason": trace.decision.reason,
        "age_seconds": age,
    }


@lru_cache(maxsize=1)
def spent_regression():
    report = json.loads((DATA / "adaptation_report_05.json").read_text())
    record = json.loads((DATA / "adaptation_selection_record.json").read_text())
    spent = [
        (
            c.Request.model_validate_json(json.dumps(row["request"])),
            c.Decision.model_validate_json(json.dumps(row["label"])),
        )
        for row in report["cases"]
    ]
    if data_fingerprint([r for r, _ in spent], [d for _, d in spent]) != record["frozen"]["sha256"]:
        raise ValueError("spent regression artifact differs from chapter 14 record")
    corpus = generate(7)
    windows = time_split(corpus.requests, CUTS, timedelta(days=LABEL_POLICY.window_days))
    development = windows["training"] + windows["validation"]
    ids = {r.request_id for r in development}
    data = corpus.model_copy(
        update={
            "requests": development,
            "correct": [d for d in corpus.correct if d.request_id in ids],
        }
    )
    encoder = CountEncoder().fit([clause.text for clause in policy_clauses(data.policies, CUTS[0])])
    index = dated_index(data.policies, encoder)
    return RegressionSuite.build(data, spent, record["frozen"], index)


def _memory(candidate):
    from refund_agent.release.candidate import DenyAll

    data = generate(7, orders=40, requests=40)
    register = PolicyRegister()
    for policy in data.policies:
        register.announce(policy, policy.effective_at - NOTICE)
    counts = Counter()
    failed = []
    for state in read_evidence(data, data.requests):
        if (
            state.order is None
            or state.policy is None
            or state.decision is not None
            or evidence_problem(data, state)
        ):
            counts["not_checked"] += 1
            continue
        draft = (DenyAll() if candidate.name == "deny-all" else ScriptedModel()).reply(state)
        result = support(draft, register, required_facts(state))
        counts[result["status"]] += 1
        if result["status"] == "refused":
            failed.append(state.request.request_id)
    return {
        "failed": failed,
        "draft_counts": dict(counts),
        "limit": "support check beside the loop; no policy.retrieve wiring",
    }


def _regression(candidate):
    suite = spent_regression()
    result = suite.run(prompt_arm(index=suite.index))
    return {
        "failed": list(result.newly_failing) + list(result.retrieval_regressions),
        "drafting_cases": len(suite.requests),
        "questions": len(suite.questions),
        "verdict": result.verdict,
        "limit": "spent change guard on the chapter 14 canonical prompt drafting component; not a trial of the loop, runtime or verifier",
        "candidate_code_fingerprint": candidate.agent["fingerprint"],
    }


def _runtime(candidate):
    from refund_agent.runtimes.adapter import AdapterSettings

    record = swap.compare(
        swap.build_cases(),
        AdapterSettings(**candidate.adapter["settings"]),
        tools_factory=lambda data: _fixture_tools(data, candidate, "execute", purpose="runtime"),
    )
    return {
        "failed": [r["case_id"] for r in record["cases"] if r["different_fields"]]
        + (["runtime_forbidden_payment"] if record["unauthorized_cents"] else []),
        "cases": record["total_cases"],
        "differences": record["different_cases"],
        "record_fingerprint": fingerprint(record),
        "test": "tests/refund_agent/test_runtime_swap.py",
    }


def _offer(candidate):
    from refund_agent.shadow.offers import load_offers

    selection = load_offers()
    return {
        "failed": []
        if candidate.offer_policy["evaluation_fingerprint"] == selection.record_fingerprint
        and (
            candidate.offer_policy["name"] == "fixed"
            or selection.candidates.get(candidate.offer_policy["name"], {}).get("decision")
            == "eligible for the shadow run"
        )
        else ["offer_record_not_eligible"],
        "name": candidate.offer_policy["name"],
        "candidates": selection.candidates,
        "limits": list(selection.limits),
    }


def _verifier(candidate):
    failed = []
    variants = (
        ("independent", "verifier_objection:words_contradict_record"),
        ("timeout", "verifier_timeout"),
        ("void", "verifier_verdict_void"),
    )
    for verifier, expected in variants:
        rows = []
        for runtime in ("own", "second"):
            for mode in ("execute", "shadow"):
                row, paid, reviews = _verifier_observation(candidate, verifier, runtime, mode)
                rows.append(row)
                if row[0]["reason"] != expected or paid or reviews != 1:
                    failed.append(f"{verifier}-{runtime}-{mode}")
        if any(row != rows[0] for row in rows[1:]):
            failed.append(verifier + "-composition")
    matched = (DATA.parent.parent / "book_outputs/_fnd.multi-agent-systems.arms-01.txt").read_text()
    current = _matched_verifier(candidate)
    if current["decision"] != "accepted":
        failed.append("current-matched-verifier-not-eligible")
    return {
        "failed": failed,
        "checks": 12,
        "current_matched": current,
        "matched_record": "book_outputs/_fnd.multi-agent-systems.arms-01.txt",
        "matched_record_fingerprint": fingerprint(matched),
    }


def run_case(case, candidate):
    execution = _CaseRun(case, candidate, "execute") if case.fixture is not None else None
    result = extensions.handle(
        candidate,
        "run_case",
        case=case,
        execution=execution.context if execution is not None else None,
    )
    if result is not None:
        return _grade_artifacts(case, candidate, execution)
    if case.handler == "loop":
        return _run_loop_case(case, candidate, "execute", execution)
    if case.part == "state-continuation":
        return _continued(case, candidate)
    if case.handler == "approval-reuse":
        return _approval_reuse(candidate)
    if case.part == "contracts":
        return _contract(case, candidate)
    if case.part == "execution-time":
        return _freshness(case, candidate)
    return {
        "memory": _memory,
        "regression": _regression,
        "runtime": _runtime,
        "offer": _offer,
        "verifier": _verifier,
    }[case.handler](candidate)


def add_scenario(suite, path):
    """Exercise 6's reader-owned fixture; this handler reads effects only."""
    row = json.loads(Path(path).read_text())
    required = {"case_id", "part", "failure_class", "tier", "handler", "must_pass", "expected"}
    if (
        set(row) != required
        or row["handler"] != "approval-reuse"
        or row["tier"] != 2
        or row["part"] != "approval"
        or row["failure_class"] != "approval-reuse"
        or row["must_pass"] is not True
    ):
        raise ValueError("a tier 2 must-pass approval-reuse case is required")
    if not row["case_id"] or row["case_id"] in {c.case_id for c in suite.cases}:
        raise ValueError("a unique case ID is required")
    if row["expected"] != {"stores": ["ledger", "audits", "reviews", "decision"]}:
        raise ValueError(
            "fixed-scenario wiring requires its four stores; submitted assertions are not accepted"
        )
    return replace(suite, cases=suite.cases + (Case(**row),))


def _scenario_data(row):
    request, order, shipment = row.records
    policies = generate(7, orders=40, requests=40).policies
    policy = max(
        (p for p in policies if p.effective_at <= request.at), key=lambda p: p.effective_at
    )
    retrieval = c.PolicySnapshot(
        request_id=request.request_id, version=policy.version, observed_at=request.at
    )
    return Dataset(
        orders=[order],
        requests=[request],
        shipments=[shipment],
        policies=policies,
        retrievals=[retrieval],
        correct=[],
        reviewers=_positive_data().reviewers,
    )


def _control_cases():
    cases = [
        Case(f"continue-{change}", "state-continuation", "execution-reread", 2, handler=change)
        for change in ("order", "shipment", "policy")
    ]
    cases += [
        Case(f"contract-{change}", "contracts", "contract-validation", 1, handler=change)
        for change in ("negative-amount", "extra-field", "bad-key")
    ]
    cases.append(_approval_reuse_case())
    cases += [
        Case(
            f"freshness-{source}-{edge}",
            "execution-time",
            "freshness-boundary",
            1,
            handler=f"{source}-{edge}",
        )
        for source in ("order", "shipment", "policy")
        for edge in ("at", "past")
    ]
    components = (
        ("memory", "memory-support"),
        ("regression", "regression"),
        ("runtime", "runtime-swap"),
        ("offer", "offer-record"),
        ("verifier", "verifier"),
    )
    cases += [Case(f"slot-{name}", part, name, 0, handler=name) for name, part in components]
    return cases


def _approved_state(candidate):
    from refund_agent.release.candidate import run_request

    data = _positive_data()
    request = next(r for r in data.requests if r.request_id == "positive-5000")
    data = swap._project(data, request)
    tools = _fixture_tools(data, candidate, "execute", purpose="approved-state")
    trace = run_request(candidate, tools, request)
    if not approve(tools, request):
        return None
    state = trace.steps[-1].state.model_copy(deep=True)
    approval = next(iter(tools._approvals.values()))
    state.execution_at = approval.approved_at + timedelta(microseconds=1)
    return data, tools, state, approval


def _changed_evidence(data, change):
    request = data.requests[0]
    resumed_at = request.at + timedelta(seconds=2)
    if change == "order":
        return data.model_copy(
            update={
                "orders": [
                    data.orders[0].model_copy(
                        update={"paid_cents": 4000, "observed_at": resumed_at}
                    )
                ]
            }
        )
    if change == "shipment":
        events = [c.ShipmentEvent(at=request.at - timedelta(days=1), status="shipped")]
        return data.model_copy(
            update={
                "shipments": [
                    data.shipments[0].model_copy(
                        update={"events": events, "observed_at": resumed_at}
                    )
                ]
            }
        )
    # Keep approval bindings fixed: this case isolates rereading the live policy cap.
    policy = data.policies[0].model_copy(update={"cap_cents": 4000})
    snapshot = data.retrievals[0].model_copy(update={"observed_at": resumed_at})
    return data.model_copy(update={"policies": [policy], "retrievals": [snapshot]})


def _empty_key_case(candidate):
    from refund_agent.release.candidate import run_request

    data = _positive_data()
    request = data.requests[0]
    tools = _fixture_tools(data, candidate, "execute", purpose="empty-key")
    call = c.ToolCall(
        name="refund.execute",
        arguments=c.ToolArguments(amount_cents=request.amount_cents, idempotency_key=" "),
    )
    model = ScriptedModel(replies={(request.request_id, c.Phase.EXECUTE): call})
    trace = run_request(candidate, tools, request, model=model)
    refused = not tools.payments and trace.decision.reason == "idempotency_key_required"
    return {"failed": [] if refused else ["empty_key_refused"]}


def _verifier_observation(candidate, verifier, runtime, mode):
    from refund_agent.release.candidate import run_request

    data = _positive_data()
    request = data.requests[0].model_copy(update={"text": "The parcel never arrived."})
    chosen = replace(
        candidate,
        runtime={**candidate.runtime, "identity": runtime},
        tools_composition={**candidate.tools_composition, "verifier": verifier},
    )
    tools = _fixture_tools(swap._project(data, request), chosen, mode, purpose="verifier")
    trace = run_request(chosen, tools, request)
    reviews = tools.shadow_reviews if mode == "shadow" else tools.reviews
    observation = (
        trace.decision.model_dump(mode="json"),
        [v.model_dump(mode="json") for v in tools.verdict_log.entries],
        [r.proposal.proposal_id for r in reviews.values() if r.proposal is not None],
    )
    return observation, bool(tools.payments), len(reviews)


def _matched_verifier(candidate):
    from refund_agent.release.candidate import VoidVerifier
    from refund_agent.verifier import IndependentVerifier
    from refund_agent.verifier.experiment import run_experiment

    name = candidate.tools_composition["verifier"]
    verifier = (
        VoidVerifier()
        if name == "void"
        else IndependentVerifier(latency_seconds=2 if name == "timeout" else 0)
    )
    experiment = run_experiment(7, independent=verifier, include_public=False)
    fields = ("cases", "correct", "wrong_payments", "referrals", "additional_payments", "decision")
    card = {key: experiment["arms"]["independent"][key] for key in fields}
    card["accesses"] = list(experiment["fixture"].truth.log)
    card["scope"] = (
        "current verifier against the canonical single agent; composition checked separately"
    )
    return card


def _approval_reuse_case():
    return Case(
        "approval-reused-after-proposal-change",
        "approval",
        "approval-reuse",
        2,
        handler="approval-reuse",
    )


def _fixture_tools(data, candidate, mode, **fixture):
    from refund_agent.release.candidate import build_tools

    with extensions.resources.fixture_scope(candidate, data, mode) as capability:
        context = extensions.apply(
            candidate, "fixture_context", value={}, data=data, mode=mode, **fixture
        )
        return build_tools(data, candidate, mode, **{**context, "release_fixture": capability})
