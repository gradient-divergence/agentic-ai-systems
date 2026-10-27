"""Chosen cases and a field-by-field swap record, not an equivalence proof."""

import inspect
import json
from collections import Counter
from dataclasses import asdict, dataclass, replace
from datetime import timedelta
from hashlib import sha256
from pathlib import Path
from typing import cast

from refund_agent import contracts as c
from refund_agent.agent import Agent
from refund_agent.data.generate import Dataset, generate
from refund_agent.evaluation import POSITIVE_CASES, _cases, _positive_data
from refund_agent.model import ScriptedModel
from refund_agent.runtimes.adapter import ADAPTER_VERSION, AdapterSettings, RefundAdapter
from refund_agent.runtimes.faults import FaultTools
from refund_agent.tools import Tools

FIELDS = (
    "decision",
    "outcome",
    "reason",
    "tier",
    "amount_cents",
    "executed",
    "tool_calls",
    "ledger_receipts",
    "audits",
    "failure_detail",
)
CONFIGURATIONS = (
    ("runtime defaults", AdapterSettings(False, False, False, False, False)),
    ("audit ending", AdapterSettings(True, False, False, False, False)),
    ("ordered calls", AdapterSettings(True, True, False, False, False)),
    ("referral ending", AdapterSettings(True, True, True, False, False)),
    ("no retries", AdapterSettings(True, True, True, True, False)),
    ("full adapter", AdapterSettings()),
)


class MalformedModel(ScriptedModel):
    def reply(self, state):
        if state.phase == c.Phase.EXECUTE:
            return c.ToolCall.model_construct(
                name="refund.execute",
                arguments=c.ToolArguments.model_construct(amount_cents=-1, idempotency_key="bad"),
            )
        return super().reply(state)


@dataclass(frozen=True)
class SwapCase:
    case_id: str
    kind: str
    data: Dataset
    request: c.Request
    allowed_cents: int
    named_denial: bool = False
    wrong_reply: str | None = None
    malformed: bool = False
    fault_tool: str | None = None
    after: bool = False
    max_steps: int = 12
    approval_role: str | None = None
    resume: str | None = None

    def model(self):
        return MalformedModel() if self.malformed else ScriptedModel(self.wrong_reply)


def _project(data, request):
    orders = [o for o in data.orders if request.order_reference in (o.order_id, o.reference)]
    ids = {o.order_id for o in orders}
    return Dataset(
        orders=orders,
        requests=[request],
        policies=data.policies,
        shipments=[s for s in data.shipments if s.order_id in ids],
        retrievals=[s for s in data.retrievals if s.request_id == request.request_id],
        correct=[d for d in data.correct if d.request_id == request.request_id],
        reviewers=data.reviewers,
    )


def build_cases(seed=7):
    if seed != 7:
        raise ValueError("this chosen public suite uses seed 7")
    data = generate(seed, orders=40, requests=40)
    positive = _positive_data()
    return (
        _public_cases(data)
        + _refusal_cases(data)
        + _positive_cases(positive)
        + _fault_cases(data)
        + _resume_cases(positive)
    )


def _public_cases(data):
    labels = {d.request_id: d for d in data.correct}
    cases = []
    for rid in data.public_scenarios:
        request = next(r for r in data.requests if r.request_id == rid)
        label = labels[rid]
        allowed = (
            request.amount_cents if label.outcome == c.Outcome.APPROVE and label.tier == 1 else 0
        )
        cases.append(
            SwapCase(
                f"public-{rid}",
                "public",
                _project(data, request),
                request,
                allowed,
                named_denial=label.tier == 3 and label.outcome == c.Outcome.DENY,
            )
        )
    return cases


def _refusal_cases(data):
    cases = []
    for i, (request, model, reason) in enumerate(_cases(data)):
        cases.append(
            SwapCase(
                f"refusal-{i + 1}-{reason}",
                "refusal",
                _project(data, request),
                request,
                0,
                wrong_reply=model.wrong_reply,
            )
        )
    return cases


def _positive_cases(positive):
    cases = []
    for request in positive.requests:
        amount, role = POSITIVE_CASES[request.request_id]
        cases.append(
            SwapCase(
                request.request_id,
                "positive",
                _project(positive, request),
                request,
                amount,
                approval_role=role,
            )
        )
    return cases


def _fault_cases(data):
    request = next(r for r in data.requests if r.request_id == "r-7-0007")
    common = SwapCase("fault", "fault", _project(data, request), request, 2499)
    return [
        replace(common, case_id="execute-before", fault_tool="refund.execute"),
        replace(common, case_id="execute-after", fault_tool="refund.execute", after=True),
        replace(common, case_id="shipment-timeout", fault_tool="shipment.status"),
        replace(common, case_id="malformed-reply", allowed_cents=0, malformed=True),
        replace(common, case_id="budget-four", max_steps=4),
    ]


def _resume_cases(positive):
    cases = []
    for direction, rid in (("own-to-second", "positive-5000"), ("second-to-own", "positive-50000")):
        request = next(r for r in positive.requests if r.request_id == rid)
        cases.append(
            SwapCase(
                f"resume-{direction}",
                "resume",
                _project(positive, request),
                request,
                request.amount_cents,
                approval_role="reviewer",
                resume=direction,
            )
        )
    return cases


def _approve(tools, request, role):
    proposal = next(
        (p for p in tools.proposals.values() if p.request_id == request.request_id), None
    )
    if proposal is None:
        return False
    approval = c.Approval(
        approval_id=f"approval-{request.request_id}",
        request_id=request.request_id,
        order_id=proposal.evidence.order.order_id,
        proposal_id=proposal.proposal_id,
        amount_cents=proposal.amount_cents,
        reason=proposal.reason,
        policy_version=proposal.evidence.policy.version,
        reviewer="alex",
        role=role,
        approved_at=proposal.created_at + timedelta(seconds=1),
    )
    return tools.record_approval(approval).ok


def observe(trace, tools):
    chosen = trace.decision.model_dump(mode="json")
    rid = trace.decision.request_id
    return {
        "decision": chosen,
        "outcome": chosen["outcome"],
        "reason": chosen["reason"],
        "tier": chosen["tier"],
        "amount_cents": chosen["amount_cents"],
        "executed": chosen["executed"],
        "tool_calls": list(tools.calls),
        "ledger_receipts": _receipts(tools, rid),
        "audits": _audits(tools, rid),
        "failure_detail": trace.failure_detail,
    }


def _receipts(tools, rid):
    return [
        {
            "request_id": rid,
            "receipt_id": r.receipt_id,
            "order_id": r.order_id,
            "proposal_id": r.proposal_id,
            "policy_version": r.evidence.policy.version if r.evidence else None,
            "amount_cents": r.amount_cents,
            "idempotency_key": r.idempotency_key,
            "approvals": [a.model_dump(mode="json") for a in r.approvals],
        }
        for r in tools.payments
        if r.request and r.request.request_id == rid
    ]


def _audits(tools, rid):
    return [
        {
            "event": a.event,
            "decision": a.decision.model_dump(mode="json"),
            "reason": a.decision.reason,
            "approval_ids": a.approval_ids,
            "proposal_id": a.proposal_id,
            "receipt_id": a.receipt_id,
            "money_moved": a.money_moved,
            "paid_amount_cents": a.paid_amount_cents,
            "failure_detail": a.failure_detail,
        }
        for a in tools.audits.values()
        if a.decision.request_id == rid
    ]


def run_case(
    case, settings=None, *, tools_factory=Tools, runner_type=None, adapter_factory=RefundAdapter
):
    tools = FaultTools(tools_factory(case.data), case.fault_tool, case.after)
    config = (
        replace(settings, max_steps=min(case.max_steps, settings.max_steps)) if settings else None
    )

    def run(second):
        if not second:
            return Agent(cast(Tools, tools), case.model(), max_steps=case.max_steps).run(
                case.request
            )
        options = {"runner_type": runner_type} if runner_type is not None else {}
        return adapter_factory(tools, case.model(), settings=config, **options).run(case.request)

    second = settings is not None
    first_second = second and case.resume != "own-to-second"
    trace = run(first_second)
    if case.approval_role and _approve(tools, case.request, case.approval_role):
        last_second = second and case.resume != "second-to-own"
        trace = run(last_second)
    return observe(trace, tools)


def compare(
    cases,
    settings,
    *,
    tools_factory=Tools,
    runner_type=None,
    adapter_factory=RefundAdapter,
    name="custom",
):
    rows = []
    for case in cases:
        own = run_case(case, tools_factory=tools_factory)
        second = run_case(
            case,
            settings,
            tools_factory=tools_factory,
            runner_type=runner_type,
            adapter_factory=adapter_factory,
        )
        rows.append(_comparison_row(case, own, second))
    return _comparison_summary(rows, settings, name)


def _comparison_summary(rows, settings, name):
    violations = _unauthorized_cents(rows)
    different = sum(bool(row["different_fields"]) for row in rows)
    return {
        "name": name,
        "settings": asdict(settings),
        "total_cases": len(rows),
        "cases": rows,
        "different_cases": different,
        "money_differences": sum(
            row["own"]["ledger_receipts"] != row["second"]["ledger_receipts"] for row in rows
        ),
        "unauthorized_cents": violations,
        "differences_by_field": {
            f: sum(f in row["different_fields"] for row in rows) for f in FIELDS
        },
        "differences_by_kind": {
            k: sum(bool(r["different_fields"]) for r in rows if r["kind"] == k)
            for k in dict.fromkeys(r["kind"] for r in rows)
        },
        **_verdict(rows, different, violations),
    }


def fingerprint(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _code_fingerprint(*paths):
    return fingerprint({str(p.name): sha256(p.read_bytes()).hexdigest() for p in paths})


def build_record(seed=7, *, tools_factory=Tools):
    cases = build_cases(seed)
    record = _record_metadata(seed, cases, tools_factory)
    record["configurations"] = [
        compare(cases, settings, tools_factory=tools_factory, name=name)
        for name, settings in CONFIGURATIONS
    ]
    return record


def _record_metadata(seed, cases, tools_factory):
    package = Path(__file__).resolve().parents[1]
    runtime = package / "runtimes"
    model = _model_identity(package, seed)
    suite = _suite_record(cases)
    return {
        "seed": seed,
        "base": "cf3f5da0 (decision-1 repair e759eb11 included)",
        "agent_fingerprint": _code_fingerprint(
            package / "agent/__init__.py", package / "tools.py", package / "contracts.py"
        ),
        **_runtime_identity(package, runtime),
        "model": model,
        "suite": {
            "fingerprint": fingerprint(suite),
            "cases": suite,
            "source_fingerprint": _code_fingerprint(runtime / "swap.py", runtime / "faults.py"),
        },
        "tools_composition": {
            "identity": f"{tools_factory.__module__}.{tools_factory.__qualname__}",
            "fingerprint": sha256(inspect.getsource(tools_factory).encode()).hexdigest(),
        },
        **_record_context(cases),
    }


def _suite_record(cases):
    return [
        {
            "case_id": case.case_id,
            "kind": case.kind,
            "data_fingerprint": fingerprint(case.data.model_dump(mode="json")),
            "wrong_reply": case.wrong_reply,
            "malformed": case.malformed,
            "fault_tool": case.fault_tool,
            "after": case.after,
            "max_steps": case.max_steps,
            "approval_role": case.approval_role,
            "resume": case.resume,
            "allowed_cents": case.allowed_cents,
        }
        for case in cases
    ]


def _record_context(cases):
    return {
        "case_counts": dict(Counter(case.kind for case in cases)),
        "fields_compared": list(FIELDS),
        "ownership": {
            "contracts": "refund_agent/contracts.py",
            "gate": "refund_agent/tools.py:Tools._authorize",
            "approval_records": "supplied Tools",
            "ledger": "supplied Tools simulated ledger",
            "audit_log": "supplied Tools",
            "semantics": "refund_agent/runtimes/adapter.py",
            "session": "Runner Session, in process",
            "spans": "Runner Session, joined by trace_id",
        },
        "data_access": {"frozen_reads": 0, "held_out_reads": 0},
        "limits": [
            "Chosen cases show coverage, not a rate or equivalence.",
            "The scripted model is held fixed; live-model variation is not measured.",
            "No concurrency, process crash, or hosted-session export is tested.",
            "No vendor SDK is run, and no shadow or deployment is authorized.",
        ],
    }


def _model_identity(package, seed):
    return {
        "identity": "ScriptedModel",
        "settings": {"seed": seed, "temperature": "not applicable"},
        "fingerprint": _code_fingerprint(package / "model/__init__.py", package / "rules.py"),
    }


def _runtime_identity(package, runtime):
    return {
        "runtimes": [
            {
                "position": "own loop",
                "identity": "Agent.run",
                "fingerprint": _code_fingerprint(package / "agent/__init__.py"),
            },
            {
                "position": "library runtime",
                "identity": "offline Runner",
                "fingerprint": _code_fingerprint(runtime / "runner.py"),
            },
        ],
        "adapter": {
            "version": ADAPTER_VERSION,
            "settings": asdict(AdapterSettings()),
            "fingerprint": _code_fingerprint(runtime / "adapter.py"),
        },
    }


def write_record(record, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _comparison_row(case, own, second):
    differences = [field for field in FIELDS if own[field] != second[field]]
    return {
        "case_id": case.case_id,
        "kind": case.kind,
        "allowed_cents": case.allowed_cents,
        "named_denial": case.named_denial,
        "own": own,
        "second": second,
        "different_fields": differences,
    }


def _unauthorized_cents(rows):
    return sum(
        max(
            0,
            sum(r["amount_cents"] for r in row["second"]["ledger_receipts"]) - row["allowed_cents"],
        )
        for row in rows
    )


def _verdict(rows, different, violations):
    return {
        "decision": "held"
        if not rows or different or violations
        else "eligible for the shadow run",
        "reasons": (
            ["missing coverage: suite is empty"]
            if not rows
            else ["observable records differ"]
            if different
            else []
        ),
    }
