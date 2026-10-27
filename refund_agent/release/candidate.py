"""One candidate record and tools construction shared by both offline runtimes.

Part VI registers components from their own modules. Authority stays in Tools.
"""

import json
import os
import sys
from dataclasses import asdict, dataclass, replace
from functools import lru_cache
from hashlib import sha256
from pathlib import Path

from refund_agent import contracts as c
from refund_agent.agent import Agent
from refund_agent.evaluation import GateRemovedTools
from refund_agent.model import ScriptedModel
from refund_agent.release import extensions
from refund_agent.rules import approval_problem, decision, execution_time, gate, tier_for
from refund_agent.runtimes.adapter import ADAPTER_VERSION, AdapterSettings, RefundAdapter
from refund_agent.selection import fingerprint
from refund_agent.tools import Tools
from refund_agent.verifier import IndependentVerifier, VerifiedTools

PACKAGE = Path(__file__).resolve().parents[1]


def source_fingerprint(*paths):
    return fingerprint({p: _source_hash(PACKAGE / p) for p in paths})


def _source_hash(path):
    # A cached digest must still authorize and record this caller's read.
    sys.audit("open", str(path), "r", os.O_RDONLY)
    return _file_hash(path, extensions.resources.file_identity(path), Path.read_bytes)


@lru_cache(maxsize=2048)
def _file_hash(path, identity, reader):
    return sha256(reader(path)).hexdigest()


# region chapter29:candidate-record
# Chapter 29 owns these fields; Parts VI's jobs add their versioned components here.
@dataclass(frozen=True)
class Candidate:
    name: str
    agent: dict
    model: dict
    runtime: dict
    adapter: dict
    tools_composition: dict
    offer_policy: dict
    suite: dict

    def record(self):
        return json.loads(json.dumps(asdict(self)))

    @property
    def fingerprint(self):
        software = {
            key: value
            for key, value in self.record().items()
            if key not in ("suite", "offer_policy")
        }
        software["offer_policy"] = {key: self.offer_policy[key] for key in ("name", "policy")}
        return fingerprint(software)


def build_candidate(
    suite,
    name="as-built",
    runtime="own",
    verifier="off",
    *,
    offers=None,
    settings=None,
    components=None,
):
    from refund_agent.shadow.offers import load_offers

    _validate_choice(name, runtime, verifier)
    offers, settings = offers or load_offers(), settings or AdapterSettings()
    selected_components = extensions.selections(
        suite.components if components is None else components
    )
    if selected_components != suite.components:
        raise ValueError("candidate components differ from the fixed suite")
    composition = _tools_record(name, verifier)
    if selected_components:
        composition["components"] = extensions.records(selected_components)
    return Candidate(
        name,
        _agent_record(),
        _model_record(name),
        _runtime_record(runtime),
        _adapter_record(settings),
        composition,
        offers.record(),
        {"fingerprint": suite.fingerprint, "version": suite.version},
    )


def _agent_record():
    return {
        "identity": "Agent",
        "fingerprint": source_fingerprint(
            "agent/__init__.py", "tools.py", "contracts.py", "rules.py"
        ),
    }


def _adapter_record(settings):
    return {
        "version": ADAPTER_VERSION,
        "settings": asdict(settings),
        "fingerprint": source_fingerprint("runtimes/adapter.py"),
    }


# endregion chapter29:candidate-record


class DenyAll(ScriptedModel):
    def reply(self, state):
        if state.phase == c.Phase.ASSESS:
            return decision(state, c.Outcome.DENY, "constructed_deny_all")
        return super().reply(state)


class DenyAllTools(Tools):
    def assess(self, state):
        chosen = super().assess(state)
        if chosen.tier != c.Tier.NAMED_APPROVER:
            return chosen.model_copy(
                update={"outcome": c.Outcome.DENY, "reason": "constructed_deny_all"}
            )
        return chosen


class ResumeFromSavedEvidence(Tools):
    """Planted payment fault: after approval, gate the saved copy instead of rereading."""

    def _save_proposal(self, state):
        result = super()._save_proposal(state)
        if not hasattr(self, "_saved_data"):
            self._saved_data = {}
        self._saved_data[state.proposal_id] = self.data
        return result

    def _execute(self, call, state):
        args = call.arguments
        approval = self._approvals.get(args.approval_id or "")
        if approval is None or state.receipt is not None or not args.idempotency_key:
            return super()._execute(call, state)
        saved = self._saved_data[state.proposal_id]
        proposal = self._proposals[state.proposal_id]
        state.execution_at = execution_time(state, approval)
        # Deliberate mutation: both the reread and the gate's live-source comparison are lost.
        problem = gate(saved, state, args.amount_cents, proposal, approval, state.refunded_cents)
        if problem:
            return c.ToolResult(ok=False, reason=problem)
        return self._commit(state, args.idempotency_key, args.amount_cents, approval)


class StaleApprovalTools(Tools):
    """Exercise candidate: only the changed-proposal approval check is bypassed."""

    def completed_state(self, request):
        stored = super().completed_state(request)
        if (
            stored
            and stored.failure_detail == "request_content_conflict"
            and any(a.request_id == request.request_id for a in self._approvals.values())
        ):
            return None  # Planted fault: forget the changed-request continuation binding.
        return stored

    def _authorize(self, state, amount, approval):
        problem = super()._authorize(state, amount, approval)
        if approval and problem and problem.startswith("proposal_changed:"):
            return None
        return problem


class VoidVerifier(IndependentVerifier):
    def verify(self, proposal, reads, *, deadline):
        verdict = super().verify(proposal, reads, deadline=deadline)
        return verdict.model_copy(update={"proposal_id": "different-proposal"})


# region chapter29:build-tools
@extensions.resource_guard(1)
def build_tools(data, candidate, mode, **context):
    if mode not in ("execute", "shadow"):
        raise ValueError("mode must be execute or shadow")
    base, kwargs = _base_tools(candidate)
    base, kwargs = extensions.compose(
        candidate, base, kwargs, mode=mode, context=context, outermost=False
    )
    if mode == "shadow":
        from refund_agent.shadow.tools import ShadowTools

        base = type("CandidateShadowTools", (ShadowTools, base), {})
    base, kwargs = extensions.compose(
        candidate, base, kwargs, mode=mode, context=context, outermost=True
    )
    if mode == "shadow":
        base = _core_shadow_tools(base, ShadowTools)
    tools = base(data, **kwargs)
    tools._release_candidate = candidate
    return tools


def _base_tools(candidate):
    base = {
        "as-built": Tools,
        "deny-all": DenyAllTools,
        "resume-saved": ResumeFromSavedEvidence,
        "gate-removed": GateRemovedTools,
        "stale-approval": StaleApprovalTools,
    }[candidate.name]
    selected = candidate.tools_composition["verifier"]
    kwargs = {}
    if selected != "off":
        # VerifiedTools reaches the supplied base's gate through super().
        base = (
            type("CandidateVerifiedTools", (VerifiedTools, base), {})
            if base != Tools
            else VerifiedTools
        )
        kwargs["verifier"] = (
            VoidVerifier()
            if selected == "void"
            else IndependentVerifier(latency_seconds=2 if selected == "timeout" else 0)
        )
    return base, kwargs


def _core_shadow_tools(base, shadow):
    # Core bindings cannot be replaced by an outermost component.
    bindings = {
        name: shadow.__dict__[name]
        for name in (
            "ledger",
            "reviews",
            "payments",
            "shadow_payments",
            "shadow_reviews",
            "_commit",
            "_review",
            "_store_review",
            "_checked_append",
        )
    }
    return type("CoreShadowTools", (base,), {**bindings, "_write": Tools._write})


# endregion chapter29:build-tools


# region chapter29:decision-validation
def validate_final_decision(trace, tools, request):
    """Recheck decision 1 against core approval records after every transform."""
    chosen = c.Decision.model_validate(trace.decision.model_dump())
    if chosen.request_id != request.request_id:
        raise ValueError("final decision names another request")
    if tier_for(request.amount_cents) != c.Tier.NAMED_APPROVER:
        return
    if chosen.amount_cents != request.amount_cents or chosen.tier != c.Tier.NAMED_APPROVER:
        raise ValueError("final decision changes the named-approver request")
    if chosen.outcome == c.Outcome.REFER and not chosen.executed:
        return
    state = trace.steps[-1].state.model_copy(deep=True)
    state.request = request
    proposal = tools.proposals.get(state.proposal_id or "")
    if proposal is None or not any(
        a.role == "named_approver"
        and state.policy is not None
        and a.reviewer == state.policy.named_approver
        and proposal.request == request
        and approval_problem(tools.data, state, proposal, a) is None
        for a in tools._approvals.values()
    ):
        raise ValueError("a named approver must decide every request over $500, including denials")


# endregion chapter29:decision-validation


@extensions.resource_guard(0)
def run_request(candidate, tools, request, *, model=None, at=None, max_steps=None, runtime=None):
    runtime = runtime or candidate.runtime["identity"]
    if runtime != candidate.runtime["identity"]:
        raise ValueError("runtime differs from the fingerprinted candidate")
    model = model or (DenyAll() if candidate.name == "deny-all" else ScriptedModel())
    settings = AdapterSettings(**candidate.adapter["settings"])
    if max_steps is not None:
        settings = replace(settings, max_steps=min(settings.max_steps, max_steps))
    model = extensions.apply(
        candidate, "wrap_model", value=model, tools=tools, runtime_settings=settings
    )
    extensions.notify(
        candidate,
        "before_run",
        tools=tools,
        request=request,
        at=at,
        runtime_settings=settings,
    )
    if runtime == "own":
        runner = Agent(tools, model, max_steps=settings.max_steps)
    else:
        runner = RefundAdapter(tools, model, settings)
    return extensions.apply(
        candidate,
        "after_run",
        value=runner.run(request, at=at),
        tools=tools,
        request=request,
        at=at,
        runtime_settings=settings,
    )


@dataclass
class CandidateRun:
    tools: Tools
    traces: list


def run(candidate, requests, runtime, **context):
    """Requests is a Dataset; its labels are stripped before mounting the tools."""
    tools = build_tools(
        requests.model_copy(update={"correct": []}), candidate, "execute", **context
    )
    return CandidateRun(
        tools, [run_request(candidate, tools, r, runtime=runtime) for r in requests.requests]
    )


def _model_record(name):
    return {
        "identity": "DenyAll" if name == "deny-all" else "ScriptedModel",
        "version": "1",
        "settings": {"seed": 7, "decoding": "deterministic"},
        "fingerprint": source_fingerprint("model/__init__.py", "release/candidate.py"),
    }


def _tools_record(name, verifier):
    from refund_agent.release.suite import IMPLEMENTATION_FILES

    return {
        "base": name,
        "verifier": verifier,
        "version": "release-tools-1",
        "verifier_version": IndependentVerifier.version if verifier != "off" else None,
        "fingerprint": source_fingerprint(*IMPLEMENTATION_FILES),
    }


def _runtime_record(runtime):
    source = "agent/__init__.py" if runtime == "own" else "runtimes/runner.py"
    return {"identity": runtime, "version": "1", "fingerprint": source_fingerprint(source)}


def _validate_choice(name, runtime, verifier):
    if (
        name not in ("as-built", "resume-saved", "deny-all", "gate-removed", "stale-approval")
        or runtime not in ("own", "second")
        or verifier not in ("off", "independent", "timeout", "void")
    ):
        raise ValueError("unknown candidate, runtime or verifier")
