"""The three arms differ only in the policy that reaches the same drafter."""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Callable

from refund_agent import contracts as c
from refund_agent.adaptation.drafter import PolicyTerms, ScriptedDrafter
from refund_agent.adaptation.prompt import StructuredPrompt
from refund_agent.data.generate import _policies
from refund_agent.model import ModelClient
from refund_agent.retrieval.index import PolicyIndex


def fit_adapter_terms(drafts, policies=None):
    """Look up the complete terms of the latest version cited by training drafts.

    This looks up that version's terms; it trains no language model and never
    chooses a version newer than any training draft. The low-rank toy is separate.
    """
    if not drafts:
        raise ValueError("training drafts required")
    versions = {d.policy_citation.split("@")[1].split("#")[0] for d in drafts if d.policy_citation}
    if not versions:
        raise ValueError("training drafts must cite a policy version")
    by_version = {p.version: p for p in (_policies() if policies is None else policies)}
    if versions - by_version.keys():
        raise ValueError("unknown training policy version")
    return PolicyTerms.from_policy(by_version[max(versions)])


def question_texts(state, wording):
    """Chapter 11's two wordings, from observed request facts, without labels."""
    r = state.request
    delivered = [e.at for e in state.shipment_events if e.status == "delivered" and e.at <= r.at]
    days = (r.at - max(delivered)).total_seconds() / 86400 if delivered else None
    age = f"{days:g}" if days is not None else "unknown"
    if wording == "policy":
        eligibility = (
            "not_delivered if no delivery"
            if r.reason == "not_delivered"
            else f"{r.reason} {age} days after delivery"
        )
        cap = f"at most {r.amount_cents} cents and the unrefunded payment balance"
    elif wording == "customer":
        reason = {
            "damaged": "it arrived broken",
            "changed_mind": "I no longer want it",
            "not_delivered": "my parcel has not arrived",
        }.get(r.reason, r.reason)
        eligibility = f"{reason}; {age} days later can I get my money back"
        cap = f"can you pay back {r.amount_cents} from what I paid"
    else:
        raise ValueError("wording must be policy or customer")
    return eligibility, cap


@dataclass(frozen=True)
class Arm:
    name: str
    prompt: StructuredPrompt = field(default_factory=StructuredPrompt)
    adapter_terms: PolicyTerms | None = None
    index: PolicyIndex | Callable[[c.State | datetime], PolicyIndex] | None = None
    k: int = 1
    wording: str = "policy"
    training_ids: tuple[str, ...] = ()
    client_factory: Callable[..., ModelClient] = ScriptedDrafter

    def policy_text(self, state):
        if self.name == "prompt":
            if state.policy is None:
                return ""
            return f"effective {state.policy.effective_at.isoformat()}; read age {state.freshness['policy']:g} seconds\n{state.policy.text}"
        if self.name == "adapter":
            return ""
        if self.index is None:
            raise ValueError("retrieval arm requires a policy index")
        index = self.index(state) if callable(self.index) else self.index
        clauses = {}
        for question in question_texts(state, self.wording):
            for hit in index.retrieve(question, state.request.at, self.k).hits:
                clauses[hit.clause.citation] = hit.clause
        return "\n".join(
            f"{clause.citation}; effective_from {clause.effective_from.isoformat()}; "
            f"effective_to {clause.effective_to.isoformat() if clause.effective_to else 'open'}; "
            f"observed_at {clause.observed_at.isoformat()}\n#{clause.section}: {clause.text}"
            for _, clause in sorted(clauses.items())
        )

    def render(self, state):
        return self.prompt.render(state, self.policy_text(state))

    def client(self, data, context) -> ModelClient:
        return self.client_factory(self.adapter_terms, data=data, context=context)

    def question_index(self, question):
        if self.index is None:
            raise ValueError("retrieval checks require a policy index")
        if callable(self.index):
            # The factory also accepts a question's time for the frozen retrieval checks.
            return self.index(question.as_of)
        return self.index


def prompt_arm(prompt=None, *, index=None):
    return Arm("prompt", prompt or StructuredPrompt(), index=index)


def retrieval_arm(index, k=1, wording="policy", prompt=None):
    if type(k) is not int or k < 1 or wording not in ("policy", "customer"):
        raise ValueError("positive k and a declared wording required")
    return Arm("retrieval", prompt or StructuredPrompt(), index=index, k=k, wording=wording)


def adapter_arm(drafts, *, policies=None, prompt=None, index=None):
    return Arm(
        "adapter",
        prompt or StructuredPrompt(),
        fit_adapter_terms(drafts, policies),
        index=index,
        training_ids=tuple(sorted(d.request_id for d in drafts)),
    )


def dated_index(policies, encoder):
    """Retrospective corpus, a simulated fresh read per request (chapter 11)."""

    def at_clock(state_or_time):
        if hasattr(state_or_time, "request"):
            at = state_or_time.request.at
            observed = at - timedelta(seconds=state_or_time.freshness.get("policy", 3600))
        else:
            at, observed = state_or_time, state_or_time - timedelta(hours=1)
        return PolicyIndex.build(policies, observed, encoder)

    return at_clock
