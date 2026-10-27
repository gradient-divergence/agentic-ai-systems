"""A scripted ModelClient: context terms first, then terms carried by an adapter."""

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone

from refund_agent import contracts as c
from refund_agent.adaptation.prompt import draft_fields, parse_draft
from refund_agent.rules import decision, draft, evidence_problem


@dataclass(frozen=True)
class PolicyTerms:
    version: str
    return_days: int
    cap_cents: int

    @classmethod
    def from_policy(cls, policy):
        return cls(policy.version, policy.return_days, policy.cap_cents)

    def policy(self):
        return c.Policy(
            version=self.version,
            effective_at=datetime.fromisoformat(self.version).replace(tzinfo=timezone.utc),
            return_days=self.return_days,
            cap_cents=self.cap_cents,
            text=f"refund_cap_v4@{self.version}: physical goods. #eligibility: damaged or changed_mind "
            f"within {self.return_days} days of delivery; not_delivered if no delivery. "
            f"#cap: at most {self.cap_cents} cents and the unrefunded payment balance.",
        )


def context_terms(context):
    # Read only the policy section, never the worked drafts or quoted request.
    block = re.search(r"\[Policy\]\n(.*?)\n\n\[Order\]\n", context, re.S)
    if block is None:
        return None
    text = block[1]
    versions = set(re.findall(r"refund_cap_v4@(\d{4}-\d{2}-\d{2})", text))
    days = re.findall(r"within (\d+) days of delivery", text)
    caps = re.findall(r"at most (\d+) cents", text)
    if len(versions) != 1 or len(set(days)) != 1 or len(set(caps)) != 1:
        return None
    return PolicyTerms(versions.pop(), int(days[0]), int(caps[0]))


class ScriptedDrafter:
    def __init__(self, adapter_terms=None, *, data=None, context=""):
        self.adapter_terms, self.data, self.context = adapter_terms, data, context

    def draft_text(self, state: c.State) -> str:
        if self.data is None:
            raise ValueError("data required to check the observed evidence")
        state = state.model_copy(deep=True)
        problem = evidence_problem(self.data, state)
        if state.decision is not None:  # A read tool already refused this evidence.
            chosen = state.decision
        elif problem:
            chosen = decision(state, c.Outcome.REFER, problem)
        else:
            terms = context_terms(self.context) or self.adapter_terms
            if terms is None:
                state.policy = None
                chosen = decision(state, c.Outcome.REFER, "missing_policy")
            else:
                state.policy = terms.policy()
                chosen = draft(state)
                if chosen.outcome == c.Outcome.APPROVE and chosen.tier != c.Tier.AGENT:
                    chosen = decision(state, c.Outcome.REFER, "approval_required")
        return json.dumps(draft_fields(chosen), separators=(",", ":"))

    def reply(self, state: c.State) -> c.Decision:
        return parse_draft(self.draft_text(state), state.request)
