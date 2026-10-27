"""A second reader of the proposal's evidence, using the gate's eligibility rules."""

import json

from refund_agent import contracts as c
from refund_agent.rules import draft
from refund_agent.selection import fingerprint
from refund_agent.verifier.record import Read, make_verdict


class AgreeingVerifier:
    name = "agreeing"
    version = "agreeing-rules-v1"

    def verify(self, proposal, reads, *, deadline):
        state = c.State(request=proposal.request, **proposal.evidence.model_dump())
        state.execution_at = reads.at
        # Use the already read evidence, without independent tool calls or a truth read.
        answer = "agree" if draft(state).outcome == c.Outcome.APPROVE else "object"
        reason = None if answer == "agree" else "proposal_ineligible"
        payload = proposal.model_dump(mode="json")
        words = len(json.dumps(payload, sort_keys=True).split())
        source = Read(
            tool="proposal",
            observed_at=proposal.created_at,
            read_at=reads.at,
            sha256=fingerprint(payload),
            words=words,
        )
        return make_verdict(proposal, state, (source,), self.name, answer, reason, reads.at, words)
