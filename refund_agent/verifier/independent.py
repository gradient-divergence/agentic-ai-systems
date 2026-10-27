"""A phrase rule standing in for a model reader, behind the unchanged payment gate."""

# region book:fnd.multi-agent-systems.verifier-01 output
from datetime import timedelta
from math import isfinite

from refund_agent import contracts as c
from refund_agent.contracts import _validated
from refund_agent.rules import tier_for
from refund_agent.tools import Tools
from refund_agent.verifier.record import (
    ReadOnlyTools,
    Verdict,
    VerdictLog,
    verdict_problem,
)


def independent_check(proposal, reads):
    state = reads.evidence(proposal)
    delivered = any(
        event.status == "delivered" for event in state.shipment_events
    )
    if "never arrived" in proposal.request.text.casefold() and delivered:
        return "object", "words_contradict_record"
    return "agree", None


class VerifiedTools(Tools):
    version = "verified-tools-v1"

    def __init__(self, data, verifier, deadline_seconds=1):
        if not isfinite(deadline_seconds) or deadline_seconds <= 0:
            raise ValueError("positive finite verifier deadline required")
        super().__init__(data)
        self.verifier, self.deadline_seconds = verifier, deadline_seconds
        self.verdict_log, self._verifications = VerdictLog(), {}

    @property
    def verification_calls(self):
        return len(self._verifications)

    def _authorize(self, state, amount, approval):
        problem = super()._authorize(state, amount, approval)
        if problem or tier_for(amount) != c.Tier.AGENT:
            return problem
        proposal = self.proposals[state.proposal_id]
        deadline = state.execution_at + timedelta(
            seconds=self.deadline_seconds
        )
        if proposal.proposal_id not in self._verifications:
            reads = ReadOnlyTools(self._data, state.execution_at)
            try:
                verdict = _validated(
                    Verdict,
                    self.verifier.verify(proposal, reads, deadline=deadline),
                )
                self.verdict_log.append(verdict)
            except TimeoutError:
                verdict = "verifier_timeout"
            except (TypeError, ValueError):
                verdict = "verifier_verdict_void"
            except Exception:
                verdict = "verifier_error"
            self._verifications[proposal.proposal_id] = verdict
        verdict = self._verifications[proposal.proposal_id]
        if isinstance(verdict, str):
            problem = verdict
        else:
            problem = verdict_problem(verdict, proposal, state, deadline)
            if problem is None and verdict.answer == "object":
                problem = f"verifier_objection:{verdict.objection_reason}"
        if problem:
            self._review(c.ToolCall(name="approval.request"), state)
        return problem


if __name__ == "__main__":
    from demos.multi_agent_systems_demo import print_verdicts

    print_verdicts(VerifiedTools)
# endregion book:fnd.multi-agent-systems.verifier-01


class IndependentVerifier:
    """Offline latency is simulated; live adapters must enforce the deadline and raise TimeoutError."""

    name = "independent"
    version = "independent-phrase-v1"

    def __init__(self, latency_seconds=0):
        if not isfinite(latency_seconds) or latency_seconds < 0:
            raise ValueError("finite nonnegative latency required")
        self.latency_seconds = latency_seconds

    def verify(self, proposal, reads, *, deadline):
        from refund_agent.verifier.record import make_verdict

        completed_at = reads.at + timedelta(seconds=self.latency_seconds)
        if completed_at > deadline:
            raise TimeoutError("verifier deadline")
        answer, reason = independent_check(proposal, reads)
        return make_verdict(
            proposal,
            reads.evidence_state,
            reads.reads,
            self.name,
            answer,
            reason,
            completed_at,
            reads.words + len(proposal.request.text.split()),
        )
