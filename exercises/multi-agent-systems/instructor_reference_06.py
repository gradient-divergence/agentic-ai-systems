"""Reference solution to exercise 6; the chapter's verifier is unchanged."""

import runpy
from datetime import datetime, timedelta
from pathlib import Path

StarterVerifier = runpy.run_path(str(Path(__file__).with_name("starter_06.py")))[
    "ProofCheckingVerifier"
]


class ProofCheckingVerifier(StarterVerifier):
    version = "proof-checking-reference-v1"

    def verify(self, proposal, reads, *, deadline):
        verdict = super().verify(proposal, reads, deadline=deadline)
        proof_date = datetime.fromisoformat(self.proof[proposal.request_id])
        dates = [
            event.at
            for event in reads.evidence_state.shipment_events
            if event.status == "delivered"
        ]
        if dates and abs(proof_date - max(dates)) > timedelta(days=1):
            return verdict.model_copy(
                update={"answer": "object", "objection_reason": "proof_contradicts_record"}
            )
        return verdict
