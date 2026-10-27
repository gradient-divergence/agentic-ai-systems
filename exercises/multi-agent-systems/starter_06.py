"""Exercise 6: add a second-source check in this subclass; leave the core verifier alone."""

import json
from pathlib import Path

from refund_agent.selection import fingerprint
from refund_agent.verifier.experiment import run_experiment
from refund_agent.verifier.independent import IndependentVerifier
from refund_agent.verifier.record import Read


class ProofCheckingVerifier(IndependentVerifier):
    name = "proof-checking"
    version = "proof-checking-starter-v1"

    def __init__(self, proof_path=None):
        super().__init__()
        path = (
            Path(proof_path) if proof_path else Path(__file__).with_name("proof_of_delivery.json")
        )
        payload = json.loads(path.read_text())
        self.proof = {row["request_id"]: row["delivered_at"] for row in payload["rows"]}
        self.proof_sha256 = fingerprint(payload)
        self.source_log = (
            {
                "entry": 1,
                "source": "proof-of-delivery-v1",
                "reader": self.name,
                "purpose": "load exercise 6's supplied second source",
                "rows": len(payload["rows"]),
                "sha256": self.proof_sha256,
            },
        )
        self.accesses = ()

    def verify(self, proposal, reads, *, deadline):
        verdict = super().verify(proposal, reads, deadline=deadline)
        access = Read(
            tool="proof.delivery",
            observed_at=reads.at,
            read_at=reads.at,
            sha256=self.proof_sha256,
            words=2,
        )
        self.accesses += (access,)
        verdict = verdict.model_copy(
            update={
                "reads": (*verdict.reads, access),
                "words_read": verdict.words_read + access.words,
            }
        )
        # TODO: compare this request's proof date with the delivered carrier event.
        return verdict


def run():
    report = run_experiment(7, independent=ProofCheckingVerifier(), include_public=False)
    arm = report["arms"]["independent"]
    print(
        f"n+ {arm['paired'].n_plus}; n- {arm['paired'].n_minus}; {arm['decision']}; "
        f"additional payments {arm['additional_payments']}; saving ${arm['saving']['mean_dollars']:.4f}"
    )
    return report


if __name__ == "__main__":
    run()
