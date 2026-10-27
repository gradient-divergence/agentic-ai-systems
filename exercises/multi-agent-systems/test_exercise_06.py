"""Run with EXERCISE_VERIFIER_FILE set to your solution; the untouched starter fails."""

import json
import os
import runpy
from datetime import timedelta
from pathlib import Path

import pytest

from refund_agent import contracts as c
from refund_agent.adaptation.experiment import agreement, decide, paired
from refund_agent.verifier.experiment import ALPHA, authority_count, run_arm, run_case
from refund_agent.verifier.fixture import build_fixture
from refund_agent.verifier.independent import IndependentVerifier


def check_acceptance(verifier_class):
    fixture = build_fixture(7)
    verifier = verifier_class()
    baseline = run_arm(fixture.data)
    original = run_arm(fixture.data, IndependentVerifier())
    rows = run_arm(fixture.data, verifier)
    by_id = {row.trace.decision.request_id: row for row in rows}
    assert authority_count(baseline, rows) == 0, "a verifier added a payment"
    for before in original:
        if before.verdicts[0].answer == "object":
            row = by_id[before.trace.decision.request_id]
            assert row.verdicts[0].answer == "object", "an original verifier objection was removed"
            assert row.verdicts[0].objection_reason == before.verdicts[0].objection_reason
            assert row.trace.decision == before.trace.decision
    for rid in fixture.groups["false_date"]:
        row = by_id[rid]
        assert row.verdicts[0].answer == "object", "the false carrier date was not caught"
        assert row.trace.receipt is None
    for rid in fixture.groups["plain"]:
        assert by_id[rid].verdicts[0].answer == "agree", "a plain request gained an objection"
    # Published constructed group labels suffice; acceptance never consults sealed truth.
    outcomes = {
        rid: c.Outcome.REFER
        if group == "contradiction"
        else (c.Outcome.DENY if group == "false_date" else c.Outcome.APPROVE)
        for group, ids in fixture.groups.items()
        for rid in ids
    }
    labels = [
        c.Decision(
            request_id=request.request_id,
            outcome=outcomes[request.request_id],
            amount_cents=2499,
            tier=c.Tier.AGENT,
            reason="constructed_fixture",
            policy_citation=f"refund_cap_v4@{fixture.data.policy_at(request.at).version}#eligibility",
        )
        for request in fixture.data.requests
    ]
    single = agreement([row.trace.decision for row in baseline], labels)
    matches = agreement([row.trace.decision for row in rows], labels)
    card = paired(single, matches, alpha=ALPHA)
    assert (sum(single), sum(matches), card.n_plus, card.n_minus) == (360, 376, 24, 8)
    assert card == paired(
        single, agreement([row.trace.decision for row in original], labels), alpha=ALPHA
    ), "the original paired result changed"
    assert decide(True, card, alpha=ALPHA) == "accepted"
    assert len(verifier.accesses) == 400
    assert len(verifier.source_log) == 1
    assert fixture.truth.log == (), "acceptance needs no sealed labels"


def test_exercise_06():
    path = os.environ.get("EXERCISE_VERIFIER_FILE", str(Path(__file__).with_name("starter_06.py")))
    check_acceptance(runpy.run_path(path)["ProofCheckingVerifier"])


def test_acceptance_rejects_removing_the_original_verifiers_catches():
    reference = runpy.run_path(str(Path(__file__).with_name("instructor_reference_06.py")))[
        "ProofCheckingVerifier"
    ]

    class DropsOriginalCatches(reference):
        def verify(self, proposal, reads, *, deadline):
            verdict = super().verify(proposal, reads, deadline=deadline)
            if verdict.objection_reason == "words_contradict_record":
                return verdict.model_copy(update={"answer": "agree", "objection_reason": None})
            return verdict

    with pytest.raises(AssertionError, match="original verifier"):
        check_acceptance(DropsOriginalCatches)


def test_acceptance_rejects_a_purchase_only_shortcut():
    starter = runpy.run_path(str(Path(__file__).with_name("starter_06.py")))[
        "ProofCheckingVerifier"
    ]

    class PurchaseOnly(starter):
        def verify(self, proposal, reads, *, deadline):
            verdict = super().verify(proposal, reads, deadline=deadline)
            if verdict.answer == "object":
                return verdict
            old = proposal.request.at - reads.evidence_state.order.purchased_at > timedelta(days=60)
            return verdict.model_copy(
                update={
                    "answer": "object" if old else "agree",
                    "objection_reason": "purchase_age" if old else None,
                }
            )

    with pytest.raises(AssertionError, match="plain request"):
        check_acceptance(PurchaseOnly)


@pytest.mark.parametrize("sign", [-1, 1])
@pytest.mark.parametrize("seconds,answer", [(86399, "agree"), (86400, "agree"), (86401, "object")])
def test_proof_difference_must_be_more_than_one_day(tmp_path, sign, seconds, answer):
    fixture = build_fixture(7)
    rid = fixture.groups["plain"][0]
    request = next(request for request in fixture.data.requests if request.request_id == rid)
    shipment = next(
        row for row in fixture.data.shipments if row.order_id == request.order_reference
    )
    payload = json.loads(Path(__file__).with_name("proof_of_delivery.json").read_text())
    for row in payload["rows"]:
        if row["request_id"] == rid:
            row["delivered_at"] = (
                shipment.events[0].at + timedelta(seconds=sign * seconds)
            ).isoformat()
    proof_path = tmp_path / "boundary-proof.json"
    proof_path.write_text(json.dumps(payload))
    path = os.environ.get("EXERCISE_VERIFIER_FILE", str(Path(__file__).with_name("starter_06.py")))
    verifier_class = runpy.run_path(path)["ProofCheckingVerifier"]
    row = run_case(fixture.data, request, verifier_class(proof_path=proof_path))
    assert row.verdicts[0].answer == answer
    assert (row.trace.receipt is None) == (answer == "object")
