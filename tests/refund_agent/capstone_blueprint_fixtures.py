"""Constructed records for chapter 39's reader tests, never operating evidence."""

from datetime import datetime, timedelta, timezone
from pathlib import Path

from refund_agent import contracts as c
from refund_agent.agent import Agent
from refund_agent.canary.cloud_native_deployment_rehearsal import fixture
from refund_agent.launch.capstone_blueprint_packet import (
    EvidenceClaims,
    EvidenceFile,
    assemble,
    write_record,
)
from refund_agent.launch.capstone_blueprint_requirement import (
    refund_requirement,
)
from refund_agent.launch.capstone_blueprint_scan import (
    EffectiveDecision,
    ReviewOutcome,
)
from refund_agent.selection import fingerprint
from refund_agent.tools import Tools

ROLES = {
    "decision_owner": ("morgan",),
    "release_owner": ("morgan",),
    "reviewer": ("alex",),
    "named_approver": ("morgan",),
}
OWNERS = {
    "roles": ROLES,
    "layers": {
        kind: "morgan"
        for kind in (
            "interface",
            "context",
            "reasoning_and_orchestration",
            "model",
            "data",
            "evaluation_and_operations",
            "control_plane",
        )
    },
}
AT = datetime(2026, 10, 1, tzinfo=timezone.utc)


class ConstructedRegistry:
    """The test's declared producer schema; no partner mechanism is reimplemented."""

    def __init__(self, requirement):
        self.requirement = requirement
        self.required = set()
        self.false = set()
        self.absent_mechanisms = set()
        self.entries = []

    def validate_record(self, kind, payload, claims):
        parsed = EvidenceClaims.model_validate(payload)
        if parsed != claims or parsed.kind != kind:
            raise ValueError("producer's claims do not match manifest")

    def covered(self, earlier, launched, kind):
        if earlier not in {"B5", "B6"} or launched != "B6":
            raise ValueError("unlinked bundle")
        return kind not in self.required

    def rollback_target(self, bundle):
        return {"B6": "B5"}[bundle]

    def mechanisms(self, bundle):
        assert bundle == "B6"
        return (
            frozenset(
                c.mechanism
                for row in self.requirement.rows
                for c in row.reversals
            )
            - self.absent_mechanisms
        )

    def found_false(self, record_id):
        return record_id in self.false

    def record_gate(self, **kwargs):
        self.entries.append(kwargs)

    def withdraw_launch(self, **kwargs):
        self.entries.append(kwargs)
        return kwargs

    def record_withdrawal(self, **kwargs):
        self.entries.append(kwargs)

    def admit_launch(self, *, bundle, gate, owner, day):
        from refund_agent.launch.capstone_blueprint_decision import Admission

        row = Admission(
            record_id="constructed-admission",
            candidate_bundle_fingerprint=bundle,
            gate_fingerprint=gate.fingerprint,
            signed_by=owner,
            signed_day=day,
        )
        self.entries.append(row)
        return row


def evidence(need, **changes):
    data = dict(
        record_id="constructed-" + need.kind,
        kind=need.kind,
        day=61,
        bundle=None if need.bundle_rule == "any" else "B6",
        signed_by="morgan",
        signer_role=need.signer_role,
        signed_day=1,
        run_started_day=2,
        verdict=need.required_verdict,
        period=need.period,
        rollback_target="B5" if need.kind == "rehearsal" else None,
    )
    data.update(changes)
    claims = EvidenceClaims(**data)
    return EvidenceFile(claims=claims, payload=claims.model_dump(mode="json"))


def packet_fixture(path, changes=None):
    from refund_agent.launch.capstone_blueprint_packet import assemble

    requirement = refund_requirement("B6")
    done = set()
    for row in requirement.rows:
        for need in row.records:
            if need.kind in done:
                continue
            done.add(need.kind)
            write_record(
                path, evidence(need, **(changes or {}).get(need.kind, {}))
            )
    registry = ConstructedRegistry(requirement)
    return requirement, assemble(path, registry, requirement), registry


def action(
    tier=2, *, outcome="approve", executed=False, actor="alex", reviewed=True
):
    amount = {1: 4999, 2: 5000, 3: 50001}[tier]
    data, request = fixture(
        "capstone-blueprint", "scan-" + str(tier), AT, amount=amount
    )
    tools = Tools(data)
    Agent(tools).run(request)
    proposal = next(iter(tools.proposals.values()))
    choice = c.Decision(
        request_id=request.request_id,
        amount_cents=amount,
        tier=c.Tier(tier),
        outcome=c.Outcome(outcome),
        reason="constructed",
        executed=executed,
        policy_citation="refund_cap_v4@"
        + data.policies[0].version
        + "#eligibility",
    )
    review = (
        ReviewOutcome(
            request_id=request.request_id,
            proposal_id=proposal.proposal_id,
            proposal_fingerprint=fingerprint(proposal.model_dump(mode="json")),
            actor=actor,
            actor_kind="person",
            outcome=outcome,
            at=AT,
        )
        if reviewed
        else None
    )
    return EffectiveDecision(
        record_id="constructed-action-" + str(tier),
        day=3,
        requested_day=3,
        request=request,
        decision=choice,
        at=AT,
        actor=actor,
        proposal=proposal,
        review=review,
    ), data.policies[0]


def payment_action(tier=2):
    row, policy = action(
        tier, executed=True, actor="morgan" if tier == 3 else "alex"
    )
    approval = c.Approval(
        approval_id="approval-" + str(tier),
        request_id=row.request.request_id,
        order_id=row.proposal.evidence.order.order_id,
        proposal_id=row.proposal.proposal_id,
        amount_cents=row.request.amount_cents,
        reason=row.request.reason,
        policy_version=policy.version,
        reviewer=row.review.actor,
        role="named_approver" if tier == 3 else "reviewer",
        approved_at=row.review.at,
    )
    receipt = c.Receipt(
        receipt_id="receipt-" + str(tier),
        request=row.request,
        order_id=approval.order_id,
        amount_cents=approval.amount_cents,
        idempotency_key="payment-" + str(tier),
        proposal_id=approval.proposal_id,
        approvals=(approval,),
        at=AT + timedelta(seconds=1),
        evidence=row.proposal.evidence,
        decision=row.decision,
    )
    return row.model_copy(
        update={"approval": approval, "receipt": receipt}
    ), policy


class RehearsalFixtureRegistry:
    """Given signatures and verdicts for the constructed earlier-bundle packet.

    This diagnostic schema checks the given claims, not predecessor evidence;
    it is used only by the demo fixture and never by the operating driver.
    """

    def __init__(self, requirement):
        self.requirement = requirement

    def validate_record(self, kind, payload, claims):
        given = EvidenceClaims.model_validate(payload)
        if given != claims or given.kind != kind:
            raise ValueError("constructed claims differ from manifest")

    def covered(self, earlier, launched, kind):
        raise ValueError(
            "constructed packet supplies no earlier covered record"
        )

    def rollback_target(self, bundle):
        if bundle != "B6":
            raise ValueError("unknown constructed bundle")
        return "B5"

    def mechanisms(self, bundle):
        if bundle != "B6":
            raise ValueError("unknown constructed bundle")
        return frozenset(
            c.mechanism for row in self.requirement.rows for c in row.reversals
        )

    def found_false(self, record_id):
        return False  # All fixture verdicts are given; only the rehearsal is stale.


def rehearsal_packet(run_dir: Path):
    """All rows are given passing; only the rehearsal names the earlier B5."""
    requirement = refund_requirement("B6")
    registry = RehearsalFixtureRegistry(requirement)
    written = set()
    for row in requirement.rows:
        for need in row.records:
            if need.kind in written:
                continue
            written.add(need.kind)
            claims = EvidenceClaims(
                record_id="constructed-" + need.kind,
                kind=need.kind,
                day=61,
                bundle="B5"
                if need.kind == "rehearsal"
                else (None if need.bundle_rule == "any" else "B6"),
                signed_by="morgan",
                signer_role=need.signer_role,
                signed_day=1,
                run_started_day=2,
                verdict=need.required_verdict,
                period=need.period,
                rollback_target="B5" if need.kind == "rehearsal" else None,
            )
            write_record(
                run_dir,
                EvidenceFile(
                    claims=claims,
                    payload=claims.model_dump(mode="json"),
                ),
            )
    return requirement, assemble(run_dir, registry, requirement), registry
