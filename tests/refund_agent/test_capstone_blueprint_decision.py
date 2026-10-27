from datetime import timedelta

import pytest
from capstone_blueprint_fixtures import AT, OWNERS, packet_fixture

from refund_agent.launch.capstone_blueprint_decision import (
    OpenItem,
    StandingControl,
    admit,
    hold_record,
    on_promotion,
    reverse,
    sign,
)
from refund_agent.launch.capstone_blueprint_gate import check
from refund_agent.launch.capstone_blueprint_scan import (
    LaunchLogs,
    ReversalEvent,
)


class ConstructedAuthority:
    def __init__(self):
        self.withdrawals = []
        self.ledger = ("already-paid",)

    def events(self):
        return ()

    def withdraw(self, condition, **fields):
        self.withdrawals.append((condition, fields))
        return fields

    def permitted(self, tier):
        return not any(
            fields["scope"] in ("all", "tier" + str(tier))
            for _, fields in self.withdrawals
        )


def test_admission_signature_and_hold_guards(tmp_path):
    req, packet, registry = packet_fixture(tmp_path)
    verdict = check(packet, req, registry, LaunchLogs(), OWNERS, 64)
    admission = admit(verdict, "morgan", 65, registry)
    assert admission.candidate_bundle_fingerprint == "B6"
    controls = (
        StandingControl(name="audit", owner="morgan", period="monthly"),
    )
    items = (
        OpenItem(
            item="generated labels", owner="morgan", accepted_by="morgan"
        ),
    )
    decision = sign(
        verdict,
        admission,
        "morgan",
        65,
        requirement=req,
        controls=controls,
        open_items=items,
    )
    assert decision.verdict == "launch" and len(decision.reversals) == 5
    for owner, day, record in (
        ("alex", 65, admission),
        ("morgan", 63, admission),
        ("morgan", 65, None),
        (
            "morgan",
            65,
            admission.model_copy(
                update={"candidate_bundle_fingerprint": "B5"}
            ),
        ),
    ):
        with pytest.raises(ValueError):
            sign(
                verdict,
                record,
                owner,
                day,
                requirement=req,
                controls=controls,
                open_items=items,
            )
    held = check(packet, req, None, LaunchLogs(), OWNERS, 64)
    previous = tuple(registry.entries)
    with pytest.raises(ValueError):
        admit(held, "morgan", 65, registry)
    assert tuple(registry.entries) == previous
    state = hold_record(held, req, registry, None, 64)
    assert state.signed_by is None and not state.authority_available
    assert "replacement model" in state.clears["1"]


def test_admission_rejects_a_different_bundle(tmp_path, monkeypatch):
    req, packet, registry = packet_fixture(tmp_path)
    verdict = check(packet, req, registry, LaunchLogs(), OWNERS, 64)
    admission = admit(verdict, "morgan", 65, registry)
    wrong_bundle = admission.model_copy(
        update={"candidate_bundle_fingerprint": "B5"}
    )
    monkeypatch.setattr(registry, "admit_launch", lambda **fields: wrong_bundle)
    with pytest.raises(ValueError, match="promotion differs"):
        admit(verdict, "morgan", 65, registry)


@pytest.mark.parametrize(
    "condition,scope",
    [
        ("forbidden_payment", "tier1"),
        ("budget_spent", "tier1"),
        ("tier2_unbound_approval", "tier2"),
        ("tier3_without_named_approver", "all"),
        ("launch_gate_held", "launch"),
    ],
)
def test_reversals_change_authority_and_never_ledger(
    tmp_path, condition, scope
):
    req, _, registry = packet_fixture(tmp_path)
    authority = ConstructedAuthority()
    event = ReversalEvent(
        record_id="reversal",
        condition_id=condition,
        day=66,
        at=AT + timedelta(days=66),
        reason="constructed violation",
    )
    reverse(
        event,
        requirement=req,
        authority=authority,
        registry=registry,
        by="morgan",
    )
    assert authority.ledger == ("already-paid",)
    if scope != "launch":
        assert authority.withdrawals[-1][1]["scope"] == scope
        assert not authority.permitted(1 if scope in ("tier1", "all") else 2)
    else:
        assert (
            not authority.withdrawals and registry.entries[-1]["to_stage"] == 6
        )


def test_promotion_reruns_gate_and_withdraws_launch_without_signing(tmp_path):
    req, packet, registry = packet_fixture(
        tmp_path, {"injection_suite": {"bundle": "B5"}}
    )
    first = check(packet, req, registry, LaunchLogs(), OWNERS, 64)
    admission = admit(first, "morgan", 64, registry)
    launch = sign(
        first,
        admission,
        "morgan",
        64,
        requirement=req,
        controls=(
            StandingControl(name="audit", owner="morgan", period="monthly"),
        ),
        open_items=(),
    )
    registry.required.add("injection_suite")
    state = {
        "packet": packet,
        "requirement": req,
        "registry": registry,
        "logs": LaunchLogs(),
        "owners": OWNERS,
        "day": 64,
        "decision": launch,
        "authority": ConstructedAuthority(),
        "at": AT,
        "owner": "morgan",
    }
    promotion = {
        "candidate_bundle_fingerprint": "B6",
        "candidate_fingerprint": "software-B6",
    }
    verdict = on_promotion(state, promotion)
    assert verdict.verdict == "hold" and registry.entries[-1]["to_stage"] == 6
    held = hold_record(verdict, req, registry, None, 64)
    state["decision"] = held
    registry.entries.clear()
    assert (
        on_promotion(state, promotion).verdict == "hold"
    )
    assert len(registry.entries) == 1 and "promotion" in registry.entries[0]


def test_promotion_records_seven_row_gate_with_distinct_identities(tmp_path):
    req, packet, registry = packet_fixture(
        tmp_path, {"comparison": {"verdict": "not better"}}
    )
    first = check(packet, req, registry, LaunchLogs(), OWNERS, 64)
    state = {
        "packet": packet,
        "requirement": req,
        "registry": registry,
        "logs": LaunchLogs(),
        "owners": OWNERS,
        "day": 64,
        "decision": hold_record(first, req, registry, None, 64),
    }
    promotion = {
        "candidate_bundle_fingerprint": "B6",
        "candidate_fingerprint": "software-B6",
    }
    verdict = on_promotion(state, promotion)
    assert verdict.verdict == "hold" and verdict.launched_bundle == "B6"
    assert [row.row for row in verdict.rows] == [
        "1", "2", "3", "4", "5", "6", "7"
    ]
    assert [(f.row, f.kind, f.check) for f in verdict.failures] == [
        ("1", "comparison", "complete")
    ]
    assert registry.entries == [{"promotion": promotion, "verdict": verdict}]

    with pytest.raises(ValueError, match="promotion needs a requirement"):
        on_promotion(
            state,
            {
                "candidate_bundle_fingerprint": "B5",
                "candidate_fingerprint": "B6",
            },
        )
    assert registry.entries == [{"promotion": promotion, "verdict": verdict}]
