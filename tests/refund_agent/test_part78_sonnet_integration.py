"""Real chapter boundaries and regressions from the Part VII/VIII review."""

import subprocess
import sys
from dataclasses import replace
from datetime import timedelta
from pathlib import Path

import pytest

pytest_plugins = ["test_future_directions_renewal"]


@pytest.mark.parametrize("component_first", [True, False])
def test_risk_component_selection_in_a_cold_interpreter(component_first):
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
assert not any(name in sys.modules for name in ('numpy', 'scipy', 'sklearn', 'zoneinfo'))
"""
            + ("import refund_agent.future_directions_risk\n" if component_first else "")
            + """
from refund_agent.release.suite import build_suite
from refund_agent.release.candidate import build_candidate
from refund_agent.future_directions_risk import selector
from refund_agent.renewal.future_directions_candidate import selected_score
component = selected_score().component
suite = build_suite(trials=1, components=(selector(component, policy_version='2026-09-01'),))
candidate = build_candidate(suite)
assert candidate.tools_composition['components'][0]['name'] == 'future-directions.risk'
print(candidate.fingerprint)
""",
        ],
        text=True,
        capture_output=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("phase", [1, 2])
def test_timeline_refuses_an_audit_before_thirty_day_maturity(tmp_path, phase):
    from test_capstone_blueprint_timeline import constructed_driver

    from refund_agent.launch.capstone_blueprint_timeline import Demand, operate

    driver = constructed_driver()
    from refund_agent.launch.capstone_blueprint_timeline import StageResult

    audits = []

    def audit(stream, out, period):
        audits.append(period)
        return StageResult(
            end_day=period.end + (0 if len(audits) == phase else 30) + 0.0001, records=()
        )

    driver.audit_period = audit
    with pytest.raises(ValueError, match="clock"):
        operate(7, demand=Demand(), out=tmp_path / "run", driver=driver)
    assert "step" + str(phase) not in driver.calls


def test_late_scan_withdraws_real_authority_and_replays_idempotently(tmp_path):
    from capstone_blueprint_fixtures import ConstructedRegistry
    from test_org_adoption_rollout import plan_fixture

    from refund_agent.launch.capstone_blueprint_decision import reverse
    from refund_agent.launch.capstone_blueprint_requirement import refund_requirement
    from refund_agent.launch.capstone_blueprint_scan import ReversalEvent
    from refund_agent.rollout.org_adoption_authority import StageAuthority

    plan, evidence = plan_fixture()
    authority = StageAuthority(plan, tmp_path / "authority.jsonl")
    at = plan.steps[0].window.start - timedelta(minutes=1)
    authority.sign_step(plan.steps[0].step_id, by="morgan", at=at, evidence=evidence)
    event = ReversalEvent(
        record_id="late-scan",
        condition_id="forbidden_payment",
        day=1,
        at=at - timedelta(hours=1),
        reason="late-discovered violation",
    )
    req = refund_requirement("B6")
    registry = ConstructedRegistry(req)
    first = reverse(event, requirement=req, authority=authority, registry=registry, by="morgan")
    assert first.at == at and first.scope == "tier1"
    authority.withdraw(
        "owner",
        by="morgan",
        at=at + timedelta(hours=1),
        record_id="later-withdrawal",
        scope="tier2",
    )
    assert (
        reverse(event, requirement=req, authority=authority, registry=registry, by="morgan")
        == first
    )
    assert len(authority.events()) == 3


def test_layer_owners_match_the_book_and_empty_owner_holds(tmp_path):
    from capstone_blueprint_fixtures import OWNERS, packet_fixture

    from refund_agent.launch.capstone_blueprint_gate import check
    from refund_agent.launch.capstone_blueprint_scan import LaunchLogs

    root = Path(__file__).resolve().parents[3]
    table = (
        (root / "mainmatter/modern-ai-stack.qmd")
        .read_text()
        .split("| Layer or plane |", 1)[1]
        .split("{#tbl-modern-ai-stack-refund-layers}", 1)[0]
    )
    layers = {
        line.split("|")[1].strip().lower().replace(" ", "_"): "morgan"
        for line in table.splitlines()
        if line.startswith("| ")
    }
    req, packet, registry = packet_fixture(tmp_path)
    owners = {**OWNERS, "layers": layers}
    assert check(packet, req, registry, LaunchLogs(), owners, 64).verdict == "launch"
    for layer in layers:
        broken = {**owners, "layers": {**layers, layer: ""}}
        result = check(packet, req, registry, LaunchLogs(), broken, 64)
        assert result.verdict == "hold"
        assert any(f.check == "signed_in_order" and "no owner" in f.reason for f in result.failures)


@pytest.mark.parametrize(
    "kind,title",
    [
        ("exposure", "Deployment and Rollback"),
        ("shadow_window", "Deployment and Rollback"),
        ("stop", "Deployment and Rollback"),
        ("rehearsal", "Deployment and Rollback"),
        ("segment_audit", "Outcome Audits and Decision Explanations"),
        ("explanation", "Outcome Audits and Decision Explanations"),
        ("comparison", "Capstone: Design, Build, Deploy, Operate"),
        ("tier3", "Capstone: Design, Build, Deploy, Operate"),
    ],
)
def test_packet_producer_titles_are_final_chapter_titles(kind, title):
    from refund_agent.launch.capstone_blueprint_requirement import refund_requirement

    needs = [
        need for row in refund_requirement("B6").rows for need in row.records
        if need.kind == kind
    ]
    assert needs and all(need.producer == title for need in needs)


def test_record_need_defaults_to_the_final_capstone_title():
    from refund_agent.launch.capstone_blueprint_requirement import RecordNeed

    need = RecordNeed(
        kind="comparison", required_verdict="pass", bundle_rule="strict",
        signer_role="decision_owner",
    )
    assert need.producer == "Capstone: Design, Build, Deploy, Operate"


def test_producer_table_matches_available_final_chapter_headings():
    import re

    from refund_agent.launch.capstone_blueprint_requirement import (
        PRODUCER_CHAPTERS,
        refund_requirement,
    )

    root = Path(__file__).resolve().parents[3]
    headings = {}
    for path in (root / "mainmatter").glob("*.qmd"):
        heading = re.match(r"# (.+) \{#(sec-[\w-]+)\}", path.read_text().splitlines()[0])
        if heading:
            headings[heading[2]] = heading[1]
    # These unmerged drafts reuse the final chapters' section labels.
    legacy_headings = {
        "sec-capstone-blueprint": "Capstone Blueprint",
        "sec-case-studies-domain-agnostic": "Domain-Agnostic Case Studies",
        "sec-future-directions": "Future Directions",
    }
    needs = [need for row in refund_requirement("B6").rows for need in row.records]
    assert {need.kind for need in needs} == {
        kind for _, kinds in PRODUCER_CHAPTERS.values() for kind in kinds
    }
    for section, (title, kinds) in PRODUCER_CHAPTERS.items():
        assert section in headings or section == "sec-audits-explanations"
        assert all(need.producer == title for need in needs if need.kind in kinds)
        if section in headings and headings[section] != legacy_headings.get(section):
            assert title == headings[section], section
    assert PRODUCER_CHAPTERS["sec-case-studies-domain-agnostic"][0] == "Domain-Agnostic Transfer Cases"
    assert PRODUCER_CHAPTERS["sec-future-directions"][0] == "Future Directions and System Renewal"


def test_selected_authority_appends_only_to_its_run_store(tmp_path):
    import json

    from test_org_adoption_rollout import plan_fixture

    plan, evidence = plan_fixture()
    path = tmp_path / "run-authority.jsonl"
    payload = json.dumps(
        {
            "plan": plan.model_dump(mode="json"),
            "evidence": [r.model_dump(mode="json") for r in evidence],
        }
    )
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import json, os, sys
from datetime import timedelta
from pathlib import Path
from refund_agent.release.candidate import build_candidate, build_tools
from refund_agent.release.suite import build_suite
from refund_agent.release.extensions import RuntimeStore
from refund_agent.rollout.org_adoption_authority import StageAuthority
from refund_agent.rollout.org_adoption_component import selector, restore_selection
from refund_agent.rollout.org_adoption_plan import RolloutPlan
from refund_agent.rollout.org_adoption_records import EvidenceRecord
from refund_agent.rollout.org_adoption_desk import balance_data, TEXT_BALANCE
raw = json.loads(sys.argv[2])
path = Path(sys.argv[1])
selected = selector(RuntimeStore(str(path), 'morgan', 'append'))
candidate = build_candidate(build_suite(trials=1, components=[selected]))
plan = RolloutPlan.model_validate(raw['plan']).model_copy(update={'candidate_fingerprint': candidate.fingerprint})
evidence = tuple(EvidenceRecord.model_validate(row).model_copy(update={'candidate_fingerprint': candidate.fingerprint}) for row in raw['evidence'])
authority = StageAuthority(plan, path)
authority.sign_step(plan.steps[0].step_id, by='morgan', evidence=evidence,
                    at=plan.steps[0].window.start - timedelta(minutes=1))
assert path.is_file() and len(authority.events()) == 1
relative = StageAuthority(plan, Path(os.path.relpath(path)))
assert not relative.path.is_absolute() and relative.path.resolve() == path
tools = build_tools(balance_data(TEXT_BALANCE), candidate, 'execute', org_adoption_authority=relative)
assert tools.stage_authority is relative and len(tools.stage_authority.events()) == 1
wrong = StageAuthority(plan, path.with_name('other.jsonl'))
try:
    build_tools(balance_data(TEXT_BALANCE), candidate, 'execute', org_adoption_authority=wrong)
except ValueError as error:
    assert 'store differs' in str(error)
else:
    raise AssertionError('different run store accepted')
assert not wrong.path.exists()
import subprocess
restart = '''
import json, sys
from pathlib import Path
from refund_agent.release.candidate import Candidate, build_tools
from refund_agent.rollout.org_adoption_component import restore_selection
from refund_agent.rollout.org_adoption_authority import StageAuthority
from refund_agent.rollout.org_adoption_plan import RolloutPlan
from refund_agent.rollout.org_adoption_desk import balance_data, TEXT_BALANCE
candidate = Candidate(**json.loads(sys.argv[1]))
restore_selection(candidate.record())
plan = RolloutPlan.model_validate_json(sys.argv[2])
authority = StageAuthority(plan, Path(sys.argv[3]))
tools = build_tools(balance_data(TEXT_BALANCE), candidate, 'execute', org_adoption_authority=authority)
assert len(tools.stage_authority.events()) == 1
'''
result = subprocess.run([sys.executable, '-c', restart, json.dumps(candidate.record()), plan.model_dump_json(), str(path)], text=True, capture_output=True, timeout=120)
assert result.returncode == 0, result.stderr
""",
            str(path),
            payload,
        ],
        text=True,
        capture_output=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "worse",
    [
        "hours_per_week",
        "wrongful_denials",
        "wrong_payment_cents",
        "payments_later_false",
        "cost_cents",
        "timeliness",
    ],
)
def test_comparison_rejects_a_gain_with_any_worse_measure(worse):
    from capstone_blueprint_fixtures import ROLES
    from test_capstone_blueprint_compare import comparison_fixture

    from refund_agent.launch.capstone_blueprint_compare import better, measure

    requirement, inputs = comparison_fixture()
    other = measure(inputs, requirement, ROLES).arms[2]
    changes = {"hours_per_week": other.hours_per_week - 0.1}
    if worse == "timeliness":
        changes["tier1_answered_in_time"] = other.tier1_answered_in_time - 1
    else:
        changes["cost_cents"] = other.cost_cents - 1
        changes[worse] = getattr(other, worse) + 1
    assert not better(other.model_copy(update=changes), other)


@pytest.mark.parametrize("request_id,cents", [("r0", 4999), ("r1", 4998)])
def test_comparison_refuses_payment_on_a_denial_or_for_a_different_amount(request_id, cents):
    from capstone_blueprint_fixtures import ROLES
    from test_capstone_blueprint_compare import comparison_fixture

    from refund_agent.launch.capstone_blueprint_compare import measure

    requirement, inputs = comparison_fixture()
    decisions = tuple(
        row.model_copy(update={"paid_cents": cents}) if row.request_id == request_id else row
        for row in inputs.agent.decisions
    )
    arm = inputs.agent.model_copy(update={"decisions": decisions})
    with pytest.raises(ValueError, match="approved requested amount"):
        measure(inputs.model_copy(update={"agent": arm}), requirement, ROLES)


def test_review_cost_refuses_float_cents():
    from refund_agent.launch.capstone_blueprint_requirement import CostMatrix

    with pytest.raises(ValueError):
        CostMatrix(review_cents=400.0, signed_by="morgan", signed_day=1)


@pytest.mark.parametrize("fault", ["controls", "acceptance"])
def test_signature_requires_controls_and_explicit_open_item_acceptance(tmp_path, fault):
    from capstone_blueprint_fixtures import OWNERS, packet_fixture

    from refund_agent.launch.capstone_blueprint_decision import (
        OpenItem,
        StandingControl,
        admit,
        sign,
    )
    from refund_agent.launch.capstone_blueprint_gate import check
    from refund_agent.launch.capstone_blueprint_scan import LaunchLogs

    requirement, packet, registry = packet_fixture(tmp_path)
    verdict = check(packet, requirement, registry, LaunchLogs(), OWNERS, 64)
    admission = admit(verdict, "morgan", 65, registry)
    controls = (
        ()
        if fault == "controls"
        else (StandingControl(name="audit", owner="morgan", period="monthly"),)
    )
    items = (
        OpenItem(
            item="generated labels",
            owner="morgan",
            accepted_by=None if fault == "acceptance" else "morgan",
        ),
    )
    with pytest.raises(ValueError, match="standing controls"):
        sign(
            verdict,
            admission,
            "morgan",
            65,
            requirement=requirement,
            controls=controls,
            open_items=items,
        )


@pytest.mark.parametrize("fault", ["no_role", "partial_roles", "time_box"])
def test_reverse_requires_every_named_role_and_a_real_violation(tmp_path, fault):
    from capstone_blueprint_fixtures import AT, ConstructedRegistry
    from test_capstone_blueprint_decision import ConstructedAuthority

    from refund_agent.launch.capstone_blueprint_decision import reverse
    from refund_agent.launch.capstone_blueprint_requirement import refund_requirement
    from refund_agent.launch.capstone_blueprint_scan import ReversalEvent

    requirement = refund_requirement("B6")
    if fault == "partial_roles":
        rows = tuple(
            row.model_copy(
                update={
                    "reversals": tuple(
                        condition.model_copy(update={"owner_roles": ("decision_owner", "reviewer")})
                        if condition.condition_id == "launch_gate_held"
                        else condition
                        for condition in row.reversals
                    )
                }
            )
            for row in requirement.rows
        )
        requirement = requirement.model_copy(update={"rows": rows})
    event = ReversalEvent(
        record_id="reverse-guard",
        condition_id="forbidden_payment" if fault == "time_box" else "launch_gate_held",
        day=65,
        at=AT,
        reason="canary ended: time_box" if fault == "time_box" else "guard",
        planned_time_box=fault == "time_box",
    )
    registry = ConstructedRegistry(requirement)
    with pytest.raises(ValueError):
        reverse(
            event,
            requirement=requirement,
            authority=ConstructedAuthority(),
            registry=registry,
            by="alex" if fault == "no_role" else "morgan",
        )
    assert not registry.entries


@pytest.mark.parametrize("fault", ["non_approver", "deadline_handler"])
def test_tier3_denial_needs_a_person_with_the_named_role(fault):
    from capstone_blueprint_fixtures import action

    from refund_agent.launch.capstone_blueprint_requirement import Period
    from refund_agent.launch.capstone_blueprint_scan import LaunchLogs, scan
    from refund_agent.launch.capstone_blueprint_tier3 import inspect_tier3

    row, policy = action(3, outcome="deny", actor="alex" if fault == "non_approver" else "morgan")
    if fault == "deadline_handler":
        row = row.model_copy(
            update={"review": row.review.model_copy(update={"actor_kind": "deadline_handler"})}
        )
    logs = LaunchLogs(decisions=(row,))
    assert len(scan(logs, None, policy, 0)) == 1
    assert inspect_tier3(logs, policy, Period(start=2, end=32)).verdict == "hold"


@pytest.mark.parametrize("identity", ["requester_id", "preparer_id"])
def test_scan_rejects_approval_by_requester_or_preparer(identity):
    from capstone_blueprint_fixtures import payment_action

    from refund_agent.launch.capstone_blueprint_scan import LaunchLogs, scan

    row, policy = payment_action(2)
    request = row.request.model_copy(update={identity: row.approval.reviewer})
    proposal = row.proposal.model_copy(update={"request": request})
    from refund_agent.selection import fingerprint

    row = row.model_copy(
        update={
            "request": request,
            "proposal": proposal,
            "review": row.review.model_copy(
                update={"proposal_fingerprint": fingerprint(proposal.model_dump(mode="json"))}
            ),
            "receipt": row.receipt.model_copy(update={"request": request}),
        }
    )
    assert len(scan(LaunchLogs(decisions=(row,)), None, policy, 0)) == 1


def test_scan_returns_chronological_violations():
    from capstone_blueprint_fixtures import action

    from refund_agent.launch.capstone_blueprint_scan import LaunchLogs, scan

    first, policy = action(3, outcome="deny", reviewed=False)
    second = first.model_copy(
        update={"record_id": "later", "day": first.day + 1, "at": first.at + timedelta(days=1)}
    )
    events = scan(LaunchLogs(decisions=(second, first)), None, policy, 0)
    assert [event.at for event in events] == [first.at, second.at]


def test_known_pending_tier3_referral_is_not_an_unauthorized_decision():
    from capstone_blueprint_fixtures import action

    from refund_agent.contracts import Outcome
    from refund_agent.launch.capstone_blueprint_requirement import Period
    from refund_agent.launch.capstone_blueprint_scan import LaunchLogs, scan
    from refund_agent.launch.capstone_blueprint_tier3 import inspect_tier3

    row, policy = action(3, outcome="deny", reviewed=False)
    row = row.model_copy(
        update={"decision": row.decision.model_copy(update={"outcome": Outcome.REFER})}
    )
    logs = LaunchLogs(decisions=(row,))
    result = inspect_tier3(
        logs, policy, Period(start=2, end=32),
        request_ids=(row.request.request_id,),
    )
    assert result.requests == 1 and result.decided_otherwise == ()
    assert result.pending == (row.request.request_id,) and result.verdict == "hold"
    assert scan(logs, None, policy, 0) == ()


def test_audit_label_day_thirty_is_the_exact_boundary():
    from test_org_adoption_rollout import plan_fixture

    from refund_agent.rollout.org_adoption_records import EvidenceRecord

    _, records = plan_fixture()
    audit = next(row for row in records if row.record_type == "audit")
    raw = audit.model_dump()
    raw["labels_mature_at"] = audit.period_end + timedelta(days=29)
    with pytest.raises(ValueError, match="30-day"):
        EvidenceRecord.model_validate(raw)
    raw["labels_mature_at"] += timedelta(days=1)
    assert EvidenceRecord.model_validate(raw).labels_mature_at == raw["labels_mature_at"]


@pytest.mark.parametrize("fault", ["pending", "disposition"])
def test_rollout_holds_on_pending_decisions_or_missing_workaround_disposition(fault):
    from test_org_adoption_rollout import plan_fixture

    from refund_agent.rollout.org_adoption_records import Pending, WindowCount
    from refund_agent.rollout.org_adoption_rule import decide

    plan, evidence = plan_fixture()
    step = plan.steps[0]
    count = WindowCount(
        window=step.window,
        received=100,
        governed=99,
        around=(),
        missing=("unpaid-workaround",) if fault == "disposition" else (),
        pending=(Pending(request_id="pending", decision_deadline=step.window.end, overdue=False),)
        if fault == "pending"
        else (),
    )
    result = decide(plan, step, count, evidence)
    assert result.outcome == "hold"
    assert (
        "pending_decisions" if fault == "pending" else "workaround_disposition_missing"
    ) in result.reasons


@pytest.mark.parametrize("fault", ["requester", "preparer", "owner", "path"])
def test_desk_identity_guards_write_nothing(fault):
    from test_human_oversight_risk_lifecycle import REVIEW_AT, fixture, issue

    _, tools, desk, ticket = fixture()
    proposal, proposed, explanation = desk.packets[ticket.review_id]
    reviewer = "riley" if fault == "path" else "alex"
    if fault in ("requester", "preparer"):
        field = "requester_id" if fault == "requester" else "preparer_id"
        proposal = proposal.model_copy(
            update={"request": proposal.request.model_copy(update={field: reviewer})}
        )
        desk.packets[ticket.review_id] = (proposal, proposed, explanation)
    if fault == "owner":
        desk.decision_owner = reviewer
    before = (dict(tools.audits), dict(desk.tickets), tuple(tools.payments))
    with pytest.raises(ValueError, match="credential_reviewer"):
        desk.record_refusal(
            ticket.review_id, reviewer, issue(reviewer, REVIEW_AT), "guard", REVIEW_AT
        )
    assert (dict(tools.audits), dict(desk.tickets), tuple(tools.payments)) == before


def test_denial_confirmation_reads_changed_delivery_evidence():
    from test_human_oversight_risk_lifecycle import REVIEW_AT, fixture, freshen, issue

    _, tools, desk, ticket = fixture(denial=True)
    freshen(tools, REVIEW_AT)
    tools.data = tools.data.model_copy(
        update={
            "shipments": [
                shipment.model_copy(
                    update={
                        "events": [
                            event.model_copy(update={"status": "in_transit"})
                            for event in shipment.events
                        ]
                    }
                )
                for shipment in tools.data.shipments
            ]
        }
    )
    result = desk.approve(ticket.review_id, "alex", issue("alex", REVIEW_AT), REVIEW_AT)
    assert result.request_state == "waiting" and result.reason == "denial_proposal_changed"
    assert not desk.confirmed_denials and not tools.payments


def test_self_appeal_and_pending_month_are_refused():
    from test_human_oversight_risk_lifecycle import fixture

    from refund_agent.contracts import Tier
    from refund_agent.oversight.human_oversight_risk_design import accept_months
    from refund_agent.oversight.human_oversight_risk_partners import appeal_route

    data, _, _, _ = fixture(amount=65000)
    with pytest.raises(ValueError, match="preappointed"):
        appeal_route(
            request_id="appeal",
            tier=Tier.NAMED_APPROVER,
            first_reviewer="morgan",
            reviewers=data.reviewers,
            policy=data.policies[0],
            appeals_approver="morgan",
            independent_assessment={
                "request_id": "appeal",
                "reviewer": "alex",
                "text": "assessment",
            },
        )
    assert not accept_months([{"late": 0, "blocking": 100, "pending": 1}])["accepted"]
    assert accept_months([{"late": 0, "blocking": 100, "pending": 0}])["accepted"]


def test_transfer_requires_all_seven_real_launch_rows():
    from case_studies_domain_agnostic_fixtures import test_packets

    from refund_agent.transfer.case_studies_domain_agnostic_check import structure

    packet, facts = test_packets()[0]
    assert not structure(packet, facts)
    short = packet.model_copy(update={"requirement": packet.requirement[:-1]})
    assert any(f.item == "rows" for f in structure(short, facts))


@pytest.mark.parametrize(
    "fault",
    [
        "zero_tolerance",
        "one_tolerance",
        "negative_age",
        "age_and_version",
        "source_neither",
        "source_both",
        "gap",
        "overlap",
    ],
)
def test_transfer_rejects_invalid_domain_contracts(fault):
    from fractions import Fraction

    from case_studies_domain_agnostic_fixtures import test_packets

    from refund_agent.transfer.case_studies_domain_agnostic_facts import Source
    from refund_agent.transfer.case_studies_domain_agnostic_packet import Interval, Rule

    packet, facts = test_packets()[0]
    with pytest.raises(ValueError):
        if fault.endswith("tolerance"):
            type(facts).model_validate(
                {**facts.model_dump(), "tolerance": 0 if fault == "zero_tolerance" else 1}
            )
        elif fault in ("negative_age", "age_and_version"):
            type(packet).model_validate(
                {
                    **packet.model_dump(),
                    "age_limits": {"order.lookup": -1}
                    if fault == "negative_age"
                    else {"policy.retrieve": 30},
                }
            )
        elif fault.startswith("source_"):
            Source(
                name="policy",
                refund_role="policy",
                observed_at="now",
                rate_per_hour=1 if fault == "source_both" else None,
                version_checked=fault == "source_both",
                basis="constructed",
            )
        else:
            Rule(
                intervals=(
                    Interval(
                        lower=0, upper=Fraction(1, 2), action="pay", right_closed=fault == "overlap"
                    ),
                    Interval(
                        lower=Fraction(3, 5) if fault == "gap" else Fraction(1, 2),
                        upper=1,
                        action="refer",
                        right_closed=True,
                    ),
                ),
                actions_alone=("pay",),
                referral="refer",
                status="constructed",
            )


@pytest.mark.parametrize(
    "fault", ["target", "candidate", "release", "budget_wiring", "confirmation_wiring", "same_day"]
)
def test_adoption_preconditions_cannot_be_bypassed(req, fault):
    from test_future_directions_renewal import evidence, ports

    from refund_agent.renewal.future_directions_adopt import RECORDS, adopt
    from refund_agent.renewal.future_directions_decide import decide

    decision = decide(req, evidence(req), 1)
    registry = []
    clients = ports(req, decision, registry)
    if fault in ("target", "candidate"):
        original = clients.rehearse
        field = "rollback_target_fingerprint" if fault == "target" else "candidate_fingerprint"
        clients = replace(
            clients,
            rehearse=lambda **kwargs: {
                **original(**kwargs).model_dump(mode="json"),
                field: "wrong-candidate",
            },
        )
    elif fault == "release":
        original = clients.release
        clients = replace(
            clients, release=lambda **kwargs: {**original(**kwargs), "decision": "hold"}
        )
    elif fault.endswith("wiring"):
        original = clients.register_watch
        field = (
            "budget_withdrawal_registered" if fault == "budget_wiring" else "confirmation_persisted"
        )
        clients = replace(
            clients, register_watch=lambda **kwargs: {**original(**kwargs), field: False}
        )
    records = {
        name: {"requirement_fingerprint": req.fingerprint, "passed": True} for name in RECORDS
    }
    result = adopt(
        decision,
        req,
        registry,
        "morgan",
        decision.day if fault == "same_day" else decision.day + 1,
        records=records,
        ports=clients,
    )
    assert result.verdict == "held" and not registry
    if fault in ("target", "candidate"):
        assert "rollback rehearsal failed or names another target" in result.reason


def test_evidence_rejects_another_incumbent_and_excess_prediction(req, score, window):
    from refund_agent.data.history import END_AT
    from refund_agent.renewal.future_directions_evidence import evaluate

    rows = [row for row in window.rows if row.paid]
    with pytest.raises(ValueError, match="another incumbent"):
        evaluate(
            rows,
            score.component,
            "other-incumbent",
            req,
            window.mature_at,
            window=window,
            expected_start=END_AT,
        )
    component = replace(score.component, intercept=10)
    requirement = req.model_copy(update={"score_fingerprint": component.fingerprint})
    ev = evaluate(
        rows,
        component,
        requirement.incumbent_candidate_fingerprint,
        requirement,
        window.mature_at,
        window=window,
        expected_start=END_AT,
    )
    assert ev.prediction > ev.interval[1]
    assert not next(check for check in ev.checks if check.name == "calibration").passed


@pytest.mark.parametrize("fault", ["unspent", "agent_decider", "live_authority"])
def test_budget_withdrawal_requires_spent_and_a_person_in_shadow(tmp_path, fault):
    from test_future_directions_watch import setup

    from refund_agent.renewal.future_directions_watch import BudgetStatus, budget_spent

    _, adoption, store = setup(tmp_path)
    budget = BudgetStatus(
        plan_fingerprint="plan",
        window_fingerprint="window",
        spent=fault != "unspent",
        status_fingerprint="budget",
    )
    record = {
        "tier1_authority": "stage6" if fault == "live_authority" else "shadow",
        "decider": "agent" if fault == "agent_decider" else "person",
    }
    with pytest.raises(ValueError):
        budget_spent(store, adoption, budget=budget, withdraw=lambda **kwargs: record, day=61)


def test_watch_accepts_labels_at_exactly_thirty_days(tmp_path):
    from test_future_directions_watch import rows, setup

    from refund_agent.renewal.future_directions_watch import watch

    req, adoption, store = setup(tmp_path)
    result = watch(rows(1, 100, 100, 20, 60), adoption, req, 60, store, period="month")
    assert result.verdict == "pending" and result.state.failed_month == "month"


def test_payment_outcomes_wait_exactly_thirty_days():
    from refund_agent.launch.capstone_blueprint_outcomes import PaymentOutcome, outcomes

    row = PaymentOutcome(
        receipt_id="paid",
        request_id="request",
        payment_day=1,
        amount_cents=4999,
        actor="human",
        mode="shadow",
        found_false_30d=False,
        generated_label=False,
    )
    assert outcomes((row,), 30).pending == ("paid",)
    assert outcomes((row,), 31).final == (row,)


def test_adoption_calls_the_real_launch_gate_with_its_full_state(req, tmp_path):
    from capstone_blueprint_fixtures import OWNERS, packet_fixture
    from test_future_directions_renewal import evidence, ports

    from refund_agent.launch.capstone_blueprint_decision import hold_record, on_promotion
    from refund_agent.launch.capstone_blueprint_gate import check
    from refund_agent.launch.capstone_blueprint_scan import LaunchLogs
    from refund_agent.renewal.future_directions_adopt import RECORDS, adopt
    from refund_agent.renewal.future_directions_decide import decide

    requirement, packet, registry = packet_fixture(
        tmp_path, {"comparison": {"verdict": "not better"}}
    )
    req = req.model_copy(update={"candidate_bundle_fingerprint": requirement.launched_bundle})
    decision = decide(req, evidence(req), 1)
    verdict = check(packet, requirement, registry, LaunchLogs(), OWNERS, 64)
    state = {
        "requirement": requirement,
        "packet": packet,
        "registry": registry,
        "logs": LaunchLogs(),
        "owners": OWNERS,
        "day": 64,
        "decision": hold_record(verdict, requirement, registry, None, 64),
    }
    promotions = []
    clients = replace(ports(req, decision, promotions), on_promotion=on_promotion)
    records = {
        name: {"requirement_fingerprint": req.fingerprint, "passed": True} for name in RECORDS
    }
    result = adopt(
        decision, req, promotions, "morgan", 91, records=records, ports=clients, launch_state=state
    )
    assert result.verdict == "adopted" and result.launch_verdict == "hold"
    assert registry.entries[-1]["verdict"].rows == verdict.rows
    partial = {
        "bundle_fingerprint": "incumbent-bundle",
        "candidate_fingerprint": "incumbent",
        "verdict": "hold",
        "signature": None,
    }
    pending = adopt(
        decision, req, [], "morgan", 91, records=records, ports=clients, launch_state=partial
    )
    assert pending.verdict == "adopted-launch-pending" and "requirement" in pending.reason


def test_bank_calls_the_actual_queue_producer(monkeypatch, capsys):
    from demos import case_studies_domain_agnostic_demo as demo
    from refund_agent.oversight.human_oversight_risk_queueing import mm1

    calls = []

    def observed(lam, mu):
        calls.append((lam, mu))
        return mm1(lam, mu)

    monkeypatch.setattr(demo, "mm1", observed, raising=False)
    demo.print_bank()
    assert calls == [(12, 15)]
    assert "M/M/1 review-open time: 20 minutes" in capsys.readouterr().out


def test_budget_exercise_key_names_the_actual_scope_decider_and_mechanism(capsys):
    from demos import capstone_blueprint_demo as demo

    answer = demo.reference()["exercise_02"]()
    demo.print_keys()
    line = next(line for line in capsys.readouterr().out.splitlines() if line.startswith("2:"))
    assert (
        answer["withdraws"] in line
        and answer["decider_after"] in line
        and answer["mechanism"] in line
    )


def test_request_before_window_routes_to_person_and_withdrawal_conflict_is_refused(tmp_path):
    from test_org_adoption_rollout import plan_fixture

    from refund_agent.contracts import Request
    from refund_agent.rollout.org_adoption_authority import StageAuthority

    plan, evidence = plan_fixture()
    authority = StageAuthority(plan, tmp_path / "clock-authority.jsonl")
    at = plan.steps[0].window.start - timedelta(minutes=30)
    authority.sign_step(plan.steps[0].step_id, by="morgan", evidence=evidence, at=at)
    from refund_agent.canary.cloud_native_deployment_plan import in_cohort

    requester = next(
        f"requester-{i}"
        for i in range(100)
        if in_cohort(plan.canary_id, f"requester-{i}", plan.steps[0].requester_share)
    )
    request = Request(
        request_id="early",
        requester_id=requester,
        order_reference="order",
        at=at + timedelta(minutes=1),
        amount_cents=4999,
        reason="damaged",
    )
    assert authority.route(request, request.at) == "person"
    assert authority.reason(request, request.at) == "org_adoption_step_not_started"
    authority.withdraw("budget_spent", by="morgan", at=at, record_id="same-id")
    with pytest.raises(ValueError, match="identity conflict"):
        authority.withdraw(
            "budget_spent", by="morgan", at=at + timedelta(seconds=1), record_id="same-id"
        )


@pytest.mark.parametrize("amount", [5000, 50001])
def test_shared_owner_is_not_a_distinct_escalation_path(amount):
    from test_human_oversight_risk_lifecycle import fixture

    _, _, desk, ticket = fixture(amount=amount)
    desk.decision_owner = "morgan"
    proposal, proposed, explanation = desk.packets[ticket.review_id]
    with pytest.raises(ValueError, match="distinct path"):
        desk.open(proposal, proposed, explanation, reviewer="alex")


def test_budget_rollback_never_regrants_real_stage_authority(tmp_path):
    from test_future_directions_watch import setup
    from test_org_adoption_rollout import plan_fixture

    from refund_agent.renewal.future_directions_watch import BudgetStatus, watch
    from refund_agent.rollout.org_adoption_authority import StageAuthority
    from refund_agent.rollout.org_adoption_rule import StepDecision

    plan, evidence = plan_fixture()
    authority = StageAuthority(plan, tmp_path / "real-authority.jsonl")
    first, second = plan.steps
    authority.sign_step(
        first.step_id, by="morgan", at=first.window.start - timedelta(minutes=1), evidence=evidence
    )
    req, adoption, store = setup(tmp_path)
    attempted = []

    def withdraw(**kwargs):
        authority.withdraw(
            "budget_spent", by="morgan", at=first.window.start, record_id="spent-budget"
        )
        return {"tier1_authority": "shadow", "decider": "person"}

    def rollback(**kwargs):
        previous = StepDecision(
            plan_id=plan.plan_id,
            step_id=first.step_id,
            window_id=first.window.window_id,
            at=first.window.closed_at,
            outcome="widen",
            reasons=(),
        )
        with pytest.raises(ValueError, match="cannot restore withdrawn authority"):
            authority.sign_step(
                second.step_id,
                by="morgan",
                at=second.window.start - timedelta(minutes=1),
                evidence=evidence,
                previous=previous,
            )
        attempted.append("restore grant refused")
        return {
            "target_fingerprint": adoption["incumbent_bundle_fingerprint"],
            "target_candidate_fingerprint": adoption["incumbent_candidate_fingerprint"],
            "stage": 6,
            "verdict": "restored",
        }

    result = watch(
        [],
        adoption,
        req,
        61,
        store,
        period="budget",
        budget=BudgetStatus(
            plan_fingerprint="plan",
            window_fingerprint="window",
            spent=True,
            status_fingerprint="budget",
        ),
        withdraw=withdraw,
        rollback=rollback,
    )
    assert result.state.grant_withdrawn and attempted == ["restore grant refused"]
    assert [event.kind for event in authority.events()] == [
        "org-adoption-grant",
        "org-adoption-withdrawal",
    ]


COMPOSED_CODE = """
import sys
from datetime import timedelta
from pathlib import Path
from refund_agent.release.candidate import build_candidate, build_tools, run_request
from refund_agent.release.suite import build_suite
from refund_agent.release.extensions import RuntimeStore
sys.path.insert(0, str(Path.cwd() / 'companion/tests/refund_agent'))
from test_human_oversight_risk_lifecycle import fixture, freshen, authenticate, explain_fixture
from test_org_adoption_rollout import plan_fixture
from test_future_directions_risk import component
from refund_agent.oversight.human_oversight_risk_component import SELECTOR as oversight
from refund_agent.rollout.org_adoption_component import selector as authority_selector
from refund_agent.rollout.org_adoption_authority import StageAuthority
from refund_agent.future_directions_risk import selector as risk_selector
from refund_agent.contracts import Outcome
amount = int(sys.argv[2])
data, prepared, _, _ = fixture(amount=amount, denial=amount > 50000)
score = component()
path = Path(sys.argv[1]) / 'authority.jsonl'
candidate = build_candidate(build_suite(trials=1, components=(
    oversight, authority_selector(RuntimeStore(str(path), 'morgan', 'append')),
    risk_selector(score, policy_version=data.policies[0].version),
)))
plan, evidence = plan_fixture(candidate.fingerprint)
first = plan.steps[0].model_copy(update={'requester_share': 1.0})
plan = plan.model_copy(update={'steps': (first, plan.steps[1])})
at = first.window.start + timedelta(hours=1)
prepared.data = data
freshen(prepared, at)
data = prepared.data
request = data.requests[0].model_copy(update={'at': at})
shipments = [s.model_copy(update={'events': [e.model_copy(update={'at': at - timedelta(days=1)}) for e in s.events]}) for s in data.shipments]
data = data.model_copy(update={'requests': [request], 'shipments': shipments})
authority = StageAuthority(plan, path)
authority.sign_step(first.step_id, by='morgan', evidence=evidence,
                    at=first.window.start - timedelta(minutes=1))
tools = build_tools(data, candidate, 'execute', reviewer_authenticate=authenticate,
                    denial_explainer=explain_fixture, decision_owner='decision-owner',
                    assigned_reviewer='alex', org_adoption_authority=authority,
                    future_directions_component=score)
trace = run_request(candidate, tools, request)
assert trace.decision.outcome == Outcome.REFER and not tools.payments
if amount > 50000:
    assert not any(step.state.decision is not None and step.state.decision.reason == 'org_adoption_person_required' for step in trace.steps)
print(trace.decision.reason)
print('tickets=' + str(len(tools.review_desk.tickets)))
"""


def test_composed_tier3_denial_retains_its_policy_reason(tmp_path):
    result = subprocess.run(
        [sys.executable, "-c", COMPOSED_CODE, str(tmp_path), "65000"],
        text=True,
        capture_output=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == ["already_delivered", "tickets=1"]


def test_timeline_signs_both_steps_through_the_real_maturity_rule(tmp_path):
    from test_capstone_blueprint_timeline import constructed_driver
    from test_org_adoption_rollout import plan_fixture

    from refund_agent.launch.capstone_blueprint_timeline import START, Demand, StageResult, operate
    from refund_agent.rollout.org_adoption_authority import StageAuthority
    from refund_agent.rollout.org_adoption_records import EvidenceRecord, Window
    from refund_agent.rollout.org_adoption_rule import StepDecision

    def instant(day):
        return START + timedelta(days=day - 1)

    base, _ = plan_fixture("B6")
    # Signed constructed schedule; each audit actually waits its full label window.
    first_audit = (6, 34)
    first_window = (64.0011, 78.0011)
    second_window = (108.0032, 122.0032)
    periods = (first_audit, first_window)
    steps = []
    for step, (start, end), (audit_start, audit_end) in zip(
        base.steps, (first_window, second_window), periods
    ):
        refs = tuple(
            ref.model_copy(
                update={"period_start": instant(audit_start), "period_end": instant(audit_end)}
            )
            if ref.record_type == "audit"
            else ref
            for ref in step.required
        )
        steps.append(
            step.model_copy(
                update={
                    "required": refs,
                    "signed_at": instant(start - 0.0001),
                    "window": Window(
                        window_id=step.window.window_id,
                        start=instant(start),
                        end=instant(end),
                        closed_at=instant(end),
                    ),
                }
            )
        )
    plan = base.model_copy(update={"steps": tuple(steps), "signed_at": instant(1)})
    out = tmp_path / "run"
    out.mkdir()
    authority = StageAuthority(plan, out / "authority.jsonl")
    driver = constructed_driver()
    audits, grants, evidence = [], [], []
    generation = []
    generate = driver.generate_stage6

    def generate_stream(seed, start, days, demand):
        generation.append(days)
        return generate(seed, start, days, demand)

    driver.generate_stage6 = generate_stream

    def audit(stream, out, period):
        index = len(audits)
        step = plan.steps[index]
        signed = period.end + 30.0001
        for ref in step.required:
            evidence.append(
                EvidenceRecord(
                    record_type=ref.record_type,
                    record_id=ref.record_id,
                    candidate_fingerprint="B6",
                    signed_by="morgan",
                    signed_at=instant(signed),
                    status="pass",
                    period_start=ref.period_start,
                    period_end=ref.period_end,
                    labels_mature_at=instant(period.end + 30)
                    if ref.record_type == "audit"
                    else None,
                )
            )
        audits.append(period)
        return StageResult(end_day=signed, records=())

    def rollout(stream, out, number, period):
        step = plan.steps[number - 1]
        assert instant(period.start) == step.window.start
        previous = (
            None
            if number == 1
            else StepDecision(
                plan_id=plan.plan_id,
                step_id=plan.steps[0].step_id,
                window_id=plan.steps[0].window.window_id,
                at=plan.steps[0].window.closed_at,
                outcome="widen",
                reasons=(),
            )
        )
        grants.append(
            authority.sign_step(
                step.step_id,
                by="morgan",
                at=step.signed_at,
                evidence=tuple(evidence),
                previous=previous,
            )
        )
        driver.clock = period.end
        return StageResult(end_day=period.end, records=())

    driver.audit_period, driver.rollout_step, driver.authority = audit, rollout, authority
    run = operate(7, demand=Demand(), out=out, driver=driver)
    assert len(grants) == 2 and run.gate_day > 124
    assert len(authority.events()) == 2
    assert generation[0] > run.gate_day


def test_timeline_refuses_month_controls_before_label_maturity(tmp_path):
    from test_capstone_blueprint_timeline import constructed_driver

    from refund_agent.launch.capstone_blueprint_timeline import Demand, StageResult, operate

    driver = constructed_driver()
    driver.month_controls = lambda stream, out, period, requirement: StageResult(
        end_day=driver.clock, records=()
    )
    with pytest.raises(ValueError, match="clock"):
        operate(7, demand=Demand(), out=tmp_path / "run", driver=driver)
    assert not (tmp_path / "run/capstone-blueprint-gate-01.json").exists()


def test_registry_port_declares_the_commands_the_launch_consumer_uses():
    import ast

    from refund_agent.launch.capstone_blueprint_gate import RegistryPort

    source = (
        Path(__file__).resolve().parents[2] / "refund_agent/launch/capstone_blueprint_decision.py"
    )
    calls = {
        node.func.attr
        for node in ast.walk(ast.parse(source.read_text()))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "registry"
    }
    assert calls <= set(vars(RegistryPort))


def test_watch_waits_without_consuming_an_immature_month(tmp_path):
    from test_future_directions_watch import rows, setup

    from refund_agent.renewal.future_directions_watch import watch

    req, adoption, store = setup(tmp_path)
    month = rows(1, 100, 100, 20, 60)
    initial = store.load()
    result = watch(month, adoption, req, 59, store, period="month")
    assert result.verdict == "pending" and result.state.status == "clear"
    assert result.state.last_month_end == initial.last_month_end == 0
    assert watch(month, adoption, req, 60, store, period="month").verdict == "pending"


def test_watch_fixture_declares_its_store_after_component_selection():
    code = """
import sys
from pathlib import Path
from refund_agent.release.candidate import build_candidate
from refund_agent.release.suite import build_suite
sys.path.insert(0, str(Path.cwd() / 'companion/tests/refund_agent'))
from test_future_directions_risk import component
from refund_agent.future_directions_risk import selector
build_candidate(build_suite(trials=1, components=(selector(component(), policy_version='2026-09-01'),)))
import pytest
raise SystemExit(pytest.main(['companion/tests/refund_agent/test_future_directions_watch.py', '-k', 'budget_exhaustion', '-p', 'no:cacheprovider', '-o', 'addopts=', '-q']))
"""
    result = subprocess.run(
        [sys.executable, "-c", code], text=True, capture_output=True, timeout=120
    )
    assert result.returncode == 0, result.stdout + result.stderr
