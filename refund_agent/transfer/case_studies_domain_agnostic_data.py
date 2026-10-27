"""Given chapter-40 inputs and packet construction over a supplied launch requirement."""

import json
import math
from fractions import Fraction as F
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from refund_agent.oversight.human_oversight_risk_queueing import mm1
from refund_agent.sources.data_systems_budget import rate_upper
from refund_agent.transfer.case_studies_domain_agnostic_check import (
    clock,
    coupling,
    optimal_rule,
    rule,
    structure,
)
from refund_agent.transfer.case_studies_domain_agnostic_domains import HospitalDay, Patient
from refund_agent.transfer.case_studies_domain_agnostic_facts import (
    Action,
    Clock,
    CostMatrix,
    DomainFacts,
    SharedResource,
    Source,
    require_inventory,
)
from refund_agent.transfer.case_studies_domain_agnostic_packet import (
    Cap,
    Cell,
    DecisionClass,
    DecisionPacket,
    Rule,
    copy_refund,
    map_requirement_roles,
    strip,
)

SLUG = "case-studies-domain-agnostic"
DATA = Path(__file__).resolve().parents[2] / "demos/data/transfer"
EXERCISES = Path(__file__).resolve().parents[2] / f"exercises/{SLUG}"
GIVEN = "constructed, given in chapter 40"
DOMAINS = ("refund", "triage", "hospital", "bank")
QUESTIONS = (
    "What can this agent do that a workflow cannot?",
    "What is the smallest safe action scope?",
    "Which failures become public or expensive?",
    "What evidence would show that the agent is improving?",
    "How do we turn incidents into evaluation cases?",
    "What can a person override, and how fast?",
    "Which rollback has actually been rehearsed?",
)
ANSWERS = {
    "refund": (
        "Comparison awaits chapter 39's operating run; launch status unverified",
        "Under $50 alone; reads under 30 minutes; designed score below 0.0800",
        "Payments cannot be recalled; denials audited by segment",
        "No forbidden payment; wrongful denials no more than baseline",
        "Each incident becomes a case in the release suite",
        "Review deadline and escalation; clock never pays or denies",
        "Rehearsal required on launched bundle; payments stand",
    ),
    "triage": (
        "Against engineers and rules engine, same incidents",
        "Cards with fewer than 50 sessions; alarm probed at restart",
        "Dropped calls; changes to emergency-call routing",
        "Time to recovery; counter-measure, restarts marked needless",
        "Each incident becomes a case in the release suite",
        "Engineer; deadline takes the branch that can be undone",
        "Rehearsal required; dropped calls stay dropped",
    ),
    "hospital": (
        "Against schedulers and a priority rule",
        "Bookings from the day's allocation, rechecked at execution",
        "Urgent follow-up missed; outcomes audited by segment",
        "Urgent follow-ups within 2 days; counter-measure, missed appointments",
        "Each incident becomes a case in the release suite",
        "Scheduler, then clinician; booking by telephone",
        "Rehearsal required; booked slots stand and people rebook",
    ),
    "bank": (
        "Against analysts' queue and rules engine, same flags",
        "Holds alone above 1/6; clears alone below 1.1%",
        "Open-card spending; wrongful hold; holds audited by segment",
        "Losses on unheld cards; counter-measure, holds lifted as mistaken",
        "Each incident becomes a case in the release suite",
        "Customer lifts hold in minutes; analyst decides",
        "Rehearsal required; holds stand until lifted",
    ),
}


def cells(facts: DomainFacts) -> tuple[Cell, ...]:
    """The short prose represents only the fixed constructed examples."""
    domain = facts.domain
    if domain not in ANSWERS or facts != domain_facts(domain):
        raise ValueError(f"{domain} table cannot represent the supplied facts")
    require_inventory(range(len(ANSWERS[domain])), range(len(QUESTIONS)), f"{domain} table answer")
    return tuple(
        Cell(
            question=q,
            answer=a,
            kind=(
                "same"
                if index == 4 or domain == "refund"
                else "redesigned"
                if index == 1 or domain == "bank" and index == 5
                else "refilled"
            ),
        )
        for index, (q, a) in enumerate(zip(QUESTIONS, ANSWERS[domain]))
    )


def _action(name, role, dimension="money", unit="cents", loss=4999, *, undo=None):
    return Action(
        name=name,
        role=role,
        effect="write",
        reversible=undo is not None,
        undo_seconds=undo,
        harm_unit=unit,
        harm_dimension=dimension,
        largest_loss=F(loss) if loss is not None else None,
        basis=GIVEN,
    )


def domain_facts(domain: str) -> DomainFacts:
    titles = {
        "refund": (
            "merchant decision owner",
            "merchant release owner",
            "refund reviewer",
            "named approver",
        ),
        "triage": (
            "network decision owner",
            "network release owner",
            "on-call engineer",
            "duty manager",
        ),
        "hospital": (
            "clinic decision owner",
            "clinic release owner",
            "scheduler",
            "clinician on duty",
        ),
        "bank": (
            "fraud decision owner",
            "fraud release owner",
            "fraud analyst",
            "fraud operations manager",
        ),
    }
    mapped = {
        "refund": (
            "order.lookup",
            "shipment.status",
            "policy.retrieve",
            "refund.propose",
            "refund.execute",
            "pay",
            "refer",
        ),
        "triage": (
            "alarm.read",
            "topology.lookup",
            "runbook.retrieve",
            "ticket.open",
            "card.restart",
            "card.restart",
            "approval.request",
        ),
        "hospital": (
            "discharge.read",
            "schedule.lookup",
            "protocol.retrieve",
            "booking.propose",
            "booking.confirm",
            "booking.confirm",
            "approval.request",
        ),
        "bank": (
            "card.lookup",
            "alert.read",
            "rules.retrieve",
            "case.open",
            "clear",
            "clear",
            "refer",
        ),
    }
    old = (
        "order.lookup",
        "shipment.status",
        "policy.retrieve",
        "refund.propose",
        "refund.execute",
        "pay",
        "refer",
    )
    tools = dict(zip(old, mapped[domain])) | {
        "approval.request": "approval.request",
        "audit.append": "audit.append",
    }
    rates = (
        (math.log(2) * 6, 0.01)
        if domain == "triage"
        else (rate_upper(0, 6000), 0.02)
        if domain == "refund"
        else (0.01, 0.01)
    )
    sources = tuple(
        Source(
            name=tools[role],
            refund_role=role,
            observed_at="source observation",
            rate_per_hour=rate,
            basis=(
                "chapter 30: 95% upper rate, 0 changes in 6000 order-hours"
                if domain == "refund" and role == "order.lookup"
                else "chapter 30: 48 first-delivery scans in 2400 parcel-hours"
                if domain == "refund"
                else GIVEN
            ),
        )
        for role, rate in zip(old[:2], rates)
    ) + (
        Source(
            name=tools["policy.retrieve"],
            refund_role="policy.retrieve",
            observed_at="versioned document",
            basis="version currency, as chapters 23 and 30",
            version_checked=True,
        ),
    )
    clocks = [
        Clock(
            name="operational reads",
            kind="read",
            seconds=1800,
            basis="specification: 30 minutes, copied unless redesigned",
        )
    ]
    matrix, allowed, shared, actions, notes = None, (), (), [], []
    if domain == "refund":
        actions = [
            _action("refund.execute", "execute"),
            _action("pay", "pay"),
            _action("refer", "refer", undo=0),
        ]
        allowed = ("pay", "refer")
        matrix = CostMatrix(
            outcomes=("sound", "false"),
            entries={"pay": (F(0), F("49.99")), "refer": (F(4), F(4))},
            unit="dollars",
            basis="chapter 5: $49.99 payment and illustrative $4 review",
        )
        shared = (
            SharedResource(
                name="order_balance",
                capacity=F(4000),
                unit="cents",
                actions=("refund.execute",),
                scope="order",
                requirement="limit",
                basis="B3: two 3000-cent requests; 4000-cent order",
            ),
        )
        notes = [
            "Reviewer minutes are consumed by review assignment beside the request loop.",
            "Designed calibrated rule held; running agent retains its text score.",
            "Over 50000 cents the named approver decides, including denials.",
        ]
        clocks += [
            Clock(
                name="policy text",
                kind="read",
                seconds=30 * 86400,
                basis="specification: 30 days; currency by version",
            ),
            Clock(
                name="payment",
                kind="undo",
                seconds=None,
                basis="specification: payments cannot be recalled",
            ),
            Clock(
                name="found false",
                kind="outcome",
                seconds=30 * 86400,
                basis="D6: 30-day label window",
            ),
        ]
    elif domain == "triage":
        actions = [
            _action("card.restart", "execute", "sessions", "sessions dropped", 50),
            _action("ticket.open", "propose", "sessions", "sessions dropped", 0, undo=0),
        ]
        notes = [
            "Main scenario: fewer than 50 sessions alone; 50 or more need an engineer.",
            "Constructed frequency constraint: at most 10 restarts per day.",
        ]
        clocks[0] = Clock(
            name="alarm freshness budget",
            kind="read",
            seconds=-math.log1p(-0.01) / rates[0] * 3600,
            basis="computed from given alarm half-life and prior 1% tolerance",
        )
        clocks += [
            Clock(
                name="dropped calls",
                kind="undo",
                seconds=None,
                basis="constructed: dropped calls cannot be restored",
            ),
            Clock(name="needless restart", kind="outcome", seconds=86400, basis=GIVEN),
        ]
    elif domain == "hospital":
        actions = [_action("booking.confirm", "execute", "bookings", "follow-ups", 1, undo=0)]
        shared = (
            SharedResource(
                name="slots",
                capacity=F(21),
                unit="follow-up slots",
                actions=("booking.confirm",),
                scope="domain",
                requirement="allocation",
                basis="constructed: 3 slots/day for 7 days",
            ),
        )
        clocks += [
            Clock(name="urgent follow-up", kind="window", seconds=2 * 86400, basis=GIVEN),
            Clock(name="routine follow-up", kind="window", seconds=7 * 86400, basis=GIVEN),
            Clock(
                name="missed window",
                kind="undo",
                seconds=None,
                basis="a missed follow-up window cannot be given back",
            ),
            Clock(
                name="readmission",
                kind="outcome",
                seconds=30 * 86400,
                basis="chapter 4: 30-day readmission label",
            ),
        ]
    elif domain == "bank":
        review = mm1(12, 15)
        if review.W is None:
            raise ValueError("bank review queue is unstable")
        referral_cost = F(4) + F(150) * review.W
        actions = [
            _action("clear", "execute", loss=40000),
            _action("refer", "refer", loss=referral_cost * 100, undo=0),
            _action("hold", "protect", loss=1400, undo=300),
            _action("account.close", "close", loss=None, undo=3 * 86400),
        ]
        allowed = ("clear", "refer", "hold")
        matrix = CostMatrix(
            outcomes=("legitimate", "fraudulent"),
            entries={"clear": (F(0), F(400)), "refer": (F(4), referral_cost), "hold": (F(14), F(4))},
            unit="dollars per card",
            basis="constructed card-level costs; $4 review + $150/hour during chapter 37 mm1(12,15).W",
        )
        notes = [
            "Score: probability a card is in fraudulent use within 60 days, given recent payments.",
            "The flagged-payment precision 4/37 transfers only under this constructed card-level definition.",
            "Human Oversight and Risk Management's mm1(12,15).W supplies the 20-minute review-open time; referral costs $4 per card, plus $150 times W if fraudulent, with W in hours; an unstable queue is refused.",
            "Account closure is level 2, decided by the named approver.",
        ]
        clocks += [
            Clock(
                name="analyst review",
                kind="review",
                seconds=review.W * 3600,
                basis="chapter 37 mm1(12,15).W * 3600; constructed arrivals and service per hour",
            ),
            Clock(
                name="card hold",
                kind="undo",
                seconds=300,
                basis="chapter 3: minutes, constructed as 5",
            ),
            Clock(
                name="account closure",
                kind="undo",
                seconds=3 * 86400,
                basis="chapter 3: days, constructed as 3",
            ),
            Clock(name="card fraud / chargeback", kind="outcome", seconds=60 * 86400, basis=GIVEN),
        ]
    known_actions = {action.name for action in actions}
    read_names = {source.name for source in sources}
    for role, name in tools.items():
        if name not in known_actions:
            actions.append(
                Action(
                    name=name,
                    role=role,
                    effect="read" if name in read_names else "write",
                    reversible=True,
                    undo_seconds=0,
                    harm_unit="cents",
                    harm_dimension="money",
                    largest_loss=F(0),
                    basis="constructed read/proposal/review/audit tool; no direct domain action",
                )
            )
            known_actions.add(name)
    planted: dict[str, tuple[Literal["clock", "coupling", "rule"], ...]] = {
        "refund": (),
        "triage": ("clock",),
        "hospital": ("coupling",),
        "bank": ("rule",),
    }
    return DomainFacts(
        domain=domain,
        constructed=True,
        sources=sources,
        actions=tuple(actions),
        actions_on_score=allowed,
        cost_matrix=matrix,
        base_rate=F(1, 100) if domain == "bank" else None,
        tolerance=0.01,
        shared_resources=shared,
        roles=dict(
            zip(("decision_owner", "release_owner", "reviewer", "named_approver"), titles[domain])
        ),
        role_map=tools,
        label_windows={
            "refund": {"found_false": 30 * 86400},
            "triage": {"needless_restart": 86400},
            "hospital": {
                "urgent_followup": 2 * 86400,
                "routine_followup": 7 * 86400,
                "readmission": 30 * 86400,
            },
            "bank": {"card_fraud": 60 * 86400},
        }[domain],
        clocks=tuple(clocks),
        planted_checks=planted[domain],
        notes=tuple(notes),
    )


def refund_rule() -> Rule:
    facts = domain_facts("refund")
    if facts.cost_matrix is None:
        raise ValueError("refund design requires its cost matrix")
    optimal = optimal_rule(facts.cost_matrix, facts.actions_on_score, "refer", ("pay",))
    return Rule.model_validate({**optimal.model_dump(), "status": "designed-held"})


def refund_packet(requirement: BaseModel) -> DecisionPacket:
    """The host supplies chapter 39's real requirement; no invented predecessor."""
    return DecisionPacket(
        domain="refund",
        form="refund",
        boundary={
            "permitted": "eligible refunds",
            "forbidden": "forbidden payments and score-only denials",
            "approval": "one reviewer to 50000 cents; named approver above, including denials",
            "reversal": "payments stand",
            "ceiling": "under 5000 cents alone",
            "logging": "audit.append",
        },
        classes=(
            DecisionClass(
                name="small payment",
                action="refund.execute",
                level=4,
                above_role="reviewer",
                condition="eligible and under 5000 cents",
            ),
            DecisionClass(
                name="reviewed payment",
                action="refund.execute",
                level=3,
                above_role="reviewer",
                condition="5000 through 50000 cents; one reviewer decides",
            ),
            DecisionClass(
                name="named-approver outcome",
                action="refund.execute",
                level=2,
                above_role="named_approver",
                condition="above 50000 cents; approver decides payments and denials",
            ),
        ),
        age_limits={"order.lookup": 1800, "shipment.status": 1800},
        version_checks=("policy.retrieve",),
        rule=refund_rule(),
        decision_unit="request",
        recheck_at_execution=True,
        caps=(
            Cap(
                action="refund.execute",
                value=F(5000),
                unit="cents",
                dimension="money",
                inclusive=False,
                basis="specification: tier 1 under 5000 cents",
            ),
        ),
        requirement=strip(requirement),
        cells=cells(domain_facts("refund")),
        label_windows={"found_false": 30 * 86400},
    )


def _redesign_fields(facts: DomainFacts) -> dict[str, object]:
    """Refill every answer from these facts; launch rows remain the host's."""
    table_cells = cells(facts)
    actions = {a.name: a for a in facts.actions}
    reviewer, approver = facts.roles["reviewer"], facts.roles["named_approver"]
    logging = facts.role_map["audit.append"]
    band = None
    if facts.domain == "triage":
        restart = actions["card.restart"]
        ceiling = restart.largest_loss
        if ceiling is None or restart.reversible or restart.harm_dimension != "sessions":
            raise ValueError("triage redesign requires bounded, irreversible session loss")
        boundary = {
            "permitted": f"restart with fewer than {ceiling} sessions and active alarm.probe at restart",
            "forbidden": "autonomous core-element or emergency-call routing changes",
            "approval": f"{reviewer} for {ceiling} or more sessions or links; {approver} for core/emergency routing",
            "reversal": "tickets can be closed; dropped calls cannot be restored",
            "ceiling": f"fewer than {ceiling} sessions per restart; 10 restarts per day, constructed",
            "logging": logging,
        }
        classes = (
            DecisionClass(
                name="small line-card restart",
                action="card.restart",
                level=4,
                above_role="reviewer",
                condition=f"active alarm.probe at restart; fewer than {ceiling} sessions; no core/emergency change",
            ),
            DecisionClass(
                name="reviewed card or link",
                action="card.restart",
                level=3,
                above_role="reviewer",
                condition=f"{ceiling} or more sessions or taking a link down; {reviewer} approves",
            ),
            DecisionClass(
                name="core or emergency routing",
                action="card.restart",
                level=2,
                above_role="named_approver",
                condition=f"{approver} decides core or emergency-call changes",
            ),
        )
        caps = (
            Cap(
                action="card.restart",
                value=ceiling,
                unit=restart.harm_unit,
                dimension=restart.harm_dimension,
                inclusive=False,
                basis="main scenario: fewer than 50 sessions per restart",
            ),
        )
    elif facts.domain == "hospital":
        slots = next(r for r in facts.shared_resources if r.name == "slots")
        booking = actions["booking.confirm"]
        boundary = {
            "permitted": "follow-up in the patient's window from the daily allocation, rechecked at execution",
            "forbidden": "diagnosis and treatment; no double booking",
            "approval": f"{reviewer} when no slot; {approver} alone changes capacity or windows",
            "reversal": "bookings can be cancelled and rebooked; a missed window cannot be restored",
            "ceiling": f"one follow-up per slot; {slots.capacity} slots",
            "logging": logging,
        }
        classes = (
            DecisionClass(
                name="allocated follow-up",
                action="booking.confirm",
                level=4,
                above_role="reviewer",
                condition="free slot in the patient's window from the daily allocation, rechecked at execution",
            ),
            DecisionClass(
                name="no free slot",
                action="booking.confirm",
                level=3,
                above_role="reviewer",
                condition=f"{reviewer} decides when no slot remains in the window",
            ),
            DecisionClass(
                name="changed window or capacity",
                action="booking.confirm",
                level=2,
                above_role="named_approver",
                condition=f"{approver} decides changes; no diagnosis or treatment by agent",
            ),
        )
        caps = (
            Cap(
                action="booking.confirm",
                value=F(1),
                unit=booking.harm_unit,
                dimension=booking.harm_dimension,
                inclusive=True,
                basis="one booking consumes one slot",
            ),
        )
    elif facts.domain == "bank":
        if facts.cost_matrix is None:
            raise ValueError("bank design requires card-level costs")
        hold_undo = actions["hold"].undo_seconds
        if hold_undo is None:
            raise ValueError("bank redesign requires a reversible card hold")
        band = optimal_rule(facts.cost_matrix, facts.actions_on_score, "refer", ("clear", "hold"))
        lower, upper = band.intervals[0].upper, band.intervals[-1].lower
        boundary = {
            "permitted": f"clear below {lower}; hold above {upper}; refer between, including ties",
            "forbidden": "account closure by agent",
            "approval": f"{reviewer} between edges and at ties; {approver} for account closure",
            "reversal": f"customer lifts hold in {hold_undo / 60:g} constructed minutes; closure takes days",
            "ceiling": f"clearing risks {facts.cost_matrix.entries['clear'][1]} dollars per card; no autonomous closure",
            "logging": logging,
        }
        classes = (
            DecisionClass(
                name="card clearing",
                action="clear",
                level=4,
                above_role="reviewer",
                condition=f"risk below {lower}; card-level rule and current status checked at execution",
            ),
            DecisionClass(
                name="analyst review",
                action="refer",
                level=3,
                above_role="reviewer",
                condition=f"risk from {lower} through {upper}, including both edges; {reviewer} decides",
            ),
            DecisionClass(
                name="card hold",
                action="hold",
                level=4,
                above_role="reviewer",
                condition=f"risk above {upper}; current status rechecked; customer can lift the hold",
            ),
            DecisionClass(
                name="account closure",
                action="account.close",
                level=2,
                above_role="named_approver",
                condition=f"only the {approver} decides closure",
            ),
        )
        caps = ()
    else:
        raise ValueError("redesign requires triage, hospital or bank facts")
    return {
        "domain": facts.domain,
        "form": "redesign",
        "boundary": boundary,
        "classes": classes,
        "age_limits": {
            source.name: 8 if facts.domain == "triage" and source.name == "alarm.read" else 1800
            for source in facts.sources
            if not source.version_checked
        },
        "version_checks": tuple(source.name for source in facts.sources if source.version_checked),
        "rule": band,
        "decision_unit": "batch:slots" if facts.domain == "hospital" else "request",
        "recheck_at_execution": True,
        "caps": caps,
        "cells": table_cells,
        "label_windows": facts.label_windows,
    }


def _validate_redesign_answers(packet: DecisionPacket, facts: DomainFacts) -> None:
    for field, expected in _redesign_fields(facts).items():
        if getattr(packet, field) != expected:
            raise ValueError(f"{facts.domain} redesign {field} contradicts domain facts")
    if clock(packet, facts) or coupling(packet, facts) or rule(packet, facts):
        raise ValueError(f"{facts.domain} redesign contradicts its clocks, resources or costs")


def validate_redesign(
    packet: DecisionPacket,
    facts: DomainFacts,
    requirement: BaseModel,
) -> None:
    """Validate all thirteen fields before publication, including real launch rows."""
    _validate_redesign_answers(packet, facts)
    if packet.requirement != map_requirement_roles(strip(requirement), facts.roles) or structure(packet, facts):
        raise ValueError(f"{facts.domain} redesign requirement contradicts supplied launch rows")


def redesign(facts: DomainFacts, copied: DecisionPacket) -> DecisionPacket:
    if copied.domain != facts.domain or copied.form != "copy":
        raise ValueError("redesign requires the matching copied packet")
    packet = DecisionPacket.model_validate(
        {**_redesign_fields(facts), "requirement": copied.requirement}
    )
    _validate_redesign_answers(packet, facts)
    return packet


def packets(requirement: BaseModel) -> tuple[tuple[DecisionPacket, DomainFacts], ...]:
    refund = refund_packet(requirement)
    rows = [(refund, domain_facts("refund"))]
    for domain in ("triage", "hospital", "bank"):
        facts = domain_facts(domain)
        copied = copy_refund(facts, refund)
        rows += [(copied, facts), (redesign(facts, copied), facts)]
    return tuple(rows)


def validate_packet_inventory(rows: tuple[tuple[DecisionPacket, DomainFacts], ...]) -> None:
    required = (("refund", "refund"),) + tuple(
        (domain, form) for domain in DOMAINS[1:] for form in ("copy", "redesign")
    )
    require_inventory(((packet.domain, packet.form) for packet, _ in rows), required, "packet")
    for packet, facts in rows:
        if packet.domain != facts.domain:
            raise ValueError(
                f"packet facts mismatch: {packet.domain}/{packet.form} paired with {facts.domain}"
            )


def constructed_day() -> HospitalDay:
    urgent = {5, 8, 10, 13, 15, 17}
    return HospitalDay(
        patients=tuple(
            Patient(
                patient_id=f"patient-{n:02d}",
                urgency="urgent" if n in urgent else "routine",
                deadline_day=2 if n in urgent else 7,
                discharged_hour=9 + (n - 1) // 3,
            )
            for n in range(1, 19)
        ),
        slots_per_day=3,
        days=7,
        batch_hour=17,
        basis=GIVEN,
    )


def write_data(destination: Path) -> None:
    """Only independent given facts and figure inputs; no synthetic LaunchRow."""
    facts_rows = tuple(domain_facts(domain) for domain in DOMAINS)
    require_inventory((facts.domain for facts in facts_rows), DOMAINS, "facts")
    outputs = {}
    for facts in facts_rows:
        domain = facts.domain
        column = cells(facts)
        require_inventory((cell.question for cell in column), QUESTIONS, f"{domain} table question")
        outputs[f"{SLUG}-{domain}.facts.json"] = facts.model_dump_json(indent=2) + "\n"
        outputs[f"{SLUG}-{domain}.cells.json"] = (
            json.dumps([c.model_dump(mode="json") for c in column], indent=2) + "\n"
        )
    outputs[f"{SLUG}-hospital-day.json"] = constructed_day().model_dump_json(indent=2) + "\n"
    outputs[f"{SLUG}-clocks.json"] = (
        json.dumps(
            {
                facts.domain: [c.model_dump(mode="json") for c in facts.clocks]
                for facts in facts_rows
            },
            indent=2,
        )
        + "\n"
    )
    destination.mkdir(parents=True, exist_ok=True)
    for name, content in outputs.items():
        (destination / name).write_text(content)


def write_packets(requirement: BaseModel, destination: Path) -> None:
    """Run only with the actual chapter-39 requirement supplied at the host."""
    rows = tuple(packets(requirement))
    validate_packet_inventory(rows)
    outputs = {}
    # All packets, including copies, must pass preflight before publication.
    for packet, facts in rows:
        require_inventory(packet.__dict__, DecisionPacket.model_fields, "packet field")
        packet = DecisionPacket.model_validate(packet.model_dump())
        if packet.form == "redesign":
            validate_redesign(packet, facts, requirement)
        serialized = packet.model_dump_json(indent=2) + "\n"
        if packet.form in ("redesign", "refund"):
            outputs[f"{SLUG}-{packet.domain}.{packet.form}.json"] = serialized
    destination.mkdir(parents=True, exist_ok=True)
    for name, content in outputs.items():
        (destination / name).write_text(content)
