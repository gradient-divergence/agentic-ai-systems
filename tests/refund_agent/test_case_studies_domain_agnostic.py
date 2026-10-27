import ast
import json
from datetime import datetime, timedelta, timezone
from fractions import Fraction as F
from pathlib import Path
from types import SimpleNamespace

import pytest
from case_studies_domain_agnostic_fixtures import test_packets as packet_inputs
from case_studies_domain_agnostic_fixtures import test_requirement as requirement_input
from pydantic import BaseModel, ValidationError

from demos.supervised_learning_demo import expected_costs
from refund_agent import contracts as c
from refund_agent.agent import Agent
from refund_agent.data.generate import Dataset
from refund_agent.tools import Tools
from refund_agent.transfer import case_studies_domain_agnostic_check as checks
from refund_agent.transfer import case_studies_domain_agnostic_data as packet_data
from refund_agent.transfer.case_studies_domain_agnostic_data import (
    constructed_day,
    domain_facts,
    refund_packet,
    write_data,
)
from refund_agent.transfer.case_studies_domain_agnostic_domains import (
    alarm_numbers,
    holds_per_fraud,
    hospital_day,
    hospital_orders,
    order_balance,
)
from refund_agent.transfer.case_studies_domain_agnostic_facts import CostMatrix, load_facts
from refund_agent.transfer.case_studies_domain_agnostic_packet import (
    Interval,
    Rule,
    copy_refund,
    dump_packet,
    load_packet,
    strip,
)


@pytest.mark.parametrize(
    "arrival,service,seconds,cost",
    [(9, 15, 600, F(29)), (14, 15, 3600, F(154))],
)
def test_bank_facts_follow_the_real_queue_inputs(monkeypatch, arrival, service, seconds, cost):
    from refund_agent.oversight.human_oversight_risk_queueing import mm1

    def changed_inputs(lam, mu):
        assert (lam, mu) == (12, 15)
        return mm1(arrival, service)

    monkeypatch.setattr(packet_data, "mm1", changed_inputs, raising=False)
    facts = domain_facts("bank")
    assert next(c.seconds for c in facts.clocks if c.kind == "review") == seconds
    assert facts.cost_matrix.entries["refer"] == (F(4), cost)
    assert next(a.largest_loss for a in facts.actions if a.name == "refer") == cost * 100


@pytest.mark.parametrize("arrival", [15, 18])
def test_bank_facts_refuse_an_unstable_real_queue(monkeypatch, arrival):
    from refund_agent.oversight.human_oversight_risk_queueing import mm1

    monkeypatch.setattr(packet_data, "mm1", lambda lam, mu: mm1(arrival, mu), raising=False)
    with pytest.raises(ValueError, match="bank review queue is unstable"):
        domain_facts("bank")


@pytest.mark.parametrize("domain", ["refund", "triage", "hospital", "bank"])
def test_transfer_table_preserves_the_launch_requirements_questions(domain):
    requirement = requirement_input()
    expected = tuple(row.question for row in requirement.rows)
    assert tuple(cell.question for cell in packet_data.cells(domain_facts(domain))) == expected
    stored = json.loads((packet_data.DATA / f"{packet_data.SLUG}-{domain}.cells.json").read_text())
    assert tuple(cell["question"] for cell in stored) == expected


def test_refund_answers_are_design_not_running_score():
    packet = refund_packet(requirement_input())
    assert packet.age_limits == {"order.lookup": 1800, "shipment.status": 1800}
    assert packet.version_checks == ("policy.retrieve",)
    assert packet.rule.status == "designed-held"
    assert packet.rule.intervals[0].upper == F(400, 4999)
    assert packet.rule.actions_alone == ("pay",)
    assert packet.caps[0].value == 5000 and not packet.caps[0].inclusive
    assert "including denials" in packet.boundary["approval"]
    rows = strip(requirement_input())
    assert len(rows) == 7 and all("launched_bundle_fingerprint" not in row for row in rows)
    assert all("gate_day" not in row and "signed_by" not in row for row in rows)


def test_copies_fail_only_the_planted_independent_check():
    refund = refund_packet(requirement_input())
    for packet, facts in packet_inputs():
        failures = [
            f
            for checker in (checks.clock, checks.coupling, checks.rule)
            for f in checker(packet, facts)
        ]
        assert {f.check for f in failures} == set(
            facts.planted_checks if packet.form == "copy" else ()
        )
        if packet.form == "copy":
            assert copy_refund(facts, refund) == packet
        if packet.domain == "triage" and packet.form == "copy":
            assert [f.item for f in failures] == ["alarm.read"]


@pytest.mark.parametrize("domain", ["triage", "hospital", "bank"])
def test_copy_maps_signers_and_reversal_owners_to_domain_titles(domain):
    from refund_agent.launch.capstone_blueprint_requirement import refund_requirement

    facts = domain_facts(domain)
    refund = refund_packet(refund_requirement("B6"))
    copied = copy_refund(facts, refund)
    for original, mapped in zip(refund.requirement, copied.requirement):
        for before, after in zip(original["records"], mapped["records"]):
            assert after["signer_role"] == facts.roles[before["signer_role"]]
        for before, after in zip(original["reversals"], mapped["reversals"]):
            assert after["owner_roles"] == [facts.roles[role] for role in before["owner_roles"]]
    assert not checks.structure(copied, facts)
    assert not checks.structure(packet_data.redesign(facts, copied), facts)
    assert refund.requirement == strip(refund_requirement("B6"))

    raw = copied.model_dump()
    raw["requirement"][0]["records"][0]["signer_role"] = "unlisted signer"
    assert checks.structure(type(copied).model_validate(raw), facts)


@pytest.mark.parametrize("existing", [False, True])
@pytest.mark.parametrize("index", [1, 6])
def test_packet_publication_rejects_unexpected_fields_before_any_write(
    tmp_path, monkeypatch, existing, index
):
    rows = list(packet_inputs())
    packet, facts = rows[index]
    rows[index] = (packet.model_copy(update={"unexpected_field": "unrequested"}), facts)
    monkeypatch.setattr(packet_data, "packets", lambda _: tuple(rows))
    destination = tmp_path / "publication"
    if existing:
        destination.mkdir()
        (destination / f"{packet_data.SLUG}-refund.refund.json").write_bytes(b"existing bytes\n")
    with pytest.raises(ValueError, match="packet field inventory"):
        packet_data.write_packets(requirement_input(), destination)
    if existing:
        assert {p.name: p.read_bytes() for p in destination.iterdir()} == {
            f"{packet_data.SLUG}-refund.refund.json": b"existing bytes\n"
        }
    else:
        assert not destination.exists()


def test_last_packet_serialization_failure_leaves_no_publication(
    tmp_path, monkeypatch
):
    from refund_agent.transfer.case_studies_domain_agnostic_packet import DecisionPacket

    serialize = DecisionPacket.model_dump_json

    def refuse_last(self, **kwargs):
        if self.domain == "bank" and self.form == "redesign":
            raise ValueError("last packet cannot serialize")
        return serialize(self, **kwargs)

    monkeypatch.setattr(DecisionPacket, "model_dump_json", refuse_last)
    destination = tmp_path / "publication"
    with pytest.raises(ValueError, match="last packet cannot serialize"):
        packet_data.write_packets(requirement_input(), destination)
    assert not destination.exists()


def test_alarm_and_shipment_arithmetic():
    import math

    values = alarm_numbers(math.log(2) * 6, 0.01)
    assert values["chance_20_minutes"] == pytest.approx(0.75)
    assert values["chance_30_minutes"] == pytest.approx(0.875)
    assert values["budget_seconds"] == pytest.approx(8.69974, abs=0.00001)
    assert checks.stale_chance(0.02, 0.5) == pytest.approx(0.00995016625)
    assert checks.budget_hours(0.02, 0.01) * 60 == pytest.approx(30.15100756)


def test_clock_uses_public_budget_functions_and_skips_versions(monkeypatch):
    packet, facts = packet_inputs()[1]
    calls = []
    chance, budget = checks.stale_chance, checks.budget_hours
    monkeypatch.setattr(
        checks, "stale_chance", lambda rate, age: (calls.append((rate, age)), chance(rate, age))[1]
    )
    monkeypatch.setattr(
        checks,
        "budget_hours",
        lambda rate, tolerance: (calls.append((rate, tolerance)), budget(rate, tolerance))[1],
    )
    failures = checks.clock(packet, facts)
    assert len(calls) == 3 and len(failures) == 1
    assert F(failures[0].numbers["chance"]) == F(7, 8)
    assert all(rate is not None for rate, _ in calls)
    raw = packet.model_dump()
    raw["age_limits"].pop("alarm.read")
    assert checks.clock(type(packet)(**raw), facts)[0].reason == "age limit missing"
    raw = packet.model_dump()
    raw["version_checks"] = ()
    assert any(f.item == "runbook.retrieve" for f in checks.clock(type(packet)(**raw), facts))


def test_limit_vs_allocation_and_execution_recheck():
    refund, refund_facts = packet_inputs()[0]
    shared = refund_facts.shared_resources[0]
    assert (shared.scope, shared.requirement, shared.capacity) == ("order", "limit", F(4000))
    assert shared.actions == ("refund.execute",)
    assert not checks.coupling(refund, refund_facts)
    assert checks.coupling(
        type(refund)(**{**refund.model_dump(), "recheck_at_execution": False}), refund_facts
    )
    hospital, facts = packet_inputs()[3]
    assert checks.coupling(hospital, facts)
    for unit in ("batch:slots", "reserve:slots"):
        good = type(hospital)(**{**hospital.model_dump(), "decision_unit": unit})
        assert not checks.coupling(good, facts)
        assert checks.coupling(
            type(good)(**{**good.model_dump(), "recheck_at_execution": False}), facts
        )
    other = type(hospital)(**{**hospital.model_dump(), "decision_unit": "batch:another_resource"})
    assert checks.coupling(other, facts)
    inactive = type(hospital)(**{**hospital.model_dump(), "classes": refund.classes})
    assert not checks.coupling(inactive, facts)


def test_two_requests_share_the_actual_execution_gate():
    at = datetime(2026, 10, 1, 12, tzinfo=timezone.utc)
    requests = [
        c.Request(
            request_id=f"request-{n}",
            requester_id="owner",
            order_reference="order",
            at=at + timedelta(seconds=n),
            amount_cents=3000,
            reason="damaged",
        )
        for n in (1, 2)
    ]
    policy = c.Policy(
        version="2026-09-30",
        effective_at=at - timedelta(days=1),
        return_days=90,
        cap_cents=100000,
        text="Constructed refund policy.",
    )
    data = Dataset(
        requests=requests,
        orders=[
            c.Order(
                order_id="order",
                reference="order",
                payment_id="original",
                paid_cents=4000,
                purchased_at=at - timedelta(days=2),
                observed_at=at,
                items=[c.Item(name="book", quantity=1, unit_cents=4000)],
            )
        ],
        policies=[policy],
        correct=[],
        shipments=[
            c.Shipment(
                order_id="order",
                observed_at=at,
                events=[c.ShipmentEvent(status="delivered", at=at - timedelta(days=1))],
            )
        ],
        retrievals=[
            c.PolicySnapshot(request_id=r.request_id, version=policy.version, observed_at=at)
            for r in requests
        ],
    )
    tools = Tools(data)
    first, second = (Agent(tools).run(request) for request in requests)
    assert first.receipt is not None and first.receipt.amount_cents == 3000
    assert second.receipt is None and second.decision.reason == "amount_over_policy"
    assert second.decision.outcome == c.Outcome.DENY
    assert sum(p.amount_cents for p in tools.payments) == 3000
    assert 4000 - sum(p.amount_cents for p in tools.payments) == 1000


def test_cost_agrees_with_chapter5_without_reading_held_out_data():
    facts = domain_facts("refund")
    for risk in (F(0), F(1, 20), F(3, 25), F(1)):
        original = expected_costs([float(1 - risk), float(risk)], [[0, 4, 20], [49.99, 4, 0]])
        for action, value in zip(("pay", "refer"), original):
            assert float(checks.expected_cost(facts.cost_matrix, action, risk)) == pytest.approx(
                value, abs=1e-12
            )


def test_bank_exact_failure_intervals_and_costs():
    packet, facts = packet_inputs()[5]
    failures = checks.rule(packet, facts)
    assert len(failures) == 2
    assert [(F(f.numbers["lower"]), F(f.numbers["upper"])) for f in failures] == [
        (F(2, 175), F(400, 4999)),
        (F(1, 6), F(1)),
    ]
    assert failures[0].numbers["left_closed"] == "True"
    assert failures[0].numbers["right_closed"] == "False"
    assert failures[1].numbers["left_closed"] == "False"
    assert failures[1].numbers["right_closed"] == "True"
    matrix = facts.cost_matrix
    assert checks.expected_cost(matrix, "clear", F(1, 20)) == 20
    assert checks.expected_cost(matrix, "refer", F(1, 20)) == F(13, 2)
    assert checks.expected_cost(matrix, "refer", F(1, 2)) == 29
    assert checks.expected_cost(matrix, "hold", F(1, 2)) == 9
    assert float(checks.expected_cost(matrix, "refer", F(4, 37))) == pytest.approx(9.4054054054)
    assert float(checks.expected_cost(matrix, "hold", F(4, 37))) == pytest.approx(12.9189189189)
    assert holds_per_fraud(F(4, 37)) == F(33, 4)
    assert facts.label_windows["card_fraud"] == 60 * 86400
    assert packet.label_windows == facts.label_windows
    assert any("card" in note and "60 days" in note for note in facts.notes)


def test_rule_checks_a_sub_grid_interval():
    packet, facts = packet_inputs()[6]
    edge, end = F(2, 175), F(2, 175) + F(1, 1_000_000)
    configured = Rule(
        intervals=(
            Interval(lower=F(0), upper=edge, action="clear"),
            Interval(lower=edge, upper=end, action="clear"),
            Interval(lower=end, upper=F(1, 6), action="refer", right_closed=True),
            Interval(
                lower=F(1, 6), upper=F(1), action="hold", left_closed=False, right_closed=True
            ),
        ),
        actions_alone=("clear", "hold"),
        referral="refer",
        status="constructed",
    )
    wrong = type(packet)(**{**packet.model_dump(), "rule": configured})
    failures = checks.rule(wrong, facts)
    assert len(failures) == 1
    assert (F(failures[0].numbers["lower"]), F(failures[0].numbers["upper"])) == (edge, end)


def test_tie_and_endpoint_are_not_hidden_in_intervals():
    packet, facts = packet_inputs()[6]
    band = packet.rule
    assert band.action_at(F(2, 175)) == "refer"
    assert band.action_at(F(1, 6)) == "refer"
    assert band.action_at(F(1)) == "hold" and band.action_at(F(0)) == "clear"
    wrong = Rule(
        intervals=(
            band.intervals[0],
            Interval(lower=F(2, 175), upper=F(1, 6), action="refer"),
            Interval(lower=F(1, 6), upper=F(1), action="hold", right_closed=True),
        ),
        actions_alone=("clear", "hold"),
        referral="refer",
        status="constructed",
    )
    failures = checks.rule(type(packet)(**{**packet.model_dump(), "rule": wrong}), facts)
    assert len(failures) == 1
    assert F(failures[0].numbers["lower"]) == F(failures[0].numbers["upper"]) == F(1, 6)


def test_bank_packet_cannot_choose_its_own_referral_at_the_tie():
    packet, facts = packet_inputs()[6]
    assert checks.check(packet, facts).verdict == "pass"
    wrong_rule = Rule(
        intervals=(
            Interval(lower=F(0), upper=F(2, 175), action="clear", right_closed=True),
            Interval(
                lower=F(2, 175), upper=F(1, 6), action="refer",
                left_closed=False, right_closed=True,
            ),
            Interval(
                lower=F(1, 6), upper=F(1), action="hold",
                left_closed=False, right_closed=True,
            ),
        ),
        actions_alone=("clear", "hold"),
        referral="clear",
        status="constructed",
    )
    wrong = packet.model_copy(update={"rule": wrong_rule})
    failures = checks.rule(wrong, facts)
    assert any(
        f.item == "referral" and f.numbers == {"declared": "clear", "required": "refer"}
        for f in failures
    )
    risk_failures = [f for f in failures if f.item == "risk"]
    assert len(risk_failures) == 1
    numbers = risk_failures[0].numbers
    assert F(numbers["lower"]) == F(numbers["upper"]) == F(2, 175)
    assert numbers["left_closed"] == numbers["right_closed"] == "True"
    assert (numbers["copied"], numbers["cheapest"]) == ("clear", "refer")
    verdict = checks.check(wrong, facts)
    assert verdict.verdict == "hold" and verdict.failures == tuple(failures)


def test_rounded_refund_edge_creates_a_real_wrong_interval():
    packet, facts = packet_inputs()[0]
    rounded = Rule(
        intervals=(
            Interval(lower=F(0), upper=F(2, 25), action="pay"),
            Interval(lower=F(2, 25), upper=F(1), action="refer", right_closed=True),
        ),
        actions_alone=("pay",),
        referral="refer",
        status="designed-held",
    )
    failures = checks.rule(type(packet)(**{**packet.model_dump(), "rule": rounded}), facts)
    assert len(failures) == 1
    assert (F(failures[0].numbers["lower"]), F(failures[0].numbers["upper"])) == (
        F(2, 25),
        F(400, 4999),
    )
    assert not checks.rule(packet, facts)


def test_zero_slope_lines_and_excluded_action():
    facts = domain_facts("refund")
    matrix = CostMatrix(
        outcomes=("sound", "false"),
        entries={"pay": (F(4), F(4)), "refer": (F(4), F(4)), "deny": (F(0), F(0))},
        unit="dollars",
        basis="constructed constant lines",
    )
    band = checks.optimal_rule(matrix, facts.actions_on_score, "refer", ())
    assert band.action_at(F(0)) == band.action_at(F(1, 2)) == band.action_at(F(1)) == "refer"
    assert len(band.intervals) == 1


def test_hospital_day_and_seeded_orders():
    day = constructed_day()
    request, batch = hospital_day(day, "request"), hospital_day(day, "batch")
    assert [n for n, p in enumerate(day.patients, 1) if p.urgency == "urgent"] == [
        5,
        8,
        10,
        13,
        15,
        17,
    ]
    assert sum(b.day is not None for b in request if b.urgency == "urgent") == 1
    assert sum(b.day is None for b in request if b.urgency == "urgent") == 5
    assert sum(b.day is not None for b in request if b.urgency == "routine") == 12
    assert all(b.day is not None for b in batch)
    assert {b.day for b in batch if b.urgency == "routine"} == {3, 4, 5, 6}
    assert batch[0].notification_hour - day.patients[0].discharged_hour == 8
    assert len({(b.day, b.slot) for b in batch}) == 18
    result = hospital_orders(day)
    assert abs(result["mean"] - 2) < 0.05
    assert result["exact_mean"] == 2
    assert result["exact_all_probability"] == F(1, 18564)


def test_transport_round_trip(tmp_path):
    write_data(tmp_path)
    assert load_facts(tmp_path / "case-studies-domain-agnostic-bank.facts.json") == domain_facts(
        "bank"
    )
    for packet, _ in packet_inputs():
        path = tmp_path / f"{packet.domain}.{packet.form}.json"
        dump_packet(packet, path)
        assert load_packet(path) == packet


@pytest.fixture
def chapter39_public_exports():
    # companion/refund_agent/launch/__init__.py
    # Source commit: 4472e45cd1fae017e855dff2e909f4914f37961f
    return {"refund_agent.launch": {"LaunchRequirement", "LaunchRow", "validate", "validate_row"}}


@pytest.mark.parametrize(
    "source",
    [
        "refund_agent/transfer/case_studies_domain_agnostic_check.py",
        "demos/case_studies_domain_agnostic_demo.py",
    ],
)
def test_launch_imports_match_chapter39_public_exports(source, chapter39_public_exports):
    tree = ast.parse((Path(__file__).resolve().parents[2] / source).read_text())
    modules, names = [], set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith(
            "refund_agent.launch"
        ):
            modules.append(node.module)
            names.update(alias.name for alias in node.names)
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "import_module"
        ):
            modules.append(ast.literal_eval(node.args[0]))
        elif (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "launch"
        ):
            names.add(node.attr)
    assert len(modules) == 1
    assert modules[0] in chapter39_public_exports, f"unexported chapter 39 module: {modules[0]}"
    assert names and names <= chapter39_public_exports[modules[0]]


def test_structure_delegates_and_preserves_external_reasons(monkeypatch):
    class ConstructedRow(BaseModel):
        question: str

    received = []

    def validator(row, roles):
        received.append((row.question, roles))
        return ["external row refusal"] if len(received) == 1 else []

    # A test-owned contract double proves consumer wiring, not producer semantics.
    monkeypatch.setattr(
        checks,
        "import_module",
        lambda name: SimpleNamespace(LaunchRow=ConstructedRow, validate_row=validator),
    )
    packet, facts = packet_inputs()[0]
    verdict = checks.check(packet, facts)
    assert verdict.verdict == "hold" and len(verdict.failures) == 1
    assert verdict.failures[0].reason == "external row refusal"
    assert len(received) == 7 and all(roles == facts.roles for _, roles in received)


def test_full_verdict_cannot_bypass_an_absent_predecessor(monkeypatch):
    def missing(name):
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(checks, "import_module", missing)
    packet, facts = packet_inputs()[0]
    with pytest.raises(ModuleNotFoundError, match=r"^refund_agent\.launch$"):
        checks.check(packet, facts)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1])
def test_invalid_age_refused(bad):
    packet, _ = packet_inputs()[0]
    raw = packet.model_dump()
    raw["age_limits"]["order.lookup"] = bad
    with pytest.raises((ValidationError, ValueError)):
        type(packet)(**raw)


@pytest.mark.parametrize("bad", [-1, F(101, 100)])
def test_invalid_risk_refused(bad):
    with pytest.raises(ValueError):
        checks.expected_cost(domain_facts("bank").cost_matrix, "clear", bad)


def test_rule_cannot_name_a_forbidden_scored_action():
    packet, facts = packet_inputs()[0]
    wrong = Rule(
        intervals=(Interval(lower=F(0), upper=F(1), action="deny", right_closed=True),),
        actions_alone=("deny",),
        referral="refer",
        status="constructed",
    )
    failures = checks.rule(type(packet)(**{**packet.model_dump(), "rule": wrong}), facts)
    assert [f.item for f in failures] == ["boundary"]


def test_insurer_has_exactly_one_known_bad_answer():
    from refund_agent.transfer.case_studies_domain_agnostic_data import EXERCISES

    fixture = json.loads((EXERCISES / "insurer_05.json").read_text())
    assert fixture["label_window_days"] == 30
    assert fixture["facts"]["all_findings_mature_by_days"] == 90
    assert fixture["facts"]["fraud_findings_after_30_days"] == 0.4
    given = fixture["other_answers"]
    assert checks.stale_chance(
        float(F(given["document_rate_per_hour"])), given["document_limit_seconds"] / 3600
    ) < float(F(given["tolerance"]))
    matrix = CostMatrix(
        outcomes=("sound", "false"),
        entries={a: tuple(F(v) for v in pair) for a, pair in given["costs_dollars"].items()},
        unit="dollars",
        basis="exercise 5",
    )
    assert checks.optimal_rule(matrix, ("pay", "refer"), "refer", ("pay",)).intervals[0].upper == F(
        given["pay_below"]
    )
    assert given["ceiling_dimension"] == "money" and given["above_ceiling_role"] == "named_approver"
    assert given["resource_requirement"] == "limit" and given["recheck_at_execution"]


def test_redesigns_keep_human_authority_and_known_loss_limits():
    for packet, facts in packet_inputs():
        assert set(facts.role_map.values()) <= {action.name for action in facts.actions}
        if packet.form == "redesign":
            assert any(c.level == 2 and c.above_role == "named_approver" for c in packet.classes)
            if packet.domain == "bank":
                closure = next(c for c in packet.classes if c.action == "account.close")
                assert closure.level == 2
                assert (
                    next(a for a in facts.actions if a.name == "account.close").largest_loss is None
                )
                assert "account.close" not in facts.actions_on_score
    refund_facts = domain_facts("refund")
    assert "6000 order-hours" in refund_facts.sources[0].basis
    assert refund_facts.sources[0].rate_per_hour < 0.001
    assert domain_facts("triage").label_windows["needless_restart"] == 86400


@pytest.mark.parametrize("domain", ["triage", "hospital", "bank"])
def test_redesign_refills_every_boundary_answer_and_decision_condition(domain):
    refund, _ = packet_inputs()[0]
    packet, _ = next(
        (p, f) for p, f in packet_inputs() if p.domain == domain and p.form == "redesign"
    )
    for field in ("permitted", "forbidden", "approval", "reversal", "ceiling"):
        assert packet.boundary[field] != refund.boundary[field], (domain, field)
    answers = " ".join(packet.boundary.values()) + " ".join(c.condition for c in packet.classes)
    assert not any(text in answers for text in ("refund", "payments stand", "5000 cents alone"))
    autonomous = [c.condition for c in packet.classes if c.level == 4]
    if domain == "triage":
        assert len(autonomous) == 1 and "active alarm.probe" in autonomous[0]
        assert "fewer than 50" in autonomous[0]
        assert "dropped calls cannot be restored" in packet.boundary["reversal"]
    elif domain == "hospital":
        assert len(autonomous) == 1 and "daily allocation" in autonomous[0]
        assert "rechecked at execution" in autonomous[0]
        assert "missed window cannot be restored" in packet.boundary["reversal"]
    else:
        assert len(autonomous) == 2
        assert any("risk below 2/175" in condition for condition in autonomous)
        assert any("risk above 1/6" in condition for condition in autonomous)
        assert "400 dollars" in packet.boundary["ceiling"]


@pytest.mark.parametrize("domain", ["triage", "hospital", "bank"])
@pytest.mark.parametrize(
    "part", ["permitted", "forbidden", "approval", "reversal", "ceiling", "logging", "condition"]
)
def test_refund_answer_refused_before_any_packet_is_published(
    tmp_path, monkeypatch, domain, part
):
    packet, facts = next(
        (p, f) for p, f in packet_inputs() if p.domain == domain and p.form == "redesign"
    )
    refund, _ = packet_inputs()[0]
    if part == "condition":
        wrong = packet.model_copy(
            update={
                "classes": (
                    packet.classes[0].model_copy(update={"condition": refund.classes[0].condition}),
                    *packet.classes[1:],
                )
            }
        )
    else:
        answer = "refund audit" if part == "logging" else refund.boundary[part]
        wrong = packet.model_copy(update={"boundary": {**packet.boundary, part: answer}})
    rows = tuple(
        (wrong, facts) if p.domain == domain and p.form == "redesign" else (p, f)
        for p, f in packet_inputs()
    )
    monkeypatch.setattr(packet_data, "packets", lambda _: rows)
    destination = tmp_path / "publication"
    with pytest.raises(ValueError, match="redesign"):
        packet_data.write_packets(requirement_input(), destination)
    assert not destination.exists()


@pytest.mark.parametrize("sessions,allowed", [(49, True), (50, False), (51, False)])
def test_main_triage_packet_excludes_fifty_sessions(sessions, allowed):
    packet, _ = packet_inputs()[2]
    (cap,) = packet.caps
    assert cap.value == 50 and not cap.inclusive
    assert (sessions < cap.value or cap.inclusive and sessions == cap.value) == allowed
    assert "fewer than 50" in packet.boundary["ceiling"]
    reviewed = next(c for c in packet.classes if c.level == 3)
    assert "50 or more" in reviewed.condition


def test_b3_execution_rechecks_after_both_assessments_approved():
    row = order_balance()
    assert row["second_assessed"] == "approve", (
        "B3 must reach execution with an initially eligible proposal"
    )
    assert row["second_execution_attempts"] == 1
    assert row["second_execution_refused"] == 1
    assert row["second_assessment_reason"] == "eligible"
    assert row["first_payment_cents"] == 3000 and row["second_payment_cents"] is None
    assert row["remaining_cents"] == 1000 and row["second_reason"] == "amount_over_policy"


@pytest.mark.parametrize("domain", ["triage", "hospital", "bank"])
@pytest.mark.parametrize(
    "field",
    [
        "domain",
        "form",
        "boundary",
        "classes",
        "age_limits",
        "version_checks",
        "rule",
        "decision_unit",
        "recheck_at_execution",
        "caps",
        "requirement",
        "cells",
        "label_windows",
    ],
)
def test_every_redesign_field_is_guarded_before_publication(
    tmp_path, monkeypatch, domain, field
):
    packet, facts = next(
        (p, f) for p, f in packet_inputs() if p.domain == domain and p.form == "redesign"
    )
    raw = packet.model_dump()
    if field == "domain":
        raw[field] = "refund"
    elif field == "form":
        raw[field] = "refund"
    elif field == "boundary":
        raw[field]["logging"] = "refund audit"
    elif field == "classes":
        raw[field][0]["condition"] = "copied refund checks pass"
    elif field == "age_limits":
        raw[field]["unobserved.source"] = 0
    elif field == "version_checks":
        raw[field] = (*packet.version_checks, "unknown.policy")
    elif field == "rule":
        raw[field] = (
            packet.rule.model_copy(update={"actions_alone": ()})
            if packet.rule
            else packet_data.refund_rule()
        )
    elif field == "decision_unit":
        raw[field] = "batch:unknown_resource"
    elif field == "recheck_at_execution":
        raw[field] = False
    elif field == "caps":
        raw[field] = () if packet.caps else packet_inputs()[0][0].caps
    elif field == "requirement":
        raw[field] = ()
    elif field == "cells":
        raw[field][1]["answer"] = "Refund answer retained"
    elif field == "label_windows":
        raw[field] = {"found_false": 30 * 86400}
    wrong = type(packet).model_validate(raw)
    rows = tuple(
        (wrong, facts) if p.domain == domain and p.form == "redesign" else (p, f)
        for p, f in packet_inputs()
    )
    monkeypatch.setattr(packet_data, "packets", lambda _: rows)
    destination = tmp_path / "publication"
    with pytest.raises(ValueError, match="redesign"):
        packet_data.write_packets(requirement_input(), destination)
    assert not destination.exists()


def test_changed_triage_ceiling_cannot_publish_the_old_table(
    tmp_path, monkeypatch
):
    facts = domain_facts("triage")
    changed = facts.model_copy(
        update={
            "actions": tuple(
                action.model_copy(update={"largest_loss": F(40)})
                if action.name == "card.restart"
                else action
                for action in facts.actions
            ),
        }
    )
    copied = copy_refund(changed, refund_packet(requirement_input()))
    destination = tmp_path / "publication"
    with pytest.raises(ValueError, match="table.*facts"):
        packet = packet_data.redesign(changed, copied)
        # Prior code accepted a boundary saying 40 beside a table saying 50.
        assert "fewer than 40" in packet.boundary["ceiling"]
        monkeypatch.setattr(packet_data, "packets", lambda _: ((packet, changed),))
        packet_data.write_packets(requirement_input(), destination)
    assert not destination.exists()


def test_publication_requires_launch_rows_validated_against_domain_roles(tmp_path, monkeypatch):
    class OwnRow(BaseModel):
        question: str

    # The external contract owns the refusal; the real publication path must honor it.
    def validator(row, roles):
        assert roles["reviewer"] == "on-call engineer"
        return ["test-owned launch row refusal"]

    monkeypatch.setattr(
        checks,
        "import_module",
        lambda _: SimpleNamespace(LaunchRow=OwnRow, validate_row=validator),
    )
    destination = tmp_path / "publication"
    with pytest.raises(ValueError, match="redesign requirement"):
        packet_data.write_packets(requirement_input(), destination)
    assert not destination.exists()


def test_valid_redesign_publication_requires_the_real_launch_contract(tmp_path, monkeypatch):
    def missing(name):
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(checks, "import_module", missing)
    destination = tmp_path / "publication"
    with pytest.raises(ModuleNotFoundError, match=r"^refund_agent\.launch$"):
        packet_data.write_packets(requirement_input(), destination)
    assert not destination.exists()


def test_valid_redesigns_publish_after_the_external_contract_accepts(
    tmp_path
):
    packet_data.write_packets(requirement_input(), tmp_path)
    windows = {
        "triage": {"needless_restart": 86400},
        "hospital": {
            "urgent_followup": 172800,
            "routine_followup": 604800,
            "readmission": 2592000,
        },
        "bank": {"card_fraud": 5184000},
    }
    for domain, labels in windows.items():
        packet = load_packet(tmp_path / f"{packet_data.SLUG}-{domain}.redesign.json")
        assert packet.domain == domain and packet.form == "redesign"
        assert packet.label_windows == labels and packet.recheck_at_execution is True
        assert [r["question"] for r in packet.requirement] == [
            r["question"] for r in strip(requirement_input())
        ]
        assert not checks.structure(packet, domain_facts(domain))


@pytest.mark.parametrize(
    "domain,form",
    [
        ("refund", "refund"),
        ("triage", "copy"),
        ("triage", "redesign"),
        ("hospital", "copy"),
        ("hospital", "redesign"),
        ("bank", "copy"),
        ("bank", "redesign"),
    ],
)
@pytest.mark.parametrize("departure", ["missing", "duplicate"])
def test_packet_inventory_refuses_missing_or_duplicate_pair(
    tmp_path, monkeypatch, domain, form, departure
):
    rows = list(packet_inputs())
    index = next(i for i, (p, _) in enumerate(rows) if (p.domain, p.form) == (domain, form))
    if departure == "missing":
        rows.pop(index)
    else:
        rows.append(rows[index])
    monkeypatch.setattr(packet_data, "packets", lambda _: tuple(rows))
    destination = tmp_path / "publication"
    with pytest.raises(ValueError, match="packet inventory") as error:
        packet_data.write_packets(requirement_input(), destination)
    assert departure in str(error.value) and domain in str(error.value) and form in str(error.value)
    assert not destination.exists()


@pytest.mark.parametrize("domain", ["triage", "hospital", "bank"])
def test_packet_inventory_refuses_redesign_labelled_copy(
    tmp_path, monkeypatch, domain
):
    rows = tuple(
        (p.model_copy(update={"form": "copy"}), f)
        if p.domain == domain and p.form == "redesign"
        else (p, f)
        for p, f in packet_inputs()
    )
    monkeypatch.setattr(packet_data, "packets", lambda _: rows)
    destination = tmp_path / "publication"
    with pytest.raises(ValueError, match="packet inventory") as error:
        packet_data.write_packets(requirement_input(), destination)
    assert "missing" in str(error.value) and "duplicate" in str(error.value)
    assert domain in str(error.value) and "redesign" in str(error.value)
    assert not destination.exists()


def test_packet_inventory_refuses_extra_domain(tmp_path, monkeypatch):
    rows = packet_inputs()
    packet, facts = rows[-1]
    extra = packet.model_copy(update={"domain": "insurer"})
    monkeypatch.setattr(packet_data, "packets", lambda _: (*rows, (extra, facts)))
    destination = tmp_path / "publication"
    with pytest.raises(ValueError, match="packet inventory") as error:
        packet_data.write_packets(requirement_input(), destination)
    assert "extra" in str(error.value) and "insurer" in str(error.value)
    assert not destination.exists()


def test_packet_inventory_refuses_mismatched_facts(tmp_path, monkeypatch):
    rows = tuple(
        (p, domain_facts("refund")) if p.domain == "bank" and p.form == "copy" else (p, f)
        for p, f in packet_inputs()
    )
    monkeypatch.setattr(packet_data, "packets", lambda _: rows)
    destination = tmp_path / "publication"
    with pytest.raises(ValueError, match="packet.*facts.*bank.*refund"):
        packet_data.write_packets(requirement_input(), destination)
    assert not destination.exists()


@pytest.mark.parametrize("departure", ["duplicate", "extra"])
def test_data_inventory_refuses_changed_domain_before_writing(tmp_path, monkeypatch, departure):
    original = packet_data.domain_facts
    wrong = (
        original("refund")
        if departure == "duplicate"
        else original("bank").model_copy(update={"domain": "insurer"})
    )
    monkeypatch.setattr(
        packet_data, "domain_facts", lambda d: wrong if d == "bank" else original(d)
    )
    destination = tmp_path / "publication"
    with pytest.raises(ValueError, match="facts inventory") as error:
        write_data(destination)
    assert "missing" in str(error.value) and "bank" in str(error.value)
    assert departure in str(error.value)
    assert not destination.exists()


@pytest.mark.parametrize("departure", ["missing", "extra"])
def test_data_refuses_incomplete_answer_inventory_before_writing(tmp_path, monkeypatch, departure):
    answers = packet_data.ANSWERS["bank"]
    wrong = answers[:-1] if departure == "missing" else (*answers, "Unrequested answer")
    monkeypatch.setitem(packet_data.ANSWERS, "bank", wrong)
    destination = tmp_path / "publication"
    with pytest.raises(ValueError, match="bank table answer inventory") as error:
        write_data(destination)
    assert departure in str(error.value)
    assert not destination.exists()


@pytest.mark.parametrize("departure", ["missing", "duplicate", "extra"])
def test_data_refuses_incomplete_cell_inventory_before_writing(tmp_path, monkeypatch, departure):
    original = packet_data.cells

    def wrong(facts):
        rows = original(facts)
        if facts.domain != "bank":
            return rows
        if departure == "missing":
            return rows[:-1]
        if departure == "duplicate":
            return (*rows, rows[0])
        return (*rows, rows[0].model_copy(update={"question": "Unrequested question"}))

    monkeypatch.setattr(packet_data, "cells", wrong)
    destination = tmp_path / "publication"
    with pytest.raises(ValueError, match="bank table question inventory") as error:
        write_data(destination)
    assert departure in str(error.value)
    assert not destination.exists()


def test_data_refusal_keeps_existing_files_untouched(tmp_path, monkeypatch):
    write_data(tmp_path)
    (tmp_path / f"{packet_data.SLUG}-refund.facts.json").write_bytes(b"existing publication\n")
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    monkeypatch.setitem(packet_data.ANSWERS, "bank", packet_data.ANSWERS["bank"][:-1])
    with pytest.raises(ValueError, match="bank table answer inventory"):
        write_data(tmp_path)
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == before


@pytest.mark.parametrize("departure", ["missing", "extra", "duplicate"])
def test_packet_serializer_inventory_refused_before_writing(tmp_path, departure):
    packet = packet_inputs()[0][0]
    if departure == "missing":
        fields = {name: getattr(packet, name) for name in type(packet).model_fields}
        fields.pop("label_windows")
        wrong = type(packet).model_construct(**fields)
    elif departure == "extra":
        wrong = packet.model_copy(update={"unexpected_field": "unrequested"})
    else:
        wrong = packet.model_copy(update={"cells": (*packet.cells[:-1], packet.cells[0])})
    destination = tmp_path / "packet.json"
    with pytest.raises(ValueError, match="packet field inventory|questions must be distinct"):
        dump_packet(wrong, destination)
    assert not destination.exists()
