import json
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from demos import case_studies_domain_agnostic_capture as capture
from demos import case_studies_domain_agnostic_demo as demo
from demos import regions as _regions
from demos.case_studies_domain_agnostic_capture import INDEPENDENT, capture_chapter
from refund_agent.launch.capstone_blueprint_requirement import refund_requirement
from refund_agent.transfer.case_studies_domain_agnostic_data import (
    DATA,
    QUESTIONS,
    SLUG,
    packets,
    write_data,
)

ROOT = Path(__file__).resolve().parents[1]


def test_bank_verdicts_follow_changed_real_queue_inputs(monkeypatch, capsys):
    from refund_agent.oversight.human_oversight_risk_queueing import mm1
    from refund_agent.transfer import case_studies_domain_agnostic_data as data

    requirement = refund_requirement("B6")
    demo.print_verdicts(requirement)
    before = capsys.readouterr().out
    monkeypatch.setattr(data, "mm1", lambda lam, mu: mm1(9, mu), raising=False)
    demo.print_verdicts(requirement)
    after = capsys.readouterr().out
    assert after != before
    assert "exact 4/375..400/4999" in after
    assert "exact 2/7..1" in after
    assert "bank redesign pass" in after


def test_verdicts_refuse_an_unstable_queue_before_printing(monkeypatch, capsys):
    from refund_agent.oversight.human_oversight_risk_queueing import mm1
    from refund_agent.transfer import case_studies_domain_agnostic_data as data

    monkeypatch.setattr(data, "mm1", lambda lam, mu: mm1(18, mu), raising=False)
    with pytest.raises(ValueError, match="bank review queue is unstable"):
        demo.print_verdicts(refund_requirement("B6"))
    assert capsys.readouterr().out == ""


@pytest.fixture(scope="module")
def captured(tmp_path_factory):
    return capture_chapter(tmp_path_factory.mktemp("case-studies-domain-agnostic"), ids=INDEPENDENT)


def test_independent_outputs_reproduce_byte_for_byte(captured):
    assert len(captured) == 6
    for name, content in captured.items():
        assert content == (ROOT / f"book_outputs/_{name}.txt").read_text()


def test_every_expected_numeric_result_is_visible(captured):
    triage = captured[f"fnd.{SLUG}.triage-01"]
    assert "0.750000" in triage and "0.875000" in triage
    assert "8.699742 seconds" in triage and "30.151008 minutes" in triage
    hospital = captured[f"fnd.{SLUG}.hospital-01"]
    assert "request urgent in time 1/6; referred 5" in hospital
    assert "batch urgent in time 6/6; referred 0" in hospital
    assert "request routine in time 12/12" in hospital and "batch routine in time 12/12" in hospital
    assert "1/18564" in hospital and "seed: 7" in hospital
    bank = captured[f"fnd.{SLUG}.bank-01"]
    assert "2/175 = 0.0114" in bank and "1/6 = 0.1667" in bank
    assert "copy clear $20.00; band refer $6.50" in bank
    assert "copy refer $29.00; band hold $9.00" in bank
    assert "$43.24/$9.41/$12.92" in bank and "8.25" in bank
    assert "card-level" in bank and "60 days" in bank
    assert "M/M/1 review-open time: 20 minutes" in bank
    balance = captured[f"fnd.{SLUG}.balance-01"]
    assert "First payment: 3000 cents" in balance
    assert "outcome: refer" in balance and "amount_over_policy" in balance
    assert "Second assessment: approve" in balance and "refused for balance: 1" in balance
    assert "remaining: 1000 cents" in balance


def test_seven_table_rows_have_twelve_word_answers(captured):
    output = captured[f"fnd.{SLUG}.packets-01"]
    assert len(output.splitlines()) == 9
    for index, question in enumerate(QUESTIONS):
        assert question in output
        for domain in ("refund", "triage", "hospital", "bank"):
            rows = json.loads((DATA / f"{SLUG}-{domain}.cells.json").read_text())
            assert len(rows[index]["answer"].split()) <= 12
            if domain != "refund":
                expected = (
                    "same"
                    if index == 4
                    else "redesigned"
                    if index == 1 or domain == "bank" and index == 5
                    else "refilled"
                )
                assert rows[index]["kind"] == expected


def test_table_refuses_a_refund_answer_before_printing(tmp_path, monkeypatch, capsys):
    write_data(tmp_path)
    source = tmp_path / f"{SLUG}-triage.cells.json"
    rows = json.loads(source.read_text())
    rows[1]["answer"] = "Under $50 alone; reads under 30 minutes; designed score below 0.0800"
    source.write_text(json.dumps(rows))
    monkeypatch.setattr(demo, "DATA", tmp_path)
    with pytest.raises(ValueError, match="triage"):
        demo.print_packets()
    assert capsys.readouterr().out == ""


def test_table_refuses_changed_facts_before_printing(tmp_path, monkeypatch, capsys):
    write_data(tmp_path)
    source = tmp_path / f"{SLUG}-triage.facts.json"
    facts = json.loads(source.read_text())
    restart = next(action for action in facts["actions"] if action["name"] == "card.restart")
    restart["largest_loss"] = "40"
    source.write_text(json.dumps(facts))
    monkeypatch.setattr(demo, "DATA", tmp_path)
    with pytest.raises(ValueError, match="table.*facts"):
        demo.print_packets()
    assert capsys.readouterr().out == ""


def test_data_reproduces_without_predecessor_substitutes(tmp_path):
    write_data(tmp_path)
    for path in tmp_path.iterdir():
        assert path.read_bytes() == (DATA / path.name).read_bytes()
    assert not list(tmp_path.glob("*.redesign.json"))


def test_printed_listing_runs_alone_and_meets_both_limits():
    extractor = vars(_regions)
    source = ROOT / "refund_agent/transfer/case_studies_domain_agnostic_check.py"
    (snippet,) = extractor["extract_snippets_from_text"](source.read_text(), source)
    assert snippet.snippet_id == f"fnd.{SLUG}.check-01"
    assert len(snippet.content.splitlines()) <= 25
    assert max(map(len, snippet.content.splitlines())) <= 79
    namespace = {"__name__": "case_studies_domain_agnostic_listing"}
    exec(compile(snippet.content, str(source), "exec"), namespace)

    own = refund_requirement("B6")
    for packet, facts in packets(own):
        clock_failures = namespace["clock"](packet, facts)
        coupling_failures = namespace["coupling"](packet, facts)
        assert bool(clock_failures) == (packet.domain == "triage" and packet.form == "copy")
        assert bool(coupling_failures) == (packet.domain == "hospital" and packet.form == "copy")
        verdict = namespace["check"](packet, facts)
        assert (verdict.verdict == "hold") == (packet.form == "copy")


def test_shared_demo_regions_outside_parts_7_8_are_identical():
    extractor = vars(_regions)
    source = ROOT / "demos/part7_8_demo.py"
    prior = subprocess.check_output(
        ["git", "show", "5dee1f07:companion/demos/part7_8_demo.py"], cwd=ROOT, text=True
    )
    old = {
        s.snippet_id: s.content
        for s in extractor["extract_snippets_from_text"](prior, source)
        # Each merged chapter retires its own legacy regions; every other region stays.
        if not s.snippet_id.startswith(tuple(f"fnd.{slug}." for slug in (
            "audits-explanations", "human-oversight-risk", "org-adoption",
            "capstone-blueprint", "case-studies-domain-agnostic", "future-directions",
        )))
    }
    new = {
        s.snippet_id: s.content
        for s in extractor["extract_snippets_from_text"](source.read_text(), source)
    }
    assert old == new


def test_unavailable_full_capture_does_not_write_fake_verdicts(tmp_path, monkeypatch):
    monkeypatch.delenv("CASE_STUDIES_DOMAIN_AGNOSTIC_REQUIREMENT", raising=False)
    with pytest.raises(ValueError) as error:
        capture_chapter(tmp_path, ids=(*INDEPENDENT, f"fnd.{SLUG}.verdicts-01"))
    assert "refund_agent.launch" in str(
        error.value
    ) or "CASE_STUDIES_DOMAIN_AGNOSTIC_REQUIREMENT" in str(error.value)
    assert not (tmp_path / f"book_outputs/_fnd.{SLUG}.verdicts-01.txt").exists()


@pytest.mark.parametrize("kind", ["facts", "cells"])
@pytest.mark.parametrize("departure", ["missing", "extra"])
def test_table_file_inventory_refused_before_printing(
    tmp_path, monkeypatch, capsys, kind, departure
):
    write_data(tmp_path)
    path = tmp_path / f"{SLUG}-bank.{kind}.json"
    if departure == "missing":
        path.unlink()
    else:
        (tmp_path / f"{SLUG}-insurer.{kind}.json").write_bytes(path.read_bytes())
    monkeypatch.setattr(demo, "DATA", tmp_path)
    with pytest.raises(ValueError, match="table file inventory") as error:
        demo.print_packets()
    assert departure in str(error.value) and kind in str(error.value)
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("policy", ["request", "batch"])
@pytest.mark.parametrize("departure", ["missing", "duplicate", "extra"])
def test_hospital_patient_inventory_refused_before_printing(monkeypatch, capsys, policy, departure):
    original = demo.hospital_day

    def wrong(day, selected_policy):
        rows = original(day, selected_policy)
        if selected_policy != policy:
            return rows
        if departure == "missing":
            return rows[:-1]
        if departure == "duplicate":
            return (*rows, rows[0])
        return (*rows, rows[0].model_copy(update={"patient_id": "unexpected-patient"}))

    monkeypatch.setattr(demo, "hospital_day", wrong)
    with pytest.raises(ValueError, match=f"hospital {policy} patient inventory") as error:
        demo.print_hospital()
    assert departure in str(error.value)
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("departure", ["missing", "duplicate", "extra"])
def test_verdict_packet_inventory_refused_before_printing(monkeypatch, capsys, departure):
    own = refund_requirement("B6")
    rows = packets(own)
    if departure == "missing":
        wrong = rows[:-1]
    elif departure == "duplicate":
        wrong = (*rows, rows[0])
    else:
        packet, facts = rows[-1]
        wrong = (*rows, (packet.model_copy(update={"domain": "insurer"}), facts))
    monkeypatch.setattr(demo, "packets", lambda _: wrong)
    with pytest.raises(ValueError, match="packet inventory") as error:
        demo.print_verdicts(own)
    assert departure in str(error.value)
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("departure", ["missing", "duplicate", "extra"])
def test_capture_request_inventory_refused_before_writing(tmp_path, departure):
    if departure == "missing":
        ids = INDEPENDENT[:-1]
    elif departure == "duplicate":
        ids = (*INDEPENDENT, INDEPENDENT[0])
    else:
        ids = (*INDEPENDENT, f"fnd.{SLUG}.unexpected-01")
    destination = tmp_path / "publication"
    with pytest.raises(ValueError, match="capture request inventory") as error:
        capture_chapter(destination, ids=ids)
    assert departure in str(error.value)
    assert not destination.exists()


@pytest.mark.parametrize("departure", ["missing", "duplicate", "extra"])
def test_capture_source_inventory_refused_before_writing(tmp_path, monkeypatch, departure):
    extract = capture._regions.extract_snippets_from_text

    def changed(text, source):
        rows = extract(text, source)
        if departure == "missing":
            return rows[:-1]
        if departure == "duplicate":
            return [*rows, rows[0]]
        return [*rows, replace(rows[0], snippet_id=f"fnd.{SLUG}.unexpected-01")]

    monkeypatch.setattr(capture._regions, "extract_snippets_from_text", changed)
    destination = tmp_path / "publication"
    with pytest.raises(ValueError, match="capture source inventory") as error:
        capture_chapter(destination, ids=INDEPENDENT)
    assert departure in str(error.value)
    assert not destination.exists()


@pytest.mark.parametrize("departure", ["missing", "extra"])
def test_capture_result_inventory_refused_before_writing(tmp_path, monkeypatch, departure):
    outputs = {name: "test-owned captured output\n" for name in INDEPENDENT}
    if departure == "missing":
        outputs.pop(INDEPENDENT[-1])
    else:
        outputs[f"fnd.{SLUG}.unexpected-01"] = "unrequested output\n"
    monkeypatch.setattr(capture._regions, "capture_outputs", lambda *args, **kwargs: outputs)
    destination = tmp_path / "publication"
    with pytest.raises(ValueError, match="capture result inventory") as error:
        capture_chapter(destination, ids=INDEPENDENT)
    assert departure in str(error.value)
    assert not destination.exists()


@pytest.mark.parametrize("departure", ["missing", "extra"])
def test_triage_result_inventory_refused_before_printing(monkeypatch, capsys, departure):
    original = demo.alarm_numbers

    def wrong(rate, tolerance):
        numbers = original(rate, tolerance)
        if departure == "missing":
            numbers.pop("budget_seconds")
        else:
            numbers["unrequested_measure"] = 1
        return numbers

    monkeypatch.setattr(demo, "alarm_numbers", wrong)
    with pytest.raises(ValueError, match="triage result inventory") as error:
        demo.print_triage()
    assert departure in str(error.value)
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("domain", ["triage", "bank"])
def test_scalar_domain_inventory_refused_before_printing(monkeypatch, capsys, domain):
    facts = demo.domain_facts(domain).model_copy(update={"domain": "insurer"})
    monkeypatch.setattr(demo, "domain_facts", lambda _: facts)
    with pytest.raises(ValueError, match=f"{domain} facts inventory"):
        getattr(demo, f"print_{domain}")()
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("departure", ["missing", "extra"])
def test_bank_cost_inventory_refused_before_printing(monkeypatch, capsys, departure):
    facts = demo.domain_facts("bank")
    entries = dict(facts.cost_matrix.entries)
    if departure == "missing":
        entries.pop("hold")
    else:
        entries["unrequested_action"] = (0, 0)
    wrong = facts.model_copy(
        update={"cost_matrix": facts.cost_matrix.model_copy(update={"entries": entries})}
    )
    monkeypatch.setattr(demo, "domain_facts", lambda _: wrong)
    with pytest.raises(ValueError, match="bank cost action inventory") as error:
        demo.print_bank()
    assert departure in str(error.value)
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("departure", ["missing", "duplicate", "extra"])
def test_bank_score_inventory_refused_before_printing(monkeypatch, capsys, departure):
    facts = demo.domain_facts("bank")
    if departure == "missing":
        actions = facts.actions_on_score[:-1]
    elif departure == "duplicate":
        actions = (*facts.actions_on_score, facts.actions_on_score[0])
    else:
        actions = (*facts.actions_on_score, "unrequested_action")
    monkeypatch.setattr(
        demo, "domain_facts", lambda _: facts.model_copy(update={"actions_on_score": actions})
    )
    with pytest.raises(ValueError, match="bank score action inventory") as error:
        demo.print_bank()
    assert departure in str(error.value)
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("departure", ["missing", "extra"])
def test_balance_result_inventory_refused_before_printing(monkeypatch, capsys, departure):
    result = demo.order_balance()
    if departure == "missing":
        result.pop("remaining_cents")
    else:
        result["unrequested_measure"] = 1
    monkeypatch.setattr(demo, "order_balance", lambda: result)
    with pytest.raises(ValueError, match="balance result inventory") as error:
        demo.print_balance()
    assert departure in str(error.value)
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("scope,field", [("fixture", "integration"), ("facts", "basis")])
@pytest.mark.parametrize("departure", ["missing", "extra"])
def test_key_fixture_inventory_refused_before_printing(
    tmp_path, monkeypatch, capsys, scope, field, departure
):
    fixture = json.loads((demo.EXERCISES / "insurer_05.json").read_text())
    values = fixture if scope == "fixture" else fixture["facts"]
    if departure == "missing":
        values.pop(field)
    else:
        values["unrequested_field"] = 1
    (tmp_path / "insurer_05.json").write_text(json.dumps(fixture))
    monkeypatch.setattr(demo, "EXERCISES", tmp_path)
    with pytest.raises(ValueError, match=f"exercise {scope} inventory") as error:
        demo.print_keys()
    assert departure in str(error.value)
    assert capsys.readouterr().out == ""
