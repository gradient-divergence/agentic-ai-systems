"""The memory lifecycle's temporal, budget and support contracts."""

from dataclasses import FrozenInstanceError
from datetime import datetime, timedelta, timezone

import pytest

from refund_agent import contracts as c
from refund_agent import rules
from refund_agent.memory.assemble import (
    assemble,
    draft_from_facts,
    fact_lines,
    last_scans_summary,
    required_facts,
)
from refund_agent.memory.funnel import funnel, stage_flags
from refund_agent.memory.memory import PolicyMemory
from refund_agent.memory.register import PolicyRegister, project
from refund_agent.memory.support import support


def at(month, day):
    return datetime(2026, month, day, tzinfo=timezone.utc)


@pytest.fixture(scope="module")
def demo():
    from demos import memory_rag_demo

    return memory_rag_demo


@pytest.fixture
def register(demo):
    return demo.demonstration_register()


def test_register_is_append_only_and_supersession_is_computed(demo):
    register = PolicyRegister()
    policies = demo.training_data()[0].policies
    register.announce(policies[0], at(1, 13))
    original = register.records
    register.announce(policies[1], at(2, 22))
    assert register.records[: len(original)] == original
    for record in original:
        assert register.superseded_at(record) == at(2, 22)
    with pytest.raises(FrozenInstanceError):
        original[0].valid_to = at(3, 1)


def test_as_of_uses_both_half_open_intervals(register):
    before = register.as_of(at(5, 12), at(4, 20))
    now = register.as_of(at(5, 12), at(5, 12))
    assert {r.version for r in before} == {"2026-03-01"}
    assert {r.version for r in now} == {"2026-05-01"}
    assert {r.version for r in register.as_of(at(5, 1), at(4, 24))} == {"2026-05-01"}
    assert all(r.version != "2026-03-01" for r in register.as_of(at(5, 1), at(5, 1)))


def test_cached_copy_refused_at_memory(register):
    memory = PolicyMemory(register, at(4, 20))
    assert memory.retrieve("damaged within days delivery", at(5, 4), 1).refusal == "beyond_notice"
    assert memory.retrieve("damaged within days delivery", at(5, 12), 1).refusal == "beyond_notice"
    memory.refresh(at(5, 12))
    assert (
        memory.retrieve("damaged within days delivery", at(5, 12), 1).hits[0].clause.version
        == "2026-05-01"
    )


def test_successor_draft_uses_refreshed_copy(demo, monkeypatch, capsys):
    demo.print_successor()
    refreshed = capsys.readouterr().out.splitlines()
    assert refreshed[3] == (
        "refreshed: approve eligible; refund_cap_v4@2026-05-01#eligibility"
    )

    refresh = PolicyMemory.refresh

    def keep_existing_copy(memory, read_at):
        if not hasattr(memory, "snapshot"):
            refresh(memory, read_at)

    monkeypatch.setattr(PolicyMemory, "refresh", keep_existing_copy)
    demo.print_successor()
    unrefreshed = capsys.readouterr().out.splitlines()
    assert unrefreshed[3] == (
        "refreshed: deny outside_return_window; refund_cap_v4@2026-03-01#eligibility"
    )
    assert len(refreshed) == len(unrefreshed) == 6
    assert refreshed[:3] + refreshed[4:] == unrefreshed[:3] + unrefreshed[4:]


def test_notice_boundary_and_future_copy(register):
    memory = PolicyMemory(register, at(5, 5))
    assert memory.retrieve("damaged", at(5, 12), 1).refusal is None
    assert (
        memory.retrieve("damaged", at(5, 12) + timedelta(seconds=1), 1).refusal == "beyond_notice"
    )
    assert memory.retrieve("damaged", at(5, 4), 1).refusal == "future_policy"


def test_seven_day_promise_covers_every_training_request(demo, register):
    data, requests = demo.training_data()
    for request in requests:
        memory = PolicyMemory(register, request.at - timedelta(days=7))
        result = memory.retrieve("damaged within days delivery", request.at, 1)
        assert result.refusal is None
        assert result.hits[0].clause.version == data.policy_at(request.at).version


def test_late_version_defeats_a_two_day_old_copy_and_audit_finds_it(demo):
    rows = demo.bank_audit()
    late = rows[1]
    assert late["age_days"] == 2 and late["usable"]
    assert late["right_then"] and not late["right_now"]


def test_late_record_passes_age_but_the_copy_is_incomplete(demo):
    data, _ = demo.training_data()
    register = PolicyRegister()
    register.announce(data.policies[0], at(1, 13))
    register.announce(data.policies[1], at(3, 5), announced_at=at(3, 4))
    memory = PolicyMemory(register, at(3, 2))
    result = memory.retrieve("damaged", at(3, 4), 1)
    assert result.refusal is None and result.hits[0].clause.version == "2026-01-20"
    assert {r.version for r in register.as_of(at(3, 4), at(3, 4))} == {"2026-01-20"}
    assert {r.version for r in register.as_of(at(3, 4), at(3, 5))} == {"2026-03-01"}
    assert register.records[-1].announced_at == at(3, 4)
    assert register.records[-1].recorded_at == at(3, 5)


def test_snapshot_and_recording_boundaries_do_not_leak_future_knowledge(register):
    before = register.snapshot(at(4, 24) - timedelta(seconds=1))
    assert not any(r.version == "2026-05-01" for r in before.records)
    exact = register.snapshot(at(4, 24))
    assert any(r.version == "2026-05-01" for r in exact.records)
    with pytest.raises(ValueError, match="knowledge time"):
        project(before, at(4, 24))


def test_snapshot_keeps_its_records_and_texts_after_correction_and_retention(register):
    snapshot = register.snapshot(at(5, 12))
    record = next(
        r for r in snapshot.records if r.citation == "refund_cap_v4@2026-05-01#eligibility"
    )
    text = snapshot.texts[record.text_hash]
    register.correct(
        record.citation,
        "damaged or changed_mind within 25 days of delivery; not_delivered if no delivery.",
        25,
        at(10, 1),
    )
    register.retain(at(10, 2))
    assert snapshot.texts[record.text_hash] == text
    assert project(snapshot, snapshot.at)[-1].return_days == 30
    with pytest.raises(TypeError):
        snapshot.texts[record.text_hash] = "changed"


def test_projection_closes_predecessor_and_preserves_corrections(register):
    before = project(register.snapshot(at(4, 20)), at(4, 20))
    assert [p.version for p in before] == ["2026-01-20", "2026-03-01"]
    after = project(register.snapshot(at(5, 12)), at(5, 12))
    assert len(after) == len({p.effective_at for p in after}) == 3
    citation = "refund_cap_v4@2026-05-01#eligibility"
    text = "damaged or changed_mind within 25 days of delivery; not_delivered if no delivery."
    original = register.records
    register.correct(citation, text, 25, at(10, 1))
    assert register.records[: len(original)] == original
    assert project(register.snapshot(at(9, 30)), at(9, 30))[2].return_days == 30
    corrected = project(register.snapshot(at(10, 1)), at(10, 1))[2]
    assert corrected.return_days == 25 and text in corrected.text


def test_text_and_structured_value_must_agree(register):
    with pytest.raises(ValueError, match="structured value"):
        register.correct("refund_cap_v4@2026-05-01#eligibility", "within 25 days", 30, at(10, 1))


def test_retention_removes_only_text_and_logs_its_hash(register):
    old = register.as_of(at(5, 12), at(5, 12))[1]
    register.correct(
        old.citation, "at most 85000 cents and the unrefunded payment balance.", 85000, at(10, 1)
    )
    records = register.records
    register.retain(at(10, 2))
    assert register.records == records
    result = register.read_text(old)
    assert result.refusal == "text_expired" and result.text_hash == old.text_hash
    assert any(old.text_hash in event.text_hashes for event in register.log)


def test_retention_allows_fresh_creation_and_refresh(register):
    memory = PolicyMemory(register, at(9, 30))
    records = register.records
    register.retain(at(10, 2))
    current = register.as_of(at(10, 2), at(10, 2))
    assert {r.version for r in current} == {"2026-09-01"}
    assert all(register.read_text(r).refusal is None for r in current)
    fresh = PolicyMemory(register, at(10, 2))
    memory.refresh(at(10, 2))
    for copy in (fresh, memory):
        hits = copy.retrieve("damaged delivery at most cents balance", at(10, 2), 2).hits
        assert {h.clause.section for h in hits} == {"eligibility", "cap"}
        assert {h.clause.version for h in hits} == {"2026-09-01"}
    assert register.records == records


def test_live_refresh_keeps_announced_successor_and_exact_effective_boundary(register):
    register.retain(at(8, 25))
    memory = PolicyMemory(register, at(8, 25))
    before = memory.retrieve("damaged delivery", at(9, 1) - timedelta(seconds=1), 1)
    exact = memory.retrieve("damaged delivery", at(9, 1), 1)
    assert before.refusal is None and before.hits[0].clause.version == "2026-07-01"
    assert exact.refusal is None and exact.hits[0].clause.version == "2026-09-01"


def test_historical_audit_needing_deleted_text_still_refuses(register):
    historical = next(r for r in register.as_of(at(5, 12), at(5, 12)) if r.section == "eligibility")
    register.retain(at(10, 2))
    assert register.read_text(historical).refusal == "text_expired"
    for read_at in (at(5, 12), at(10, 2)):
        with pytest.raises(ValueError, match="^text_expired$"):
            project(register.snapshot(read_at), at(5, 12))


def test_funnel_telescopes_and_zero_denominator_is_undefined():
    flags = [{"wording": "plain", "ret": True, "rank": i < 8, "kept": i < 6} for i in range(10)]
    row = funnel(flags, "wording")[0]
    assert row["product"] == pytest.approx(row["end_to_end"]) == 0.6
    zero = funnel([{"wording": "none", "ret": False, "rank": False, "kept": False}], "wording")[0]
    assert zero["rates"][1:] == [None, None] and zero["product"] is None
    assert zero["end_to_end"] == 0
    with pytest.raises(ValueError, match="nested"):
        funnel([{"wording": "bad", "ret": False, "rank": True, "kept": False}], "wording")


def test_reranker_cannot_change_candidates(demo):
    library, questions = demo.telecom_library()
    first, rerank, assembler = demo.telecom_stages(library, 5, True)
    before = stage_flags(questions, first, lambda q, candidates: candidates[:5], assembler)
    after = stage_flags(questions, first, rerank, assembler)
    assert [r["ret"] for r in before] == [r["ret"] for r in after]


def test_constraint_preservation_on_every_training_request(demo):
    data, requests = demo.training_data()
    for request in requests:
        state = demo.request_state(data, request)
        result = assemble(state, [], 2048)
        assert result.refusal is None
        assert set(fact_lines(required_facts(state))) <= set(result.lines)
        assert draft_from_facts(result.facts) == rules.draft(state)


def test_last_five_scans_lose_request_b_delivery(demo):
    state, records = demo.request_b()
    full = rules.draft(state)
    assert assemble(state, records, 2048).refusal is None
    assert full.outcome == c.Outcome.APPROVE
    compact = state.model_copy(
        update={"shipment_events": last_scans_summary(state.shipment_events, 5)}
    )
    assert rules.draft(compact).reason == "delivery_unknown"


def test_required_facts_exceed_budget(demo):
    state, records = demo.request_b()
    assert assemble(state, records, 1).refusal == "required_facts_exceed_budget"


def test_future_evidence_survives_cut(demo):
    state, records = demo.request_b()
    later = c.ShipmentEvent(at=state.request.at + timedelta(seconds=1), status="return_scan")
    state.shipment_events.append(later)
    result = assemble(state, records, 2048)
    assert result.facts["future_evidence"]
    assert rules.draft(state).reason == draft_from_facts(result.facts).reason == "future_evidence"


def test_training_drafts_are_supported_with_all_needed_clauses(demo, register):
    data, requests = demo.training_data()
    for request in requests:
        facts = required_facts(demo.request_state(data, request))
        draft = draft_from_facts(facts)
        report = support(draft, register, facts)
        if draft.outcome != c.Outcome.REFER:
            assert report["status"] == "supported", report
        if draft.outcome == c.Outcome.APPROVE:
            assert [row["section"] for row in report["clauses"]] == ["eligibility", "cap"]


def test_every_planted_unsupported_draft_is_refused(demo, register):
    rows = demo.planted_support(register)
    assert [r["reason"] for r in rows] == [
        "clause_does_not_entail",
        "version_not_in_force",
        "clause_does_not_entail",
        "citation_not_in_register",
        "clause_does_not_entail",
    ]
    assert all(r["status"] == "refused" for r in rows)
    assert [r["section"] for r in rows[-1]["clauses"]] == ["eligibility", "cap"]


def test_missing_or_expired_support_is_incomplete(demo, register):
    state, _ = demo.request_b()
    facts = required_facts(state)
    draft = rules.draft(state)
    cap = next(r for r in register.as_of(state.request.at, state.request.at) if r.section == "cap")
    register._texts.pop(cap.text_hash)
    assert support(draft, register, facts)["reason"] == "support_incomplete"


def test_supported_but_wrong_policy_is_not_caught(demo, register):
    data, requests = demo.training_data()
    row = demo.support_report(data, requests, register)["wrong_register"]
    assert row["status"] == "supported" and row["decision"] == "deny"


def test_failed_append_does_not_change_the_register(demo, register):
    records = register.records
    with pytest.raises(ValueError, match="duplicate"):
        register.announce(demo.training_data()[0].policies[2], at(10, 1))
    assert register.records == records


def test_tier_three_denials_are_referred_to_named_approver(demo):
    data, requests = demo.training_data()
    denials = [demo.request_state(data, r) for r in requests if r.amount_cents > 50000]
    assert denials
    for state in denials:
        routed = demo.routed_draft(state)
        assert routed.outcome == c.Outcome.REFER and not routed.executed


def test_seeded_library_is_deterministic(demo):
    assert demo.telecom_library(7) == demo.telecom_library(7)
    assert demo.telecom_library(7) != demo.telecom_library(8)
