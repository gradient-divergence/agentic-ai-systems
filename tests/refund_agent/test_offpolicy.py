"""Behavior contracts for importance weighting and predeclared policy judgments.

The cap-zero package test copies canonical sealed evidence and requires
FND_ALLOW_SEALED=1, set only for a separately authorized sealed-execution run.
"""

import itertools
import shutil

import numpy as np
import pytest

from refund_agent.offpolicy.estimators import (
    Record,
    doubly_robust,
    effective_sample_size,
    ips,
    self_normalized_interval,
    self_normalized_ips,
)
from refund_agent.offpolicy.frozen import FrozenLog
from refund_agent.offpolicy.policies import Policy
from refund_agent.offpolicy.record import evaluate
from refund_agent.offpolicy.reward_model import CellModel
from refund_agent.offpolicy.rule import Rule, exact_check


@pytest.fixture(autouse=True)
def temporary_working_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


def population_expectation(logged, target, rewards, model):
    total_ips = total_dr = 0.0
    for actions in itertools.product(range(2), repeat=len(logged)):
        chance = np.prod([logged[i][a] for i, a in enumerate(actions)])
        if not chance:
            continue
        rows = [
            Record(str(i), ("text", "call")[a], rewards[i][a], dict(zip(("text", "call"), b)))
            for i, (a, b) in enumerate(zip(actions, logged))
        ]
        ratios = [target.probabilities(r.context)[r.action] / r.logged[r.action] for r in rows]
        total_ips += chance * ips(ratios, [r.reward for r in rows])
        total_dr += chance * doubly_robust(target, model, rows)
    return total_ips, total_dr


def test_exact_expectation_and_missing_mass():
    policy = Policy(
        "target", (("0", (("text", 0.4), ("call", 0.6))), ("1", (("text", 0.4), ("call", 0.6))))
    )
    wrong = CellModel(
        (("0", "text", 0.1, 1), ("0", "call", 0.9, 1), ("1", "text", 0.2, 1), ("1", "call", 0.3, 1))
    )
    rewards = ((0.5, 1.0), (0.2, 0.7))
    truth = np.mean([0.4 * a + 0.6 * b for a, b in rewards])
    a, b = population_expectation(((0.8, 0.2), (0.7, 0.3)), policy, rewards, wrong)
    assert a == pytest.approx(truth)
    assert b == pytest.approx(truth)
    a, _ = population_expectation(((1, 0), (0.7, 0.3)), policy, rewards, wrong)
    assert truth - a == pytest.approx(0.6 * rewards[0][1] / 2)


def test_estimator_ranges_and_zero_matches():
    weights, rewards = [3, 1 / 6, 2, 1], [1, 0, 1, 1]
    assert ips(weights, rewards) == 1.5
    assert min(rewards) <= self_normalized_ips(weights, rewards) <= max(rewards)
    assert effective_sample_size([0, 0]) == 0
    assert self_normalized_ips([0, 0], [0, 1]) is None
    assert effective_sample_size([2] * 5) == 5
    rng = np.random.default_rng(7)
    for _ in range(20):
        w = rng.random(30) * rng.integers(0, 2, 30)
        assert effective_sample_size(w) <= np.count_nonzero(w) + 1e-12


def test_five_identical_rewards_have_zero_width_self_normalized_interval():
    w = [200] * 5 + [0] * 995
    r = [1] * 5 + [0] * 995
    assert ips(w, r) == 1
    assert self_normalized_interval(w, r) == (1, 1)
    assert effective_sample_size(w) == 5


def test_sparse_balanced_outcomes_hold_despite_floor_and_interval():
    target = Policy.deterministic("candidate", {"bank": "call"}, ("text", "call"))
    baseline = Policy.deterministic("baseline", {"bank": "text"}, ("text", "call"))
    rows = [
        Record("bank", "call" if i < 500 else "text", int(i < 5), {"text": 0.5, "call": 0.5})
        for i in range(1000)
    ]
    assert exact_check(rows, target, baseline, seed=7, draws=100000)["p_value"] == 1 / 32
    result = evaluate({"candidate": target}, baseline, FrozenLog.from_records(rows), Rule())
    card = result.data["policies"]["candidate"]
    assert card["n_eff"] == 500
    assert card["interval"][0] > 0
    assert card["decision"] == "hold"
    assert card["reasons"] == ["exact check exceeds 0.025"]


def test_unsupported_and_no_matching_records_are_not_judged():
    candidate = Policy.deterministic("candidate", {"bank": "call"}, ("text", "call"))
    baseline = Policy.deterministic("baseline", {"bank": "text"}, ("text", "call"))
    rows = [Record("bank", "text", 1, {"text": 1.0, "call": 0.0}) for _ in range(500)]
    result = evaluate({"candidate": candidate}, baseline, FrozenLog.from_records(rows), Rule())
    card = result.data["policies"]["candidate"]
    assert card["ips"] == 0
    assert card["n_eff"] == 0
    assert card["sn"] is None
    assert card["sn_reason"] == "no positive importance ratios"
    assert not card["judged"]
    assert card["decision"] == "hold"
    assert result.data["access"][0]["kind"] == "rule"
    assert result.data["access"][1]["kind"] == "outcomes"
    assert len(result.data["access"]) == 2


def test_doubly_robust_zero_model_and_paired_terms():
    from demos.practical_rl_demo import bank_model, bank_policies, bank_records
    from refund_agent.offpolicy.estimators import improvement_terms, per_record_terms, policy_ratios

    candidate, baseline = bank_policies()
    rows = bank_records()
    w = policy_ratios(candidate, rows)
    assert doubly_robust(candidate, CellModel(), rows) == ips(w, [r.reward for r in rows])
    assert doubly_robust(candidate, bank_model(), rows) == pytest.approx(0.72)
    np.testing.assert_allclose(
        improvement_terms(candidate, baseline, bank_model(), rows),
        per_record_terms(candidate, bank_model(), rows)
        - per_record_terms(baseline, bank_model(), rows),
    )


@pytest.mark.parametrize("scale,decision", [(1, "hold"), (10, "eligible for shadow")])
def test_bank_rule_and_seeded_randomization(scale, decision):
    from demos.practical_rl_demo import bank_policies, bank_records

    candidate, baseline = bank_policies()
    rows = bank_records(scale)
    check = exact_check(rows, candidate, baseline, seed=7, draws=100000)
    assert check == exact_check(rows, candidate, baseline, seed=7, draws=100000)
    result = evaluate(
        {"bank": candidate}, baseline, FrozenLog.from_records(rows), Rule(estimator="ips")
    )
    card = result.data["policies"]["bank"]
    assert card["decision"] == decision
    assert card["n_eff"] == pytest.approx(50 * scale, rel=1e-12)
    assert card["improvement"] == pytest.approx(0.08)
    if scale == 10:
        assert 0 < check["p_value"] < 0.025
        assert check["method"] == "seeded redraws"


def test_exact_check_with_independent_wrong_model_matches_exhaustive_assignments():
    from dataclasses import replace

    from refund_agent.offpolicy.estimators import improvement_terms

    target = Policy.deterministic("candidate", {"a": "call", "b": "call"}, ("text", "call"))
    baseline = Policy.deterministic("baseline", {"a": "text", "b": "text"}, ("text", "call"))
    model = CellModel(
        (("a", "text", 0.1, 2), ("a", "call", 0.9, 2), ("b", "text", 0.7, 2), ("b", "call", 0.2, 2))
    )
    rows = [
        Record("a", "call", 1, {"text": 0.8, "call": 0.2}),
        Record("b", "text", 0, {"text": 0.3, "call": 0.7}),
    ]
    observed = improvement_terms(target, baseline, model, rows).mean()
    probability = 0.0
    for actions in itertools.product(("text", "call"), repeat=2):
        chance = np.prod([row.logged[a] for row, a in zip(rows, actions)])
        redrawn = [replace(row, action=a) for row, a in zip(rows, actions)]
        if improvement_terms(target, baseline, model, redrawn).mean() >= observed - 1e-12:
            probability += chance
    check = exact_check(rows, target, baseline, model=model)
    assert check["method"] == "enumeration"
    assert check["p_value"] == pytest.approx(probability)


@pytest.mark.parametrize("failure", ["pending", "version", "fingerprint", "seed"])
def test_rule_holds_each_invalid_window_condition(failure):
    from dataclasses import replace

    from demos.practical_rl_demo import bank_policies, bank_records

    candidate, baseline = bank_policies()
    rows = bank_records(10)
    changes = {
        "pending": dict(final=False),
        "version": dict(logger_version="second"),
        "fingerprint": dict(logger_fingerprint="second"),
        "seed": dict(logger_seed=8),
    }
    rows[0] = replace(rows[0], **changes[failure])
    result = evaluate(
        {"bank": candidate}, baseline, FrozenLog.from_records(rows), Rule(estimator="ips")
    )
    card = result.data["policies"]["bank"]
    assert card["decision"] == "hold"
    assert any("outcomes not final" in r or "more than one logging" in r for r in card["reasons"])


def test_no_matches_can_have_support_and_still_hold():
    candidate = Policy.deterministic("candidate", {"bank": "call"}, ("text", "call"))
    baseline = Policy.deterministic("baseline", {"bank": "text"}, ("text", "call"))
    rows = [Record("bank", "text", 1, {"text": 0.99, "call": 0.01}) for _ in range(500)]
    result = evaluate({"candidate": candidate}, baseline, FrozenLog.from_records(rows), Rule())
    card = result.data["policies"]["candidate"]
    assert card["judged"]
    assert card["ips"] == 0
    assert card["sn"] is None and card["n_eff"] == 0
    assert card["decision"] == "hold"


def test_support_table_checks_every_distribution_and_mask():
    from dataclasses import replace

    from refund_agent.offpolicy.support import support_table, supported

    candidate = Policy.deterministic("candidate", {"bank": "call"}, ("text", "call"))
    rows = [
        Record("bank", "text", 1, {"text": 1.0, "call": 0.0}),
        Record("bank", "text", 1, {"text": 0.9, "call": 0.1}),
    ]
    call = next(r for r in support_table(rows, candidate) if r["offer"] == "call")
    assert call["records"] == 2 and call["positive_share"] == 0.5 and call["min_positive"] == 0.1
    assert not supported(support_table(rows, candidate))
    rows = [replace(rows[1], eligible=("text",))]
    assert not supported(support_table(rows, candidate))


def test_model_fit_is_smoothed_fingerprinted_and_rejects_evaluation_rows():
    from dataclasses import replace

    from refund_agent.offpolicy.reward_model import fit_cells

    rows = [
        Record("bank", "text", r, {"text": 0.8, "call": 0.2}, decided_at="2026-01-15")
        for r in (1, 1, 0)
    ]
    model = fit_cells(rows)
    assert model.predict("bank", "text") == 3 / 5
    assert model.predict("bank", "call") == 0.5
    assert model.prediction("bank", "call")["assumption"]
    assert model.prediction("bank", "text", supported=False)["assumption"]
    assert not model.prediction("bank", "text")["assumption"]
    assert model.fingerprint == fit_cells(rows).fingerprint
    with pytest.raises(ValueError, match="learning-window"):
        fit_cells([replace(rows[0], decided_at="2026-02-01")])


def test_unseen_context_prediction_is_a_zero_assumption():
    from refund_agent.offpolicy.reward_model import fit_cells

    model = fit_cells(
        [Record("bank", "text", 1, {"text": 0.8, "call": 0.2}, decided_at="2026-01-15")]
    )
    assert model.prediction("unseen", "call") == {
        "value": 0.0,
        "assumption": True,
        "training_records": 0,
    }
    assert model.predict("bank", "call") == 0.5
    assert CellModel().predict("bank", "text") == 0.0


@pytest.mark.parametrize("kind", ["empty", "unresolved", "fully overridden"])
def test_missing_evidence_has_undefined_improvement_and_holds(kind, tmp_path):
    from dataclasses import replace

    from demos.practical_rl_demo import bank_policies, bank_records
    from refund_agent.offpolicy.record import EvaluationRecord

    candidate, baseline = bank_policies()
    rows = bank_records()
    if kind == "empty":
        rows = []
    elif kind == "unresolved":
        rows[0] = replace(rows[0], reward=None, final=False)
    else:
        rows = [replace(row, override=True) for row in rows]
    result = evaluate({"bank": candidate}, baseline, FrozenLog.from_records(rows), Rule())
    card = result.data["policies"]["bank"]
    assert card["improvement"] is None
    assert card["interval"] is None and card["exact"]["p_value"] is None
    assert all(card[key] is None for key in ("ips", "sn", "dm", "dr"))
    assert card["decision"] == "hold"
    assert "bank improvement undefined; interval undefined; exact p undefined" in result.render()
    assert result.data["overrides"] == (100 if kind == "fully overridden" else 0)
    written = EvaluationRecord.read(result.write(tmp_path / "record.json"))
    assert written.data == result.data
    assert written.render() == result.render()


def test_overrides_have_no_used_chance_and_are_counted_apart():
    from dataclasses import replace

    from demos.practical_rl_demo import bank_policies, bank_records

    candidate, baseline = bank_policies()
    rows = bank_records(10)
    rows.append(
        replace(
            rows[0],
            action="call",
            logged={"text": 1, "call": 0},
            override=True,
            decision_id="changed-by-person",
        )
    )
    result = evaluate(
        {"bank": candidate}, baseline, FrozenLog.from_records(rows), Rule(estimator="ips")
    )
    assert result.data["overrides"] == 1
    assert result.data["override_ids"] == ["changed-by-person"]
    assert result.data["randomized_records"] == 1000
    assert result.data["policies"]["bank"]["decision"] == "eligible for shadow"
    assert result.data["policies"]["bank"]["n_eff"] == pytest.approx(500, rel=1e-12)


def test_pre_read_rule_journal_is_durable_and_evaluation_reads_once(tmp_path):
    import json

    from demos.practical_rl_demo import bank_policies, bank_records

    candidate, baseline = bank_policies()
    window = FrozenLog.from_records(bank_records())
    window.journal_path = tmp_path / "access.jsonl"
    original = window._consult
    calls = []

    def guarded(reader, purpose):
        journal = [json.loads(line) for line in window.journal_path.read_text().splitlines()]
        assert journal[0]["kind"] == "rule"
        assert journal[0]["fingerprint"] == Rule().fingerprint
        assert journal[-1]["kind"] == "outcomes"
        calls.append((reader, purpose))
        return original(reader, purpose)

    window._consult = guarded
    record = evaluate({"bank": candidate}, baseline, window, Rule())
    assert len(calls) == window.reads == record.data["outcome_reads"] == 1
    with pytest.raises(ValueError, match="one outcome"):
        window.journal("rule", fingerprint=Rule().fingerprint)
        window.consult("second-reader", "not allowed")


def test_frozen_fingerprint_detects_mutation():
    rows = [Record("bank", "text", 1, {"text": 0.8, "call": 0.2})]
    window = FrozenLog.from_records(rows)
    assert window.intact()
    rows[0].logged.update(text=0.7, call=0.3)
    assert not window.intact()


def test_five_record_floor_holds_even_if_other_gates_would_pass():
    from refund_agent.offpolicy.rule import decide

    card = dict(
        judged=True,
        n_eff=5,
        baseline_n_eff=995,
        outcomes_final=True,
        one_logging_policy=True,
        interval=[0.4, 0.4],
        exact=dict(p_value=0.001),
    )
    assert decide(card, Rule()) == ("hold", ["candidate effective size below 400"])


def test_capped_logger_deterministic_candidate_has_about_n_over_sixty_effective_records():
    from refund_agent.offers.notices import OFFERS

    rng = np.random.default_rng(7)
    n = 60000
    probabilities = [29 / 30, 1 / 60, 1 / 60]
    choices = rng.choice(3, size=n, p=probabilities)
    weights = np.where(choices == OFFERS.index("dated"), 60.0, 0.0)
    assert effective_sample_size(weights) == pytest.approx(n / 60, rel=0.1)


@pytest.mark.sealed
def test_cap_zero_package_reports_only_fixed_rule_as_judged(tmp_path):
    from demos.practical_rl_demo import PACKAGE_DIRECTORY
    from refund_agent.offers.evaluation import OfferPackage
    from refund_agent.offpolicy.policies import offer_candidates

    directory = shutil.copytree(PACKAGE_DIRECTORY, tmp_path / "package")
    package = OfferPackage.load(directory)
    candidates, baseline = offer_candidates(package)
    window = FrozenLog.from_offer_window(
        package.cap_zero, journal_path=tmp_path / "practical_rl_evaluation.access.jsonl"
    )
    result = evaluate(candidates, baseline, window, Rule(), reader="chapter-21-cap-zero-test")
    assert result.data["policies"]["fixed"]["judged"]
    for name in ("A", "B"):
        assert not result.data["policies"][name]["judged"]
        assert result.data["policies"][name]["decision"] == "hold"
    assert len(result.data["assumptions"]) == 4


def test_record_json_roundtrip(tmp_path):
    from demos.practical_rl_demo import bank_policies, bank_records
    from refund_agent.offpolicy.record import EvaluationRecord

    candidate, baseline = bank_policies()
    record = evaluate({"bank": candidate}, baseline, FrozenLog.from_records(bank_records()), Rule())
    path = record.write(tmp_path / "record.json")
    assert EvaluationRecord.read(path).data == record.data
    assert EvaluationRecord.read(path).render() == record.render()


def test_repeated_sealed_access_is_a_noop_and_distinct_accesses_are_retained(tmp_path):
    import json

    from refund_agent.offers.evaluation import SealedRecords
    from refund_agent.offpolicy.frozen import DistinctAccess

    original = SealedRecords("outcomes", [{"reward": 1}])
    original.persist(tmp_path)
    journal = tmp_path / "outcomes.access.jsonl"
    purpose = "evaluate under predeclared rule " + "a" * 64
    store = DistinctAccess(original)
    assert store.consult("chapter-21", purpose) == [{"reward": 1}]
    first = journal.read_bytes()
    reloaded = DistinctAccess(SealedRecords.load(tmp_path, "outcomes"))
    assert reloaded.consult("chapter-21", purpose) == [{"reward": 1}]
    assert journal.read_bytes() == first
    for reader, other_purpose in (
        ("another-reader", purpose),
        ("chapter-21", "another purpose; rule " + "a" * 64),
        ("chapter-21", "evaluate under predeclared rule " + "b" * 64),
    ):
        assert reloaded.consult(reader, other_purpose) == [{"reward": 1}]
    entries = [json.loads(line) for line in journal.read_text().splitlines()]
    assert len(entries) == 4
    assert [entry["entry"] for entry in entries] == [1, 2, 3, 4]


def test_recreated_evaluation_preserves_receipt_bytes_but_records_a_new_rule_or_reader(tmp_path):
    import json

    from demos.practical_rl_demo import bank_policies, bank_records

    candidate, baseline = bank_policies()
    journal = tmp_path / "evaluation.access.jsonl"

    def run(rule, reader="chapter-21"):
        window = FrozenLog.from_records(bank_records())
        window.journal_path = journal
        record = evaluate({"bank": candidate}, baseline, window, rule, reader=reader)
        assert record.data["outcome_reads"] == 1
        return record.data

    first = run(Rule())
    receipt = journal.read_bytes()
    assert run(Rule()) == first
    assert journal.read_bytes() == receipt
    run(Rule(), "another-reader")
    run(Rule(n_eff_floor=401))
    entries = [json.loads(line) for line in journal.read_text().splitlines()]
    assert [entry["kind"] for entry in entries] == [
        "rule",
        "outcomes",
        "outcomes",
        "rule",
        "outcomes",
    ]
