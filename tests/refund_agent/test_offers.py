"""Maturity, probability and sealed-window contracts shared with chapter 21.

The committed-package test requires FND_ALLOW_SEALED=1, set only for a separately
authorized sealed-execution run. Other tests generate their own rows and packages.
"""

import hashlib
import json
import shutil
from dataclasses import fields, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

from refund_agent.offers.budget import propensity_floor
from refund_agent.offers.evaluation import (
    CappedUniform,
    cap_zero_fixture,
    evaluation_log,
    freeze_candidate,
    split_windows,
)
from refund_agent.offers.log import OfferLog, OfferRow
from refund_agent.offers.notices import OFFERS, eligible
from refund_agent.offers.policies import (
    UCB1,
    Capped,
    EpsilonGreedy,
    EstimateTable,
    Thompson,
    best_probabilities,
)
from refund_agent.offers.replay import replay, report, training_rows
from refund_agent.offers.response import answers


@pytest.fixture(scope="module")
def rows():
    return training_rows()


def test_only_paid_tier_one_requests_are_eligible(rows):
    assert eligible(rows[0]) == OFFERS
    assert eligible(replace(rows[0], paid=False)) == ()
    assert eligible(replace(rows[0], amount_cents=5000)) == ()


@pytest.mark.parametrize(
    "policy", [EpsilonGreedy(0.1), UCB1(), Thompson(), Capped(EpsilonGreedy(0.3), 0.05)]
)
def test_probabilities_valid_and_masks_applied(policy):
    for counts, successes in (
        ([0, 0, 0], [0, 0, 0]),
        ([40, 10, 5], [32, 9, 4]),
        ([6000, 12, 2], [4800, 10, 1]),
    ):
        table = EstimateTable(counts, successes)
        for offers in (OFFERS, ("standard", "referenced")):
            probabilities = policy.probabilities(table, 100, offers)
            assert sum(probabilities) == pytest.approx(1)
            assert np.all(probabilities >= 0)
            if "dated" not in offers:
                assert probabilities[1] == 0
            if isinstance(policy, UCB1):
                assert set(probabilities) <= {0, 1}


def test_capped_distribution_is_the_exact_mixture():
    table = EstimateTable([5, 5, 5], [2, 5, 3])
    q = EpsilonGreedy(0.3).probabilities(table, 20)
    for cap in (0, 0.0125, 0.05, 1):
        p = Capped(EpsilonGreedy(0.3), cap).probabilities(table, 20)
        np.testing.assert_allclose(p, (1 - cap) * np.array([1, 0, 0]) + cap * q)
    assert propensity_floor(0.05, 0.3, 3) == pytest.approx(0.005)


def test_thompson_integration_matches_seeded_monte_carlo():
    counts, successes = np.array([40, 10, 5]), np.array([32, 9, 4])
    draws = np.random.default_rng(7).beta(1 + successes, 1 + counts - successes, size=(300000, 3))
    frequencies = np.bincount(np.argmax(draws, axis=1), minlength=3) / len(draws)
    np.testing.assert_allclose(
        best_probabilities(successes, counts), frequencies, atol=0.005, rtol=0
    )
    np.testing.assert_allclose(best_probabilities([0, 0, 0], [0, 0, 0]), [1 / 3] * 3)


def test_injected_early_answer_leaves_table_unchanged(rows):
    row = rows[0]
    table = EstimateTable()
    answer = answers(row)["standard"]
    fingerprint = table.fingerprint
    assert not table.observe(answer, row.at + timedelta(days=13))
    assert table.fingerprint == fingerprint
    assert table.observe(answer, row.at + timedelta(days=14))
    assert table.answers[0] == 1


def test_answers_are_policy_independent_and_replay_is_deterministic(rows):
    initial = {row.request_id: answers(row) for row in rows[:30]}
    a = replay(rows[:30], Capped(EpsilonGreedy(0.3), 0.05), 7)
    b = replay(rows[:30], UCB1(), 90)
    assert (
        a.log.fingerprint == replay(rows[:30], Capped(EpsilonGreedy(0.3), 0.05), 7).log.fingerprint
    )
    assert initial == {row.request_id: answers(row) for row in rows[:30]}
    assert a.log.rows != b.log.rows


@pytest.mark.parametrize(
    "policy", [EpsilonGreedy(0.1), UCB1(), Thompson(), Capped(EpsilonGreedy(0.3), 0.05)]
)
def test_each_logged_propensity_recomputes_from_decision_time_snapshot(rows, policy):
    result = replay(rows[:900], policy, 7)
    for index, row in enumerate(result.log.rows, 1):
        table = EstimateTable(row.table_answers, row.table_no_contacts)
        assert row.table_fingerprint == table.fingerprint
        p = policy.probabilities(table, index, row.eligible_offers)
        np.testing.assert_array_equal(p, row.probabilities)
        assert row.propensity == p[OFFERS.index(row.offer)]
        assert sum(row.probabilities) == pytest.approx(1)
    # The first two weeks have no matured answers, even when a contact arrived early.
    for row in result.log.rows:
        if row.decided_at < rows[0].at + timedelta(days=14):
            assert row.table_answers == (0, 0, 0)
    assert any(sum(row.table_answers) > 0 for row in result.log.rows)


def test_offer_log_round_trip_is_append_only(rows, tmp_path):
    result = replay(rows[:20], Capped(EpsilonGreedy(0.3), 0.05), 7)
    path = tmp_path / "offers.jsonl"
    result.log.write(path)
    before = path.read_bytes()
    result.log.write(path)
    assert path.read_bytes() == before
    loaded = OfferLog.read(path)
    assert loaded.fingerprint == result.log.fingerprint
    assert loaded.rows == result.log.rows
    with pytest.raises(ValueError):
        loaded.append(replace(loaded.rows[0], offer="dated"))
    assert path.read_bytes() == before


def test_cap_zero_sends_only_standard(rows):
    result = replay(rows[:100], Capped(EpsilonGreedy(0.3), 0), 7)
    assert all(row.offer == "standard" and row.propensity == 1 for row in result.log.rows)


def test_masks_keep_the_canonical_tie_order():
    for policy in (EpsilonGreedy(0), UCB1()):
        np.testing.assert_array_equal(
            policy.probabilities(EstimateTable(), 1, ("referenced", "standard")),
            [1, 0, 0],
        )


def test_zero_probability_sent_offer_requires_a_human_override(rows):
    row = replay(rows[:1], Capped(EpsilonGreedy(0.3), 0), 7).log.rows[0]
    with pytest.raises(ValueError):
        replace(row, offer="dated", propensity=0.0)
    assert replace(row, offer="dated", propensity=0.0, override=True).override


def test_chapter21_can_load_persist_and_consult_a_test_copy(rows, tmp_path):
    from refund_agent.offers.evaluation import OfferPackage, fixed_rule

    learning, evaluation = split_windows(rows)
    result = replay(learning[-10:], Capped(EpsilonGreedy(0.3), 0.05), 7)
    candidate = freeze_candidate(
        result.tables, result.finished_at, last_decision_at=learning[-1].at
    )
    logger = CappedUniform(0.05)
    original = OfferPackage(
        result.log,
        candidate,
        fixed_rule(candidate),
        logger,
        evaluation_log(evaluation[:20], logger, 7, candidate=candidate),
        cap_zero_fixture(evaluation[:20]),
    )
    original.persist(tmp_path)
    loaded = OfferPackage.load(tmp_path)
    loaded.persist(tmp_path)
    assert loaded.evaluation.outcomes.log == loaded.evaluation.truth.log == []
    resolved = loaded.evaluation.resolved_log("chapter21-test", "evaluate a test copy")
    assert len(resolved.rows) == 20 and all(row.reward is not None for row in resolved.rows)
    assert len(loaded.evaluation.outcomes.log) == 1
    assert loaded.evaluation.truth.log == []
    assert len(OfferPackage.load(tmp_path).evaluation.outcomes.log) == 1


def test_regeneration_never_opens_an_existing_sealed_payload(tmp_path, monkeypatch):
    from pathlib import Path

    from refund_agent.offers.evaluation import SealedRecords

    records = [{"decision_id": "test-decision", "reward": 1}]
    SealedRecords("outcomes", records).persist(tmp_path)
    original_open = Path.open

    def guarded_open(path, mode="r", *args, **kwargs):
        if path.name.endswith(".sealed.jsonl") and mode != "x":
            raise AssertionError("regeneration read a sealed payload")
        return original_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    regenerated = SealedRecords("outcomes", records)
    regenerated.persist(tmp_path)
    loaded = SealedRecords.load(tmp_path, "outcomes")
    loaded.persist(tmp_path)
    assert regenerated.log == loaded.log == []


def test_report_has_intervals_for_zero_and_three_answers_and_never_adopts(rows):
    same_reason = [row for row in rows if row.reason == "damaged"][:3]
    small = replay(same_reason, Capped(EpsilonGreedy(0.3), 0), 7)
    card = report(small.log)
    cells = card["cells"]
    assert any(cell["matured"] == 3 for cell in cells)
    assert all(cell["wilson"] is not None for cell in cells)
    assert all(cell["wilson"] == (0.0, 1.0) for cell in cells if cell["matured"] == 0)
    twenty = report(replay(rows[:20], Capped(EpsilonGreedy(0.3), 0.05), 7).log)
    assert twenty["coin"]["baseline_count"] + twenty["coin"]["learner_count"] == 20
    assert "interval" in twenty["coin"] and "approximate" in twenty["coin"]
    assert not {"winner", "adopt", "adopted", "promote"} & twenty.keys()


def test_split_is_by_decision_date_alone(rows):
    learning, evaluation = split_windows(rows)
    assert len(rows) == 13766
    assert len(learning) + len(evaluation) == len(rows)
    assert max(row.at for row in learning) < datetime(2026, 2, 1, tzinfo=timezone.utc)
    assert min(row.at for row in evaluation) >= datetime(2026, 2, 1, tzinfo=timezone.utc)
    edge = replace(rows[0], at=datetime(2026, 2, 1, tzinfo=timezone.utc))
    assert split_windows([edge]) == ([], [edge])
    with pytest.raises(ValueError):
        split_windows([replace(edge, at=datetime(2026, 6, 1, tzinfo=timezone.utc))])


def test_candidate_frozen_before_evaluation_and_immutable(rows):
    learning, evaluation = split_windows(rows)
    result = replay(learning, Capped(EpsilonGreedy(0.3), 0.05), 7)
    candidate = freeze_candidate(
        result.tables, result.finished_at, last_decision_at=learning[-1].at
    )
    assert candidate.frozen_at.date().isoformat() == "2026-02-14"
    fingerprint = candidate.fingerprint
    log = evaluation_log(evaluation, CappedUniform(0.05), 7, candidate=candidate)
    result.tables["damaged"].no_contacts[:] = 0
    assert candidate.fingerprint == fingerprint
    assert log.candidate_fingerprint == fingerprint
    with pytest.raises(ValueError):
        freeze_candidate(result.tables, learning[-1].at, last_decision_at=learning[-1].at)
    assert log.outcomes.log == log.truth.log == []


@pytest.mark.parametrize("cap", [0.05, 0.0125])
def test_frozen_logger_probabilities_fields_and_empty_journals(rows, cap, tmp_path):
    _, evaluation = split_windows(rows)
    logger = CappedUniform(cap)
    window = evaluation_log(evaluation[:30], logger, 7)
    required = {
        "decision_id",
        "request_id",
        "customer_id",
        "decided_at",
        "reason",
        "amount_cents",
        "eligible_offers",
        "probabilities",
        "offer",
        "propensity",
        "coin",
        "cap",
        "policy_version",
        "policy_fingerprint",
        "seed",
        "override",
        "contact_at",
        "matures_at",
        "reward",
    }
    assert required <= {field.name for field in fields(OfferRow)}
    for row in window.log.rows:
        assert row.probabilities == (1 - 2 * cap / 3, cap / 3, cap / 3)
        assert row.policy_fingerprint == logger.fingerprint
        assert row.policy_version == logger.version and row.seed == 7
        assert row.propensity == row.probabilities[OFFERS.index(row.offer)]
        assert row.reward is None and row.contact_at is None and not row.override
        assert row.matures_at == row.decided_at + timedelta(days=14)
    window.persist(tmp_path)
    assert window.outcomes.log == window.truth.log == []
    assert all(path.read_bytes() == b"" for path in tmp_path.glob("*.access.jsonl"))


def test_sealed_consultations_are_durable_and_cap_zero_is_same_window(rows, tmp_path):
    _, evaluation = split_windows(rows)
    window = cap_zero_fixture(evaluation[:10])
    window.persist(tmp_path)
    public = window.log.rows
    assert all(row.probabilities == (1, 0, 0) and row.offer == "standard" for row in public)
    resolved = window.resolved_log("chapter21-test", "pin the explicit outcome consultation")
    assert all(row.reward in (0, 1) for row in resolved.rows)
    window.truth.consult("chapter21-test", "after the evaluation record")
    window.persist(tmp_path)
    assert len(window.outcomes.log) == len(window.truth.log) == 1
    for path in tmp_path.glob("*.access.jsonl"):
        assert json.loads(path.read_text())["entry"] == 1


def test_chapter17_never_reads_false_labels_or_frozen_sets(monkeypatch):
    from refund_agent.data import history
    from refund_agent.offers.evaluation import SealedRecords
    from refund_agent.split import GoldenSet

    actual = history.generate_history()
    fingerprint = hashlib.sha256(
        "\n".join(row.model_dump_json() for row in actual).encode()
    ).hexdigest()

    class Guard:
        def __init__(self, row):
            self.row = row

        def __getattr__(self, name):
            if name == "found_false_30d":
                raise AssertionError("offer code read a false-claim outcome")
            if self.row.at >= datetime(2026, 6, 1, tzinfo=timezone.utc) and name != "at":
                raise AssertionError("offer code read outside the training window")
            return getattr(self.row, name)

    def forbidden(*args, **kwargs):
        raise AssertionError("chapter17 consulted a frozen set")

    monkeypatch.setattr(GoldenSet, "consult", forbidden)
    monkeypatch.setattr(SealedRecords, "consult", forbidden)
    from demos.exploration_bandits_demo import build_experiment

    experiment = build_experiment(source=[Guard(row) for row in actual])
    assert experiment["package"].evaluation.outcomes.log == []
    assert experiment["package"].evaluation.truth.log == []
    assert experiment["accesses"] == []
    assert (
        hashlib.sha256(
            "\n".join(row.model_dump_json() for row in history.generate_history()).encode()
        ).hexdigest()
        == fingerprint
    )


@pytest.mark.sealed
def test_committed_package_pins_windows_logger_and_every_learning_update(monkeypatch, tmp_path):
    from refund_agent.offers.evaluation import WINDOWS, OfferPackage

    source = Path(__file__).resolve().parents[2] / "demos/data/exploration_bandits_package"
    source_journals = {path: path.read_bytes() for path in source.glob("*/*.access.jsonl")}
    directory = shutil.copytree(source, tmp_path / "package")
    journals = {path: path.read_bytes() for path in directory.glob("*/*.access.jsonl")}
    original_open = Path.open

    def guarded_open(path, *args, **kwargs):
        if path.name.endswith(".sealed.jsonl"):
            raise AssertionError("the chapter 17 package check opened a sealed payload")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    package = OfferPackage.load(directory)
    assert WINDOWS == {
        "learning": ("2025-04-01", "2026-02-01"),
        "evaluation": ("2026-02-01", "2026-06-01"),
    }
    assert len(package.learning.rows) == 9837
    assert len(package.evaluation.log.rows) == len(package.cap_zero.log.rows) == 3929
    boundary = datetime(2026, 2, 1, tzinfo=timezone.utc)
    assert all(row.decided_at < boundary for row in package.learning.rows)
    assert all(
        boundary <= row.decided_at < datetime(2026, 6, 1, tzinfo=timezone.utc)
        for row in package.evaluation.log.rows
    )
    assert package.candidate.frozen_at == max(row.matures_at for row in package.learning.rows)
    assert package.candidate.frozen_at.date().isoformat() == "2026-02-14"

    policy = Capped(EpsilonGreedy(0.3), 0.05)
    tables = {}
    t = 0
    for event in package.learning.events:
        table = tables.setdefault(event.table_key, EstimateTable())
        if event.reward is None:
            t += 1
            assert event.table_key == event.reason
            assert event.table_fingerprint == table.fingerprint
            assert event.table_answers == tuple(table.answers)
            assert event.table_no_contacts == tuple(table.no_contacts)
            assert event.policy_fingerprint == policy.fingerprint
            np.testing.assert_array_equal(event.probabilities, policy.probabilities(table, t))
        else:
            assert event.observed_at >= event.matures_at
            index = OFFERS.index(event.offer)
            table.answers[index] += 1
            table.no_contacts[index] += event.reward
    assert t == 9837 and len(package.learning.events) == 2 * t
    for key, offer, n, k, sha in package.candidate.tables:
        assert n == tuple(tables[key].answers) and k == tuple(tables[key].no_contacts)
        assert sha == tables[key].fingerprint
        assert offer == OFFERS[int(np.argmax(tables[key].means))]
    assert all(offer == "standard" for _, offer, *_ in package.baseline.tables)

    for row, zero in zip(package.evaluation.log.rows, package.cap_zero.log.rows):
        assert row.request_id == zero.request_id and row.customer_id == zero.customer_id
        assert row.decided_at == zero.decided_at and row.matures_at == zero.matures_at
        assert row.eligible_offers == zero.eligible_offers == OFFERS
        assert row.probabilities == (1 - 2 * 0.05 / 3, 0.05 / 3, 0.05 / 3)
        assert row.policy_version == "frozen-capped-uniform-1" and row.seed == 7
        assert row.policy_fingerprint == package.logger.fingerprint
        assert row.propensity == row.probabilities[OFFERS.index(row.offer)]
        assert zero.offer == "standard" and zero.probabilities == (1, 0, 0)
        assert not row.override and not zero.override
        assert row.reward is zero.reward is row.contact_at is zero.contact_at is None
    assert len(journals) == 4
    # The Part IV plan's 17-to-21 interface reserves judgment of these sets for
    # Chapter 21. Its observed-outcome evaluation precedes its simulation check.
    purposes = {
        "outcomes": "evaluate under predeclared rule ",
        "truth": "post-record simulation check; rule ",
    }
    for path, payload in journals.items():
        window = package.evaluation if path.parent.name == "evaluation" else package.cap_zero
        kind = path.name.split(".")[0]
        for number, line in enumerate(payload.splitlines(), 1):
            receipt = json.loads(line)
            assert receipt["entry"] == number
            assert receipt["reader"] == "chapter-21"  # No Chapter 17 or development reader.
            assert receipt["set"] == kind
            assert receipt["sha256"] == getattr(window, kind).manifest["sha256"]
            assert receipt["purpose"].startswith(purposes[kind])
            rule_hash = receipt["purpose"][len(purposes[kind]) :]
            assert len(rule_hash) == 64 and all(c in "0123456789abcdef" for c in rule_hash)
    assert journals == {path: path.read_bytes() for path in journals}
    assert source_journals == {path: path.read_bytes() for path in source_journals}
