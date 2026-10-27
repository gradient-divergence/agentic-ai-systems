"""Training-only scenario counts, independent labels and reproducible suites."""

import hashlib
import itertools
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from refund_agent.scenarios.checks import check_provenance, check_schema, policy_label
from refund_agent.scenarios.materialize import SchemaError, to_records
from refund_agent.scenarios.model import PARENTS, ScenarioModel, training_data
from refund_agent.scenarios.suite import ScenarioSet, build_suite, list_suite, regenerate
from refund_agent.scenarios.tokens import FIELDS, VALUES, case_tokens, valid_cases
from refund_agent.split import CUTS, GoldenSet

AS_OF = datetime(2026, 9, 15, tzinfo=timezone.utc)


@pytest.fixture(scope="module")
def model():
    return ScenarioModel.fit()


def test_every_conditional_sums_to_one_including_unseen_contexts(model):
    for t, parents in enumerate(PARENTS):
        for context in itertools.product(*(VALUES[FIELDS[i]] for i in parents)):
            p = model.conditional(t, dict(zip((FIELDS[i] for i in parents), context)))
            assert np.all(p > 0)
            assert p.sum() == pytest.approx(1)


def test_log_prob_is_the_sum_of_counted_conditionals(model):
    for tokens in model.draw(np.random.default_rng(7), 1, 25, AS_OF):
        context = {}
        expected = 0.0
        for t, value in enumerate(tokens):
            expected += np.log(model.conditional(t, context)[VALUES[FIELDS[t]].index(value)])
            context[FIELDS[t]] = value
        assert model.log_prob(tokens) == pytest.approx(expected)


def test_small_temperature_draws_the_decision_mode(model):
    # The supported decision counts are separated enough to approach greedy here.
    # Near-tied reason/amount counts can still draw a non-mode at positive tau.
    draws = model.draw(np.random.default_rng(19), 0.01, 100, AS_OF)
    for tokens in draws:
        context = {}
        for t, value in enumerate(tokens):
            p = model.conditional(t, context)
            if t == 5:
                assert p[VALUES[FIELDS[t]].index(value)] == pytest.approx(p.max())
            context[FIELDS[t]] = value
    # Preserve the counterexample to an all-token argmax claim at positive tau.
    p = model.conditional("amount", {})
    assert any(p[VALUES["amount"].index(tokens[3])] < p.max() for tokens in draws)


def test_valid_cases_roundtrip_and_contracts_validate(model):
    data = training_data()
    for case in valid_cases():
        records = to_records(case, AS_OF, 7)
        label = policy_label(records, data)
        assert case_tokens(records.request, records.order, records.shipment, label)[:5] == case
        for record in records:
            assert type(record).model_validate_json(record.model_dump_json()) == record
        assert all(event.at <= records.shipment.observed_at for event in records.shipment.events)


def test_fractional_delivery_ages_follow_the_policy_cutoffs():
    from refund_agent.contracts import ShipmentEvent

    records = to_records(("damaged", True, "0-14", 2499, "none"), AS_OF, 7)
    data = training_data()
    for age, band in ((14, "0-14"), (14.25, "15-21"), (21.25, "22-30"), (30.25, "31-45")):
        event = ShipmentEvent(at=AS_OF - timedelta(days=age), status="delivered")
        shipment = records.shipment.model_copy(update={"events": [event]})
        label = policy_label(records._replace(shipment=shipment), data)
        assert case_tokens(records.request, records.order, shipment, label)[2] == band


@pytest.mark.parametrize(
    "case",
    [
        ("damaged", False, "0-14", 2499, "none"),
        ("damaged", True, None, 2499, "none"),
        ("damaged", True, "unknown", 2499, "none"),
    ],
)
def test_impossible_pairings_are_typed_schema_rejections(case):
    with pytest.raises(SchemaError):
        to_records(case, AS_OF, 7)


@pytest.mark.parametrize("tau", [1, 2])
def test_kept_scenarios_have_the_current_policy_label_and_separate_generator_decision(tau):
    suite = build_suite(AS_OF, tau, 200, 7)
    data = training_data()
    assert suite.manifest["policy_version"] == "2026-09-01"
    assert suite.manifest["counts"]["rejected"] > 0
    assert suite.manifest["counts"]["overruled"] > 0
    for scenario in suite.scenarios:
        assert scenario.label == policy_label(scenario.records, data)
        assert scenario.origin == "generated"
        assert scenario.generator_decision == scenario.tokens[-1]
    assert check_provenance(suite, suite.model, before=(), after=())
    assert not check_provenance(suite, suite.model, before=(), after=({"entry": 1},))


def test_fit_reads_no_heldout_labels_features_or_golden_rows(monkeypatch):
    from refund_agent.data.generate import generate

    def forbidden(*args, **kwargs):
        raise AssertionError("scenario construction consulted the frozen golden set")

    monkeypatch.setattr(GoldenSet, "consult", forbidden)
    data = generate()

    # Keep time/id metadata for splitting, but make every later feature access fail.
    class Guard:
        def __init__(self, record, allowed):
            self.record, self.allowed = record, allowed

        def __getattr__(self, name):
            assert name in self.allowed, f"held-out field read: {name}"
            return getattr(self.record, name)

    safe_requests, safe_orders, safe_shipments = [], [], []
    for i, request in enumerate(data.requests):
        held = request.at >= CUTS[0]
        safe_requests.append(Guard(request, {"at", "request_id"}) if held else request)
        safe_orders.append(Guard(data.orders[i], set()) if held else data.orders[i])
        safe_shipments.append(Guard(data.shipments[i], set()) if held else data.shipments[i])

    # The model never needs Dataset.correct; all its labels are computed on training.
    class TrainingSource:
        requests = safe_requests
        orders = safe_orders
        shipments = safe_shipments
        policies = data.policies
        policy_at = data.policy_at

        @property
        def correct(self):
            raise AssertionError("read all labels instead of training labels")

    guarded = ScenarioModel.fit(TrainingSource())
    assert all(at < CUTS[0] for at in guarded.training_times)
    assert guarded.fingerprint == ScenarioModel.fit().fingerprint
    build_suite(AS_OF, 1, 200, 7)
    list_suite(AS_OF)


def test_listing_is_exactly_the_valid_case_space():
    suite = list_suite(AS_OF)
    cases = [row.tokens[:5] for row in suite.scenarios]
    assert len(cases) == len(set(cases)) == 252
    assert set(cases) == set(valid_cases())
    assert suite.manifest["counts"]["rejected"] == 0
    assert all(row.generator_decision is None and row.origin == "listed" for row in suite.scenarios)
    data = training_data()
    assert all(row.label == policy_label(row.records, data) for row in suite.scenarios)


@pytest.mark.parametrize("tau", [None, 1, 2])
def test_suite_json_roundtrip_and_regeneration_are_byte_identical(tau, tmp_path):
    suite = list_suite(AS_OF) if tau is None else build_suite(AS_OF, tau, 200, 7)
    assert suite.to_json() == regenerate(suite.manifest).to_json()
    path = tmp_path / "suite.json"
    suite.write(path)
    assert ScenarioSet.read(path).to_json() == path.read_text() == suite.to_json()
    altered = dict(suite.manifest, model_sha256="wrong")
    with pytest.raises(ValueError, match="manifest"):
        regenerate(altered)


def test_same_seed_is_deterministic_and_review_sample_is_only_a_selection(model):
    first = build_suite(AS_OF, 1, 200, 7)
    assert first.to_json() == build_suite(AS_OF, 1, 200, 7).to_json()
    assert first.to_json() != build_suite(AS_OF, 1, 200, 8).to_json()
    chosen = first.review_sample(5, 31)
    assert chosen == first.review_sample(5, 31)
    assert len(chosen) == 5
    assert all(row in first.scenarios for row in chosen)
    assert first.manifest["human_review"]["status"] == "pending"


def test_existing_consultation_journal_is_unchanged(monkeypatch, tmp_path):
    # No frozen rows are needed to verify that an existing journal is preserved.
    journal = tmp_path / "golden-1.access.jsonl"
    prior = b'{"entry":1,"reader":"earlier-chapter","set":"golden-1"}\n'
    journal.write_bytes(prior)
    monkeypatch.setenv("BOOK_GOLDEN_DIRECTORY", str(tmp_path))

    def forbidden(*args, **kwargs):
        raise AssertionError("scenario construction consulted the frozen set")

    monkeypatch.setattr(GoldenSet, "consult", forbidden)
    for tau in (1, 2):
        build_suite(AS_OF, tau, 200, 7)
    list_suite(AS_OF)
    assert journal.read_bytes() == prior
    assert list(tmp_path.iterdir()) == [journal]


@pytest.mark.parametrize("listed", [False, True])
@pytest.mark.parametrize("prior_reads", [0, 2])
def test_golden_consultation_is_counted_and_fails_provenance(monkeypatch, listed, prior_reads):
    from refund_agent.scenarios import suite as module
    from refund_agent.split import freeze

    tracked = []
    checked = []

    def track(*args, **kwargs):
        golden = freeze(*args, **kwargs)
        for _ in range(prior_reads):
            golden.consult("earlier reader", "preserve prior history")
        tracked.append(golden)
        return golden

    def consult_then_check(*args, **kwargs):
        if tracked and len(tracked[0].log) == prior_reads:
            tracked[0].consult("scenario regression", "detect a forbidden read")
        return check_schema(*args, **kwargs)

    def provenance(suite, model):
        checked.append(suite.manifest["golden_reads"])
        return check_provenance(suite, model)

    monkeypatch.setattr(module, "freeze", track, raising=False)
    monkeypatch.setattr(module, "check_schema", consult_then_check)
    monkeypatch.setattr(module, "check_provenance", provenance)
    with pytest.raises(ValueError, match="scenario provenance failed"):
        list_suite(AS_OF) if listed else build_suite(AS_OF, 1, 25, 7)
    assert checked == [len(tracked[0].log)] == [prior_reads + 1]


def test_canonical_generator_requests_and_labels_are_unchanged():
    from refund_agent.data.generate import generate

    data = generate()
    assert len(data.requests) == 600
    assert hashlib.sha256(data.model_dump_json().encode()).hexdigest() == (
        "5fbe654e0f0140b9f5836ba220714d6d4afd36bce50edbeb819e3dafdd7f6b4a"
    )
