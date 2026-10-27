"""Dates, provenance, reconstruction and offline ranking contracts."""

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from refund_agent.data.generate import generate
from refund_agent.retrieval.clauses import policy_clauses
from refund_agent.retrieval.encoders import CountEncoder, LinearAutoencoder, PairEncoder, cosine
from refund_agent.retrieval.evaluate import mean_reciprocal_rank, recall_at_k, superseded_returned
from refund_agent.retrieval.index import PolicyIndex
from refund_agent.retrieval.questions import policy_questions, question_requests

AT = datetime(2026, 8, 15, tzinfo=timezone.utc)


@pytest.fixture(scope="module")
def data():
    return generate(7)


@pytest.fixture(scope="module")
def index(data):
    texts = [c.text for c in policy_clauses(data.policies, AT)]
    return PolicyIndex.build(data.policies, AT, CountEncoder().fit(texts))


def test_count_vectors_and_zero_or_unseen_words(index):
    clauses = index.clauses
    assert len(index.encoder.vocabulary) == 26
    eligibility = [c for c in clauses if c.section == "eligibility"]
    assert cosine(eligibility[0].vector, eligibility[1].vector) == pytest.approx(13 / 14)
    assert cosine(eligibility[2].vector, eligibility[3].vector) == pytest.approx(1)
    assert cosine(eligibility[0].vector, clauses[1].vector) == 0
    query = index.encoder.transform(["changed_mind 25 days after delivery"])[0]
    assert [cosine(query, c.vector) for c in eligibility] == pytest.approx([4 / np.sqrt(42)] * 5)
    assert cosine(query, np.zeros_like(query)) == 0
    assert index.retrieve("xylophone", AT, 3).hits == ()
    assert index.retrieve("", AT, 3).hits == ()


@pytest.mark.parametrize("offset", [-1, 0, 1])
def test_no_clause_outside_its_interval_at_every_boundary(data, offset):
    encoder = CountEncoder().fit([c.text for c in policy_clauses(data.policies, AT)])
    for policy in data.policies:
        at = policy.effective_at + timedelta(seconds=offset)
        built = PolicyIndex.build(data.policies, at, encoder)
        result = built.retrieve("damaged days delivery", at, 20)
        if at < data.policies[0].effective_at:
            assert result.refusal == "no_clause_in_force"
        else:
            assert result.hits
            for hit in result.hits:
                clause = hit.clause
                assert clause.effective_from <= at
                assert clause.effective_to is None or at < clause.effective_to
                assert clause.version == data.policy_at(at).version


@pytest.mark.parametrize(
    "age,refusal", [(0, None), (2592000, None), (2592001, "stale_policy"), (-1, "future_policy")]
)
def test_freshness_uses_the_exact_30_day_boundary(data, age, refusal):
    encoder = CountEncoder().fit([c.text for c in policy_clauses(data.policies, AT)])
    built = PolicyIndex.build(data.policies, AT - timedelta(seconds=age), encoder)
    result = built.retrieve("damaged delivery", AT, 1)
    assert result.refusal == refusal
    assert bool(result.hits) == (refusal is None)


def test_every_hit_keeps_its_record_and_manifest_tracks_the_build(index, data):
    result = index.retrieve("changed_mind 40 days after delivery", AT, 1)
    record = result.hits[0].clause.record()
    assert record["citation"] == "refund_cap_v4@2026-07-01#eligibility"
    assert record["effective_from"] == "2026-07-01T00:00:00+00:00"
    assert record["effective_to"] == "2026-09-01T00:00:00+00:00"
    assert record["observed_at"] == AT.isoformat()
    same = PolicyIndex.build(data.policies, AT, index.encoder)
    assert same.manifest == index.manifest
    assert len(index.manifest["sha256"]) == 64
    later = PolicyIndex.build(data.policies, AT + timedelta(seconds=1), index.encoder)
    assert later.manifest["sha256"] != index.manifest["sha256"]
    changed = CountEncoder(rarity=True).fit([c.text for c in index.clauses])
    assert (
        PolicyIndex.build(data.policies, AT, changed).manifest["sha256"] != index.manifest["sha256"]
    )


def test_encoder_is_detached_when_index_is_built(data):
    encoder = CountEncoder().fit([c.text for c in policy_clauses(data.policies, AT)])
    built = PolicyIndex.build(data.policies, AT, encoder)
    before = built.retrieve("delivery", AT, 1)
    encoder.fit(["entirely different text"])
    assert built.retrieve("delivery", AT, 1) == before


def test_ties_are_stable_under_permuted_input_and_unknown_intervals(data):
    encoder = CountEncoder().fit([c.text for c in policy_clauses(data.policies, AT)])
    assert (
        PolicyIndex.build(list(reversed(data.policies)), AT, encoder).manifest
        == PolicyIndex.build(data.policies, AT, encoder).manifest
    )
    with pytest.raises(ValueError, match="duplicate"):
        policy_clauses(data.policies + data.policies[:1], AT)
    with pytest.raises(ValueError, match="anchors"):
        policy_clauses([data.policies[0].model_copy(update={"text": "no sections"})], AT)
    with pytest.raises(ValueError, match="timezone"):
        policy_clauses(data.policies, AT.replace(tzinfo=None))
    with pytest.raises(ValueError):
        PolicyIndex.build(data.policies, AT, encoder).retrieve("delivery", AT, 0)


@pytest.mark.parametrize("width", [1, 2])
def test_closed_form_and_untied_gradient_descent_agree(data, width):
    encoder = CountEncoder().fit([c.text for c in policy_clauses(data.policies, AT)])
    x = encoder.transform([c.text for c in policy_clauses(data.policies, AT)])
    closed = LinearAutoencoder(width).fit(x)
    trained = LinearAutoencoder(width).fit(x, method="gradient_descent", seed=7)
    assert trained.error(x) == pytest.approx(closed.error(x), abs=1e-7)
    same = LinearAutoencoder(width).fit(x, method="gradient_descent", seed=7)
    assert np.array_equal(same.weights, trained.weights)
    assert np.array_equal(same.decoder, trained.decoder)
    assert closed.error(x) == pytest.approx(sum(closed.eigenvalues[width:]))


def test_projection_pythagoras_and_centering():
    x = np.array([[2.0, 0.0], [-2.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
    ae = LinearAutoencoder(1).fit(x)
    assert ae.eigenvalues.tolist() == pytest.approx([2, 0.5])
    assert ae.error(x) == pytest.approx(0.5)
    h = ae.transform([[3, 4]])
    assert h.tolist() == [[3.0]]
    assert ae.reconstruct(h).tolist() == [[3.0, 0.0]]
    assert 25 == float(np.sum(h**2)) + 16
    shifted = x + [10, -6]
    moved = LinearAutoencoder(1).fit(shifted)
    assert moved.error(shifted) == pytest.approx(ae.error(x))


def test_two_questions_use_input_facts_and_only_the_labels_version(data):
    requests = data.requests[:30]
    labels = [d for d in data.correct if d.request_id in {r.request_id for r in requests}]
    rows = question_requests(data, requests)
    for wording in ("policy", "customer"):
        questions = policy_questions(rows, labels, wording)
        assert len(questions) == 2 * len(requests)
        for row, pair in zip(rows, zip(questions[::2], questions[1::2])):
            assert [q.section for q in pair] == ["eligibility", "cap"]
            assert all(q.as_of == row.request.at for q in pair)
            assert all(
                q.relevant.startswith(f"refund_cap_v4@{data.policy_at(row.request.at).version}#")
                for q in pair
            )
            assert all("refund_cap_v4" not in q.text and "2026-" not in q.text for q in pair)
    changed = [
        d.model_copy(update={"reason": "not a feature", "outcome": d.outcome}) for d in labels
    ]
    assert policy_questions(rows, changed, "policy") == policy_questions(rows, labels, "policy")
    # A shipment event after the request may not provide an age of delivery.
    r = requests[5]
    order = next(o for o in data.orders if o.order_id == r.order_reference)
    shipment = next(s for s in data.shipments if s.order_id == order.order_id)
    future = shipment.model_copy(
        update={
            "events": [
                e.model_copy(update={"at": r.at + timedelta(days=1)}) for e in shipment.events
            ]
        }
    )
    changed_data = data.model_copy(update={"shipments": [future]})
    assert question_requests(changed_data, [r])[0].days_since_delivery is None


def test_pair_encoder_is_seeded_and_learns_pairs():
    pairs = [("broken parcel", "damaged goods"), ("maximum payback", "payment cap")]
    first = PairEncoder(2).fit(pairs, seed=7)
    second = PairEncoder(2).fit(pairs, seed=7)
    assert np.array_equal(first.weights, second.weights)
    assert first.losses[-1] < first.losses[0]
    vectors = first.transform([text for pair in pairs for text in pair])
    assert cosine(vectors[0], vectors[1]) > cosine(vectors[0], vectors[3])


def test_pair_loss_gradient_matches_a_finite_difference():
    model = PairEncoder(2).fit(
        [("broken parcel", "damaged goods"), ("maximum payback", "payment cap")], steps=1
    )
    q = model.counts.transform(["broken parcel", "maximum payback"])
    p = model.counts.transform(["damaged goods", "payment cap"])
    targets = np.array([0, 1])
    _, gradient = model.loss_gradient(q, p, targets)
    for coordinate in ((0, 0), (3, 1), (6, 0)):
        old = model.weights[coordinate]
        step = 1e-6
        model.weights[coordinate] = old + step
        plus = model.loss_gradient(q, p, targets)[0]
        model.weights[coordinate] = old - step
        minus = model.loss_gradient(q, p, targets)[0]
        model.weights[coordinate] = old
        assert gradient[coordinate] == pytest.approx((plus - minus) / (2 * step), abs=1e-6)


def test_ranking_measures_use_question_denominators(index):
    hit = index.retrieve("delivery", AT, 1).hits[0]
    rankings = [[hit], [], [index.retrieve("payment", AT, 1).hits[0], hit]]
    relevant = [hit.clause.citation] * 3
    assert recall_at_k(rankings, relevant, 1) == pytest.approx(1 / 3)
    assert recall_at_k(rankings, relevant, 3) == pytest.approx(2 / 3)
    assert mean_reciprocal_rank(rankings, relevant) == pytest.approx(0.5)
    assert superseded_returned(rankings, [AT] * 3) == 0
    assert superseded_returned([[hit]], [datetime(2026, 9, 1, tzinfo=timezone.utc)]) == 1
    with pytest.raises(ValueError):
        recall_at_k([], relevant, 1)


def test_generator_and_labels_are_unchanged(data):
    from demos import autoencoders_nlp_demo as demo

    before = data.model_dump_json()
    demo.build_experiment(7)
    assert data.model_dump_json() == before == generate(7).model_dump_json()
