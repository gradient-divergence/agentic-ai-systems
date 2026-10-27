"""Chapter 14's compression and cost examples use fixed, offline setups."""

import contextlib
import io
from pathlib import Path

import numpy as np
import pytest

from demos import adaptation_demo as demo
from demos import regions as _regions

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "demos/adaptation_demo.py"
PREFIX = "fnd.prompting-finetuning-rag."
TOOLS = vars(_regions)


@pytest.mark.parametrize("bits", [4, 8])
def test_quantization_packs_real_codes_and_respects_half_step(bits):
    weights = np.array([[-1.0, -0.63], [0.12, 0.87]])
    original = weights.copy()
    rounded, step, payload = demo.quantize_weights(weights, bits)
    limit = 2 ** (bits - 1) - 1
    assert step == pytest.approx(1 / limit)
    assert len(payload) == weights.size * bits // 8
    assert np.max(np.abs(rounded - weights)) <= step / 2 + 1e-15
    raw = np.unpackbits(np.frombuffer(payload, dtype=np.uint8))
    codes = raw.reshape(-1, bits) @ (2 ** np.arange(bits - 1, -1, -1))
    decoded = (codes.astype(float) - limit).reshape(weights.shape) * step
    np.testing.assert_array_equal(rounded, decoded)
    np.testing.assert_array_equal(weights, original)


@pytest.mark.parametrize("weights,bits", [([], 8), ([np.nan], 8), ([1.0], 3)])
def test_quantization_rejects_invalid_inputs(weights, bits):
    with pytest.raises(ValueError):
        demo.quantize_weights(weights, bits)


def test_quantization_handles_zero_weights_without_inventing_a_scale():
    rounded, step, payload = demo.quantize_weights(np.zeros((2, 2)), 4)
    np.testing.assert_array_equal(rounded, np.zeros((2, 2)))
    assert step == 0
    assert len(payload) == 2


@pytest.fixture
def lowrank_data():
    return (
        np.eye(16) * 0.1,
        np.zeros((16, 16)),
        np.full((768, 16), 0.25),
        np.full((768, 16), 0.5),
        np.full((2048, 16), 0.75),
        np.full((2048, 16), 1.0),
        0.1,
    )


def test_quantization_fits_only_the_existing_training_rows(monkeypatch, lowrank_data):
    original = demo.fit
    expected_w0, _, x_train, y_train, *_ = lowrank_data
    calls = []

    def controlled_data(seed):
        assert seed == 7
        return lowrank_data

    def guarded(x, y, w0, rank, **settings):
        assert x.shape == y.shape == (768, 16)
        np.testing.assert_array_equal(x, x_train)
        np.testing.assert_array_equal(y, y_train)
        np.testing.assert_array_equal(w0, expected_w0)
        assert w0.shape == (16, 16)
        assert rank == 2
        assert settings == {"nu": 8, "seed": 7, "steps": 4000, "step_size": 0.1}
        calls.append(x)
        return original(x, y, w0, rank, **settings)

    monkeypatch.setattr(demo, "lowrank_data", controlled_data)
    monkeypatch.setattr(demo, "fit", guarded)
    report = demo.quantization_experiment(7)
    assert len(calls) == 1
    assert report["n_train"] == 768 and report["n_test"] == 2048
    assert report["noise_floor"] == pytest.approx(0.01)
    assert report["reference_bytes"] == 16 * 16 * 4
    assert [row["bits"] for row in report["results"]] == [8, 4]
    assert [row["payload_bytes"] for row in report["results"]] == [256, 128]
    for row in report["results"]:
        assert row["max_error"] <= row["half_step"] + 1e-12
        assert row["metadata_bytes"] == 8
        assert row["test_mse"] > 0


def test_soft_targets_match_the_example_and_are_shift_invariant():
    for temperature, expected in (
        (1, [0.66524096, 0.24472847, 0.09003057]),
        (2, [0.50648039, 0.30719589, 0.18632372]),
    ):
        result = demo.soft_targets([2, 1, 0], temperature)
        np.testing.assert_allclose(result, expected, atol=5e-9)
        np.testing.assert_allclose(result, demo.soft_targets([1002, 1001, 1000], temperature))
        assert result.sum() == pytest.approx(1)


@pytest.mark.parametrize("temperature", [0, -1, np.nan])
def test_soft_targets_require_a_positive_finite_temperature(temperature):
    with pytest.raises(ValueError):
        demo.soft_targets([2, 1, 0], temperature)


def test_cost_includes_human_rework_and_checks_the_acceptance_share():
    assert demo.cost_per_accepted_draft(500, 0.9, 0.001, 4) == pytest.approx(0.9)
    assert demo.cost_per_accepted_draft(400, 0.8, 0.001, 4) == pytest.approx(1.2)
    assert demo.cost_per_accepted_draft(500, 1, 0, 4) == 0
    with pytest.raises(ValueError):
        demo.cost_per_accepted_draft(500, 1.1, 0.001, 4)


@pytest.fixture(scope="module")
def experiment():
    return demo.build_experiment(7)


def test_refund_costs_use_unrounded_run_counts_and_word_means(experiment):
    costs = demo.adaptation_costs(experiment, word_price=0.001, review_cost=4)
    for row in costs["arms"]:
        name = row["arm"]
        words = demo.words_per_draft(experiment["runs"][name])["context"]
        correct = sum(experiment["agreements"][name])
        assert row["mean_words"] == words
        assert row["accepted"] == correct
        assert row["cases"] == len(experiment["spent_rows"])
        assert row["word_cost"] == pytest.approx(0.001 * words)
        assert row["rework_cost"] == pytest.approx(4 * (1 - correct / row["cases"]))
        assert row["total_cost"] == pytest.approx(row["word_cost"] + row["rework_cost"])
    prompt, _, adapter = costs["arms"]
    price = costs["break_even_word_price"]
    assert demo.cost_per_accepted_draft(prompt["mean_words"], 1, price, 4) == pytest.approx(
        demo.cost_per_accepted_draft(adapter["mean_words"], 83 / 121, price, 4)
    )


@pytest.mark.parametrize("name", ["quantize-01", "cost-01", "keys-02"])
def test_new_printable_regions_run_alone_and_match_stored_output(name):
    regions = TOOLS["extract_snippets_from_text"](SOURCE.read_text(), SOURCE)
    region = next(region for region in regions if region.snippet_id == PREFIX + name)
    assert region.output
    assert max(map(len, region.content.splitlines())) <= 79
    with contextlib.redirect_stdout(io.StringIO()) as stdout:
        exec(compile(region.content, str(SOURCE), "exec"), {"__name__": "__main__"})
    stored = ROOT / "book_outputs" / f"_{PREFIX}{name}.txt"
    assert stdout.getvalue().encode() == stored.read_bytes()
