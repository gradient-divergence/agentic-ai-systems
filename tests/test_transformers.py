import builtins
import contextlib
import io
import json
import math
import re
import runpy
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from test_regressions import region

from demos import regions as _regions

SOURCE = "demos/deep_learning_demo.py"
PREFIX = "fnd.transformers-foundation-models."
ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / SOURCE
TOOLS = vars(_regions)
OUTPUT_NAMES = (
    "recurrence-01",
    "bptt-01",
    "lstm-01",
    "rollout-01",
    "core-01",
    "example-01",
    "scale-01",
    "positions-01",
    "block-01",
    "loss-01",
    "decoder-01",
    "contrastive-01",
    "icl-01",
    "rollout-02",
    "sota-01",
    "serving-01",
    "keys-07",
    "keys-12",
)
PRINTED = OUTPUT_NAMES


def snippets():
    return TOOLS["extract_snippets_from_text"](DEMO.read_text(), DEMO)


def test_printed_listings_run_in_chapter_order_without_hidden_setup() -> None:
    namespace: dict[str, Any] = {"__name__": "__main__"}
    for name in PRINTED:
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            region(SOURCE, PREFIX + name, namespace)
        stored = ROOT / "book_outputs" / f"_{PREFIX}{name}.txt"
        assert stdout.getvalue().encode() == stored.read_bytes()
    assert namespace["block_output"].shape == (5, 8)


@pytest.fixture(scope="module")
def captured():
    return TOOLS["capture_outputs"]({s.snippet_id: s for s in snippets()}, sys.executable)


@pytest.mark.parametrize("name", OUTPUT_NAMES)
def test_every_stored_transformer_output_regenerates_byte_for_byte(name, captured):
    stored = ROOT / "book_outputs" / f"_{PREFIX}{name}.txt"
    assert captured[PREFIX + name].encode() == stored.read_bytes()


@pytest.mark.parametrize("name", (
    "lstm-01", "contrastive-01", "icl-01", "rollout-02", "serving-01", "keys-12",
))
def test_enrichment_output_lines_fit_the_printed_page(name, captured):
    for number, line in enumerate(captured[PREFIX + name].splitlines(), 1):
        assert len(line) <= 79, f"{name}:{number}: {len(line)} columns: {line}"


def test_capture_is_deterministic_in_fresh_processes(captured):
    assert captured == TOOLS["capture_outputs"](
        {s.snippet_id: s for s in snippets()}, sys.executable
    )


def test_new_output_regions_follow_outline_order_and_preserve_existing_outputs():
    found = [s for s in snippets() if s.output]
    assert [s.snippet_id for s in found] == [PREFIX + name for name in OUTPUT_NAMES]


def test_recurrence_reads_both_orders_and_starts_fresh(capsys):
    ns = region(SOURCE, PREFIX + "recurrence-01")
    assert capsys.readouterr().out == (
        "w=0.5; h0=0.0; paid=1.0; returned=0.5\n"
        "paid, returned: states 0.762 0.707; code sum 1.5\n"
        "returned, paid: states 0.462 0.843; code sum 1.5\n"
    )
    run = ns["run_recurrence"]
    start = np.array([0.0])
    forward = np.array([[1.0], [0.5]])
    reverse = forward[::-1]
    weights = (np.array([[1.0]]), np.array([[0.5]]), np.array([0.0]))
    first = run(forward, *weights, start)
    second = run(reverse, *weights, start)
    np.testing.assert_allclose(first[:, 0], [0.7616, 0.7068], rtol=0, atol=1e-4)
    np.testing.assert_allclose(second[:, 0], [0.4621, 0.8429], rtol=0, atol=1e-4)
    assert first[-1, 0] != second[-1, 0]
    assert forward.sum() == reverse.sum() == 1.5
    np.testing.assert_array_equal(start, [0.0])
    np.testing.assert_array_equal(run(forward, *weights, start), first)
    assert run(np.empty((0, 1)), *weights, start).shape == (0, 1)


def test_recurrent_step_matches_the_column_formula_on_seeded_arrays():
    step = region(SOURCE, PREFIX + "recurrence-01")["recurrent_step"]
    rng = np.random.default_rng(7)
    x, h = rng.normal(size=4), rng.normal(size=3)
    W_xh, W_hh, b = rng.normal(size=(3, 4)), rng.normal(size=(3, 3)), rng.normal(size=3)
    result = step(x, h, W_xh, W_hh, b)
    assert result.shape == (3,)
    np.testing.assert_array_equal(result, np.tanh(W_hh @ h + W_xh @ x + b))


@pytest.mark.parametrize("w", [0.5, 1.0, 1.2, -1.2])
def test_gradient_is_a_power_without_tanh_and_a_bounded_checked_product_with_it(w):
    ns = region(SOURCE, PREFIX + "recurrence-01")
    ns = region(SOURCE, PREFIX + "bptt-01", ns)
    gradient, difference = ns["gradient_through_time"], ns["central_difference"]
    linear, final = gradient(w, 0.1, 10, tanh=False)
    assert linear == w**10
    assert final == 0.1 * w**10
    product, final = gradient(w, 0.1, 10)
    assert abs(product) <= abs(w) ** 10
    assert product == pytest.approx(difference(w, 0.1, 10, 1e-6), abs=1e-8, rel=0)
    assert abs(final) <= 1


def test_bptt_prints_the_products_checks_and_path_lengths(capsys):
    ns = region(SOURCE, PREFIX + "recurrence-01")
    capsys.readouterr()
    region(SOURCE, PREFIX + "bptt-01", ns)
    assert capsys.readouterr().out == (
        "n=10; h0=0.1; eps=0.000001\n"
        "w=0.5: linear=0.000977; tanh=0.000973; central=0.000973\n"
        "w=1.0: linear=1.000000; tanh=0.907591; central=0.907591\n"
        "w=1.2: linear=6.191736; tanh=2.483603; central=2.483603\n"
        "tokens 5: recurrent links 4, causal-attention links 1\n"
        "tokens 2048: recurrent links 2047, causal-attention links 1\n"
    )


def test_rollout_compounds_only_when_it_reads_its_predictions(capsys):
    ns = region(SOURCE, PREFIX + "recurrence-01")
    capsys.readouterr()
    ns = region(SOURCE, PREFIX + "rollout-01", ns)
    assert capsys.readouterr().out == (
        "level=100; factor=1.05; steps=7\n"
        "step | truth prediction error | free input prediction error\n"
        "   1 |   105.00  5.00 |   100.00   105.00  5.00\n"
        "   2 |   105.00  5.00 |   105.00   110.25 10.25\n"
        "   3 |   105.00  5.00 |   110.25   115.76 15.76\n"
        "   4 |   105.00  5.00 |   115.76   121.55 21.55\n"
        "   5 |   105.00  5.00 |   121.55   127.63 27.63\n"
        "   6 |   105.00  5.00 |   127.63   134.01 34.01\n"
        "   7 |   105.00  5.00 |   134.01   140.71 40.71\n"
        "largest input: truth=100.00; free=134.01\n"
        "day-7 / day-1 free error: 8.1\n"
    )
    rollout = ns["rollout"]
    forced, free = ns["forced"], ns["free"]
    np.testing.assert_array_equal(forced, np.tile([100.0, 105.0, 5.0], (7, 1)))
    np.testing.assert_allclose(free[:, 2], 100 * (1.05 ** np.arange(1, 8) - 1), atol=1e-12)
    np.testing.assert_array_equal(free[1:, 0], free[:-1, 1])
    np.testing.assert_array_equal(rollout(100, 1, 7, False)[:, 2], 0)
    assert rollout(100, 1.05, 0, False).shape == (0, 3)


def test_exercise_seven_prints_the_power_and_free_running_errors(capsys):
    ns = region(SOURCE, PREFIX + "recurrence-01")
    ns = region(SOURCE, PREFIX + "bptt-01", ns)
    ns = region(SOURCE, PREFIX + "rollout-01", ns)
    capsys.readouterr()
    ns = region(SOURCE, PREFIX + "keys-07", ns)
    assert ns["gradient_through_time"](0.8, 0.1, 20, tanh=False)[0] == 0.8**20
    assert ns["exercise_rollout"][0, 2] == 1
    assert ns["exercise_rollout"][-1, 2] == pytest.approx(50 * (1.02**10 - 1), abs=1e-12)
    assert capsys.readouterr().out == (
        "gradient factor (w=0.8, n=20): 0.0115\n"
        "level=50; factor=1.02; steps=10\n"
        "one-step error: 1.00\n"
        "ten-step free-running error: 10.95\n"
    )


def test_chapter_arithmetic_reads_no_external_or_frozen_data(monkeypatch):
    found = {s.snippet_id: s for s in snippets()}

    def forbidden(*args, **kwargs):
        raise AssertionError("Chapter arithmetic must not read data files.")

    real_import = builtins.__import__

    def numpy_only(name, *args, **kwargs):
        assert name == "numpy", f"Unexpected chapter dependency: {name}"
        return real_import(name, *args, **kwargs)

    namespace = {
        "__builtins__": {**vars(builtins), "open": forbidden, "__import__": numpy_only},
        "__name__": "__main__",
    }
    with monkeypatch.context() as guard:
        guard.setattr(builtins, "open", forbidden)
        guard.setattr(Path, "open", forbidden)
        for name in ("load", "loadtxt", "genfromtxt", "fromfile"):
            guard.setattr(np, name, forbidden)
        with contextlib.redirect_stdout(io.StringIO()):
            for name in OUTPUT_NAMES:
                exec(compile(found[PREFIX + name].content, str(DEMO), "exec"), namespace)


def test_chapter_outputs_do_not_depend_on_ambient_numpy_seed():
    found = {s.snippet_id: s for s in snippets()}
    outputs = []
    state = np.random.get_state()
    try:
        for seed in (7, 7, 29):
            np.random.seed(seed)
            namespace = {"__name__": "__main__"}
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                for name in OUTPUT_NAMES:
                    exec(compile(found[PREFIX + name].content, str(DEMO), "exec"), namespace)
            outputs.append(stdout.getvalue())
    finally:
        np.random.set_state(state)
    assert outputs[0] == outputs[1] == outputs[2]


def enrichment_namespace(name):
    found = {s.snippet_id: s for s in snippets()}
    assert PREFIX + name in found, f"Missing region: {name}"
    namespace = {"__name__": "__main__"}
    with contextlib.redirect_stdout(io.StringIO()):
        for current in OUTPUT_NAMES[:OUTPUT_NAMES.index(name) + 1]:
            assert PREFIX + current in found, f"Missing region: {current}"
            exec(compile(found[PREFIX + current].content, str(DEMO), "exec"), namespace)
    return namespace


def test_lstm_and_gru_gates_match_hand_formulas_with_nonzero_previous_state(captured):
    ns = enrichment_namespace("lstm-01")
    assert captured[PREFIX + "lstm-01"].splitlines()[4:6] == [
        "LSTM gates: forget=0.731; input=0.622; output=0.818",
        "LSTM values: candidate=0.762; cell=0.474; state=0.361",
    ]

    def sigmoid(x):
        return 1 / (1 + math.exp(-x))

    W = np.array([1., 2., -1., .5])
    U = np.array([.5, -1., 2., 1.])
    biases = np.array([.1, .2, -.3, .4])
    result = ns["lstm_step"](.5, .2, -.4, W, U, biases)
    f, i, o, candidate = sigmoid(.7), sigmoid(1.), sigmoid(-.4), math.tanh(.85)
    cell = f * -.4 + i * candidate
    np.testing.assert_allclose(
        list(result.values()), [f, i, o, candidate, cell, o * math.tanh(cell)]
    )
    result = ns["gru_step"](.5, .2, W[:3], U[:3], biases[:3])
    r, z = sigmoid(.7), sigmoid(1.)
    candidate = math.tanh(-.5 + 2 * (r * .2) - .3)
    np.testing.assert_allclose(
        list(result.values()), [r, z, candidate, z * .2 + (1 - z) * candidate]
    )
    assert ns["lstm_values"]["cell"] == pytest.approx(sigmoid(.5) * math.tanh(1))
    assert ns["gru_values"]["state"] == pytest.approx((1 - sigmoid(.5)) * math.tanh(1))


@pytest.mark.parametrize("d,d_h", [(1, 1), (3, 2), (512, 512)])
def test_recurrent_weight_counts_include_each_gate_bias(d, d_h):
    ns = enrichment_namespace("lstm-01")
    count = ns["recurrent_weight_count"]
    plain = d_h * (d + d_h) + d_h
    assert [count(d, d_h, gates) for gates in (4, 3, 1)] == [4 * plain, 3 * plain, plain]
    assert [count(512, 512, gates) for gates in (4, 3, 1)] == [2_099_200, 1_574_400, 524_800]
    np.testing.assert_allclose(ns["cell_paths"], .95 ** np.arange(1, 11))
    np.testing.assert_allclose(ns["plain_paths"], .5 ** np.arange(1, 11))


def test_contrastive_pairs_normalize_both_directions_and_swapping_raises_loss():
    ns = enrichment_namespace("contrastive-01")
    np.testing.assert_allclose(np.linalg.norm(ns["image_vectors"], axis=1), 1)
    np.testing.assert_allclose(np.linalg.norm(ns["caption_vectors"], axis=1), 1)
    cosine, rows, columns, losses = ns["contrastive_results"]["matched"]
    np.testing.assert_allclose(cosine, [[.8, 0, .6], [.6, .8, 0], [0, .6, .8]])
    np.testing.assert_array_equal(cosine.argmax(axis=1), [0, 1, 2])
    denominator = math.exp(8) + math.exp(6) + 1
    np.testing.assert_allclose(rows, np.exp(cosine / .1) / denominator)
    np.testing.assert_allclose(rows.sum(axis=1), 1)
    np.testing.assert_allclose(columns.sum(axis=0), 1)
    np.testing.assert_allclose(losses, [math.log(denominator) - 8] * 3)
    swapped = ns["contrastive_results"]["swapped"]
    np.testing.assert_allclose(swapped[0], cosine[:, [0, 2, 1]])
    np.testing.assert_allclose(swapped[1].sum(axis=1), 1)
    np.testing.assert_allclose(swapped[2].sum(axis=0), 1)
    np.testing.assert_allclose(swapped[3], [losses[0] + 10 / 3] * 3)
    assert swapped[3][2] > losses[2]
    # Unequal column normalizers catch accidentally using row loss twice.
    images = np.eye(3)
    captions = np.array([[1., 0, 0], [.6, .8, 0], [0, 0, 1.]])
    cosine, rows, columns, losses = ns["contrastive_comparison"](images, captions, .5)
    scores = np.exp(cosine / .5)
    expected_rows = scores / scores.sum(axis=1, keepdims=True)
    expected_columns = scores / scores.sum(axis=0, keepdims=True)
    np.testing.assert_allclose(rows, expected_rows)
    np.testing.assert_allclose(columns, expected_columns)
    np.testing.assert_allclose(
        losses[:2],
        [-np.log(np.diag(expected_rows)).mean(), -np.log(np.diag(expected_columns)).mean()],
    )
    assert losses[2] == pytest.approx(sum(losses[:2]) / 2)


def test_in_context_reading_moves_predictions_while_projection_weights_stay_fixed():
    ns = enrichment_namespace("icl-01")
    first, second = ns["icl_results"]
    a, b = math.exp(1 / math.sqrt(2)), 1.
    expected_a = np.array([a, b, a]) / (2 * a + b)
    expected_b = np.array([b, a, b]) / (a + 2 * b)
    np.testing.assert_allclose(first[0], expected_a)
    np.testing.assert_allclose(second[0], expected_b)
    assert first[1] == pytest.approx(expected_a @ [0, 10, 20])
    assert second[1] == pytest.approx(expected_b @ [10, 0, 40])
    assert first[1] != second[1]
    assert first[2] == second[2] == 5.
    for expected, after in zip((np.eye(2), np.eye(2), np.ones((1, 1))), ns["icl_projections"]):
        np.testing.assert_array_equal(expected, after)


def test_generation_rollout_reuses_the_original_random_head_and_appends_greedy_tokens():
    ns = enrichment_namespace("rollout-02")
    trace = ns["generation_trace"]
    np.testing.assert_array_equal(trace[0][0], ns["token_ids"])
    assert trace[0][1] == ns["appended_ids"][-1] == 0
    assert trace[0][2] == ns["decoder_probabilities"][-1, 0]
    assert trace[0][2] == pytest.approx(.262324, abs=5e-7)
    assert len(trace) == 3
    for context, action, probability, appended in trace:
        hidden = ns["decoder_forward"](
            context, ns["embedding_table"], ns["decoder_weights"]
        )["block output"][-1]
        scores = hidden @ ns["head_before"]
        expected = np.exp(scores - scores.max())
        expected /= expected.sum()
        assert action == expected.argmax()
        assert probability == expected[action]
        np.testing.assert_array_equal(appended, np.r_[context, action])
    for earlier, later in zip(trace, trace[1:]):
        np.testing.assert_array_equal(earlier[3], later[0])


def test_serving_weight_cache_bytes_intensity_and_batches_follow_closed_forms():
    ns = enrichment_namespace("serving-01")
    assert ns["serving_weight_count"] == 75_497_472
    for row, expected_batch in zip(ns["serving_rows"], [9, 19, 41]):
        b, weights, cache_token, cache_request, left, batch = row
        assert weights == 75_497_472 * b
        assert cache_token == 2 * 24 * 8 * 64 * b
        assert cache_request == cache_token * 2048
        assert left == 2**30 - weights
        assert batch == left // cache_request == expected_batch
        assert weights + batch * cache_request <= 2**30
        assert weights + (batch + 1) * cache_request > 2**30
    assert ns["prefill_intensity"] == 2048 / 2
    assert ns["batch_table"] == [(n, n / 2) for n in (1, 2, 4, 8, 16, 32, 64)]


def test_independent_slow_call_formula_agrees_with_constructed_latency_simulation():
    ns = enrichment_namespace("serving-01")
    for k, chance in ns["tail_table"]:
        assert chance == pytest.approx(1 - .99**k, abs=1e-15)
    assert {1, 10, 100} <= {k for k, _ in ns["tail_table"]}
    assert [k for k, _ in ns["tail_simulation"]] == [10, 100]
    for k, share in ns["tail_simulation"]:
        # Four binomial standard errors at 10,000 runs are at most 0.02.
        assert share == pytest.approx(1 - .99**k, abs=.02)
    assert set(np.unique(ns["constructed_latencies"])) == {1, 10}


def test_new_exercise_keys_use_different_givens_and_match_hand_arithmetic(captured):
    ns = enrichment_namespace("keys-12")
    assert captured[PREFIX + "keys-12"].splitlines()[5:7] == [
        "LSTM key gates: forget=0.622; input=0.500; output=0.622",
        "LSTM key values: candidate=0.462; cell=0.231; state=0.141",
    ]

    def sigmoid(x):
        return 1 / (1 + math.exp(-x))

    f, i, o = sigmoid(.5), sigmoid(0), sigmoid(.5)
    cell = i * math.tanh(.5)
    np.testing.assert_allclose(
        list(ns["exercise_lstm"].values()),
        [f, i, o, math.tanh(.5), cell, o * math.tanh(cell)],
    )
    assert ns["exercise_cell_path"] == .9**20
    expected = np.exp([0., .5, 1.])
    expected /= expected.sum()
    for weights in ns["exercise_icl_weights"]:
        np.testing.assert_allclose(weights, expected)
    first, second = ns["exercise_icl_predictions"]
    assert first == pytest.approx(expected @ [2, 4, 8])
    assert second == pytest.approx(expected @ [2, 10, 8])
    assert second - first == pytest.approx(6 * expected[1])
    for row, expected_batch in zip(ns["exercise_serving_rows"], [13, 13]):
        b, weights, cache_request, left, batch = row
        assert weights == 12 * (4 * 768**2 + 2 * 768 * 3072) * b
        assert cache_request == 2 * 12 * 12 * 64 * 4096 * 2 == 150_994_944
        assert left == 2**31 - weights
        assert batch == left // cache_request == expected_batch
        assert weights + batch * cache_request <= 2**31
        assert weights + (batch + 1) * cache_request > 2**31
    assert ns["exercise_tail"] == pytest.approx(1 - .98**20)


def test_figure_records_parse_into_the_required_series(captured):
    paths = re.findall(
        r"^path t=(\d+): cell=([\d.]+); recurrence=([\d.]+)$",
        captured[PREFIX + "lstm-01"], re.M,
    )
    expected_paths = [(t, .95**t, .5**t) for t in range(1, 11)]
    np.testing.assert_allclose(np.array(paths, dtype=float), expected_paths, atol=5e-7)
    cosine = re.findall(
        r"^matched cosine row \d+: ([\d. ]+)$",
        captured[PREFIX + "contrastive-01"], re.M,
    )
    np.testing.assert_allclose(
        [[float(v) for v in row.split()] for row in cosine],
        [[.8, 0, .6], [.6, .8, 0], [0, .6, .8]],
    )
    weights = re.findall(
        r"^context [AB] weights: ([\d. ]+)$", captured[PREFIX + "icl-01"], re.M
    )
    np.testing.assert_allclose(
        [[float(v) for v in row.split()] for row in weights],
        [[.401112, .197776, .401112], [.248255, .503490, .248255]],
    )
    predictions = re.findall(
        r"^context [AB] prediction: ([\d.]+)$", captured[PREFIX + "icl-01"], re.M
    )
    np.testing.assert_allclose(np.array(predictions, dtype=float), [10, 12.412754])
    batches = re.findall(
        r"^decode n_req=(\d+); b=2; multiplications/weight byte=([\d.]+)$",
        captured[PREFIX + "serving-01"], re.M,
    )
    np.testing.assert_allclose(
        np.array(batches, dtype=float), [(n, n / 2) for n in (1, 2, 4, 8, 16, 32, 64)]
    )
    tail = re.findall(
        r"^tail k=(\d+): chance=([\d.]+); percent=([\d.]+)$",
        captured[PREFIX + "serving-01"], re.M,
    )
    assert [int(k) for k, _, _ in tail] == [1, 2, 5, 10, 20, 50, 100, 200, 500]
    for k, chance, percent in tail:
        expected = 1 - .99**int(k)
        assert float(chance) == pytest.approx(expected, abs=5e-7)
        assert float(percent) == pytest.approx(100 * expected, abs=5e-4)


@pytest.fixture(scope="module")
def sequence_plotter():
    pytest.importorskip("matplotlib")
    pytest.importorskip("PIL")
    return runpy.run_path(str(ROOT.parent / "figures/plots/fig_transformers_foundation_models.py"))


def test_sequence_figure_data_matches_the_constructed_graph_and_captured_counts(sequence_plotter):
    path = ROOT / "demos/data/transformers_sequence_paths.json"
    data = sequence_plotter["sequence_path_data"]()
    assert path.read_text() == json.dumps(data, indent=2) + "\n"
    assert data["tokens"] == [1, 2, 3, 4, 5]
    assert data["recurrent_edges"] == [[1, 2], [2, 3], [3, 4], [4, 5]]
    assert data["causal_attention_edges"] == [[1, 5], [2, 5], [3, 5], [4, 5], [5, 5]]
    output = (ROOT / "book_outputs" / f"_{PREFIX}bptt-01.txt").read_text()
    for n in (5, 2048):
        graph = sequence_plotter["sequence_path_data"](n)
        assert (
            f"tokens {n}: recurrent links {graph['recurrent_path_length']}, "
            f"causal-attention links {graph['causal_attention_path_length']}\n"
        ) in output


def test_sequence_figure_exports_a_fixed_page_and_a_greyscale_image(sequence_plotter, tmp_path):
    from PIL import Image

    sequence_plotter["sequence_paths_figure"](out=tmp_path)
    name = "fig-transformers-foundation-models-paths"
    pdf = (tmp_path / f"{name}.pdf").read_bytes()
    page = re.search(rb"/MediaBox\s*\[\s*0\s+0\s+([\d.]+)\s+([\d.]+)\s*\]", pdf)
    assert page is not None
    sizes = (ROOT.parent / "figures/page-sizes.tsv").read_text().splitlines()
    width, height = next(row.split("\t")[1:] for row in sizes if row.startswith(f"{name}\t"))
    assert tuple(float(v) for v in page.groups()) == (float(width), float(height))
    with (
        Image.open(tmp_path / f"{name}.png") as colour,
        Image.open(tmp_path / f"{name}-gray.png") as grey,
    ):
        assert colour.size == grey.size
        assert grey.mode == "L"
        assert grey.tobytes() == colour.convert("L").tobytes()
    repeat = tmp_path / "repeat"
    sequence_plotter["sequence_paths_figure"](out=repeat)
    for suffix in (".pdf", ".png", "-gray.png"):
        assert (tmp_path / f"{name}{suffix}").read_bytes() == (
            repeat / f"{name}{suffix}"
        ).read_bytes()


def test_core_attention_normalizes_visible_keys_and_rejects_empty_rows() -> None:
    ns = region(SOURCE, PREFIX + "core-01")
    attention = ns["scaled_dot_product_attention"]
    queries = np.array([[1.0, 0.0], [0.0, 1.0]])
    keys = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    values = np.array([[10.0], [0.0], [20.0]])
    mask = np.array([[0.0, -np.inf, -np.inf], [0.0, 0.0, -np.inf]])
    output, weights = attention(queries, keys, values, mask)
    assert output.shape == (2, 1)
    np.testing.assert_array_equal(weights[np.isneginf(mask)], 0.0)
    np.testing.assert_allclose(weights.sum(axis=1), 1.0)
    np.testing.assert_allclose(output[:, 0], [10.0, 3.3023845067])
    changed_values = values.copy()
    changed_values[2] = 1e9
    np.testing.assert_allclose(attention(queries, keys, changed_values, mask)[0], output)
    expected = [[0.7310585786, 0.2689414214], [0.5, 0.5]]
    np.testing.assert_allclose(
        ns["softmax"](np.array([[1000.0, 999.0], [-1000.0, -1000.0]])), expected
    )
    with pytest.raises(ValueError, match="finite"):
        attention(queries, keys, values, np.full((2, 3), -np.inf))


def test_example_prints_the_masked_and_unmasked_weighted_averages(
    capsys: pytest.CaptureFixture[str],
) -> None:
    ns = region(SOURCE, PREFIX + "core-01")
    ns = region(SOURCE, PREFIX + "example-01", ns)
    np.testing.assert_allclose(ns["weights"], [[0.4011120927, 0.1977758146, 0.4011120927]])
    assert ns["output"].item() == pytest.approx(12.0333627804)
    assert ns["masked_output"].item() == pytest.approx(6.6976154933)
    assert capsys.readouterr().out == (
        "unmasked weights: 0.401 0.198 0.401\n"
        "unmasked output: 12.033\n"
        "masked weights: 0.670 0.330 0.000\n"
        "masked output: 6.698\n"
    )


def test_scale_variance_grows_with_width_and_scaled_variance_stays_near_one(
    capsys: pytest.CaptureFixture[str],
) -> None:
    ns = region(SOURCE, PREFIX + "core-01")
    region(SOURCE, PREFIX + "scale-01", ns)
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 3
    previous_variance = 0.0
    for width, line in zip((4, 64, 512), lines):
        label, unscaled, scaled = line.replace(",", "").split()
        assert label == f"d_k={width}:"
        variance = float(unscaled.split("=")[1])
        assert variance == pytest.approx(width, rel=0.05)
        assert float(scaled.split("=")[1]) == pytest.approx(1.0, rel=0.05)
        assert variance > previous_variance
        previous_variance = variance


def test_positions_break_attention_permutation_equivariance(
    capsys: pytest.CaptureFixture[str],
) -> None:
    ns = region(SOURCE, PREFIX + "core-01")
    ns = region(SOURCE, PREFIX + "positions-01", ns)
    np.testing.assert_allclose(
        ns["positions"][1], [math.sin(1), math.cos(1), math.sin(0.01), math.cos(0.01)]
    )
    assert capsys.readouterr().out == (
        "position 1: 0.841 0.540 0.010 1.000\n"
        "equivariant without positions: True\n"
        "equivariant with positions: False\n"
    )
    attention = ns["scaled_dot_product_attention"]
    tokens = ns["token_vectors"]
    permutation = [1, 2, 0]
    plain = attention(tokens, tokens, tokens)[0]
    reordered = tokens[permutation]
    np.testing.assert_allclose(attention(reordered, reordered, reordered)[0], plain[permutation])
    positioned = tokens + ns["positions"]
    reordered = tokens[permutation] + ns["positions"]
    assert not np.allclose(
        attention(reordered, reordered, reordered)[0],
        attention(positioned, positioned, positioned)[0][permutation],
    )


def test_block_preserves_token_shape_and_counts_projection_weights() -> None:
    ns = region(SOURCE, PREFIX + "core-01")
    ns = region(SOURCE, PREFIX + "block-01", ns)
    for name in (
        "token_vectors",
        "attention_output",
        "attention_residual",
        "feed_forward_output",
        "block_output",
    ):
        assert ns[name].shape == (5, 8)
    assert ns["hidden"].shape == (5, 16)
    for name in ("attention_residual", "block_output"):
        np.testing.assert_allclose(ns[name].mean(axis=1), 0.0, atol=1e-12)
        np.testing.assert_allclose(ns[name].var(axis=1), 1.0, atol=2e-4)
    matrices = (
        "query_projection",
        "key_projection",
        "value_projection",
        "output_projection",
        "feed_forward_in",
        "feed_forward_out",
    )
    assert ns["block_weight_count"](8, 16) == sum(ns[name].size for name in matrices) == 512
    assert ns["block_weight_count"](512, 2048) == 3_145_728
    np.testing.assert_allclose(
        ns["layer_norm"](np.array([[2.0, 4.0, 6.0, 8.0]])),
        [[-1.341639445, -0.447213148, 0.447213148, 1.341639445]],
    )


def test_loss_is_negative_log_probability_and_perplexity_is_inverse_geometric_mean(
    capsys: pytest.CaptureFixture[str],
) -> None:
    ns = region(SOURCE, PREFIX + "core-01")
    ns = region(SOURCE, PREFIX + "loss-01", ns)
    np.testing.assert_allclose(
        ns["losses"], [0.6931471806, 1.3862943611, 0.6931471806, 2.0794415417]
    )
    assert ns["mean_loss"] == pytest.approx(1.21300756598)
    assert ns["perplexity"] == pytest.approx(3.36358566101)
    assert capsys.readouterr().out == (
        "The: 0.693\nsky: 1.386\nis: 0.693\nblue: 2.079\nmean loss: 1.213\nperplexity: 3.364\n"
    )


def decoder_namespace():
    ns = region(SOURCE, PREFIX + "core-01")
    ns = region(SOURCE, PREFIX + "block-01", ns)
    return region(SOURCE, PREFIX + "decoder-01", ns)


def test_decoder_connects_shifted_targets_to_causal_hidden_states():
    ns = decoder_namespace()
    trace = ns["decoder_trace"]
    np.testing.assert_array_equal(ns["target_ids"], [1, 2, 3, 4])
    assert [ns["vocabulary"][i] for i in ns["token_ids"]] == ["(start)", "The", "sky", "is"]
    for name in ("embeddings", "positions", "input", "attention", "attention add/norm", "feed-forward output", "block output"):
        assert trace[name].shape == (4, 8)
    assert trace["feed-forward hidden"].shape == (4, 16)
    attention = trace["attention weights"]
    np.testing.assert_array_equal(attention[np.triu_indices(4, 1)], 0)
    np.testing.assert_allclose(attention.sum(axis=1), 1)
    for end in range(1, 5):
        prefix = ns["decoder_forward"](ns["token_ids"][:end], ns["embedding_table"], ns["decoder_weights"])
        np.testing.assert_allclose(prefix["block output"], trace["block output"][:end], atol=1e-12)
    probabilities = ns["decoder_probabilities"]
    np.testing.assert_allclose(probabilities.sum(axis=1), 1)
    chosen = probabilities[np.arange(4), ns["target_ids"]]
    np.testing.assert_array_equal(ns["target_probabilities"], chosen)
    assert ns["decoder_loss"] == pytest.approx(-np.log(chosen).mean())
    assert ns["decoder_perplexity"] == pytest.approx(1 / np.prod(chosen) ** 0.25)
    assert ns["appended_ids"][-1] == int(np.argmax(probabilities[-1]))


def test_decoder_head_gradient_matches_finite_differences_and_only_head_moves():
    ns = decoder_namespace()
    hidden = ns["decoder_trace"]["block output"]
    head = ns["head_before"]
    targets = ns["target_ids"]

    def loss_for(weights):
        probabilities = ns["softmax"](hidden @ weights)
        return -np.log(probabilities[np.arange(len(targets)), targets]).mean()

    numeric = np.empty_like(head)
    for index in np.ndindex(head.shape):
        step = np.zeros_like(head)
        step[index] = 1e-5
        numeric[index] = (loss_for(head + step) - loss_for(head - step)) / 2e-5
    np.testing.assert_allclose(ns["head_gradient"], numeric, atol=1e-9, rtol=0)
    np.testing.assert_array_equal(ns["output_head"], head - ns["head_learning_rate"] * ns["head_gradient"])
    assert ns["updated_decoder_loss"] < ns["decoder_loss"]
    for name, before in ns["frozen_weights"].items():
        after = ns["embedding_table"] if name == "embedding" else ns["decoder_weights"][name]
        np.testing.assert_array_equal(before, after)


def test_cache_bytes_and_generation_score_counts_follow_closed_forms(
    capsys: pytest.CaptureFixture[str],
) -> None:
    ns = region(SOURCE, PREFIX + "sota-01")
    assert ns["cache_bytes"] == 100_663_296
    assert ns["cached_scores"] == 2_098_176
    assert ns["uncached_scores"] == 2_865_409_024
    for n in (0, 1, 2, 17, 2048):
        cached, uncached = ns["attention_score_counts"](n)
        assert cached == sum(range(1, n + 1))
        assert uncached == sum(t * t for t in range(1, n + 1))
        assert ns["kv_cache_bytes"](24, 8, 64, n, 2) == 49_152 * n
    with pytest.raises(ValueError):
        ns["attention_score_counts"](-1)
    with pytest.raises(ValueError):
        ns["kv_cache_bytes"](24, 8, 0, 2048, 2)
    assert capsys.readouterr().out == (
        "cache bytes: 100663296\ncache MiB: 96.000\n"
        "cached scores: 2098176\nuncached scores: 2865409024\n"
    )
