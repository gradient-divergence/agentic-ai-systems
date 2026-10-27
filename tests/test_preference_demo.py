"""Numerical contracts, independent listings and captured preference outputs."""

import contextlib
import importlib.util
import io
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from demos import regions as _regions

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "demos/preference_demo.py"
TOOLS = vars(_regions)
PREFIX = "fnd.rlhf-alignment."
NAMES = ("pair-01", "optimum-01", "dpo-01", "train-01", "delivered-01", "keys-01")
PRINTED = NAMES[:3]


def demo():
    from demos import preference_demo

    return preference_demo


def regions():
    return TOOLS["extract_snippets_from_text"](DEMO.read_text(), DEMO)


def test_printed_listings_run_in_chapter_order_without_hidden_setup():
    namespace = {"__name__": "__main__"}
    snippets = {s.snippet_id: s for s in regions()}
    for name in PRINTED:
        with contextlib.redirect_stdout(io.StringIO()) as stdout:
            exec(compile(snippets[PREFIX + name].content, name, "exec"), namespace)
        path = ROOT / "book_outputs" / f"_{PREFIX}{name}.txt"
        assert stdout.getvalue().encode() == path.read_bytes()


@pytest.fixture(scope="module")
def captured():
    return TOOLS["capture_outputs"]({s.snippet_id: s for s in regions()}, sys.executable)


@pytest.mark.parametrize("name", NAMES)
def test_every_stored_output_regenerates_byte_for_byte(name, captured):
    path = ROOT / "book_outputs" / f"_{PREFIX}{name}.txt"
    assert captured[PREFIX + name].encode() == path.read_bytes()


def test_output_regions_are_in_outline_order():
    assert [s.snippet_id for s in regions()] == [PREFIX + n for n in NAMES]
    assert all(s.output for s in regions())


def test_capture_is_deterministic_in_fresh_processes(captured):
    assert captured == TOOLS["capture_outputs"](
        {s.snippet_id: s for s in regions()}, sys.executable
    )


def test_import_has_no_output_or_file_effects(tmp_path):
    result = subprocess.run(
        [sys.executable, "-c", "import demos.preference_demo"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == ""
    assert list(tmp_path.iterdir()) == []


def test_pair_shift_disagreement_and_stable_logistic():
    m = demo()
    for shift in (0, 5, -100):
        pair = m.PreferencePair(chosen_score=1.2 + shift, rejected_score=0.4 + shift)
        assert m.bradley_terry_loss(pair) == pytest.approx(np.logaddexp(0, -0.8))
    assert m.preference_probability(1000) == 1
    assert m.preference_probability(-1000) == 0
    assert m.preference_probability(m.logit(0.7)) == pytest.approx(0.7)


@pytest.mark.parametrize("beta", [0.25, 0.5, 1, 10, 1e12])
def test_optimum_support_normalization_value_and_random_simplex_check(beta):
    m = demo()
    ref, rewards = np.array([0.5, 0.45, 0.05]), np.array([1, 0.5, -2])
    result = m.kl_optimum(ref, rewards, beta)
    p = result["probabilities"]
    assert p.sum() == pytest.approx(1, abs=1e-14)
    assert (p > 0).all()
    value = result["expected_reward"] - beta * result["kl"]
    assert value == pytest.approx(result["beta_log_z"], abs=1e-10)
    rng = np.random.default_rng(7)
    candidates = rng.dirichlet(np.ones(3), size=2000)
    candidates = np.vstack([candidates, 0.99 * p + 0.01 * candidates])
    scores = candidates @ rewards - beta * np.sum(candidates * np.log(candidates / ref), axis=1)
    assert np.max(scores) <= value + 1e-10
    if beta == 1e12:
        np.testing.assert_allclose(p, ref, rtol=0, atol=1e-12)


def test_constant_reward_shift_preserves_optimum_and_shifts_value_only():
    m = demo()
    a = m.kl_optimum([0.5, 0.45, 0.05], [1, 0.5, -2], 0.5)
    b = m.kl_optimum([0.5, 0.45, 0.05], [8, 7.5, 5], 0.5)
    np.testing.assert_array_equal(a["probabilities"], b["probabilities"])
    assert b["beta_log_z"] - a["beta_log_z"] == pytest.approx(7)


def test_optimum_rejects_an_unnormalized_reference():
    with pytest.raises(ValueError, match="normalized reference"):
        demo().kl_optimum([0.500001, 0.45, 0.05], [1, 0.5, -2], 1e12)


def test_optimum_normalizes_extreme_positive_reference_in_log_space():
    result = demo().kl_optimum([1e-310, 1], [0, -710], 1)
    p = result["probabilities"]
    # Divide both unnormalized weights by exp(-710), without the demo's helpers.
    ratio = math.exp(math.log(1e-310) + 710)
    expected = np.array([ratio, 1]) / (1 + ratio)
    assert np.isfinite(p).all()
    assert p.sum() == pytest.approx(1, abs=1e-12)
    np.testing.assert_allclose(p, expected, rtol=0, atol=1e-12)


@pytest.mark.parametrize("beta", [0.25, 0.5, 1, 2])
def test_two_answer_sharpening_equation_to_nine_decimals(beta):
    m = demo()
    for mu in (0.2, 0.4, 0.5, 0.6, 0.8, 0.9):
        p = m.kl_optimum([mu, 1 - mu], [m.logit(mu), 0], beta)["probabilities"][0]
        assert m.logit(p) == pytest.approx((1 + 1 / beta) * m.logit(mu), abs=1e-9)


def test_sequence_probabilities_are_sums_not_length_means():
    m = demo()
    assert m.sequence_logprob([-0.4, -0.9, -0.2]) == pytest.approx(-1.5)
    assert m.sequence_logprob([-0.3, -0.6]) == pytest.approx(-0.9)
    batch = m.PreferenceBatch(
        chosen_logprob=-1.1,
        rejected_logprob=-1.8,
        ref_chosen_logprob=-1.4,
        ref_rejected_logprob=-1.6,
    )
    np.testing.assert_allclose(m.implicit_rewards(batch, m.DPOConfig(beta=0.2)), [0.06, -0.04])
    same = m.PreferenceBatch(
        chosen_logprob=-1.4,
        rejected_logprob=-1.6,
        ref_chosen_logprob=-1.4,
        ref_rejected_logprob=-1.6,
    )
    assert m.dpo_loss(same, m.DPOConfig(beta=0.2)) == pytest.approx(np.log(2))
    assert m.preference_accuracy(same) == 0.5


def test_clean_test_probability_endpoints_and_bound():
    m = demo()
    assert m.clean_test_chance(0, 300) == 1
    assert m.clean_test_chance(1, 300) == 0
    assert m.clean_test_chance(0.01, 0) == 1
    bound = m.zero_failure_bound(300)
    assert m.clean_test_chance(bound, 300) == pytest.approx(0.05)


@pytest.fixture(scope="module")
def experiment():
    return demo().build_experiment(7)


def test_seeded_windows_and_random_draft_order_are_reproducible_and_independent():
    m = demo()
    first, second = m.generate(7), m.generate(7)
    changed = m.generate(8)
    for window in first:
        for key in first[window]:
            np.testing.assert_array_equal(first[window][key], second[window][key])
    assert not np.array_equal(first["labeled"]["true_cause"], changed["labeled"]["true_cause"])
    smaller = m.generate(7, n_heldout_reports=1, n_heldout_pairs=1)
    for window in ("labeled", "pairs"):
        for key in first[window]:
            np.testing.assert_array_equal(first[window][key], smaller[window][key])
    pairs = first["pairs"]
    assert set(pairs["first"]) == {0, 1}
    np.testing.assert_array_equal(pairs["second"], 1 - pairs["first"])
    np.testing.assert_array_equal(
        np.where(pairs["chosen_position"] == 0, pairs["first"], pairs["second"]), pairs["chosen"]
    )


@pytest.mark.parametrize("beta", [0.25, 0.5, 1])
def test_dpo_reaches_same_optimum_as_fitted_reward_model(beta, experiment):
    m = demo()
    ref, gaps = experiment["reference"], experiment["reward_gaps"]
    fit = m.train_dpo(beta, reference=ref, pairs=experiment["data"]["pairs"])
    assert fit["gradient_norm"] < 1e-8
    np.testing.assert_allclose(fit["logits"], ref + gaps / beta, rtol=0, atol=1e-6)
    np.testing.assert_allclose(fit["probabilities"], m.rlhf_optimum(ref, gaps, beta), atol=1e-8)


def test_dpo_gradient_matches_central_difference(experiment):
    from demos.appendix_math_demo import central_difference

    m = demo()
    ref, pairs = experiment["reference"], experiment["data"]["pairs"]
    theta = ref + np.array([0.2, -0.3, 0.1, 0.4, -0.2])
    gradient = m.dpo_gradient(theta, ref, pairs, 0.5)
    numeric = central_difference(lambda x: m.dpo_objective(x, ref, pairs, 0.5), theta)
    np.testing.assert_allclose(gradient, numeric, rtol=0, atol=1e-9)


@pytest.mark.parametrize("beta", [0.25, 0.5, 1])
def test_exact_expectation_policy_gradient_reaches_the_same_probabilities(beta, experiment):
    # Score-function gradient of E[r] - beta KL, with both actions enumerated.
    m = demo()
    ref, gaps = experiment["reference"], experiment["reward_gaps"]
    theta = ref.copy()
    for _ in range(50000):
        p = m.preference_probability(theta)
        theta += 10 * p * (1 - p) * (gaps - beta * (theta - ref)) / len(theta)
    np.testing.assert_allclose(
        m.preference_probability(theta), m.rlhf_optimum(ref, gaps, beta), rtol=0, atol=1e-3
    )


def test_heldout_reference_calibration_and_unchanged_answers(experiment):
    m = demo()
    ref = m.preference_probability(experiment["reference"])
    assert experiment["evaluations"]["reference"]["calibration_error"] < 0.02
    for fit in experiment["fits"].values():
        np.testing.assert_array_equal(fit["probabilities"] >= 0.5, ref >= 0.5)
    for result in experiment["evaluations"].values():
        assert result["correctness"] == experiment["evaluations"]["reference"]["correctness"]
    assert experiment["evaluations"]["reference"]["preference_accuracy"] == 0.5
    assert experiment["evaluations"]["0.25"]["calibration_error"] > 0.18


def test_evaluation_kl_stays_nonzero_when_seed_seven_beta_point_zero_five_saturates():
    m = demo()
    data = m.generate(7)
    reference = m.fit_reference(data["labeled"])
    fit = m.train_dpo(0.05, reference=reference, pairs=data["pairs"])
    assert fit["converged"]
    assert (fit["probabilities"] == 1).any()
    result = m.evaluate(
        fit["probabilities"], reference, data["heldout_reports"], data["heldout_pairs"]
    )
    # Scalar Bernoulli definition, omitting zero-mass terms independently.
    terms = []
    for p, theta in zip(fit["probabilities"], reference):
        q = 1 / (1 + math.exp(-float(theta)))
        terms.append(
            math.fsum(
                mass * math.log(mass / ref_mass)
                for mass, ref_mass in ((p, q), (1 - p, 1 - q))
                if mass > 0
            )
        )
    counts = np.bincount(data["heldout_reports"]["kind"], minlength=len(terms))
    expected = math.fsum(count * term for count, term in zip(counts, terms)) / counts.sum()
    assert result["mean_kl"] > 0
    assert result["mean_kl"] == pytest.approx(expected, rel=0, abs=1e-12)
    assert result["mean_kl"] == pytest.approx(0.323019731095, rel=0, abs=1e-12)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_evaluation_rejects_nonfinite_kl(value):
    m = demo()
    data = m.generate(7, n_heldout_reports=1, n_heldout_pairs=1)
    reference = np.zeros(len(m.KINDS.shares))
    reference[0] = value
    with pytest.raises(ValueError, match="KL must be finite"):
        m.evaluate(np.full(5, 0.5), reference, data["heldout_reports"], data["heldout_pairs"])


def test_evaluation_rejects_a_nonfinite_mean_of_finite_kl_terms():
    m = demo()
    data = m.generate(7, n_heldout_reports=10, n_heldout_pairs=1)
    with pytest.raises(ValueError, match="KL must be finite"):
        m.evaluate(
            np.full(5, 0.5), np.full(5, 1e308), data["heldout_reports"], data["heldout_pairs"]
        )


@pytest.mark.parametrize("window", ["reports", "comparisons"])
def test_evaluation_rejects_empty_windows(window):
    m = demo()
    sizes = {"n_heldout_reports": 0} if window == "reports" else {"n_heldout_pairs": 0}
    data = m.generate(7, **sizes)
    reference = m.fit_reference(data["labeled"])
    with pytest.raises(ValueError, match="evaluation needs"):
        m.evaluate(
            m.preference_probability(reference),
            reference,
            data["heldout_reports"],
            data["heldout_pairs"],
        )


def test_evaluation_rejects_a_missing_report_kind():
    m = demo()
    data = m.generate(7, n_heldout_reports=1, n_heldout_pairs=1)
    reference = m.fit_reference(data["labeled"])
    reports = data["heldout_reports"]
    reports = {name: values[reports["kind"] != 2] for name, values in reports.items()}
    with pytest.raises(ValueError, match="report for every kind"):
        m.evaluate(m.preference_probability(reference), reference, reports, data["heldout_pairs"])


def test_fitting_never_reads_heldout_or_frozen_data(monkeypatch):
    from refund_agent import split

    m = demo()

    def forbidden(*args, **kwargs):
        raise AssertionError("training read held-out or frozen data")

    monkeypatch.setattr(split.GoldenSet, "consult", forbidden)
    monkeypatch.setattr(Path, "read_text", forbidden)
    monkeypatch.setattr(Path, "read_bytes", forbidden)
    data = m.generate(7, n_heldout_reports=0, n_heldout_pairs=0)
    ref = m.fit_reference(data["labeled"])
    before = ref.copy()
    gaps = m.fit_reward_gaps(data["pairs"])
    fit = m.train_dpo(0.5, reference=ref, pairs=data["pairs"])
    np.testing.assert_array_equal(ref, before)
    assert not ref.flags.writeable
    np.testing.assert_allclose(fit["probabilities"], m.rlhf_optimum(ref, gaps, 0.5), atol=1e-8)
    poisoned = {
        "labeled": data["labeled"],
        "pairs": data["pairs"],
        "heldout_reports": object(),
        "heldout_pairs": object(),
    }
    trained = m.fit_training(poisoned)
    np.testing.assert_array_equal(trained["reference"], ref)


@pytest.mark.parametrize(
    "name,function",
    [
        ("rlhf_alignment_figures.json", "figure_data"),
        ("rlhf_alignment_delivered.json", "delivered_record"),
        ("rlhf_alignment_report_05.json", "exercise_five_record"),
    ],
)
def test_data_and_fixtures_regenerate_byte_for_byte(name, function):
    expected = json.dumps(getattr(demo(), function)(), indent=2, allow_nan=False) + "\n"
    assert (DEMO.with_name("data") / name).read_bytes() == expected.encode()


def test_exercise_five_has_exactly_the_one_selected_label_error():
    m = demo()
    record = m.exercise_five_record()
    rows = record["heldout_messages"]
    y = np.array([r["true_urgent"] for r in rows])
    p = np.array([r["probability_urgent"] for r in rows])
    report = record["agent_report"]
    assert report["mean_probability_on_urgent"] == pytest.approx(p[y == 1].mean())
    assert report["preference_accuracy"] == pytest.approx(
        np.mean(
            [0.5 if r["margin"] == 0 else r["margin"] > 0 for r in record["heldout_preferences"]]
        )
    )
    assert report["mean_probability_on_urgent"] == pytest.approx(0.91, abs=0.005)
    assert report["calibration_rows"] == "true_urgent == 1"
    assert report["calibration_verdict"] == "calibrated"
    assert m.calibration_error(m.reliability(y, p, bins=5)) > 0.2
    assert len({r["message_id"] for r in rows}) == len(rows)


def test_plot_data_folds_both_axes_to_the_likelier_answer(experiment):
    data = demo().figure_data(experiment)
    for series in data["sharpening"]["points"].values():
        assert len(series["confidence"]) == 5
        assert min(series["confidence"]) >= 0.5
        assert min(series["observed_correctness"]) >= 0.5
    assert data["sharpening"]["truth"] == [0.8, 0.6, 0.6, 0.8, 0.9]


def test_plotter_writes_colour_and_grayscale_artifacts(tmp_path):
    pytest.importorskip("matplotlib", reason="plotting requires Matplotlib")
    spec = importlib.util.spec_from_file_location(
        "preference_plot", ROOT.parent / "figures/plots/fig_rlhf_alignment.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.OUT = tmp_path
    module.main()
    assert len(list(tmp_path.glob("*.pdf"))) == 2
    assert len(list(tmp_path.glob("*.png"))) == 4


def test_exercise_six_acceptance_against_the_instructor_solution():
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-p",
            "no:cacheprovider",
            str(ROOT / "exercises/rlhf-alignment/test_acceptance.py"),
        ],
        env={**os.environ, "RLHF_ALIGNMENT_SOLUTION": str(DEMO)},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_exercise_six_later_labels_never_fit_the_calibration_map(monkeypatch):
    m = demo()
    expected = m.exercise_six()
    training, calibration, later = m.exercise_windows()
    changed = {**later, "true_cause": 1 - later["true_cause"]}
    monkeypatch.setattr(m, "exercise_windows", lambda seed: (training, calibration, changed))
    actual = m.exercise_six()
    for key in ("logits", "target_logits", "slope", "intercept", "mapped_choices"):
        assert actual[key] == expected[key]
    assert actual["after_error"] != expected["after_error"]


def test_exercise_acceptance_rejects_an_invented_report(tmp_path):
    fake = tmp_path / "fake_submission.py"
    report = dict(
        seed=7,
        beta=0.5,
        logits=[0] * 5,
        target_logits=[0] * 5,
        population_check=dict(logit_gaps=[0] * 5, max_logit_gap=0, tolerance=0.02, passed=True),
        slope=1 / 3,
        intercept=0,
        reference_error=0,
        before_error=0.1,
        after_error=0,
        reference_choices=[0] * 5,
        tuned_choices=[0] * 5,
        mapped_choices=[0] * 5,
        window_order=["training", "calibration", "later check"],
        calibration_reports=10000,
        later_reports=10000,
    )
    fake.write_text(f"def exercise_six(seed=7):\n    return {report!r}\n")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-p",
            "no:cacheprovider",
            str(ROOT / "exercises/rlhf-alignment/test_acceptance.py"),
        ],
        env={**os.environ, "RLHF_ALIGNMENT_SOLUTION": str(fake)},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1, result.stdout + result.stderr
    assert "5 failed, 1 passed" in result.stdout
