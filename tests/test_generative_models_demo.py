"""Chapter-order listings, numerical identities and captured generative outputs."""

import contextlib
import importlib
import io
import json
import math
import runpy
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from demos import regions as _regions

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "demos/generative_models_demo.py"
TOOLS = vars(_regions)
PREFIX = "fnd.generative-models."
PRINTED = ("sample-01", "likelihood-01", "elbo-01", "gan-01", "diffusion-01", "reverse-01", "scenarios-01")
NAMES = PRINTED[:3] + ("vae-01",) + PRINTED[3:6] + ("schedule-01",) + PRINTED[6:] + ("keys-01", "keys-gan-01")


def demo():
    return importlib.import_module("demos.generative_models_demo")


def snippets():
    return {s.snippet_id: s for s in TOOLS["extract_snippets_from_text"](DEMO.read_text(), DEMO)}


def test_printed_listings_run_in_chapter_order_without_hidden_setup():
    namespace = {"__name__": "__main__"}
    regions = snippets()
    for name in PRINTED:
        with contextlib.redirect_stdout(io.StringIO()) as stdout:
            exec(compile(regions[PREFIX + name].content, str(DEMO), "exec"), namespace)
        stored = ROOT / "book_outputs" / f"_{PREFIX}{name}.txt"
        assert stdout.getvalue().encode() == stored.read_bytes()
    assert namespace["listed"].manifest["counts"]["distinct"] == 252


def test_output_regions_follow_outline_order():
    regions = snippets()
    assert tuple(regions) == tuple(PREFIX + name for name in NAMES)
    assert all(s.output for s in regions.values())


@pytest.mark.parametrize("name", ["gan-01", "keys-gan-01"])
def test_new_gan_regions_run_alone_with_explicit_imports(name):
    regions = snippets()
    assert PREFIX + name in regions
    region = regions[PREFIX + name]
    assert max(map(len, region.content.splitlines())) <= 79
    with contextlib.redirect_stdout(io.StringIO()) as stdout:
        exec(compile(region.content, str(DEMO), "exec"), {"__name__": "__main__"})
    stored = ROOT / "book_outputs" / f"_{PREFIX}{name}.txt"
    assert stdout.getvalue().encode() == stored.read_bytes()


@pytest.fixture(scope="module")
def captured():
    return TOOLS["capture_outputs"](snippets(), sys.executable)


@pytest.mark.parametrize("name", NAMES)
def test_each_stored_output_regenerates_byte_for_byte(name, captured):
    stored = ROOT / "book_outputs" / f"_{PREFIX}{name}.txt"
    assert captured[PREFIX + name].encode() == stored.read_bytes()


def test_capture_is_deterministic(captured):
    assert captured == TOOLS["capture_outputs"](snippets(), sys.executable)


def test_import_has_no_printing_generation_or_writes(tmp_path):
    code = """from refund_agent.data import generate
def forbidden(*args, **kwargs):
    raise AssertionError('import generated data')
generate.generate = forbidden
import demos.generative_models_demo
"""
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=tmp_path, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == ""
    assert list(tmp_path.iterdir()) == []


def test_top_p_keeps_token_that_crosses_cutoff():
    # The case moved from test_regressions.py, without changing the three scores.
    p = demo().sampling_distribution(np.log([0.6, 0.35, 0.05]), {"name": "nucleus", "p": 0.9})
    assert np.flatnonzero(p).tolist() == [0, 1]
    np.testing.assert_allclose(p, [0.6 / 0.95, 0.35 / 0.95, 0])


@pytest.mark.parametrize(
    "rule",
    [
        {"name": "greedy"},
        {"name": "temperature", "tau": 0.5},
        {"name": "top-k", "k": 2},
        {"name": "nucleus", "p": 0.8},
    ],
)
def test_sampling_distributions_are_normalized_and_keep_ties_stable(rule):
    p = demo().sampling_distribution(np.array([2.0, 1.0, 0.5, 0.0, -1.0]), rule)
    assert p.sum() == pytest.approx(1)
    assert np.all(p >= 0)
    assert np.argmax(p) == 0
    tied = demo().sampling_distribution(np.zeros(3), {"name": "top-k", "k": 1})
    np.testing.assert_array_equal(tied, [1, 0, 0])


def test_gaussian_kl_and_the_worked_exact_posterior():
    m = demo()
    assert m.gaussian_kl([1], [1]) == pytest.approx(0.5)
    assert m.evidence_lower_bound(-2, [1], [1]) == pytest.approx(-2.5)
    x = np.array([[2.0]])
    for variance in (1.0, 0.5):
        terms = m.linear_vae_terms(x, np.array([1.0]), np.array([0.5]), variance, 1.0)
        assert terms["log_likelihood"] == pytest.approx(-2.2655121234846454)
        assert terms["log_likelihood"] == pytest.approx(terms["elbo"] + terms["posterior_kl"])
    assert terms["posterior_kl"] == pytest.approx(0, abs=1e-12)


def test_gan_optimum_confirms_the_two_point_example():
    discriminator, value, divergence = demo().gan_optimum([0.5, 0.5], [0.8, 0.2])
    np.testing.assert_allclose(discriminator, [5 / 13, 5 / 7])
    expected = (
        0.5 * math.log(5 / 13) + 0.5 * math.log(5 / 7)
        + 0.8 * math.log(8 / 13) + 0.2 * math.log(2 / 7)
    )
    assert value == pytest.approx(expected)
    assert value == pytest.approx(-1.285, abs=0.0005)
    assert divergence == pytest.approx(0.051, abs=0.0005)
    assert value == pytest.approx(-math.log(4) + 2 * divergence)


@pytest.mark.parametrize("p", [[0.5, 0.5], [0.6, 0.3, 0.1]])
def test_gan_optimum_matching_distributions_reach_the_floor(p):
    discriminator, value, divergence = demo().gan_optimum(p, p)
    np.testing.assert_allclose(discriminator, 0.5)
    assert value == pytest.approx(-math.log(4))
    assert divergence == pytest.approx(0, abs=1e-15)


def test_gan_optimum_discriminator_maximizes_the_fixed_generator_game():
    p, q = np.array([0.6, 0.4]), np.array([0.3, 0.7])
    discriminator, value, divergence = demo().gan_optimum(p, q)
    np.testing.assert_allclose(discriminator, [2 / 3, 4 / 11])
    assert value == pytest.approx(p @ np.log(discriminator) + q @ np.log1p(-discriminator))
    alternatives = np.array(np.meshgrid(np.linspace(0.05, 0.95, 19),
                                       np.linspace(0.05, 0.95, 19))).reshape(2, -1).T
    assert np.all(np.log(alternatives) @ p + np.log1p(-alternatives) @ q <= value)
    assert value > -math.log(4)
    assert value == pytest.approx(-math.log(4) + 2 * divergence)
    assert demo().gan_optimum(q, p)[2] == pytest.approx(divergence)


def test_gan_optimum_zero_masses_use_the_zero_log_zero_limit():
    discriminator, value, divergence = demo().gan_optimum([1, 0, 0], [0, 1, 0])
    np.testing.assert_allclose(discriminator[:2], [1, 0])
    assert discriminator[2] == 0.5  # Neither distribution visits this point.
    assert np.all((discriminator >= 0) & (discriminator <= 1))
    assert value == pytest.approx(0)
    assert divergence == pytest.approx(math.log(2))


@pytest.mark.parametrize(
    "p,q",
    [([], []), ([[0.5, 0.5]], [[0.5, 0.5]]), ([1], [0.5, 0.5]),
     ([-0.1, 1.1], [0.5, 0.5]), ([0.5, 0.5], [0.2, 0.2]),
     ([np.nan, 0.5], [0.5, 0.5]), ([0.5, 0.5], [np.inf, 0])],
)
def test_gan_optimum_rejects_invalid_distributions(p, q):
    with pytest.raises(ValueError):
        demo().gan_optimum(p, q)


def test_elbo_listing_computes_the_example_and_the_encoder_to_posterior_gap():
    namespace = {"__name__": "__main__", "np": np}
    with contextlib.redirect_stdout(io.StringIO()) as stdout:
        exec(compile(snippets()[PREFIX + "elbo-01"].content, str(DEMO), "exec"), namespace)
    output = stdout.getvalue()
    assert "expected log-likelihood: -1.919\n" in output
    assert "evidence lower bound: -2.419\n" in output
    assert "log p(x): -2.266; gap KL(q || posterior): 0.153\n" in output
    expected = demo().linear_vae_terms(np.array([[2.0]]), np.ones(1), np.array([0.5]), 1, 1)
    assert namespace["expected_log_likelihood"] == pytest.approx(expected["reconstruction"])
    assert namespace["worked_bound"] == pytest.approx(expected["elbo"])
    assert namespace["worked_log_likelihood"] == pytest.approx(expected["log_likelihood"])
    assert namespace["posterior_mean"] == pytest.approx(1)
    assert namespace["posterior_variance"] == pytest.approx(0.5)
    assert namespace["posterior_kl"] == pytest.approx(0.5 * (1 - np.log(2)))
    assert namespace["worked_log_likelihood"] - namespace["worked_bound"] == pytest.approx(namespace["posterior_kl"])


def test_reverse_step_uses_algorithm_two_mean_and_beta_variance():
    m = demo()
    betas = np.full(3, 0.36)
    x_t = m.noise_jump([1, -2], [0.5, -1], betas)
    predicted, z = np.array([0.4, -0.7]), np.array([0.25, -0.5])
    mean, sample = m.reverse_step(x_t, predicted, betas, z)
    # Independently substitute the printed beta and t=3 into algorithm 2.
    expected_mean = (x_t - 0.36 * predicted / np.sqrt(1 - 0.64**3)) / 0.8
    np.testing.assert_allclose(mean, expected_mean, atol=1e-12)
    np.testing.assert_allclose(sample, mean + 0.6 * z, atol=1e-12)
    np.testing.assert_array_equal(m.reverse_step(x_t, predicted, betas, z)[1], sample)
    np.testing.assert_array_equal(m.reverse_step(x_t, predicted, betas, np.zeros(2))[1], mean)
    # The affine noise part has covariance beta*I for unit-covariance z.
    draws = np.array([[np.sqrt(2), 0], [-np.sqrt(2), 0], [0, np.sqrt(2)], [0, -np.sqrt(2)]])
    samples = np.array([m.reverse_step(x_t, predicted, betas, draw)[1] for draw in draws])
    np.testing.assert_allclose((samples - mean).T @ (samples - mean) / 4, 0.36 * np.eye(2), atol=1e-12)


@pytest.mark.parametrize("betas", [[], [0], [1], [-0.1], [np.nan], [[0.36]]])
def test_reverse_step_rejects_invalid_noise_schedules(betas):
    with pytest.raises(ValueError):
        demo().reverse_step([1, -2], [0.4, -0.7], betas, [0.25, -0.5])


@pytest.mark.parametrize("predicted,z", [([0.4], [0.25, -0.5]), ([0.4, -0.7], [0.25]), ([np.nan, -0.7], [0.25, -0.5]), ([0.4, -0.7], [np.inf, -0.5])])
def test_reverse_step_rejects_mismatched_or_nonfinite_vectors(predicted, z):
    with pytest.raises(ValueError):
        demo().reverse_step([1, -2], predicted, [0.36] * 3, z)


def test_linear_vae_exact_likelihood_is_bound_plus_posterior_kl():
    m = demo()
    rng = np.random.default_rng(31)
    x = rng.normal(size=(19, 2))
    for variance in (0.2, 0.7, 1.4):
        terms = m.linear_vae_terms(x, np.array([1.2, -0.4]), np.array([0.6, 0.1]), variance, 0.5)
        assert terms["log_likelihood"] == pytest.approx(
            terms["elbo"] + terms["posterior_kl"], abs=1e-12
        )
        assert terms["posterior_kl"] >= 0
    curve = m.train_linear_vae()["steps"]
    assert all(row["elbo"] <= row["log_likelihood"] + 1e-12 for row in curve)
    assert curve[-1]["elbo"] > curve[0]["elbo"]
    assert curve[-1]["posterior_kl"] < curve[0]["posterior_kl"]


def test_exact_bound_gradients_match_central_differences():
    m = demo()
    x = np.random.default_rng(11).normal(size=(27, 2))
    theta = np.array([0.8, -0.3, 0.4, 0.1, -0.5])

    def bound(value):
        return m.linear_vae_terms(x, value[:2], value[2:4], np.exp(value[4]), 0.5)["elbo"]

    dw, da, dv = m.linear_vae_gradient(x, theta[:2], theta[2:4], np.exp(theta[4]), 0.5)
    numeric = []
    for i in range(len(theta)):
        step = np.zeros_like(theta)
        step[i] = 1e-5
        numeric.append((bound(theta + step) - bound(theta - step)) / 2e-5)
    np.testing.assert_allclose(np.r_[dw, da, dv], numeric, rtol=0, atol=1e-9)


def test_jump_mean_and_variance_equal_composed_steps():
    betas = np.array([0.05, 0.2, 0.36])
    coefficient, variance = 1.0, 0.0
    for beta in betas:
        coefficient *= np.sqrt(1 - beta)
        variance = (1 - beta) * variance + beta
    x = np.array([1.0, -2.0])
    np.testing.assert_allclose(demo().noise_jump(x, np.zeros(2), betas), coefficient * x)
    np.testing.assert_allclose(demo().noise_jump(np.zeros(2), np.ones(2), betas), np.sqrt(variance))
    assert coefficient**2 + variance == pytest.approx(1)


def test_figure_data_and_exercise_fixture_regenerate():
    m = demo()
    for name, expected in (
        ("generative_models_figures.json", m.figure_data()),
        ("generative_models_scenarios_06.json", m.exercise_fixture()),
    ):
        stored = ROOT / "demos/data" / name
        assert (
            stored.read_text()
            == json.dumps(expected, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )


@pytest.mark.parametrize("name", ["listed", "tau1", "tau2"])
def test_stored_scenario_suites_regenerate_byte_for_byte(name):
    from refund_agent.scenarios.suite import ScenarioSet, regenerate

    path = ROOT / "demos/data" / f"generative_models_scenarios_{name}.json"
    stored = ScenarioSet.read(path)
    assert regenerate(stored.manifest).to_json().encode() == path.read_bytes()


def test_fixture_uses_one_stale_policy_for_all_labels():
    from refund_agent.contracts import Decision
    from refund_agent.labels import _label
    from refund_agent.scenarios.checks import policy_label
    from refund_agent.scenarios.materialize import to_records
    from refund_agent.scenarios.model import training_data

    fixture = demo().exercise_fixture()
    assert fixture["manifest"]["policy_version"] == "2026-07-01"
    assert fixture["manifest"]["as_of"].startswith("2026-09-15")
    data = training_data()
    stale = next(p for p in data.policies if p.version == "2026-07-01")
    wrong = []
    for row in fixture["scenarios"]:
        records = to_records(tuple(row["tokens"]), demo().AS_OF, fixture["manifest"]["seed"])
        recorded = Decision.model_validate_json(json.dumps(row["policy_label"]))
        assert recorded == _label(*records, stale)
        actual = policy_label(records, data)
        if (recorded.outcome, recorded.reason) != (actual.outcome, actual.reason):
            wrong.append(row)
            assert recorded.outcome.value == "deny"
            assert actual.outcome.value in ("approve", "refer")
            assert actual.reason in ("eligible", "approval_required")
            assert row["tokens"][2] == "31-45"
    assert wrong
    assert any(row["policy_check_passed"] for row in wrong)


def test_plots_render_when_matplotlib_is_installed(tmp_path):
    pytest.importorskip("matplotlib", reason="plot rendering needs Matplotlib")
    plot = runpy.run_path(str(ROOT.parent / "figures/plots/fig_generative_models.py"))
    plot["main"](tmp_path)
    for name in ("sampling", "elbo-terms", "schedule"):
        for suffix in (".pdf", ".png", "-gray.png"):
            assert (tmp_path / f"fig-generative-models-{name}{suffix}").stat().st_size > 0
