"""Chapter-order listings and byte-identical captured exploration results."""

import contextlib
import importlib
import io
import json
import runpy
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from demos import regions as _regions

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "demos/exploration_bandits_demo.py"
TOOLS = vars(_regions)
PREFIX = "fnd.exploration-bandits."
NAMES = (
    "trace-01",
    "epsilon-01",
    "ucb-01",
    "thompson-01",
    "cap-01",
    "delay-01",
    "adaptive-01",
    "run-01",
    "regret-01",
    "keys-01",
)


def demo():
    return importlib.import_module("demos.exploration_bandits_demo")


def regions():
    return TOOLS["extract_snippets_from_text"](DEMO.read_text(), DEMO)


def test_printed_listings_run_in_chapter_order_without_hidden_setup():
    namespace = {"__name__": "__main__"}
    snippets = {s.snippet_id: s for s in regions()}
    for name in ("epsilon-01", "cap-01"):
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            exec(compile(snippets[PREFIX + name].content, name, "exec"), namespace)
        assert (
            stdout.getvalue().encode()
            == (ROOT / "book_outputs" / f"_{PREFIX}{name}.txt").read_bytes()
        )
    mean, cap = namespace["incremental_mean"], namespace["cap_from_budget"]
    assert callable(mean) and callable(cap)
    assert mean(0.3, 4, 1) == pytest.approx(0.44)
    assert cap(0.01, 0.8, 0.6) == pytest.approx(0.05)


def test_regions_follow_the_outline():
    assert [s.snippet_id for s in regions()] == [PREFIX + name for name in NAMES]
    assert all(s.output for s in regions())


@pytest.fixture(scope="module")
def captured():
    snippets = {s.snippet_id: s for s in regions()}
    first = TOOLS["capture_outputs"](snippets, sys.executable)
    assert first == TOOLS["capture_outputs"](snippets, sys.executable)
    return first


@pytest.mark.parametrize("name", NAMES)
def test_every_stored_output_regenerates_byte_for_byte(name, captured):
    assert (
        captured[PREFIX + name].encode()
        == (ROOT / "book_outputs" / f"_{PREFIX}{name}.txt").read_bytes()
    )


def test_import_has_no_output_or_file_effects(tmp_path):
    result = subprocess.run(
        [sys.executable, "-c", "import demos.exploration_bandits_demo"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == "" and list(tmp_path.iterdir()) == []


def test_adaptive_enumeration_and_reweighting():
    assert demo().enumerated_means() == (0.4375, 0.4375)
    result = demo().adaptive_experiment()
    assert all(x < 0.5 for x in result["plain"])
    for mean, se in zip(result["reweighted"], result["standard_errors"]):
        assert abs(mean - 0.5) < 3 * se
    assert result == demo().adaptive_experiment()


def test_formal_box_cap_and_thompson_numbers():
    from refund_agent.offers.budget import cap_from_budget
    from refund_agent.offers.policies import best_probabilities, hoeffding_radius

    assert cap_from_budget(0.01, 0.8, 0) == pytest.approx(0.0125)
    assert cap_from_budget(0.01, 0.8, 0.6) == pytest.approx(0.05)
    assert round(hoeffding_radius(25, 0.05), 4) == 0.2448
    assert round(hoeffding_radius(25, 0.05, counts=100), 4) == 0.3899
    assert round(hoeffding_radius(25, 100**-4), 4) == 0.6070
    assert [f"{x:.3f}" for x in best_probabilities([32, 9, 4], [40, 10, 5])] == [
        "0.219",
        "0.574",
        "0.207",
    ]


def test_delay_batch_does_not_call_pending_silence_a_success():
    rows = {row["day"]: row for row in demo().delay_batch()}
    assert rows[7]["silent_success"] == 0.1
    assert rows[7]["arrived_only"] == 1.0
    assert rows[7]["matured"] is None
    assert rows[14]["silent_success"] == rows[14]["arrived_only"] == rows[14]["matured"] == 0.2


def test_figure_data_regenerates_and_keeps_all_twenty_seed_paths():
    data = demo().figure_data(demo().build_experiment())
    stored = (ROOT / "demos/data/exploration_bandits_figures.json").read_bytes()
    assert (json.dumps(data, sort_keys=True, indent=2) + "\n").encode() == stored
    for row in data["strategies"].values():
        assert len(row["seeds"]) == len(row["paths"]) == 20
        np.testing.assert_allclose(row["mean"], np.mean(row["paths"], axis=0))
        np.testing.assert_allclose(row["p10"], np.quantile(row["paths"], 0.1, axis=0))
        np.testing.assert_allclose(row["p90"], np.quantile(row["paths"], 0.9, axis=0))


def test_exercise_five_contains_only_the_uncapped_probability_error():
    expected = demo().exercise_five(correct=True)
    stored = (ROOT / "demos/data/exploration_bandits_log_05.json").read_bytes()
    actual = json.loads(stored)
    assert (json.dumps(demo().exercise_five(), sort_keys=True, indent=2) + "\n").encode() == stored
    assert len(actual["rows"]) == 2000
    for good, bad in zip(expected["rows"], actual["rows"]):
        assert {k: v for k, v in good.items() if k not in ("probabilities", "propensity")} == {
            k: v for k, v in bad.items() if k not in ("probabilities", "propensity")
        }
        assert sum(bad["probabilities"]) == pytest.approx(1)
        if bad["offer"] != "standard":
            assert bad["propensity"] / good["propensity"] == pytest.approx(20)


def test_plots_render_at_stated_sizes(tmp_path):
    pytest.importorskip("matplotlib", reason="plotting requires Matplotlib")
    plot = runpy.run_path(str(ROOT.parent / "figures/plots/fig_exploration_bandits.py"))
    plot["main"](tmp_path)
    for name in ("regret", "delay"):
        for suffix in (".pdf", ".png", "-gray.png"):
            assert (tmp_path / f"fig-exploration-bandits-{name}{suffix}").stat().st_size > 0
