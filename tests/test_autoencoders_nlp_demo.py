"""Printed listings and captured results for the representations chapter."""

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
DEMO = ROOT / "demos/autoencoders_nlp_demo.py"
TOOLS = vars(_regions)
PREFIX = "fnd.autoencoders-nlp."


def snippets():
    return TOOLS["extract_snippets_from_text"](DEMO.read_text(), DEMO)


def test_printed_listings_run_in_chapter_order_without_hidden_setup():
    # This namespace contains only the three printed regions, never the whole demo.
    namespace = {"__name__": "__main__"}
    found = {s.snippet_id.removeprefix(PREFIX): s for s in snippets()}
    for name in ("counts-01", "autoencoder-01", "index-01"):
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(found[name].content, str(DEMO), "exec"), namespace)
    assert len(namespace["clauses"]) == 10
    assert namespace["stale"].refusal == "stale_policy"
    assert namespace["result"].hits[0].clause.version == "2026-07-01"


def test_four_point_covariance_and_trace_are_printed(capsys):
    namespace = {"__name__": "__main__"}
    found = {s.snippet_id.removeprefix(PREFIX): s for s in snippets()}
    for name in ("counts-01", "autoencoder-01"):
        exec(compile(found[name].content, str(DEMO), "exec"), namespace)
    printed = capsys.readouterr().out
    assert all(len(line) <= 79 for line in printed.splitlines())
    assert (
        "four-point covariance: [[2.0, 0.0], [0.0, 0.5]]; trace 2.5; eigenvectors (1, 0), (0, 1)"
        in " ".join(printed.split())
    )
    points = namespace["points"]
    covariance = points.T @ points / len(points)
    np.testing.assert_array_equal(covariance, [[2, 0], [0, 0.5]])
    assert np.trace(covariance) == pytest.approx(namespace["toy"].eigenvalues.sum())


def test_listings_and_figure_ask_at_the_worked_requests_own_time():
    # One clock: the printed question, its time and the figure's question are the
    # worked request's, read from the frozen set's one logged consultation.
    module = importlib.import_module("demos.autoencoders_nlp_demo")
    report = module.build_experiment(7)
    worked = report["worked"]
    assert worked.request_id == "r-7-0117"
    namespace = {"__name__": "__main__"}
    found = {s.snippet_id.removeprefix(PREFIX): s for s in snippets()}
    for name in ("counts-01", "autoencoder-01", "index-01"):
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(found[name].content, str(DEMO), "exec"), namespace)
    for name in ("counts-01", "index-01"):
        assert f'"{worked.text}"' in found[name].content
    assert namespace["at"] == worked.as_of
    hit = namespace["result"].hits[0]
    assert hit.clause.citation == report["worked_hits"]["dated"][0].clause.citation
    assert hit.clause.observed_at == worked.as_of - module.timedelta(hours=1)
    question = module.figure_data(report)["question"]
    assert question["at"] == worked.as_of.isoformat()
    assert question["text"] == worked.text
    assert question["source_request"] == worked.request_id


@pytest.fixture(scope="module")
def captured():
    regions = {s.snippet_id: s for s in snippets()}
    first = TOOLS["capture_outputs"](regions, sys.executable)
    assert first == TOOLS["capture_outputs"](regions, sys.executable)
    return first


@pytest.mark.parametrize(
    "name",
    [
        "counts-01",
        "autoencoder-01",
        "recall-01",
        "recall-02",
        "versions-01",
        "index-01",
        "keys-01",
    ],
)
def test_each_stored_output_regenerates_byte_for_byte(name, captured):
    path = ROOT / "book_outputs" / f"_{PREFIX}{name}.txt"
    assert captured[PREFIX + name].encode() == path.read_bytes()


def test_import_does_not_print_or_generate_data():
    code = """import contextlib, io
from refund_agent.data import generate
def forbidden(*args, **kwargs):
    raise AssertionError('import generated data')
generate.generate = forbidden
with contextlib.redirect_stdout(io.StringIO()) as buffer:
    import demos.autoencoders_nlp_demo
assert buffer.getvalue() == ''
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_no_fit_or_choice_reads_the_frozen_rows(monkeypatch):
    module = importlib.import_module("demos.autoencoders_nlp_demo")
    from refund_agent.retrieval.encoders import CountEncoder, LinearAutoencoder, PairEncoder

    reads = []
    original_freeze = module.freeze

    class GuardedSet:
        def __init__(self, held):
            self.held = held
            self.manifest = held.manifest
            self.log = held.log

        def consult(self, reader, purpose):
            reads.append((reader, purpose))
            return self.held.consult(reader, purpose)

        def __getattr__(self, name):
            raise AssertionError(f"unlogged frozen access: {name}")

    monkeypatch.setattr(module, "freeze", lambda *a, **k: GuardedSet(original_freeze(*a, **k)))
    for cls in (CountEncoder, LinearAutoencoder, PairEncoder):
        original_fit = cls.fit

        def guarded_fit(self, *args, _fit=original_fit, **kwargs):
            assert not reads, "a model was fitted after consulting the golden set"
            return _fit(self, *args, **kwargs)

        monkeypatch.setattr(cls, "fit", guarded_fit)
    report = module.build_experiment(7)
    assert len(reads) == 1
    assert len(report["golden_questions"]["policy"]) == 242
    assert report["selection"]["chosen_on"] == "validation (July)"
    assert report["selection"]["encoder"] == "counts"
    assert len(report["access_log"]) == 1
    # Every undated top-three hit is a tie behind an earlier version of the same clause.
    undated = {
        (row["wording"], row["encoder"]): row["similarity"]
        for row in report["metrics"]
        if row["window"] == "golden"
    }
    wide = undated["policy", "linear-2"]
    assert (wide["hits"][1], wide["hits"][3], wide["tied_below_first"]) == (0, 60, 60)
    assert undated["policy", "linear-1"]["hits"][3] == 0
    assert all(card["tied_below_first"] == card["hits"][3] for card in undated.values())
    assert report["pair_training"]["pairs"] == 4 * 398
    assert report["pair_training"]["policy_versions"] == ["2026-01-20", "2026-03-01", "2026-05-01"]


def test_exercise_fixture_has_only_the_planted_interval_error():
    module = importlib.import_module("demos.autoencoders_nlp_demo")
    from refund_agent.retrieval.encoders import CountEncoder
    from refund_agent.retrieval.index import PolicyIndex

    path = ROOT / "demos/data/autoencoders_nlp_index_report_05.json"
    fixture = json.loads(path.read_text())
    policies = module.exercise_policies()
    encoder = CountEncoder().fit(
        [c.text for c in module.policy_clauses(policies, module.EXERCISE_AT)]
    )
    index = PolicyIndex.build(policies, module.EXERCISE_AT, encoder)
    correct = [c.record() for c in index.clauses]
    altered = fixture["records"]
    assert len(correct) == len(altered) == 12
    differences = []
    for expected, actual in zip(correct, altered):
        for key in expected:
            if expected[key] != actual[key]:
                differences.append((actual["version"], key, actual[key]))
    assert differences == [("2026-09-01", "effective_to", None)] * 2
    assert fixture["golden_recall_at_1"] == 1.0
    assert fixture["golden_last"] < "2026-10-01"


def test_figure_data_is_generated_by_the_demo():
    module = importlib.import_module("demos.autoencoders_nlp_demo")
    path = ROOT / "demos/data/autoencoders_nlp_figures.json"
    assert json.loads(path.read_text()) == module.figure_data(module.build_experiment(7))


def test_sixth_version_exercise_and_rarity_preserve_the_historical_result():
    module = importlib.import_module("demos.autoencoders_nlp_demo")
    for check in module.exercise_acceptance(module.build_experiment(7)):
        assert check["hits"] == check["questions"] == 20
        assert check["out_of_interval"] == 0
        assert check["records_complete"]
        assert check["stale_refusal"] == "stale_policy"
        assert check["golden_hits"] == 242


def test_plots_render_when_matplotlib_is_installed(tmp_path):
    pytest.importorskip("matplotlib", reason="plot rendering needs Matplotlib")
    plot = runpy.run_path(str(ROOT.parent / "figures/plots/fig_autoencoders_nlp.py"))
    plot["main"](tmp_path)
    for name in ("reconstruction", "versions", "reconstruction-width"):
        for suffix in (".pdf", ".png", "-gray.png"):
            assert (tmp_path / f"fig-autoencoders-nlp-{name}{suffix}").stat().st_size > 0
