"""Printed listings, generated artifacts and plots for chapter 14."""

import contextlib
import importlib
import io
import json
import re
import runpy
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from demos import regions as _regions

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "demos/adaptation_demo.py"
TOOLS = vars(_regions)
PREFIX = "fnd.prompting-finetuning-rag."
NAMES = (
    "prompt-01",
    "objective-01",
    "lora-01",
    "lowrank-01",
    "paired-01",
    "experiment-01",
    "results-01",
    "resources-01",
    "suite-01",
    "keys-01",
    "quantize-01",
    "cost-01",
    "keys-02",
)
PRINTED = ("prompt-01", "lora-01", "paired-01")


def snippets():
    return TOOLS["extract_snippets_from_text"](DEMO.read_text(), DEMO)


def test_printed_listings_run_in_chapter_order_without_hidden_setup():
    namespace = {"__name__": "__main__"}
    regions = {s.snippet_id: s for s in snippets()}
    for name in PRINTED:
        with contextlib.redirect_stdout(io.StringIO()) as stdout:
            exec(compile(regions[PREFIX + name].content, str(DEMO), "exec"), namespace)
        assert (
            stdout.getvalue().encode()
            == (ROOT / "book_outputs" / f"_{PREFIX}{name}.txt").read_bytes()
        )


def test_regions_follow_the_outline_and_all_declare_output():
    assert [s.snippet_id for s in snippets()] == [PREFIX + name for name in NAMES]
    assert all(s.output for s in snippets())


def test_prompt_prints_parseable_json_fragments_within_book_measure(capsys):
    snippet = next(s for s in snippets() if s.snippet_id == PREFIX + "prompt-01")
    exec(compile(snippet.content, str(DEMO), "exec"), {"__name__": "__main__"})
    output = capsys.readouterr().out
    decoder = json.JSONDecoder()
    fragments = [
        decoder.raw_decode(output[match.end():])[0]
        for match in re.finditer(
            r"(?m)(?:^|items |section totals: |draft: )(?=[{\[])", output
        )
    ]
    assert fragments == [
        {
            "outcome": "deny", "amount_cents": 120000, "tier": 3,
            "reason": "amount_over_policy", "citation": "refund_cap_v4@2026-07-01#cap",
        },
        {
            "outcome": "refer", "amount_cents": 5000, "tier": 2,
            "reason": "approval_required", "citation": "refund_cap_v4@2026-07-01#eligibility",
        },
        [{"name": "physical desk lamp", "quantity": 1, "unit_cents": 138720}],
        {"evidence": 40, "instructions": 264, "order": 65, "policy": 63, "request": 42},
        {
            "outcome": "approve", "amount_cents": 4999, "tier": 1,
            "reason": "eligible", "citation": "refund_cap_v4@2026-09-01#eligibility",
        },
    ]
    assert max(map(len, output.splitlines())) <= 79


def test_split_tables_keep_the_same_quality_safety_and_word_counts(capsys):
    from demos.adaptation_demo import print_results

    record = SimpleNamespace(evidence=[{
        "arm": "prompt", "correct": 121, "cases": 121,
        "agreement_interval": (0.969, 1.0), "paired": None,
        "forbidden": 0, "zero_failure_bound": 0.0245,
        "words": {"context": 469.16, "draft": 44.92},
    }])
    print_results(record, compact=True)
    quality = capsys.readouterr().out
    assert quality.splitlines()[0].count("|") == 8
    assert quality.splitlines()[-1] == "| prompt | 121/121 [0.969, 1.000] | -- | -- | -- | 0/121 | 0.0245 |"
    from demos.adaptation_demo import print_resources

    print_resources(record)
    assert capsys.readouterr().out == (
        "| Arm | Context words | Draft words |\n"
        "|---|---|---|\n"
        "| prompt | 469.16 | 44.92 |\n"
    )


def test_hospital_prints_both_diagonal_cells(capsys):
    namespace = {"__name__": "__main__"}
    snippet = next(s for s in snippets() if s.snippet_id == PREFIX + "paired-01")
    exec(compile(snippet.content, str(DEMO), "exec"), namespace)
    assert "both complete 166; neither 16\n" in capsys.readouterr().out
    card = namespace["hospital"]
    assert 166 + card.n_plus + card.n_minus + 16 == card.n == 200
    assert 166 + card.n_minus == sum(namespace["base"]) == 170
    assert 166 + card.n_plus == sum(namespace["challenger"]) == 180


@pytest.fixture(scope="module")
def captured():
    return TOOLS["capture_outputs"]({s.snippet_id: s for s in snippets()}, sys.executable)


@pytest.mark.parametrize("name", NAMES)
def test_every_stored_output_regenerates_byte_for_byte(name, captured):
    stored = ROOT / "book_outputs" / f"_{PREFIX}{name}.txt"
    assert captured[PREFIX + name].encode() == stored.read_bytes()


def test_capture_is_deterministic_in_a_fresh_process(captured):
    assert captured == TOOLS["capture_outputs"](
        {s.snippet_id: s for s in snippets()}, sys.executable
    )


@pytest.mark.parametrize("compact", [False, True])
@pytest.mark.parametrize(
    "card, expected",
    [
        (None, "--"),
        ({"n_plus": 0, "n_minus": 0, "exact_chance": 1}, "--"),
        (
            {
                "n_plus": 2,
                "n_minus": 0,
                "lead": 0.02,
                "interval": (0.01, 0.03),
                "exact_chance": 0.5,
            },
            "0.020 [0.010, 0.030]",
        ),
        (
            {
                "n_plus": 0,
                "n_minus": 2,
                "lead": -0.02,
                "interval": (-0.03, -0.01),
                "exact_chance": 0.5,
            },
            "-0.020 [-0.030, -0.010]",
        ),
    ],
)
def test_results_hide_the_lead_interval_only_without_discordance(compact, card, expected, capsys):
    from demos.adaptation_demo import print_results

    row = {
        "arm": "challenger",
        "correct": 90,
        "cases": 100,
        "agreement_interval": (0.8, 0.95),
        "paired": card,
        "forbidden": 0,
        "zero_failure_bound": 0.03,
        "words": {"context": 20, "draft": 10},
    }
    print_results(SimpleNamespace(evidence=[row]), compact=compact)
    printed = capsys.readouterr().out
    if compact:
        assert all(len(line) <= 79 for line in printed.splitlines())
    rows = [line.split("|") for line in printed.splitlines()[2:]]
    columns = [" ".join(row[i].strip() for row in rows if row[i].strip()) for i in range(len(rows[0]))]
    assert columns[4] == expected
    if card is not None:
        assert columns[3] == f"{card['n_plus']} / {card['n_minus']}"
        assert columns[5] == format(card["exact_chance"], ".2g" if compact else ".6g")


def test_import_has_no_output_or_data_effects(tmp_path):
    code = """from refund_agent.data import generate
def forbidden(*args, **kwargs):
    raise AssertionError('import generated data')
generate.generate = forbidden
import demos.adaptation_demo
"""
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=tmp_path, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == ""
    assert list(tmp_path.iterdir()) == []


@pytest.fixture(scope="module")
def demo_report():
    module = importlib.import_module("demos.adaptation_demo")
    return module, module.build_experiment(7)


def test_generated_data_and_selection_record_are_reproducible(demo_report, tmp_path):
    demo, report = demo_report
    demo.write_data(report, tmp_path)
    for path in sorted(tmp_path.glob("*.json")):
        assert path.read_bytes() == (ROOT / "demos/data" / path.name).read_bytes()
    record = report["record"]
    assert record.intact()
    assert record.selected == "prompt"
    assert report["model_choice_record"]["adaptation_selection_record"] == record.sha256


def test_exercise_report_drops_only_the_three_parse_failures(demo_report):
    demo, _ = demo_report
    path = ROOT / "demos/data/adaptation_report_05.json"
    fixture = json.loads(path.read_text())
    scored = demo.rescore_exercise_report(fixture)
    assert fixture["reported"]["adapter"] == {"correct": 115, "total": 118}
    assert fixture["reported"]["prompt"] == {"correct": 116, "total": 121}
    assert scored["correct"] == {"adapter": 115, "prompt": 116}
    assert scored["parse_failures"] == 3
    assert scored["decision"] == "not adopted"


def test_plot_inputs_come_from_the_run(demo_report):
    demo, report = demo_report
    data = json.loads((ROOT / "demos/data/adaptation_figures.json").read_text())
    assert data == demo.figure_data(report)
    assert len(data["golden"]["cases"]) == 121
    assert sum(not row["agreement"]["adapter"] for row in data["golden"]["cases"]) == 38


def test_plots_render_when_matplotlib_is_installed(tmp_path):
    pytest.importorskip("matplotlib", reason="plot rendering needs Matplotlib")
    plot = runpy.run_path(str(ROOT.parent / "figures/plots/fig_prompting_finetuning_rag.py"))
    plot["main"](tmp_path)
    for name in ("golden-drafts", "rank-error", "paired-intervals"):
        for suffix in (".pdf", ".png", "-gray.png"):
            assert (tmp_path / f"fig-prompting-finetuning-rag-{name}{suffix}").stat().st_size > 0
