"""Chapter 27's printed listings, captures and generated exercise inputs."""

import contextlib
import io
import json
import re
import runpy
from itertools import product
from pathlib import Path
from xml.etree import ElementTree

import pytest

from demos import regions as _regions

ROOT = Path(__file__).resolve().parents[1]
COMPANION = ROOT
PREFIX = "fnd.multi-agent-systems."
EXTRACT = vars(_regions)
SOURCES = (
    COMPANION / "refund_agent/verifier/independent.py",
    COMPANION / "demos/agentic_systems_demo.py",
    COMPANION / "demos/multi_agent_systems_demo.py",
)


def chapter_regions():
    found = {}
    for source in SOURCES:
        assert source.exists(), f"chapter source missing: {source.relative_to(ROOT)}"
        found.update(
            (snippet.snippet_id, snippet)
            for snippet in EXTRACT["extract_snippets_from_text"](source.read_text(), source)
            if snippet.snippet_id.startswith(PREFIX)
        )
    return found


def test_printed_listings_run_in_chapter_order_without_hidden_setup():
    regions = chapter_regions()
    namespace = {"__name__": "__main__"}
    for name in ("verifier-01", "sota-01"):
        snippet = regions[PREFIX + name]
        with contextlib.redirect_stdout(io.StringIO()) as output:
            exec(compile(snippet.content, str(snippet.source_path), "exec"), namespace)
        assert output.getvalue().strip()
    assert namespace["weighted_consensus"](namespace["votes"], namespace["config"]) == {
        "approved": True,
        "approvals": 2.0,
        "weight": 1.6,
    }


def test_every_chapter_output_regenerates_byte_for_byte():
    outputs = EXTRACT["capture_outputs"](chapter_regions(), __import__("sys").executable)
    assert set(outputs) == {
        PREFIX + name
        for name in (
            "verifier-01",
            "sota-01",
            "shared-01",
            "trace-01",
            "arms-01",
            "handoff-01",
            "keys-01",
            "shared-02",
            "diagnostic-01",
            "saving-01",
            "exercise-06-01",
            "strategic-01",
            "contract-01",
        )
    }
    for snippet_id, output in outputs.items():
        stored = (COMPANION / "book_outputs" / f"_{snippet_id}.txt").read_text()
        if snippet_id == PREFIX + "diagnostic-01":
            # Only the new diagnostic's measured wall-clock scalar is nondeterministic.
            pattern = r"(?<=on this machine: )\d+\.\d+(?=; not a model call)"
            assert len(re.findall(pattern, output)) == len(re.findall(pattern, stored)) == 1
            assert float(re.search(pattern, output).group()) > 0
            assert re.sub(pattern, "<measured>", output) == re.sub(pattern, "<measured>", stored)
        else:
            assert output.encode() == stored.encode()
        if snippet_id in {PREFIX + name for name in ("shared-02", "trace-01", "handoff-01")}:
            assert max(map(len, output.splitlines())) <= 79


def test_importing_new_modules_prints_and_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with contextlib.redirect_stdout(io.StringIO()) as output:
        runpy.run_path(str(SOURCES[0]), run_name="chapter_import")
        runpy.run_path(str(SOURCES[2]), run_name="chapter_import")
    assert output.getvalue() == ""
    assert list(tmp_path.iterdir()) == []


def test_figure_data_and_exercise_fixtures_regenerate(tmp_path):
    from demos.multi_agent_systems_demo import write_data

    write_data(tmp_path)
    paths = (
        "demos/data/multi_agent_systems_figures.json",
        "exercises/multi-agent-systems/review_05.json",
        "exercises/multi-agent-systems/proof_of_delivery.json",
    )
    for path in paths:
        assert (tmp_path / path).read_bytes() == (COMPANION / path).read_bytes()
    data = json.loads((tmp_path / paths[0]).read_text())
    assert data["exact"][0]["pair"] == pytest.approx(0.04)
    assert data["exact"][-1]["majority"] == pytest.approx(0.2)


def test_exercise_starter_fails_acceptance_and_reference_passes():
    path = COMPANION / "exercises/multi-agent-systems"
    starter = runpy.run_path(str(path / "starter_06.py"))
    reference = runpy.run_path(str(path / "instructor_reference_06.py"))
    acceptance = runpy.run_path(str(path / "test_exercise_06.py"))["check_acceptance"]
    with pytest.raises(AssertionError):
        acceptance(starter["ProofCheckingVerifier"])
    acceptance(reference["ProofCheckingVerifier"])


def test_bank_record_exposes_agreement_as_the_planted_error():
    from demos.multi_agent_systems_demo import bank_review, bank_summary

    record = bank_review()
    summary = bank_summary(record)
    assert summary["agreements"] == 118
    assert summary["counts"] == {"00": 109, "01": 2, "10": 0, "11": 9}
    assert summary["rho_p"] == pytest.approx(9 / 120)
    assert summary["rho_v"] == pytest.approx(11 / 120)
    assert summary["correlation"] == pytest.approx(0.8963480370, abs=1e-10)


def test_reads_table_and_exercise_two_decide_the_supplied_events():
    from demos.multi_agent_systems_demo import read_cases

    rows = read_cases()
    assert {name: row["paid"] for name, row in rows.items()} == {
        "stale carrier, 45 minutes": (0, 0, 0),
        "words contradict delivery": (2499, 2499, 0),
        "fresh false carrier date": (2499, 2499, 2499),
        "planted instruction with ignore": (0, 0, 0),
        "instruction without ignore": (2499, 2499, 2499),
        "independent verifier timeout": (2499, 2499, 0),
    }
    assert rows["stale carrier, 45 minutes"]["agreeing"] == "not asked"
    assert rows["independent verifier timeout"]["independent"] == "refers:verifier_timeout"


def test_printed_simulation_uses_the_same_cases_as_the_figure():
    from refund_agent.verifier.shared_error import simulation

    points = json.loads((COMPANION / "demos/data/multi_agent_systems_figures.json").read_text())[
        "simulation"
    ]
    rows = simulation(7)
    assert [row["observed"] for row in rows] == [
        points[0]["pair"],
        points[0]["majority"],
        points[2]["majority"],
        points[4]["majority"],
    ]


def test_shared_excerpt_is_only_the_existing_heading_and_simulation(capsys):
    from demos import multi_agent_systems_demo as demo

    assert PREFIX + "shared-02" in chapter_regions(), "the excerpt must have its own capture"
    demo.print_shared_excerpt()
    expected = (COMPANION / "book_outputs" / f"_{PREFIX}shared-01.txt").read_text().splitlines()
    output = capsys.readouterr().out
    assert output.split() == "\n".join([expected[0], *expected[13:17]]).split()
    assert max(map(len, output.splitlines())) <= 79


def test_saving_plot_uses_the_equation_and_labels_illustrative_values():
    from demos import multi_agent_systems_demo as demo

    assert PREFIX + "saving-01" in chapter_regions(), "V8's values must be captured"
    assert demo.saving_cents(24 / 400, 8 / 400) == pytest.approx(117.94)
    assert demo.saving_cents(1 / 200, 0.02) == pytest.approx(2.495)
    for price in (0, 25, 100):
        boundary = (0.02 * 400 + price) / 2099
        assert demo.saving_cents(boundary, 0.02, price) == pytest.approx(0)
        assert demo.saving_cents(boundary + 0.001, 0.02, price) > 0
        assert demo.saving_cents(boundary - 0.001, 0.02, price) < 0
    svg = ElementTree.fromstring(demo.saving_plot_svg())
    labels = " ".join(svg.itertext())
    for value in ("25 cents", "100 cents", "capacity", "2499 cents", "400 cents"):
        assert value in labels and "illustrative" in labels.casefold()
    assert "fixture" in labels and "rare mix" in labels
    assert "saves money below" in labels
    paths = {node.attrib["id"]: node for node in svg.iter() if "id" in node.attrib}
    assert {
        "call-0",
        "call-25",
        "call-100",
        "review-capacity",
        "fixture",
        "rare-mix",
    } <= paths.keys()
    assert float(paths["fixture"].attrib["cx"]) > float(paths["rare-mix"].attrib["cx"])


def test_diagnostic_capture_labels_real_local_timing_and_no_model_calls(capsys):
    from demos import multi_agent_systems_demo as demo

    assert PREFIX + "diagnostic-01" in chapter_regions(), "the diagnostic must have its own capture"
    demo.print_diagnostic()
    text = capsys.readouterr().out
    assert "correct 376/400; paid 368; paid wrongly 16; referred 32" in text
    assert "additional payments 0; changed against single 32; differs from independent 0" in text
    assert "verifier calls 0; verdict records 0; review items 32" in text
    assert re.search(
        r"local scripted check wall-clock seconds/request on this machine: \d+\.\d+", text
    )
    assert "not a model call" in text and "not a challenger; no rule decision" in text


def test_exercise_six_matches_the_captured_cost_and_agreement(capsys):
    from demos import multi_agent_systems_demo as demo

    assert PREFIX + "exercise-06-01" in chapter_regions()
    demo.print_exercise_cost()
    output = capsys.readouterr().out
    assert "saving $1.1794 -> $2.0190/request before the call; increase $0.8396" in output
    assert "agreement 376/400 -> 376/400; n+ 24; n- 8; accepted" in output
    assert "break-even call price $2.0190; saves money strictly below it" in output
    stored = (COMPANION / "book_outputs/_fnd.multi-agent-systems.exercise-06-01.txt").read_text()
    assert output == stored


@pytest.mark.parametrize(
    "charge, expected, equilibrium, response",
    [
        (0, {("decide", "decide"): (2, 2),
             ("decide", "refer"): (3.5, 1.5),
             ("refer", "decide"): (1.5, 3.5),
             ("refer", "refer"): (3, 3)}, ("refer", "refer"), "refer"),
        (1.5, {("decide", "decide"): (2, 2),
               ("decide", "refer"): (3.5, 3),
               ("refer", "decide"): (3, 3.5),
               ("refer", "refer"): (4.5, 4.5)},
         ("decide", "decide"), "decide"),
    ],
)
def test_strategic_tables_best_responses_and_enumerated_equilibria(
    charge, expected, equilibrium, response,
):
    from demos import multi_agent_systems_demo as demo

    table = demo.referral_costs(charge)
    assert table == expected
    for agent, other in product(range(2), ("decide", "refer")):
        assert demo.best_responses(table, agent, other) == (response,)
    assert demo.pure_equilibria(table) == [equilibrium]
    # Check the system-cost comparison separately from individual best responses.
    assert sum(table[("decide", "decide")]) < sum(table[("refer", "refer")])


def test_referral_charge_boundary_keeps_indifference_explicit():
    from demos import multi_agent_systems_demo as demo

    assert demo.pure_equilibria(demo.referral_costs(0.49)) == [("refer", "refer")]
    assert set(demo.pure_equilibria(demo.referral_costs(0.50))) == set(
        product(("decide", "refer"), repeat=2)
    )
    assert demo.pure_equilibria(demo.referral_costs(0.51)) == [("decide", "decide")]


@pytest.mark.parametrize(
    "rule, bids, winner, payment, gain",
    [
        ("pay-your-bid", (4, 6, 9), 0, 4, 0),
        ("pay-your-bid", (5.99, 6, 9), 0, 5.99, 1.99),
        ("second-price", (4, 6, 9), 0, 6, 2),
        ("second-price", (5.99, 6, 9), 0, 6, 2),
        ("second-price", (6.01, 6, 9), 1, 6.01, 0.01),
        ("second-price", (10, 11, 9), 2, 10, 1),
    ],
)
def test_contract_winner_payment_and_gain(rule, bids, winner, payment, gain):
    from demos.multi_agent_systems_demo import award_task

    award = award_task((4, 6, 9), bids, rule)
    assert award["winner"] == winner
    assert award["payment"] == pytest.approx(payment)
    assert award["gain"] == pytest.approx(gain)


@pytest.mark.parametrize("agent", range(3))
def test_second_price_truth_is_best_against_all_cent_deviations(agent):
    from demos.multi_agent_systems_demo import award_task

    costs = (4, 6, 9)
    truthful = award_task(costs, costs, "second-price")["gains"][agent]
    for cents in range(1001):
        bids = list(costs)
        bids[agent] = cents / 100
        assert award_task(costs, bids, "second-price")["gains"][agent] <= truthful


@pytest.mark.parametrize("agent", range(3))
def test_second_price_truth_is_best_with_other_reports_and_ties(agent):
    from demos.multi_agent_systems_demo import award_task

    costs = (4, 6, 9)
    # Include either side of each true cost, equal bids, and a zero bid.
    reports = (0, 3.99, 4, 4.01, 5.99, 6, 6.01, 8.99, 9, 9.01, 10)
    others = [i for i in range(3) if i != agent]
    for pair in product(reports, repeat=2):
        bids = list(costs)
        for other, bid in zip(others, pair):
            bids[other] = bid
        truthful = award_task(costs, bids, "second-price")["gains"][agent]
        for bid in reports:
            bids[agent] = bid
            assert award_task(costs, bids, "second-price")["gains"][agent] <= truthful


def test_second_price_truth_is_weak_and_misreports_can_hurt():
    from demos.multi_agent_systems_demo import award_task

    costs = (4, 6, 9)
    assert award_task(costs, costs, "second-price")["gains"] == (2, 0, 0)
    assert award_task(costs, (5.99, 6, 9), "second-price")["gains"] == (2, 0, 0)
    assert award_task(costs, (6.01, 6, 9), "second-price")["gains"][0] == 0
    assert award_task(costs, (4, 3.99, 9), "second-price")["gains"][1] == -2
    assert award_task(costs, (6, 6, 9), "second-price")["winner"] == 0


@pytest.mark.parametrize(
    "bids, winner, payment, gains",
    [
        ((4, 6, 9, 11), 0, 6, (2, 0, 0, 0)),
        ((12, 13, 14, 11), 3, 12, (0, 0, 0, 1)),
    ],
)
def test_four_bidder_exercise_uses_the_same_award_rule(bids, winner, payment, gains):
    from demos.multi_agent_systems_demo import award_task

    row = award_task((4, 6, 9, 11), bids, "second-price")
    assert row["winner"] == winner
    assert row["payment"] == payment
    assert row["gain"] == gains[winner]
    assert row["gains"] == gains


@pytest.mark.parametrize("name", ("strategic-01", "contract-01"))
def test_enrichment_regions_run_alone_with_explicit_imports(name, capsys):
    import ast

    snippet = chapter_regions()[PREFIX + name]
    assert any(isinstance(n, (ast.Import, ast.ImportFrom))
               for n in ast.parse(snippet.content).body)
    assert max(map(len, snippet.content.splitlines())) <= 79
    exec(compile(snippet.content, str(snippet.source_path), "exec"),
         {"__name__": "__main__"})
    assert capsys.readouterr().out == (
        COMPANION / "book_outputs" / f"_{PREFIX}{name}.txt"
    ).read_text()
