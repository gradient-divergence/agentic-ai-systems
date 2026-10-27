"""Runnable listings, independent bounds and captured chapter 15 results."""

import ast
import contextlib
import io
import json
import runpy
import sys
from pathlib import Path

import numpy as np
import pytest

from demos import regions as _regions

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "demos/constrained_optimization_demo.py"
TOOLS = vars(_regions)
PREFIX = "fnd.constrained-optimization."
NAMES = (
    "certificate-01",
    "relax-01",
    "branches-01",
    "cp-01",
    "search-01",
    "plan-01",
    "window-01",
    "robust-01",
    "uncertain-01",
    "absence-01",
    "relief-01",
    "keys-01",
)
PRINTED = (
    "certificate-01", "relax-01", "cp-01", "search-01", "plan-01",
    "uncertain-01",
)


def regions():
    return {s.snippet_id: s for s in TOOLS["extract_snippets_from_text"](DEMO.read_text(), DEMO)}


def test_printed_listings_run_in_chapter_order_without_hidden_setup():
    namespace = {"__name__": "__main__"}
    found = regions()
    for name in PRINTED:
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            exec(compile(found[PREFIX + name].content, str(DEMO), "exec"), namespace)
        assert (
            stdout.getvalue().encode()
            == (ROOT / "book_outputs" / f"_{PREFIX}{name}.txt").read_bytes()
        )


def test_search_report_runs_the_listings_own_search_for_both_methods():
    listing = ast.parse(regions()[PREFIX + "search-01"].content)
    search = next(
        node for node in listing.body
        if isinstance(node, ast.FunctionDef) and node.name == "cost_search"
    )
    search.body = ast.parse("return dict(plan=[], cost=-1, expanded=0)").body
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        exec(
            compile(ast.fix_missing_locations(listing), str(DEMO), "exec"),
            {"__name__": "__main__"},
        )
    for name in ("uniform-cost", "A*"):
        assert f"{name}: expanded 0; optimal cost -1" in stdout.getvalue()


@pytest.fixture(scope="module")
def captured():
    return TOOLS["capture_outputs"](regions(), sys.executable)


@pytest.mark.parametrize("name", NAMES)
def test_every_stored_output_regenerates_byte_for_byte(name, captured):
    assert (
        captured[PREFIX + name].encode()
        == (ROOT / "book_outputs" / f"_{PREFIX}{name}.txt").read_bytes()
    )


def test_capture_is_deterministic_and_in_outline_order(captured):
    assert list(regions()) == [PREFIX + name for name in NAMES]
    assert captured == TOOLS["capture_outputs"](regions(), sys.executable)


def test_lab_and_integer_certificates():
    from demos.constrained_optimization_demo import branches, laboratory, whole_laboratory

    lab = laboratory()
    assert lab["plan"] == pytest.approx([0.5, 3])
    assert lab["multipliers"] == pytest.approx([1.5, 0, 0.5])
    assert lab["check"]["bound"] == 7.5
    assert lab["check"]["optimal"]
    whole = whole_laboratory()
    assert whole["plan"] == [1, 2]
    assert whole["value"] == whole["rounded_bound"] == 7
    assert [(r["plan"], r["value"]) for r in branches()] == [
        ([0.5, 3], 7.5),
        ([0, 3], 6),
        ([1, 2], 7),
    ]


def test_incident_orders_and_independent_counting_proof():
    from demos.constrained_optimization_demo import counting_proof, incident_orders

    orders = incident_orders()
    assert len(orders) == 6
    assert [row["order"] for row in orders if row["feasible"]] == [["B", "A", "C"]]
    proof = counting_proof()
    assert proof == {"due_slot": 2, "tasks": ["A", "B"], "work_slots": 3, "available_slots": 2}


def test_constraint_solver_returns_the_unique_order_and_refuses_stricter_due_time():
    pytest.importorskip("ortools", reason="cp-01 requires the opt extra (OR-Tools)")
    from demos.constrained_optimization_demo import incident_schedule

    result = incident_schedule()
    assert result["status"] == "optimal"
    assert result["starts"] == {"A": 1, "B": 0, "C": 3}
    assert incident_schedule(strict=True)["status"] == "infeasible"


@pytest.mark.parametrize(
    "multipliers,plan,permitted,certificate",
    [
        ([-1, 0, 0.5], [0.5, 3], True, False),
        ([1, 0, 0], [0.5, 3], True, False),
        ([1.5, 0, 0.5], [1, 3], False, True),
    ],
)
def test_arithmetic_check_rejects_bad_certificates_or_plans(
    multipliers, plan, permitted, certificate
):
    from refund_agent.schedule.certificate import check_certificate

    check = check_certificate([[2, 1], [1, 0], [0, 1]], [4, 3, 3], [3, 2], plan, multipliers)
    assert check["permitted"] is permitted
    assert check["certificate"] is certificate
    assert not check["optimal"]


def test_relief_cost_and_certificate():
    from demos.constrained_optimization_demo import relief

    result = relief()
    assert result["shipments"] == [[3, 1], [0, 3]]
    assert result["cost"] == 10
    assert result["check"]["optimal"]
    assert result["check"]["bound"] == -10  # maximize negative cost


def test_transportation_uses_generic_nodes():
    from demos.constrained_optimization_demo import solve_transportation

    result = solve_transportation([("depot", 10)], [("shelter", 4)], [[2]])
    assert result["cost"] == 8
    assert result["shipments"] == [[4]]


def test_figure_data_and_exercise_fixture_regenerate():
    from demos.constrained_optimization_demo import exercise_five, figure_data

    for name, data in (
        ("constrained_optimization_figures", figure_data()),
        ("review_plan_05", exercise_five()),
    ):
        assert json.loads((DEMO.parent / "data" / f"{name}.json").read_text()) == data


def test_all_json_artifacts_regenerate_byte_for_byte(tmp_path, monkeypatch):
    from demos import constrained_optimization_demo as demo

    destination = tmp_path / "companion"
    (destination / "demos/data").mkdir(parents=True)
    (destination / "book_outputs").mkdir()
    monkeypatch.setattr(
        demo, "__file__", str(destination / "demos/constrained_optimization_demo.py")
    )
    demo.write_data()
    generated_files = sorted(destination.rglob("*.json"))
    assert {str(p.relative_to(destination)) for p in generated_files} == {
        "book_outputs/_fnd.constrained-optimization.solve-records.json",
        "demos/data/constrained_optimization_figures.json",
        "demos/data/constrained_optimization_enrichment_figures.json",
        "demos/data/review_plan_05.json",
    }
    for generated in generated_files:
        original = ROOT / generated.relative_to(destination)
        assert generated.read_bytes() == original.read_bytes()


def test_plots_render_in_colour_and_grayscale(tmp_path):
    pytest.importorskip("matplotlib", reason="plot rendering needs Matplotlib")
    plot = runpy.run_path(str(ROOT.parent / "figures/plots/fig_constrained_optimization.py"))
    plot["main"](tmp_path)
    for name in ("region", "monday"):
        for suffix in (".pdf", ".png", "-gray.png"):
            assert (tmp_path / f"fig-constrained-optimization-{name}{suffix}").stat().st_size > 0


def test_nonfinite_certificate_inputs_are_rejected():
    from refund_agent.schedule.certificate import check_certificate

    with pytest.raises(ValueError, match="finite"):
        check_certificate([[1]], [1], [1], [0], [np.nan])


def test_engineer_tree_counts_and_pruning_preserve_the_only_plan():
    from demos import constrained_optimization_demo as demo

    full, cut = demo.engineer_tree(), demo.engineer_tree(prune=True)
    assert (full["generated"], full["examined"], full["expanded"]) == (9, 9, 6)
    assert (cut["generated"], cut["examined"], cut["expanded"]) == (5, 5, 3)
    assert (full["pruned"], cut["pruned"]) == (0, 1)
    assert full["plans"] == cut["plans"] == [["B", "A", "C"]]
    assert [n["order"] for n in cut["nodes"] if n["pruned"]] == [["A"]]
    assert {tuple(n["order"]) for n in full["nodes"]} == {
        (), ("A",), ("B",), ("A", "B"), ("A", "C"), ("B", "A"),
        ("A", "B", "C"), ("A", "C", "B"), ("B", "A", "C"),
    }


def brute_remaining(tasks, state):
    from itertools import permutations

    done, clock = state
    costs = []
    for order in permutations(set(tasks) - done):
        seen, time, cost = set(done), clock, 0
        for name in order:
            task = tasks[name]
            if not set(task["after"]) <= seen:
                break
            time += task["length"]
            cost += task["weight"] * max(0, time - task["due"])
            seen.add(name)
        else:
            costs.append(cost)
    return min(costs)


def test_relaxed_heuristic_is_admissible_on_every_reachable_state():
    from itertools import permutations

    from demos import constrained_optimization_demo as demo

    tasks, states = demo.oncall_tasks(), {(frozenset(), 0)}
    for order in permutations(tasks):
        done, clock = frozenset(), 0
        for name in order:
            if not set(tasks[name]["after"]) <= done:
                break
            done, clock = done | {name}, clock + tasks[name]["length"]
            states.add((done, clock))
    assert any(demo.relaxed_cost(tasks, state) > 0 for state in states)
    for state in states:
        assert 0 <= demo.relaxed_cost(tasks, state) <= brute_remaining(tasks, state)
    optimum = brute_remaining(tasks, (frozenset(), 0))
    ucs, astar = demo.cost_search(tasks), demo.cost_search(tasks, demo.relaxed_cost)
    assert ucs["cost"] == astar["cost"] == optimum
    assert astar["expanded"] < ucs["expanded"]
    for result in (ucs, astar):
        done, time, cost = set(), 0, 0
        for name in result["plan"]:
            task = tasks[name]
            assert set(task["after"]) <= done
            time += task["length"]
            cost += task["weight"] * max(0, time - task["due"])
            done.add(name)
        assert done == set(tasks) and cost == optimum


def test_root_children_report_real_step_costs_and_relaxed_priorities(capsys):
    from demos import constrained_optimization_demo as demo

    tasks, priorities = demo.oncall_tasks(), {}
    for name, task in tasks.items():
        if task["after"]:
            continue
        g = demo.cost_search({name: task})["cost"]
        h = demo.relaxed_cost(tasks, (frozenset({name}), task["length"]))
        priorities[name] = (g, h, g + h)
    assert priorities == {"credential": (0, 2, 2), "database": (0, 8, 8)}
    demo.search_report(demo.cost_search)
    assert (
        "root's children: credential g 0, h 2, f 2; database g 0, h 8, f 8"
        in capsys.readouterr().out
    )


@pytest.mark.parametrize("budget,coverage", [(5, 211 / 243), (12, 527345 / 531441)])
def test_sampling_coverage_does_not_remove_false_acceptances(budget, coverage):
    from demos import constrained_optimization_demo as demo

    result = demo.proposal_sampling(budget)
    assert result == demo.proposal_sampling(budget)
    assert result["coverage"] == pytest.approx(coverage)
    assert result["minimum_k"] == 12
    assert result["invalid_share"] == pytest.approx(1 / 6)
    assert result["simulated_coverage"] == pytest.approx(coverage, abs=0.02)
    assert result["simulated_invalid_share"] == pytest.approx(1 / 6, abs=0.02)
    assert result["accepted"] + result["exhausted"] == 10000
    assert result["acceptance"] == pytest.approx(1 - 0.6**budget)
    assert result["accepted"] / 10000 == pytest.approx(result["acceptance"], abs=0.02)
    assert demo.proposal_sampling(budget, false_pass=0)["invalid_share"] == 0


@pytest.fixture(scope="module")
def uncertain():
    from demos import constrained_optimization_demo as demo

    return demo.uncertain_results()


def test_chance_rows_and_all_four_checked_monday_plans(uncertain):
    from demos import constrained_optimization_demo as demo

    row = demo.normal_row([3, 3], 60, 0.95)
    assert row["mean"] == 60
    assert row["spread"] == pytest.approx(10.6066017178)
    assert row["quantile"] == pytest.approx(1.64485362695)
    assert row["required"] == pytest.approx(77.4463073052)
    assert row["overrun"] == 0.5
    assert not row["holds"]
    assert [len(p["plan"]) for p in uncertain["stances"].values()] == [9, 9, 10, 10]
    for stance in uncertain["stances"].values():
        assert not stance["failures"]
        assert stance["rows"]["alex"]["overrun"] == pytest.approx(
            7.6872989721e-13, abs=1e-20
        )
    for name in ("robust", "chance-95"):
        assert uncertain["stances"][name]["rows"]["morgan"]["overrun"] == pytest.approx(
            0.0000316712418331
        )
    for name in ("chance-50", "scenarios"):
        assert uncertain["stances"][name]["rows"]["morgan"]["overrun"] == 0.5


def test_fifty_percent_boundary_uses_the_schedule_tolerance():
    from demos import constrained_optimization_demo as demo
    from refund_agent.schedule.inputs import TOL

    assert demo.normal_row([3, 3], 60, 0.5)["holds"]
    assert demo.normal_row([3, 3], 60 - TOL / 2, 0.5)["holds"]
    assert not demo.normal_row([3, 3], 60 - 2 * TOL, 0.5)["holds"]
    assert demo.normal_row([3, 3], 60, 0.5)["quantile"] == 0


def test_ninety_percent_exercise_with_ten_minute_tier_three_spread():
    from demos import constrained_optimization_demo as demo

    spreads = {1: 1.25, 2: 3.75, 3: 10}
    two = demo.normal_row([3, 3], 60, 0.9, spreads)
    one = demo.normal_row([3], 60, 0.9, spreads)
    assert two["quantile"] == pytest.approx(1.28155156554)
    assert two["required"] == pytest.approx(78.1238760487)
    assert one["required"] == pytest.approx(42.8155156554)
    assert not two["holds"] and one["holds"]
    plan = demo.chance_plan(demo.monday_queue(), 0.9, spreads)
    assert len(plan) == 9


def test_scenarios_choose_two_with_seeded_and_exact_completion_counts(uncertain):
    from demos import constrained_optimization_demo as demo

    assert uncertain == demo.uncertain_results()
    one, two = uncertain["one"], uncertain["two"]
    assert one["sample_average"] == pytest.approx(9, abs=0.01)
    assert two["sample_average"] == pytest.approx(9.5, abs=0.03)
    assert one["exact_average"] == pytest.approx(8.99996832876)
    assert two["exact_average"] == pytest.approx(9.49996832876)
    assert uncertain["stances"]["scenarios"]["plan"] == uncertain["stances"]["chance-50"]["plan"]


def test_scenario_completion_rule_counts_only_finishes_within_each_block():
    from demos import constrained_optimization_demo as demo
    from refund_agent.schedule.rule import ordered_plan

    queue = [d for d in demo.monday_queue() if d.tier == 3][:2]
    plan = ordered_plan(queue, {d.request_id: "morgan" for d in queue})
    times = np.array([[45, 20], [30, 30], [61, 1]], dtype=float)
    assert demo.scenario_average(plan, queue, times) == 1


def test_new_printable_regions_fit_the_width_limit():
    found = regions()
    for name in ("search-01", "uncertain-01"):
        assert max(map(len, found[PREFIX + name].content.splitlines())) <= 79


def test_enrichment_figure_data_matches_the_curve_panels_and_normal_quantile():
    from demos import constrained_optimization_demo as demo

    data = demo.enrichment_figure_data()
    curve, panels, bell = data["F2"], data["F4"], data["F5"]
    assert curve["coverage"][4] == pytest.approx(211 / 243)
    assert curve["coverage"][11] == pytest.approx(527345 / 531441)
    assert curve["invalid_share"] == pytest.approx([1 / 6] * len(curve["N"]))
    assert [p["reviews"] for p in panels] == [1, 1, 2]
    assert [p["minutes_required"] for p in panels] == pytest.approx(
        [45, 42.3364022021, 60]
    )
    assert panels[-1]["normal_overrun"] == 0.5
    assert bell["mean"] == bell["capacity"] == 60
    assert bell["quantile_95"] == pytest.approx(77.4463073052)
    index = bell["minutes"].index(bell["quantile_95"])
    assert bell["cdf"][index] == pytest.approx(0.95)



def test_hours_range_is_the_interval_of_the_solved_binding_rows():
    from demos.constrained_optimization_demo import laboratory

    low,high=laboratory()["hours_range"]
    assert (low,high)==(3,9)
    for hours in (low+0.01,high-0.01):
        assert laboratory(hours)["multipliers"][0]==pytest.approx(1.5)
    assert laboratory(low-0.01)["multipliers"][0]!=pytest.approx(1.5)
    assert laboratory(high+0.01)["multipliers"][0]!=pytest.approx(1.5)
