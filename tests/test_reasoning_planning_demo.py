"""The reasoning and planning chapter's validator, checked on the plans it prints."""

import contextlib
import io
import math
import runpy
from collections import deque
from itertools import product
from pathlib import Path
from random import Random
from typing import Any

import pytest

DEMO = Path(__file__).resolve().parents[1] / "demos" / "reasoning_planning_demo.py"


@pytest.fixture(scope="module")
def demo() -> dict[str, Any]:
    with contextlib.redirect_stdout(io.StringIO()):
        return runpy.run_path(str(DEMO), run_name="reasoning_planning_demo")


def plan(demo: dict[str, Any], *calls: tuple[str, int] | str) -> list[Any]:
    step = demo["Step"]
    return [step(call) if isinstance(call, str) else step(*call) for call in calls]


READS = ("order.lookup", "shipment.status", "policy.retrieve")


def test_pinned_tier1_plan_is_valid_and_bounded(demo: dict[str, Any]) -> None:
    facts, limits = demo["facts"], demo["limits7"]
    assert len(demo["plan7"]) == 6 and limits.steps == 12
    found = demo["validate_plan"](demo["plan7"], facts(), facts("paid", "audited"), limits)
    assert found is None


def test_second_run_must_read_again(demo: dict[str, Any]) -> None:
    validate, start, limits = demo["validate_plan"], demo["start9"], demo["limits9"]
    goal = demo["facts"]("paid", "audited")
    assert start == demo["facts"]("approved", "proposed")  # the review is closed
    assert not start & demo["READS"]
    refused = validate(demo["at_once"], start, goal, limits, 5000)
    assert refused == "step 1: lacks order_read, policy_read, shipment_read"
    assert validate(demo["again"], start, goal, limits, 5000) is None


def test_each_planted_plan_is_refused_by_its_own_check(demo: dict[str, Any]) -> None:
    validate, facts = demo["validate_plan"], demo["facts"]
    found = [
        validate(plan, facts(), facts("paid", "audited"), limits)
        for _, plan, limits in demo["planted"]
    ]
    assert found == [
        "step 4: policy.check is not a tool",
        "step 4: lacks proposed",
        "step 5: lacks approved",
        "step 6: a second payment",
        "step 5: pays 2499, most 1000",
        "goal unmet: audited",
        "13 steps, budget 12",
    ]


def test_payment_above_its_proposal_needs_its_own_tier(demo: dict[str, Any]) -> None:
    facts, limits = demo["facts"], demo["limits7"]
    steps = plan(demo, *READS, ("refund.propose", 2499), ("refund.execute", 12000), "audit.append")
    found = demo["validate_plan"](steps, facts(), facts("paid", "audited"), limits)
    assert found == "step 5: lacks approved"
    approved = demo["validate_plan"](
        steps, facts("approved"), facts("paid", "audited"), limits, 2499
    )
    assert approved == "step 5: pays 12000, most 2499"


@pytest.mark.parametrize("cents", [-2499, 0, 24.99, True])
def test_payment_must_be_a_positive_whole_amount(demo: dict[str, Any], cents: object) -> None:
    facts, limits = demo["facts"], demo["limits7"]
    steps = plan(demo, *READS, ("refund.propose", 2499), ("refund.execute", cents), "audit.append")
    found = demo["validate_plan"](steps, facts(), facts("paid", "audited"), limits)
    assert found == f"step 5: {cents} is not an amount"


def test_payment_may_not_exceed_its_proposal(demo: dict[str, Any]) -> None:
    facts, limits = demo["facts"], demo["limits7"]
    steps = plan(demo, *READS, ("refund.propose", 2499), ("refund.execute", 4000), "audit.append")
    found = demo["validate_plan"](steps, facts(), facts("paid", "audited"), limits)
    assert found == "step 5: pays 4000, most 2499"


def test_approval_does_not_carry_to_a_new_proposal(demo: dict[str, Any]) -> None:
    facts, limits = demo["facts"], demo["limits9"]
    start = facts("approved", "proposed")
    steps = plan(demo, *READS, ("refund.propose", 6000), ("refund.execute", 6000), "audit.append")
    found = demo["validate_plan"](steps, start, facts("paid", "audited"), limits, 5000)
    assert found == "step 5: lacks approved"
    same = plan(demo, *READS, ("refund.propose", 5000), ("refund.execute", 5000), "audit.append")
    assert demo["validate_plan"](same, start, facts("paid", "audited"), limits, 5000) is None


def test_audit_record_must_come_last(demo: dict[str, Any]) -> None:
    facts, limits = demo["facts"], demo["limits7"]
    steps = plan(demo, "audit.append", *READS, ("refund.propose", 2499), ("refund.execute", 2499))
    found = demo["validate_plan"](steps, facts(), facts("paid", "audited"), limits)
    assert found == "step 1: audit.append must be the last call"


def test_fractional_proposal_is_not_an_amount(demo: dict[str, Any]) -> None:
    facts, limits = demo["facts"], demo["limits7"]
    steps = plan(demo, *READS, ("refund.propose", 2499.5), ("refund.execute", 2499), "audit.append")
    found = demo["validate_plan"](steps, facts(), facts("paid", "audited"), limits)
    assert found == "step 4: 2499.5 is not an amount"


def test_audit_record_comes_last_whatever_the_start(demo: dict[str, Any]) -> None:
    facts, limits = demo["facts"], demo["limits7"]
    goal = facts("paid", "audited")
    steps = plan(demo, *READS, ("refund.propose", 2499), ("refund.execute", 2499))
    found = demo["validate_plan"](steps, facts("audited"), goal, limits)
    assert found == "the plan does not end with audit.append"
    closed = [*steps, demo["Step"]("audit.append")]
    assert demo["validate_plan"](closed, facts("audited"), goal, limits) is None


def test_another_table_through_operators(demo: dict[str, Any]) -> None:
    operator, facts, limits = demo["Operator"], demo["facts"], demo["limits7"]
    table = {
        "open": operator(facts(), facts("opened")),
        "pour": operator(facts("opened"), facts("poured")),
    }
    validate, goal = demo["validate_plan"], facts("poured")
    assert validate(plan(demo, "open", "pour"), facts(), goal, limits, operators=table) is None
    refused = validate(plan(demo, "pour"), facts(), goal, limits, operators=table)
    assert refused == "step 1: lacks opened"
    assert validate(plan(demo, "open"), facts(), goal, limits) == "step 1: open is not a tool"


def test_forty_public_runs_are_valid_plans(demo: dict[str, Any]) -> None:
    assert sum(demo["runs"].values()) == 40
    assert all(found == "valid" for (_, found) in demo["runs"])


def test_operators_and_their_rules(demo: dict[str, Any]) -> None:
    operators = demo["OPERATORS"]
    assert len(operators) == 7
    assert not any(op.delete for op in operators.values())
    assert [name for name, op in operators.items() if op.irreversible] == ["refund.execute"]
    assert [name for name, op in operators.items() if op.proposes] == ["refund.propose"]
    assert [name for name, op in operators.items() if op.last] == ["audit.append"]
    assert all("approved" not in op.add for op in operators.values())


def test_validator_upper_bound_and_execution_exact_amount(demo: dict[str, Any]) -> None:
    assert "boundary_problem" in demo, "the approved underpayment comparison is missing"
    assert demo["boundary_request"].amount_cents == 12000
    assert demo["boundary_approval"].amount_cents == 12000
    assert demo["boundary_start"] == demo["facts"]("approved")
    assert demo["boundary_plan"][-2].cents == 11000
    assert demo["boundary_problem"] is None
    result, tools = demo["boundary_result"], demo["boundary_tools"]
    assert not result.ok and result.reason == "amount_mismatch"
    assert not tools.payments
    refusals = [a for a in tools.audits.values() if a.decision.reason == "amount_mismatch"]
    assert len(refusals) == 1 and not refusals[0].money_moved
    assert refusals[0].paid_amount_cents == 0


@pytest.fixture(scope="module")
def planning():
    from demos import reasoning_planning_search

    return reasoning_planning_search


def oracle_distance(table, start, goal, relaxed=False):
    """Independent deque BFS, including the refund table's terminal audit rule."""
    frontier, seen = deque([(start, 0)]), {start}
    while frontier:
        state, distance = frontier.popleft()
        if goal <= state:
            return distance
        for operator in table.values():
            if not operator.pre <= state:
                continue
            after = (state if relaxed else state - operator.delete) | operator.add
            if operator.last and not goal <= after:
                continue
            if after not in seen:
                seen.add(after)
                frontier.append((after, distance + 1))
    return math.inf


@pytest.mark.parametrize("case", ["refund", "network"])
@pytest.mark.parametrize("algorithm", ["BFS", "A*"])
def test_search_finds_a_shortest_valid_plan(planning, case, algorithm):
    table, start, goal = planning.search_problems()[case]
    names, expanded = planning.solve(table, start, goal, algorithm)
    assert len(names) == oracle_distance(table, start, goal) == 6
    assert expanded > 0
    amount = 2499 if case == "refund" else 0
    steps = [planning.Step(name, amount if table[name].takes_cents else 0) for name in names]
    assert planning.validate_plan(
        steps, start, goal, planning.Limits(12, 80000, 137006), operators=table
    ) is None
    if case == "refund":
        assert names[-1] == "audit.append"


@pytest.mark.parametrize("case", ["refund", "network"])
def test_relaxed_heuristic_never_overestimates_any_reachable_state(planning, case):
    table, start, goal = planning.search_problems()[case]
    reachable, pending = {start}, [start]
    while pending:
        state = pending.pop()
        for _, after in planning.successors(state, table, goal):
            if after not in reachable:
                reachable.add(after)
                pending.append(after)
    for state in reachable:
        estimate = planning.relaxed_length(state, goal, table)
        assert estimate == oracle_distance(table, state, goal, relaxed=True)
        assert estimate <= oracle_distance(table, state, goal)
    assert planning.check_admissibility(table, start, goal) == len(reachable)


@pytest.mark.parametrize("on_backup,real_length", [(False, 6), (True, 5)])
def test_network_relaxed_and_real_lengths(planning, on_backup, real_length):
    table, start, goal = planning.search_problems()["network"]
    if on_backup:
        start = planning.facts("link_up", "traffic_on_backup", "backup_up", "leased_up")
    assert planning.relaxed_length(start, goal, table) == 4
    assert oracle_distance(table, start, goal) == real_length
    relaxed, _ = planning.best_first_search(
        start, goal, lambda state: planning.successors(state, table, goal, relaxed=True)
    )
    if not on_backup:
        assert relaxed == ("reroute", "check_idle", "take_down", "replace_optic")


@pytest.mark.skipif(
    not (DEMO.parents[2] / "_quarto.yml").is_file(),
    reason="requires the book repository",
)
def test_network_operators_match_the_manuscript(planning):
    import re

    chapter = (DEMO.parents[2] / "mainmatter/reasoning-planning.qmd").read_text()
    table_text = chapter.split("| `reroute` |", 1)[1].split("\n\n", 1)[0]
    rows = ("| `reroute` |" + table_text).splitlines()
    for row, (name, operator) in zip(rows, planning.NETWORK_OPERATORS.items(), strict=True):
        fields = row.strip("|").split("|")
        assert fields[0].strip() == f"`{name}`"
        for field, expected in zip(fields[1:], operator[:3], strict=True):
            assert frozenset(re.findall(r"`([^`]+)`", field)) == expected


def test_search_fifo_duplicates_goals_and_unreachable(planning):
    edges = {"s": [("a", "a"), ("b", "b")], "a": [("finish", "g")],
             "b": [("finish", "g")], "g": []}
    result, expanded = planning.best_first_search(
        frozenset("s"), frozenset("g"),
        lambda state: [(name, frozenset(after)) for name, after in edges[next(iter(state))]],
    )
    assert result == ("a", "finish") and expanded == 3
    assert planning.best_first_search(frozenset("g"), frozenset("g"), lambda _: []) == ((), 0)
    assert planning.best_first_search(frozenset(), frozenset("g"), lambda _: []) == (None, 1)


def test_astar_rediscovery_keeps_the_shorter_path(planning):
    edges = {"s": ("a", "c"), "a": ("b",), "b": ("x",),
             "c": ("x",), "x": ("g",), "g": ()}
    estimates = {"s": 1, "a": 0, "b": 0, "c": 2, "x": 1, "g": 0}
    # This consistent heuristic queues x via a,b before rediscovering it via c.
    result, _ = planning.best_first_search(
        frozenset("s"), frozenset("g"),
        lambda state: [(after, frozenset(after)) for after in edges[next(iter(state))]],
        lambda state: estimates[next(iter(state))],
    )
    assert result == ("c", "x", "g")


def test_pool_has_one_valid_plan_with_one_fixed_fault_per_other_plan(planning, demo):
    base, start, goal, limits = planning.refund_problem()
    assert base == demo["plan7"] and start == frozenset()
    assert limits == demo["limits7"]
    pool = planning.proposal_pool(base)
    found = [planning.validate_plan(steps, start, goal, limits) for _, steps in pool]
    assert len(pool) == 8
    assert found == [None, "step 4: policy.check is not a tool", "step 4: lacks proposed",
                     "step 3: lacks shipment_read", "step 6: a second payment",
                     "step 5: pays 2499, most 2498", "goal unmet: audited", "13 steps, budget 12"]
    assert all(step.cents == 2499 for _, steps in pool for step in steps
               if step.tool == "refund.execute")


def test_budget_formulas_match_exhaustive_three_proposal_streams(planning):
    base, start, goal, limits = planning.refund_problem()
    valid = [planning.validate_plan(steps, start, goal, limits) is None
             for _, steps in planning.proposal_pool(base)]
    used, referred = 0, 0
    for stream in product(range(len(valid)), repeat=3):
        accepted_at = next((i for i, choice in enumerate(stream, 1) if valid[choice]), None)
        used += accepted_at or 3
        referred += accepted_at is None
    expected_used, expected_referred = planning.budget_expectations(sum(valid) / len(valid), 3)
    assert used / len(valid) ** 3 == expected_used
    assert referred / len(valid) ** 3 == expected_referred


def test_seeded_budget_stops_on_validation_and_matches_the_formulas(planning):
    pool, rows = planning.budget_experiment()
    assert planning.budget_experiment()[1] == rows
    _, start, goal, limits = planning.refund_problem()
    rng = Random(7)
    counts = {budget: [0, 0] for budget in (3, 6, 12, 24)}
    for _ in range(10000):
        accepted = False
        for used in range(1, 25):
            _, steps = rng.choice(pool)
            if planning.validate_plan(steps, start, goal, limits) is None:
                accepted = True
                break
        for budget in counts:
            counts[budget][0] += min(used, budget)
            counts[budget][1] += not (accepted and used <= budget)
    for row in rows:
        assert [row.used, row.referred] == counts[row.budget]
        assert row.requests == 10000
        assert row.expected_used == pytest.approx(8 * (1 - (7 / 8) ** row.budget))
        assert row.expected_referred == pytest.approx((7 / 8) ** row.budget)
        se = math.sqrt(row.expected_referred * (1 - row.expected_referred) / row.requests)
        assert abs(row.referred / row.requests - row.expected_referred) < 5 * se


@pytest.mark.parametrize("p,budget", [(0, 3), (1, 3), (0.125, 0)])
def test_budget_boundary_cases(planning, p, budget):
    expected = (float(budget), 1.0) if p == 0 else ((0.0, 1.0) if budget == 0 else (1.0, 0.0))
    assert planning.budget_expectations(p, budget) == expected


@pytest.mark.parametrize("name", ["search-02", "budget-01", "keys-01"])
def test_new_output_regions_run_independently_and_match_stored_outputs(planning, name):
    from demos.regions import extract_snippets_from_text

    snippet_id = f"fnd.reasoning-planning.{name}"
    snippets = extract_snippets_from_text(DEMO.read_text(), DEMO)
    snippet = next(region for region in snippets if region.snippet_id == snippet_id)
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        exec(compile(snippet.content, snippet_id, "exec"), {})
    stored = DEMO.parents[1] / "book_outputs" / f"_{snippet_id}.txt"
    assert stdout.getvalue() == stored.read_text()


def test_search_listing_runs_with_only_its_printed_imports(planning):
    from demos.regions import extract_snippets_from_text

    path = Path(planning.__file__)
    snippet = next(region for region in extract_snippets_from_text(path.read_text(), path)
                   if region.snippet_id == "fnd.reasoning-planning.search-01")
    assert len(snippet.content.splitlines()) <= 26
    assert max(map(len, snippet.content.splitlines())) <= 79
    namespace = {}
    exec(compile(snippet.content, snippet.snippet_id, "exec"), namespace)
    assert namespace["best_first_search"](frozenset(), frozenset(), lambda _: []) == ((), 0)


def test_network_figure_trace_reports_every_decision_and_cost(planning):
    """Catch lost duplicates, wrong pop order, or mislabelled figure costs."""
    import re

    from demos.regions import extract_snippets_from_text

    regions = {
        region.snippet_id: region
        for region in extract_snippets_from_text(DEMO.read_text(), DEMO)
    }
    name = "fnd.reasoning-planning.search-03"
    assert name in regions, "the independent figure-data output is missing"
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        exec(compile(regions[name].content, name, "exec"), {})
    output = stdout.getvalue()
    assert max(map(len, output.splitlines())) <= 79
    stored = DEMO.parents[1] / "book_outputs" / f"_{name}.txt"

    comparison = io.StringIO()
    with contextlib.redirect_stdout(comparison):
        exec(compile(regions["fnd.reasoning-planning.search-02"].content,
                     "search-02", "exec"), {})
    network_comparison = comparison.getvalue().split("network: start", 1)[1]
    table, start, goal = planning.search_problems()["network"]
    operator_order = (
        "reroute", "reroute_leased", "check_idle", "take_down",
        "replace_optic", "bring_up", "restore",
    )
    expected_plan = (
        "reroute", "check_idle", "take_down", "replace_optic",
        "bring_up", "restore",
    )
    sections = output.split("\nBFS\n", 1)[1].split("\nA* (h+)\n")
    assert len(sections) == 2
    for algorithm, section, expected_expanded in zip(
        ("BFS", "A*"), sections, (7, 6), strict=True,
    ):
        discovered = {0: (start, ())}
        popped, priorities = [], []
        blocks = section.strip().split("\n\n")
        assert len(blocks) == expected_expanded + 1
        for index, block in enumerate(blocks):
            lines = block.splitlines()
            header = re.fullmatch(
                r"(Expansion (\d+)|Goal popped): N(\d+); "
                r"g=(\d+); h\+=(\d+); f=(\d+)", lines[0],
            )
            assert header, lines[0]
            node, g, h, f = map(int, header.group(3, 4, 5, 6))
            facts_at = next(i for i, line in enumerate(lines)
                            if line.startswith("  facts: "))
            prefix = " ".join(line.strip() for line in lines[1:facts_at])
            assert prefix.startswith("prefix: ")
            calls = prefix.removeprefix("prefix: ")
            path = () if calls == "start" else tuple(calls.split(" -> "))
            end = facts_at + 1
            while end < len(lines) and lines[end].startswith("    "):
                end += 1
            printed_facts = " ".join(
                [lines[facts_at].removeprefix("  facts: ")]
                + [line.strip() for line in lines[facts_at + 1:end]]
            )
            state = frozenset(printed_facts.split(", "))
            assert (state, path) == discovered[node]
            assert node not in popped
            popped.append(node)
            assert g == len(path) and f == g + h
            assert h == planning.relaxed_length(state, goal, table)
            assert h == oracle_distance(table, state, goal, relaxed=True)
            assert h <= oracle_distance(table, state, goal)
            priorities.append((g if algorithm == "BFS" else f, node))

            if header[1] == "Goal popped":
                assert index == expected_expanded and goal <= state
                assert path == expected_plan and g == 6
                assert lines[end:] == [
                    f"Totals: expanded={expected_expanded}; calls={g}",
                ]
                summary = re.search(
                    rf"  {re.escape(algorithm)}: expanded=(\d+); calls=(\d+)",
                    network_comparison,
                )
                assert summary
                assert tuple(map(int, summary.groups())) == (index, g)
                assert planning.solve(table, start, goal, algorithm) == (
                    path, index,
                )
                continue

            assert int(header[2]) == index + 1 and not goal <= state
            verdicts = [line.strip().split(": ", 1) for line in lines[end:]]
            assert tuple(name for name, _ in verdicts) == operator_order
            for name, verdict in verdicts:
                operator = table[name]
                missing = sorted(operator.pre - state)
                if missing:
                    assert verdict == f"refused: needs {missing[0]}"
                    steps = [planning.Step(call) for call in (*path, name)]
                    problem = planning.validate_plan(
                        steps, start, goal, planning.Limits(12, 0, 0),
                        operators=table,
                    )
                    first_missing = problem.split("lacks ", 1)[1].split(", ")[0]
                    assert first_missing == missing[0]
                    continue
                child = (state - operator.delete) | operator.add
                earlier = next((n for n, (s, _) in discovered.items()
                                if s == child), None)
                if earlier is not None:
                    assert verdict == (
                        f"duplicate state, not added; held by N{earlier}"
                    )
                else:
                    added = re.fullmatch(r"child added N(\d+); h\+=(\d+)", verdict)
                    assert added, verdict
                    child_node, child_h = map(int, added.groups())
                    assert child_node == len(discovered)
                    assert child_h == planning.relaxed_length(child, goal, table)
                    assert child_h == oracle_distance(
                        table, child, goal, relaxed=True,
                    )
                    assert child_h <= oracle_distance(table, child, goal)
                    discovered[child_node] = (child, (*path, name))
            if index == 0:
                assert dict(verdicts)["take_down"] == "refused: needs link_idle"
                assert dict(verdicts)["reroute_leased"] == (
                    "duplicate state, not added; held by N1"
                )
        assert priorities == sorted(priorities)
    assert output == stored.read_text()
