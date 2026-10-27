"""Chapter 24's unit-cost search and checked-proposal experiments.

Load the canonical definition prefix exactly as the release process grader does.
Importing this helper never executes the demo's output regions. Search fixes a
tier-1 amount; its state is just the operator model's facts, not execution data.
"""

from math import inf
from pathlib import Path
from random import Random
from textwrap import fill
from typing import NamedTuple

_SOURCE = Path(__file__).with_name("reasoning_planning_demo.py")
_PREFIX, _MARKER, _ = _SOURCE.read_text().partition(
    "# region book:fnd.reasoning-planning.operators-01 output"
)
if not _MARKER:
    raise ValueError("canonical planning definition boundary is missing")
_DEFINITIONS = {"__name__": "chapter24_planning", "__file__": str(_SOURCE)}
exec(compile(_PREFIX, str(_SOURCE), "exec"), _DEFINITIONS)
Operator, Step, Limits, OPERATORS, facts, validate_plan, Agent, Tools, generate, plan_of, limits_of = (
    _DEFINITIONS[name]
    for name in (
        "Operator", "Step", "Limits", "OPERATORS", "facts", "validate_plan",
        "Agent", "Tools", "generate", "plan_of", "limits_of",
    )
)


# region book:fnd.reasoning-planning.search-01
from heapq import heappop, heappush
from itertools import count


def best_first_search(start, goal, successors, heuristic=lambda state: 0):
    serial = count()  # FIFO for equal priorities; each call costs one.
    frontier = [(heuristic(start), next(serial), start, ())]
    best, expanded = {start: 0}, 0
    while frontier:
        _, _, state, plan = heappop(frontier)
        if len(plan) != best[state]:
            continue
        if goal <= state:
            return plan, expanded  # The goal is not expanded.
        expanded += 1
        for action, after in successors(state):
            path = (*plan, action)
            if len(path) < best.get(after, float("inf")):
                best[after] = len(path)
                heappush(frontier, (
                    len(path) + heuristic(after), next(serial), after, path,
                ))
    return None, expanded
# endregion book:fnd.reasoning-planning.search-01


NETWORK_OPERATORS = {
    "reroute": Operator(facts("backup_up", "link_up"), facts("traffic_on_backup"),
                        facts("traffic_on_link")),
    "reroute_leased": Operator(facts("leased_up", "link_up"), facts("traffic_on_backup"),
                               facts("traffic_on_link")),
    "check_idle": Operator(facts("traffic_on_backup"), facts("link_idle")),
    "take_down": Operator(facts("link_idle"), facts("link_down"), facts("link_up")),
    "replace_optic": Operator(facts("link_down"), facts("optic_new")),
    "bring_up": Operator(facts("link_down", "optic_new"), facts("link_up"),
                         facts("link_down", "link_idle")),
    "restore": Operator(facts("link_up", "traffic_on_backup"), facts("traffic_on_link"),
                        facts("traffic_on_backup", "link_idle")),
}


def search_problems():
    return {
        "refund": (OPERATORS, facts(), facts("paid", "audited")),
        "network": (
            NETWORK_OPERATORS,
            facts("link_up", "traffic_on_link", "backup_up", "leased_up"),
            facts("optic_new", "link_up", "traffic_on_link"),
        ),
    }


def successors(state, table, goal, relaxed=False):
    """Progress fact states; an audit is allowed only as the closing call."""
    for name, operator in table.items():
        if operator.pre <= state:
            after = (state if relaxed else state - operator.delete) | operator.add
            if not operator.last or goal <= after:
                yield name, after


def relaxed_length(state, goal, table):
    """Exact h+: shortest unit-cost plan after ignoring all delete lists."""
    plan, _ = best_first_search(
        state, goal, lambda now: successors(now, table, goal, relaxed=True)
    )
    return inf if plan is None else len(plan)


def solve(table, start, goal, algorithm="BFS"):
    if algorithm not in {"BFS", "A*"}:
        raise ValueError("algorithm must be BFS or A*")
    return best_first_search(
        start, goal, lambda state: successors(state, table, goal),
        (lambda state: relaxed_length(state, goal, table)) if algorithm == "A*" else lambda state: 0,
    )


def check_admissibility(table, start, goal):
    """Check every reachable fact state, including unreachable-goal states."""
    seen, pending = {start}, [start]
    while pending:
        state = pending.pop()
        real, _ = solve(table, state, goal)
        exact = inf if real is None else len(real)
        if relaxed_length(state, goal, table) > exact:
            raise AssertionError(f"relaxed estimate overestimates at {sorted(state)}")
        for _, after in successors(state, table, goal):
            if after not in seen:
                seen.add(after)
                pending.append(after)
    return len(seen)


def print_search_comparison():
    print("Unit-cost fact states; FIFO ties; popped goals are not expanded.")
    print("Expansion counts exclude the searches that compute h+.")
    for case, (table, start, goal) in search_problems().items():
        print(f"{case}: start h+={relaxed_length(start, goal, table)}")
        for algorithm in ("BFS", "A*"):
            plan, expanded = solve(table, start, goal, algorithm)
            if plan is None:
                raise AssertionError(f"{case} has no plan")
            print(f"  {algorithm}: expanded={expanded}; calls={len(plan)}")
            print(fill(" -> ".join(plan), width=76, initial_indent="    ", subsequent_indent="    "))
        checked = check_admissibility(table, start, goal)
        print(f"  h+ never overestimates: {checked}/{checked} reachable states")


def refund_problem():
    """The original request, plan and bounds; no altered order or request."""
    data = generate(7)
    request = next(row for row in data.requests if row.request_id == "r-7-0007")
    trace = Agent(Tools(data)).run(request)
    return plan_of(trace), facts(), facts("paid", "audited"), limits_of(trace)


def proposal_pool(base):
    """One valid plan and seven single-fault variants of this same request."""
    proposal = base[3]._replace(cents=base[3].cents - 1)
    return (
        ("valid", base.copy()),
        ("unknown tool", base[:3] + [Step("policy.check")] + base[3:]),
        ("payment before proposal", base[:3] + [base[4], base[3], base[5]]),
        ("missing shipment read", [base[0]] + base[2:]),
        ("second payment", base[:5] + [base[4], base[5]]),
        ("proposal below payment", base[:3] + [proposal] + base[4:]),
        ("missing audit", base[:5]),
        ("thirteen calls", [base[0]] * 7 + base),
    )


def budget_expectations(p_valid, budget):
    if not 0 <= p_valid <= 1 or type(budget) is not int or budget < 0:
        raise ValueError("probability must be in [0, 1] and budget a nonnegative integer")
    if p_valid == 0:
        return float(budget), 1.0
    referred = (1 - p_valid) ** budget
    return (1 - referred) / p_valid, referred


class BudgetRow(NamedTuple):
    budget: int
    expected_used: float
    expected_referred: float
    used: int
    referred: int
    requests: int


def budget_experiment(requests=10000, seed=7, budgets=(3, 6, 12, 24)):
    base, start, goal, limits = refund_problem()
    pool = proposal_pool(base)
    valid = sum(validate_plan(plan, start, goal, limits) is None for _, plan in pool)
    if valid != 1:
        raise AssertionError("the fixed eight-plan pool must have exactly one valid plan")
    totals = {budget: [0, 0] for budget in budgets}
    rng = Random(seed)
    for _ in range(requests):
        accepted = False
        for used in range(1, max(budgets) + 1):
            _, plan = rng.choice(pool)
            if validate_plan(plan, start, goal, limits) is None:
                accepted = True
                break
        for budget, counts in totals.items():
            counts[0] += min(used, budget)
            counts[1] += not (accepted and used <= budget)
    rows = [BudgetRow(budget, *budget_expectations(valid / len(pool), budget),
                      *totals[budget], requests) for budget in budgets]
    return pool, rows


def print_budget_comparison():
    pool, rows = budget_experiment()
    _, start, goal, limits = refund_problem()
    valid = sum(validate_plan(plan, start, goal, limits) is None for _, plan in pool)
    print(f"pool={len(pool)}; valid={valid}; p_v={valid / len(pool):.3f}")
    print(f"seed=7; requests={rows[0].requests}; uniform draws with replacement")
    print("Same request, start, goal and limits; each other plan has one fault:")
    for name, plan in pool:
        problem = validate_plan(plan, start, goal, limits)
        print(f"  {name}: {'valid' if problem is None else problem}")
    print("E[used]=(1-(1-p_v)^k_max)/p_v; P[refer]=(1-p_v)^k_max")
    print("budget  E[used] mean_used P[refer] share_ref used_total referred")
    for row in rows:
        print(f"{row.budget:6} {row.expected_used:8.6f} {row.used / row.requests:9.6f} "
              f"{row.expected_referred:8.6f} {row.referred / row.requests:9.6f} "
              f"{row.used:10} {row.referred:8}")


def print_search_key():
    table, _, goal = search_problems()["network"]
    start = facts("link_up", "traffic_on_backup", "backup_up", "leased_up")
    real, _ = solve(table, start, goal)
    if real is None:
        raise AssertionError("exercise start has no plan")
    print(f"already on backup: h+={relaxed_length(start, goal, table)}; real={len(real)} calls")
    p_valid, budget = 1 / 8, 12
    _, referred = budget_expectations(p_valid, budget)
    print(f"p_v={p_valid:.3f}; budget={budget}; referred={referred:.6f}")


def print_network_search_tree():
    """Figure data for the network table; N labels follow discovery order."""
    table, start, goal = search_problems()["network"]
    print("Network; unit costs; FIFO ties; popped goals are not expanded.")
    print("Duplicate fact states are not re-added; h+ work is excluded from totals.")
    for algorithm, expected_expanded in (("BFS", 7), ("A*", 6)):
        print(f"\n{algorithm}" + (" (h+)" if algorithm == "A*" else ""))
        start_h = relaxed_length(start, goal, table)
        frontier = [(start_h if algorithm == "A*" else 0, 0, start, ())]
        discovered, expanded = {start: 0}, 0
        while frontier:
            _, node, state, plan = heappop(frontier)
            g, h = len(plan), relaxed_length(state, goal, table)
            reached = goal <= state
            if not reached:
                expanded += 1
            label = "Goal popped" if reached else f"Expansion {expanded}"
            print(f"\n{label}: N{node}; g={g}; h+={h}; f={g + h}")
            print(fill(
                " -> ".join(plan) or "start", width=79,
                initial_indent="  prefix: ", subsequent_indent="    ",
                break_long_words=False, break_on_hyphens=False,
            ))
            print(fill(
                ", ".join(sorted(state)), width=79,
                initial_indent="  facts: ", subsequent_indent="    ",
            ))
            if reached:
                assert (plan, expanded) == solve(table, start, goal, algorithm)
                assert expanded == expected_expanded and g == 6
                print(f"Totals: expanded={expanded}; calls={g}")
                break
            for name, operator in table.items():
                missing = sorted(operator.pre - state)  # validate_plan order
                if missing:
                    verdict = f"refused: needs {missing[0]}"
                else:
                    child = (state - operator.delete) | operator.add
                    if child in discovered:
                        verdict = (
                            "duplicate state, not added; "
                            f"held by N{discovered[child]}"
                        )
                    else:
                        child_node = len(discovered)
                        child_plan = (*plan, name)
                        child_h = relaxed_length(child, goal, table)
                        discovered[child] = child_node
                        priority = len(child_plan)
                        if algorithm == "A*":
                            priority += child_h
                        heappush(frontier, (
                            priority, child_node, child, child_plan,
                        ))
                        verdict = f"child added N{child_node}; h+={child_h}"
                print(f"  {name}: {verdict}")
        else:
            raise AssertionError("the network goal was not reached")
