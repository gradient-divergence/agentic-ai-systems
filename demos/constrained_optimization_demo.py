"""Constrained decisions and independent bounds, with an offline review desk.

Seed 7, constructed review-inputs-v1, no held-out requests or outcome labels read.
Run --write-data to generate plot inputs, the planted plan and solve records.
"""

# region book:fnd.constrained-optimization.certificate-01 output
import numpy as np
from scipy.optimize import linprog

from demos.output_wrap import print_wrapped
from refund_agent.schedule.certificate import check_certificate


def laboratory(hours=4):
    rows, rhs, objective = [[2, 1], [1, 0], [0, 1]], [hours, 3, 3], [3, 2]
    solved = linprog(-np.array(objective), A_ub=rows, b_ub=rhs, method="highs")
    if not solved.success:
        raise RuntimeError(f"linear solve status {solved.status}")
    multipliers = np.where(
        solved.ineqlin.marginals == 0, 0, -solved.ineqlin.marginals
    )
    check = check_certificate(rows, rhs, objective, solved.x, multipliers)
    # Hold the active upper-bound row fixed while the hours row varies.
    fixed = np.flatnonzero(multipliers[1:] > 0)
    hours_range = None
    if len(fixed) == 1 and multipliers[0] > 0:
        upper = np.array(rhs[1:])
        coefficient = np.array(rows[0])
        index = fixed[0]
        hours_range = [
            float(coefficient[index] * upper[index]),
            float(coefficient @ upper),
        ]
    return dict(
        hours_range=hours_range,
        plan=solved.x.tolist(),
        value=check["value"],
        multipliers=multipliers.tolist(),
        check=check,
    )


if __name__ == "__main__":
    lab = laboratory()
    print_wrapped(
        "laboratory: 4 analyzer hours; X 2 hours, 3 results; Y 1 hour, 2 "
        "results; 3 of each waiting"
    )
    print_wrapped(
        f"plan {lab['plan']}; value {lab['value']:g}; multipliers "
        f"{lab['multipliers']}"
    )
    checked = lab["check"]
    print_wrapped(
        f"permitted {checked['permitted']}; largest violation "
        f"{checked['largest_violation']:g}"
    )
    print_wrapped(
        f"nonnegative {checked['nonnegative']}; weighted rows "
        f"{checked['weighted_rows']}; covering {checked['covering']}"
    )
    print_wrapped(
        f"bound {checked['bound']:g}; absolute gap "
        f"{checked['absolute_gap']:g}; optimal {checked['optimal']}; "
        f"tolerance {checked['tolerance']:g}"
    )
    corners = [[0, 0], [4 / 2, 0], lab["plan"], [0, 3]]
    print_wrapped(
        f"corners {corners}; values "
        f"{[float(np.dot([3, 2], p)) for p in corners]}"
    )
    print_wrapped(
        f"hours multiplier {lab['multipliers'][0]:g} valid with the same "
        f"binding rows for hours {lab['hours_range'][0]:g} "
        f"through {lab['hours_range'][1]:g}"
    )
    actual = check_certificate(
        [[2, 1], [1, 0], [0, 1]],
        [3.5, 3, 3],
        [3, 2],
        lab["plan"],
        lab["multipliers"],
    )
    print_wrapped(
        f"if actual hours 3.5: plan permitted {actual['permitted']}; largest "
        f"violation {actual['largest_violation']:g}; written 4-hour model "
        f"still passes {checked['optimal']}"
    )
# endregion book:fnd.constrained-optimization.certificate-01


# region book:fnd.constrained-optimization.relax-01 output
from math import floor

from scipy.optimize import Bounds, LinearConstraint, milp


def whole_laboratory(hours=4):
    solved = milp(
        -np.array([3, 2]),
        integrality=[1, 1],
        bounds=Bounds(0, np.inf),
        constraints=LinearConstraint(
            [[2, 1], [1, 0], [0, 1]], -np.inf, [hours, 3, 3]
        ),
        options={"mip_rel_gap": 0.0},
    )
    if not solved.success:
        raise RuntimeError(f"whole solve status {solved.status}")
    lab = laboratory(hours)
    plan = np.rint(solved.x).astype(int).tolist()
    check = check_certificate(
        [[2, 1], [1, 0], [0, 1]],
        [hours, 3, 3],
        [3, 2],
        plan,
        lab["multipliers"],
    )
    return dict(
        status="optimal",
        plan=plan,
        value=check["value"],
        solver_bound=float(-solved.mip_dual_bound),
        relaxation_bound=check["bound"],
        rounded_bound=floor(check["bound"] + check["tolerance"]),
        check=check,
    )


if __name__ == "__main__":
    whole = whole_laboratory()
    print(
        f"status {whole['status']}; plan {whole['plan']}; value "
        f"{whole['value']:g}; solver bound {whole['solver_bound']:g}"
    )
    rounded = whole["rounded_bound"]
    print(
        f"independent relaxation bound {whole['relaxation_bound']:g}; rounded "
        f"bound {rounded}; gap {rounded - whole['value']:g}; proven "
        f"{whole['check']['permitted'] and whole['value'] == rounded}"
    )
    down, up = np.floor(lab["plan"]), np.ceil(lab["plan"])
    print(
        f"round plan down {down.astype(int).tolist()}: value "
        f"{np.dot([3, 2], down):g}"
    )
    print(
        f"round plan up {up.astype(int).tolist()}: hours "
        f"{np.dot([2, 1], up):g}, permitted {np.dot([2, 1], up) <= 4}"
    )
# endregion book:fnd.constrained-optimization.relax-01


def branches():
    rows, rhs, objective = [[2, 1], [1, 0], [0, 1]], [4, 3, 3], [3, 2]
    result = []
    for name, bounds in (
        ("root", [(0, None), (0, None)]),
        ("X <= 0", [(0, 0), (0, None)]),
        ("X >= 1", [(1, None), (0, None)]),
    ):
        solved = linprog(-np.array(objective), A_ub=rows, b_ub=rhs, bounds=bounds, method="highs")
        if not solved.success:
            raise RuntimeError("branch solve failed")
        result.append(dict(branch=name, plan=solved.x.tolist(), value=float(-solved.fun)))
    return result


# region book:fnd.constrained-optimization.branches-01 output
if __name__ == "__main__":
    print("branch: plan; relaxation value")
    for row in branches():
        print(f"{row['branch']}: {row['plan']}; {row['value']:g}")
    best = max(r["value"] for r in branches()[1:])
    value = whole_laboratory()["value"]
    print(f"best whole {value:g}; best branch bound {best:g}; gap {best - value:g}")
    incumbent, bound = (
        np.dot([3, 2], np.floor(laboratory()["plan"])),
        laboratory()["check"]["bound"],
    )
    print(
        f"illustration, stopped at root: incumbent {incumbent:g}; bound {bound:g}; relative gap {(bound - incumbent) / incumbent:.6f}"
    )
    print(
        f"after rounding bound {floor(bound)}: absolute gap {floor(bound) - incumbent:g}; relative gap {(floor(bound) - incumbent) / incumbent:.6f}"
    )
# endregion book:fnd.constrained-optimization.branches-01


def incident_orders(strict=False):
    from itertools import permutations

    lengths, due = {"A": 2, "B": 1, "C": 2}, {"A": 2 if strict else 3, "B": 2, "C": 5}
    result = []
    for order in permutations(lengths):
        time, starts = 0, {}
        for task in order:
            starts[task], time = time, time + lengths[task]
        feasible = (
            all(starts[j] + lengths[j] <= due[j] for j in lengths)
            and starts["A"] + lengths["A"] <= starts["C"]
        )
        result.append(dict(order=list(order), starts=starts, feasible=feasible))
    return result


def counting_proof():
    lengths, due = {"A": 2, "B": 1, "C": 2}, {"A": 2, "B": 2, "C": 5}
    tasks = [j for j in lengths if due[j] <= 2]
    return dict(
        due_slot=2, tasks=tasks, work_slots=sum(lengths[j] for j in tasks), available_slots=2
    )


# region book:fnd.constrained-optimization.cp-01 output
from demos import constrained_optimization_demo as incident_checks
from demos.output_wrap import print_wrapped


def incident_schedule(strict=False):
    from ortools.sat.python import cp_model

    lengths, due = (
        {"A": 2, "B": 1, "C": 2},
        {"A": 2 if strict else 3, "B": 2, "C": 5},
    )
    model = cp_model.CpModel()
    starts = {
        j: model.new_int_var(0, due[j] - lengths[j], f"start_{j}")
        for j in lengths
    }
    intervals = [
        model.new_fixed_size_interval_var(starts[j], lengths[j], j)
        for j in lengths
    ]
    model.add_no_overlap(intervals)
    model.add(starts["A"] + lengths["A"] <= starts["C"])
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers, solver.parameters.random_seed = 1, 7
    solver.parameters.max_deterministic_time = 1
    status = solver.solve(model)
    result = dict(status=solver.status_name(status).lower(), starts={})
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        result["starts"] = {j: solver.value(starts[j]) for j in lengths}
    return result


if __name__ == "__main__":
    print_wrapped(
        "incident tasks A, B, C: lengths 2, 1, 2; due slots 3, 2, 5; A before "
        "C; 30-minute slots from 09:00"
    )
    for order in incident_checks.incident_orders():
        print_wrapped(
            f"order {', '.join(order['order'])}: feasible {order['feasible']}"
        )
    try:
        incident = incident_schedule()
        print_wrapped(
            f"status {incident['status']}; starts {incident['starts']}"
        )
        from datetime import datetime, timedelta, timezone

        for task, length in (("B", 1), ("A", 2), ("C", 2)):
            start = incident["starts"][task]
            clock = datetime(2026, 3, 23, 9, tzinfo=timezone.utc) + timedelta(
                minutes=30 * start
            )
            print_wrapped(
                f"{task}: slots {start}-{start + length}; "
                f"{clock:%H:%M}-{clock + timedelta(minutes=30 * length):%H:%M}"
            )
        print_wrapped(
            f"stricter A due slot 2: status "
            f"{incident_schedule(strict=True)['status']}"
        )
    except ImportError:
        print_wrapped(
            "constraint solver unavailable; enumeration and counting proof "
            "remain available"
        )
    proof = incident_checks.counting_proof()
    print_wrapped(
        f"counting proof: {', '.join(proof['tasks'])} need "
        f"{proof['work_slots']} slots by slot {proof['due_slot']}; available "
        f"{proof['available_slots']}"
    )
    print_wrapped(
        f"work due {30 * proof['work_slots']} minutes; available "
        f"{30 * proof['available_slots']} minutes"
    )
# endregion book:fnd.constrained-optimization.cp-01


def engineer_tree(prune=False):
    lengths, due = {"A": 2, "B": 1, "C": 2}, {"A": 3, "B": 2, "C": 5}
    nodes, plans, expanded, complete = [], [], 0, 0

    def visit(order, clock):
        nonlocal expanded, complete
        done = set(order)
        remaining = [j for j in lengths if j not in done]
        cut = prune and any(
            sum(lengths[j] for j in remaining if due[j] <= deadline)
            > deadline - clock
            for deadline in {due[j] for j in remaining}
        )
        nodes.append(dict(order=list(order), done=sorted(done), clock=clock, pruned=cut))
        if cut:
            return
        if not remaining:
            complete += 1
            # Full-tree leaves retain their path to audit every earlier finish.
            time, feasible = 0, True
            for j in order:
                time += lengths[j]
                feasible = feasible and time <= due[j]
            if feasible:
                plans.append(list(order))
            return
        expanded += 1
        for j in remaining:
            if j != "C" or "A" in done:
                visit(order + [j], clock + lengths[j])

    visit([], 0)
    return dict(
        generated=len(nodes), examined=len(nodes), expanded=expanded,
        pruned=sum(n["pruned"] for n in nodes), complete=complete,
        plans=plans, nodes=nodes,
    )


def oncall_tasks():
    # Constructed once from an outage checklist, not selected by expansion counts.
    rows = (
        ("credential", 1, 2, 8, ()),
        ("database", 2, 4, 6, ()),
        ("batch", 2, 7, 3, ("database",)),
        ("probe", 1, 3, 2, ("database",)),
        ("backup", 2, 8, 2, ("database",)),
        ("patch", 2, 6, 4, ("credential",)),
        ("report", 1, 9, 1, ("probe", "batch")),
    )
    return {
        name: dict(length=length, due=due, weight=weight, after=after)
        for name, length, due, weight, after in rows
    }


def relaxed_cost(tasks, state):
    # Drop only the single engineer: unlimited parallel work retains precedence.
    done, clock = state
    finishes = {j: clock for j in done}

    def earliest(j):
        if j not in finishes:
            ready = max([clock] + [earliest(p) for p in tasks[j]["after"]])
            finishes[j] = ready + tasks[j]["length"]
        return finishes[j]

    return sum(
        task["weight"] * max(0, earliest(j) - task["due"])
        for j, task in tasks.items() if j not in done
    )


def proposal_sampling(k, seed=15, draws=10000, false_pass=0.1):
    valid_chance = 1 / 3
    samples = np.random.default_rng(seed).random((draws, k, 2))
    valid = samples[:, :, 0] < valid_chance
    passed = valid | (samples[:, :, 1] < false_pass)
    accepted = passed.any(axis=1)
    first_valid = valid[np.arange(draws), passed.argmax(axis=1)]
    acceptance = valid_chance + (1 - valid_chance) * false_pass
    minimum = 1
    while 1 - (1 - valid_chance)**minimum < 0.99:
        minimum += 1
    return dict(
        coverage=1 - (1 - valid_chance)**k,
        minimum_k=minimum,
        invalid_share=(1 - valid_chance) * false_pass / acceptance,
        acceptance=1 - (1 - acceptance)**k,
        simulated_coverage=float(valid.any(axis=1).mean()),
        simulated_invalid_share=float((~first_valid[accepted]).mean()),
        accepted=int(accepted.sum()), exhausted=int((~accepted).sum()),
    )


def search_report(search):
    print(
        "engineer state: order of tasks done, and clock; actions keep A before C"
    )
    print("A, B, C: lengths 2, 1, 2; due slots 3, 2, 5")
    print("full tree defers due checks to leaves; counted checks work still due")
    print("generated includes root; examined tests each node, even a goal")
    print("expanded generates successors; pruned fails count; goals not expanded")
    print("complete means all tasks done; feasible plans pass every due time")
    for name, result in (("full", engineer_tree()), ("counted", engineer_tree(True))):
        print(
            f"{name}: generated {result['generated']}; examined {result['examined']}; "
            f"expanded {result['expanded']}; pruned {result['pruned']}; "
            f"complete {result['complete']}"
        )
        print(f"  feasible plans {result['plans']}")
    tasks = oncall_tasks()
    print("constructed IT morning; 10-minute slots; task: length, due, weight, after")
    for name, task in tasks.items():
        print(
            f"{name}: {task['length']}, {task['due']}, {task['weight']}, "
            f"{list(task['after'])}"
        )
    print("step cost: weight times late slots at completion; due times are soft")
    print("h: unlimited parallel engineers, with all precedence retained")
    print("expanded: non-goal, non-stale heap pops; equal f uses insertion order")
    children = []
    for name, task in tasks.items():
        if task["after"]:
            continue
        end = task["length"]
        g = task["weight"] * max(0, end - task["due"])
        h = relaxed_cost(tasks, (frozenset({name}), end))
        children.append(f"{name} g {g}, h {h}, f {g + h}")
    print("root's children: " + "; ".join(children))
    for name, heuristic in (("uniform-cost", None), ("A*", relaxed_cost)):
        result = search(tasks, heuristic)
        print(
            f"{name}: expanded {result['expanded']}; optimal cost {result['cost']}"
        )
        print(f"  plan {', '.join(result['plan'])}")
    print("proposals: pv=1/3, pf=0.1; valid proposals always pass the check")
    print("one draw: N independent proposals; record coverage and first passing plan")
    print("invalid share conditions on acceptance; exhausted draws are excluded")
    print("seed 15; 10000 draws per N; exact and simulated values")
    for k in (5, 12):
        result = proposal_sampling(k)
        print(
            f"N={k}: coverage {result['coverage']:.9f} / "
            f"{result['simulated_coverage']:.9f}"
        )
        print(
            f"  accepted invalid "
            f"{result['invalid_share']:.9f} / "
            f"{result['simulated_invalid_share']:.9f}"
        )
        print(f"  accepted {result['accepted']}; exhausted {result['exhausted']}")
    print(f"smallest N for coverage >= 0.99: {result['minimum_k']}")


# region book:fnd.constrained-optimization.search-01 output
from heapq import heappop, heappush
from itertools import count

from demos import constrained_optimization_demo as search_examples


def cost_search(tasks, heuristic=None):
    h = (lambda state: heuristic(tasks, state)) if heuristic else lambda s: 0
    start, serial = (frozenset(), 0), count()
    frontier = [(h(start), next(serial), 0, start, [])]
    best, expanded = {start: 0}, 0
    while frontier:
        _, _, cost, state, plan = heappop(frontier)
        if cost != best[state]:
            continue
        done, clock = state
        if len(done) == len(tasks):
            return dict(plan=plan, cost=cost, expanded=expanded)
        expanded += 1
        for j, task in tasks.items():
            if j in done or not set(task["after"]) <= done:
                continue
            end = clock + task["length"]
            child = (done | {j}, end)
            value = cost + task["weight"] * max(0, end - task["due"])
            if value < best.get(child, float("inf")):
                best[child] = value
                heappush(frontier, (
                    value + h(child), next(serial), value, child, plan + [j],
                ))


if __name__ == "__main__":
    search_examples.search_report(cost_search)
# endregion book:fnd.constrained-optimization.search-01


# region book:fnd.constrained-optimization.plan-01 output
from collections import Counter
from datetime import date

from demos.output_wrap import print_wrapped
from refund_agent.data.generate import generate
from refund_agent.schedule.check import check_plan
from refund_agent.schedule.inputs import BUDGET, MINUTES
from refund_agent.schedule.model import solve_morning
from refund_agent.schedule.queue import waiting
from refund_agent.schedule.rule import plain_rule
from refund_agent.schedule.window import run_window

if __name__ == "__main__":
    monday = next(
        d
        for d in run_window("count_then_wait")["days"]
        if d["day"] == "2026-03-23"
    )
    queue, record = monday["queue"], monday["record"]
    print_wrapped(
        f"Monday 2026-03-23 09:00 UTC: {len(queue)} waiting; tiers "
        f"{dict(sorted(Counter(d.tier for d in queue).items()))}"
    )
    print_wrapped(
        f"arrivals: Friday {sum(d.at.weekday() == 4 for d in queue)}, weekend "
        f"{sum(d.at.weekday() >= 5 for d in queue)}, Monday "
        f"{sum(d.at.weekday() == 0 for d in queue)}"
    )
    print_wrapped(
        "as given: tier minutes 5, 15, 30; alex 120 minutes, morgan 60; tier "
        "3 to named approver morgan"
    )
    for name, plan in (
        ("plain rule", plain_rule(queue, MINUTES)),
        ("count then wait", record.plan),
    ):
        print_wrapped(
            f"{name}: {len(plan)} answered; by person "
            f"{dict(sorted(Counter(a.person for a in plan).items()))}; check "
            f"{check_plan(plan, queue, MINUTES)}"
        )
        for a in plan:
            print_wrapped(
                f"  {a.person} {a.start:%H:%M}-{a.end:%H:%M}: {a.request_id}"
            )
        used = {
            p: sum(
                (a.end - a.start).total_seconds() / 60
                for a in plan
                if a.person == p
            )
            for p in MINUTES
        }
        print_wrapped(f"  minutes used {used}")
    print_wrapped(
        "certificate draft multipliers: "
        + ", ".join(
            f"{d.request_id}={v:g}" for d, v in zip(queue, record.multipliers)
        )
    )
    print_wrapped(
        "certificate minute multipliers: "
        + ", ".join(
            f"{p}={v:.6f}"
            for p, v in zip(MINUTES, record.multipliers[len(queue) :])
        )
    )
    print_wrapped(
        f"bound {record.bound:.6f}; certificate "
        f"{record.certificate['certificate']}; absolute gap "
        f"{record.absolute_gap:g}; relative gap {record.relative_gap:g}; "
        f"tolerance {record.tolerance:g}"
    )
    more = solve_morning(queue, {**MINUTES, "morgan": 90}, BUDGET)
    print_wrapped(
        f"what-if, morgan 90 minutes: {more.value} answered; bound "
        f"{more.bound:g}"
    )
    print_wrapped(
        f"solve record: {record.inputs_version}; fingerprint "
        f"{record.queue_fingerprint}; budget {record.budget} nodes; status "
        f"{record.status}; waiting-time status {record.secondary_status}; "
        f"plan check {record.plan_check}"
    )
# endregion book:fnd.constrained-optimization.plan-01


def window_results():
    return {p: run_window(p) for p in ("plain", "count", "count_then_wait")}


# region book:fnd.constrained-optimization.window-01 output
if __name__ == "__main__":
    windows = window_results()
    print(
        f"training only; seed 7; {len(windows['plain']['days'])} working mornings, 2026-01-22 through 2026-06-30"
    )
    print(
        "policy: mornings at rounded bound; answered / arrivals; tier 3 mean wait, longest wait (days to planned start)"
    )
    for policy, result in windows.items():
        print(
            f"{policy}: {result['at_bound']}; {result['answered']} / {result['requests']}; {result['tier3']['mean_days']:.6f}, {result['tier3']['longest_days']:.6f}"
        )
    for day in windows["plain"]["days"]:
        if day["answered"] < day["rounded_bound"]:
            print(
                f"plain shortfall {day['day']}: answered {day['answered']}; bound {day['bound']:.6f}; rounded {day['rounded_bound']}"
            )
    print(
        "count-only and count-then-wait counts equal every morning: "
        + str(
            all(
                a["answered"] == b["answered"]
                for a, b in zip(windows["count"]["days"], windows["count_then_wait"]["days"])
            )
        )
    )
# endregion book:fnd.constrained-optimization.window-01


def monday_queue():
    return next(
        d["queue"] for d in run_window("count_then_wait")["days"] if d["day"] == "2026-03-23"
    )


def robust_plan():
    from dataclasses import replace

    queue = [replace(d, minutes=45 if d.tier == 3 else d.minutes) for d in monday_queue()]
    return solve_morning(queue, MINUTES, BUDGET)


# region book:fnd.constrained-optimization.robust-01 output
if __name__ == "__main__":
    robust = robust_plan()
    print(
        f"Monday 2026-03-23, tier 3 slow case 45 minutes: answered {robust.value}; bound {robust.bound:.6f}; rounded {floor(robust.bound + robust.tolerance)}"
    )
    print(
        f"plan check {robust.plan_check}; certificate {robust.certificate['certificate']}; absolute gap {robust.absolute_gap:.6f}; relative gap {robust.relative_gap:.6f}"
    )
# endregion book:fnd.constrained-optimization.robust-01


REVIEW_SPREADS = {1: 1.25, 2: 3.75, 3: 7.5}


def normal_row(tiers, minutes, confidence, spreads=None):
    from math import erfc, sqrt
    from statistics import NormalDist

    from refund_agent.schedule.inputs import REVIEW_MINUTES, TOL

    spreads = REVIEW_SPREADS if spreads is None else spreads
    mean = sum(REVIEW_MINUTES[t] for t in tiers)
    spread = sqrt(sum(spreads[t]**2 for t in tiers))
    quantile = NormalDist().inv_cdf(confidence)
    required = mean + quantile * spread
    overrun = erfc((minutes - mean) / (spread * sqrt(2))) / 2 if spread else 0.0
    return dict(
        mean=mean, spread=spread, quantile=quantile, required=required,
        overrun=overrun, holds=required <= minutes + TOL,
    )


def tier_count_plans(queue):
    from itertools import product

    from refund_agent.schedule.rule import ordered_plan

    groups = {t: sorted((d for d in queue if d.tier == t), key=lambda d: (d.at, d.request_id))
              for t in (1, 2, 3)}
    options = [
        [(a, b) for a in range(len(group) + 1) for b in range(len(group) - a + 1)
         if all("alex" in d.allowed for d in group[:a])
         and all("morgan" in d.allowed for d in group[a:a + b])]
        for group in groups.values()
    ]
    for counts in product(*options):
        chosen = {}
        for group, (a, b) in zip(groups.values(), counts):
            chosen.update((d.request_id, "alex") for d in group[:a])
            chosen.update((d.request_id, "morgan") for d in group[a:a + b])
        yield ordered_plan(queue, chosen)


def normal_plan_rows(plan, queue, confidence=0.5, spreads=None):
    drafts = {d.request_id: d for d in queue}
    return {
        person: normal_row(
            [drafts[a.request_id].tier for a in plan if a.person == person],
            minutes, confidence, spreads,
        )
        for person, minutes in MINUTES.items()
    }


def check_chance_plan(plan, queue, confidence, spreads=None):
    failures = check_plan(plan, queue, MINUTES)
    rows = normal_plan_rows(plan, queue, confidence, spreads)
    return failures + [f"{p}: chance row exceeds minutes" for p, r in rows.items() if not r["holds"]]


def plan_rank(plan, queue):
    drafts = {d.request_id: d for d in queue}
    waited = sum(
        (drafts[a.request_id].plan_at - drafts[a.request_id].at).total_seconds()
        for a in plan
    )
    shared_minutes = sum(
        drafts[a.request_id].minutes for a in plan
        if a.person == "morgan" and drafts[a.request_id].tier != 3
    )
    return len(plan), waited, -shared_minutes


def chance_plan(queue, confidence, spreads=None):
    # Exact counts by tier and person; oldest representatives break count ties.
    feasible = (
        plan for plan in tier_count_plans(queue)
        if not check_chance_plan(plan, queue, confidence, spreads)
    )
    return max(feasible, key=lambda plan: plan_rank(plan, queue))


def scenario_average(plan, queue, times):
    index = {d.request_id: j for j, d in enumerate(queue)}
    counts = np.zeros(len(times), dtype=int)
    for person, minutes in MINUTES.items():
        ordered = sorted((a for a in plan if a.person == person), key=lambda a: a.start)
        columns = [index[a.request_id] for a in ordered]
        # Every assigned review consumes its drawn time, including an overrun.
        ends = np.cumsum(times[:, columns], axis=1)
        counts += (ends <= minutes).sum(axis=1)
    return float(counts.mean())


def exact_average(plan, queue):
    drafts, total = {d.request_id: d for d in queue}, 0.0
    for person, minutes in MINUTES.items():
        tiers = []
        for a in sorted((a for a in plan if a.person == person), key=lambda a: a.start):
            tiers.append(drafts[a.request_id].tier)
            total += 1 - normal_row(tiers, minutes, 0.5)["overrun"]
    return total


def uncertain_results(seed=15, draws=10000):
    from dataclasses import replace

    from refund_agent.schedule.inputs import REVIEW_MINUTES

    queue = monday_queue()
    one, two = chance_plan(queue, 0.95), chance_plan(queue, 0.5)
    times = np.random.default_rng(seed).normal(
        [REVIEW_MINUTES[d.tier] for d in queue],
        [REVIEW_SPREADS[d.tier] for d in queue], size=(draws, len(queue)),
    )
    candidates = [p for p in tier_count_plans(queue) if not check_plan(p, queue, MINUTES)]
    scores = [scenario_average(p, queue, times) for p in candidates]
    best = max(range(len(candidates)), key=lambda i: (scores[i], plan_rank(candidates[i], queue)))
    robust = robust_plan()
    worst = [replace(d, minutes=45 if d.tier == 3 else d.minutes) for d in queue]
    stances = {}
    for name, plan, failures in (
        ("robust", robust.plan, check_plan(robust.plan, worst, MINUTES)),
        ("chance-95", one, check_chance_plan(one, queue, 0.95)),
        ("chance-50", two, check_chance_plan(two, queue, 0.5)),
        ("scenarios", candidates[best], check_plan(candidates[best], queue, MINUTES)),
    ):
        stances[name] = dict(plan=plan, failures=failures, rows=normal_plan_rows(plan, queue))
    return dict(
        stances=stances, candidates=len(candidates),
        one=dict(sample_average=scenario_average(one, queue, times), exact_average=exact_average(one, queue)),
        two=dict(sample_average=scenario_average(two, queue, times), exact_average=exact_average(two, queue)),
        two95=normal_row([3, 3], MINUTES["morgan"], 0.95),
    )


# region book:fnd.constrained-optimization.uncertain-01 output
from demos import constrained_optimization_demo as uncertain_examples

if __name__ == "__main__":
    uncertain = uncertain_examples.uncertain_results()
    print("Monday 2026-03-23; constructed independent normal review times")
    print("tiers 1, 2, 3: means 5, 15, 30; spreads 1.25, 3.75, 7.5 minutes")
    print("robust: reuse the bounded 45-minute tier 3 case; other tiers fixed")
    print("normal tails are unbounded, including a small negative-time tail")
    print("boundary: required <= capacity + 1e-7 minutes; z(0.50) = 0")
    row = uncertain["two95"]
    print(f"two tier 3: mean {row['mean']}; spread {row['spread']:.6f}; "
          f"z(0.95) {row['quantile']:.6f}; required {row['required']:.6f}")
    print("chance solve: enumerate tier/person counts, then check every plan")
    print("ties: oldest drafts, then shared-tier work stays with alex")
    print("scenarios: maximize average over all mean-feasible count plans")
    print("one oldest-first representative per tier/person count vector")
    print(f"seed 15; 10000 draws; {uncertain['candidates']} candidate plans")
    print("in person's order, count a review only if its end <= capacity")
    print("each review consumes its drawn time; no clipping or reordering")
    for name in ("one", "two"):
        result = uncertain[name]
        print(f"morning total, morgan {name} tier 3: "
              f"sample {result['sample_average']:.6f}; "
              f"exact {result['exact_average']:.12f} answers")
    print("row overruns below are exact under the normal model, even robust")
    for name, stance in uncertain["stances"].items():
        print(f"{name}: planned {len(stance['plan'])}; "
              f"plan check {stance['failures']}")
        for person, row in stance["rows"].items():
            print(f"  {person}: mean {row['mean']}; "
                  f"spread {row['spread']:.6f}; overrun {row['overrun']:.12g}")
    print("scenario average picks two tier 3 reviews for morgan")
# endregion book:fnd.constrained-optimization.uncertain-01


def absence_case():
    from dataclasses import replace
    from datetime import timedelta

    from refund_agent.schedule.inputs import morning
    from refund_agent.schedule.queue import Draft

    monday, tuesday = morning(date(2026, 3, 23)), morning(date(2026, 3, 24))
    queue = [
        Draft(
            f"absence-{i}",
            monday - timedelta(days=1, minutes=i),
            tier,
            ("alex", "morgan") if tier == 1 else ("morgan",),
            5 if tier == 1 else 30,
            monday,
        )
        for i, tier in enumerate([1] * 6 + [3] * 2)
    ]
    result = {}
    for policy in ("count_then_wait", "tier3_first"):
        chosen = queue if policy == "count_then_wait" else [d for d in queue if d.tier == 3]
        first = solve_morning(chosen, {"alex": 0, "morgan": 60}, BUDGET)
        ids = {a.request_id for a in first.plan}
        carried = [replace(d, plan_at=tuesday) for d in queue if d.request_id not in ids]
        fresh = [
            Draft(f"tuesday-{i}", tuesday - timedelta(minutes=60 - i), 3, ("morgan",), 30, tuesday)
            for i in range(2)
        ]
        second = solve_morning(carried + fresh, MINUTES, BUDGET)
        answered = {a.request_id for a in second.plan}
        remaining = [d.request_id for d in carried + fresh if d.request_id not in answered]
        result[policy] = dict(monday=first, tuesday=second, remaining=remaining)
    return result


# region book:fnd.constrained-optimization.absence-01 output
if __name__ == "__main__":
    print(
        "constructed absence: Monday alex 0 minutes, morgan 60; 6 tier 1 and 2 tier 3; Tuesday 2 new tier 3, alex back"
    )
    for policy, result in absence_case().items():
        first, second = result["monday"], result["tuesday"]
        print(
            f"{policy}: Monday {first.value}, Tuesday {second.value}; answered {first.value + second.value} / 10; waiting into Wednesday {len(result['remaining'])}"
        )
# endregion book:fnd.constrained-optimization.absence-01


def solve_transportation(depots, shelters, costs):
    # Maximize negative cost, with shelter needs written as negative <= rows.
    m, n = len(depots), len(shelters)
    cost = np.asarray(costs, dtype=float)
    if not m or not n or cost.shape != (m, n) or not np.isfinite(cost).all() or np.any(cost < 0):
        raise ValueError("finite nonnegative depot-by-shelter costs required")
    if any(v < 0 for _, v in depots + shelters):
        raise ValueError("supply and demand must be nonnegative")
    rows = np.zeros((m + n, m * n))
    for i in range(m):
        for j in range(n):
            rows[i, i * n + j], rows[m + j, i * n + j] = 1, -1
    rhs = [v for _, v in depots] + [-v for _, v in shelters]
    solved = linprog(cost.ravel(), A_ub=rows, b_ub=rhs, method="highs")
    if not solved.success:
        raise RuntimeError(f"relief solve status {solved.status}")
    multipliers = np.where(solved.ineqlin.marginals == 0, 0, -solved.ineqlin.marginals)
    check = check_certificate(rows, rhs, -cost.ravel(), solved.x, multipliers)
    return dict(
        shipments=solved.x.reshape(m, n).tolist(),
        cost=float(solved.fun),
        multipliers=multipliers.tolist(),
        check=check,
    )


def relief():
    return solve_transportation(
        [("depot A", 4), ("depot B", 3)], [("shelter X", 3), ("shelter Y", 4)], [[1, 4], [3, 1]]
    )


# region book:fnd.constrained-optimization.relief-01 output
if __name__ == "__main__":
    relief_result = relief()
    print("relief: depot supplies 4, 3; shelter needs 3, 4; unit costs 1, 4, 3, 1")
    print(
        f"shipments {relief_result['shipments']}; cost {relief_result['cost']:g}; multipliers {relief_result['multipliers']}"
    )
    print(
        f"maximize negative cost: value {relief_result['check']['value']:g}; bound {relief_result['check']['bound']:g}; cost lower bound {-relief_result['check']['bound']:g}; certificate {relief_result['check']['certificate']}; gap {relief_result['check']['absolute_gap']:g}"
    )
# endregion book:fnd.constrained-optimization.relief-01


def exercise_five():
    from dataclasses import replace

    queue = monday_queue()
    record = solve_morning(queue, MINUTES, BUDGET)
    taken = {a.request_id for a in record.plan}
    extra = next(d for d in queue if d.tier == 3 and d.request_id not in taken)
    # The agent modeled one forbidden pair as allowed. Its certificate is consistent
    # with that wrong model, which is why the independent policy check is necessary.
    wrong_queue = [replace(d, allowed=("alex", "morgan")) if d == extra else d for d in queue]
    planted = solve_morning(wrong_queue, MINUTES, BUDGET)
    return dict(
        day="2026-03-23",
        status=planted.status,
        answered=planted.value,
        solver_bound=planted.bound,
        plan=[a.to_dict() for a in planted.plan],
        solve_record=planted.to_dict(),
    )


def respond_by_day(at):
    from datetime import timedelta

    day = at.date() + timedelta(days=1)
    while day.weekday() >= 5:
        day += timedelta(days=1)
    return day


def deadline_report():
    """Exercise 8's instructor key: never silently relax a respond-by rule.

    The first working day that begins after arrival means the next working calendar
    day, ending at 17:00 UTC. Morning blocks end earlier. Already-late drafts are due
    now. On infeasibility, the ordinary plan is explicitly marked as missing the rule.
    """
    from datetime import timedelta

    from refund_agent.schedule.inputs import END, START, WORKING_DAYS

    data, day, carried, proofs, answered, flags = generate(), START, [], [], 0, 0
    while day <= END:
        if day.weekday() in WORKING_DAYS:
            queue = waiting(data, day, carried)
            mandatory = [d for d in queue if respond_by_day(d.at) <= day]
            tier3 = sum(d.minutes for d in mandatory if d.tier == 3)
            other = sum(d.minutes for d in mandatory if d.tier != 3)
            flags += tier3 > MINUTES["morgan"]
            # This training window has no shortage of alex's minutes. Check this
            # premise rather than presenting a tier-3 proof for a different conflict.
            if other > MINUTES["alex"]:
                raise RuntimeError("deadline key needs an additional shared-capacity proof")
            record = solve_morning(
                queue, MINUTES, BUDGET, required=[d.request_id for d in mandatory]
            )
            if (record.status == "infeasible") != (tier3 > MINUTES["morgan"]):
                raise RuntimeError("solver status and independent counting check disagree")
            if record.status == "infeasible":
                if tier3 <= MINUTES["morgan"]:
                    raise RuntimeError("infeasibility does not match the counting proof")
                record = solve_morning(queue, MINUTES, BUDGET)
                taken = {a.request_id for a in record.plan}
                extra = tier3 - MINUTES["morgan"]
                repaired = solve_morning(
                    queue,
                    {**MINUTES, "morgan": MINUTES["morgan"] + extra},
                    BUDGET,
                    required=[d.request_id for d in mandatory],
                )
                if repaired.status != "optimal":
                    raise RuntimeError("claimed extra time does not repair the deadline")
                proofs.append(
                    dict(
                        day=day.isoformat(),
                        status="infeasible",
                        tier3_minutes=tier3,
                        morgan_minutes=MINUTES["morgan"],
                        extra_morgan_minutes=extra,
                        required=[d.request_id for d in mandatory if d.tier == 3],
                        missing_rule=[d.request_id for d in mandatory if d.request_id not in taken],
                        plan_check=record.plan_check,
                        plan=[a.to_dict() for a in record.plan],
                    )
                )
            taken = {a.request_id for a in record.plan}
            if record.plan_check:
                raise RuntimeError("deadline plan is not permitted")
            carried = [d for d in queue if d.request_id not in taken]
            answered += len(taken)
        day += timedelta(days=1)
    from refund_agent.schedule.queue import training_requests

    return dict(
        infeasible_mornings=len(proofs),
        counting_flags=flags,
        proofs=proofs,
        answered=answered,
        remaining=len(carried),
        requests=len(training_requests(data)),
    )


# region book:fnd.constrained-optimization.keys-01 output
if __name__ == "__main__":
    print(
        "exercise 1: hourly charge/release are decisions; cost is the objective; capacity, rate, reserve are constraints"
    )
    root = branches()[0]["value"]
    print(
        f"exercise 2: incumbent 6, bound {root:g}, gap {(root - 6) / 6:.6f}; exceeds 10%, reject; plan grants no approval"
    )
    lab5, whole5 = laboratory(5), whole_laboratory(5)
    print(
        f"exercise 3: 5 hours, linear plan {lab5['plan']}, value {lab5['value']:g}; multipliers {lab5['multipliers']}; bound {lab5['check']['bound']:g}"
    )
    print(
        f"whole plan {whole5['plan']}, value {whole5['value']:g}; whole gain {whole5['value'] - whole_laboratory()['value']:g}; hours multiplier {lab5['multipliers'][0]:g}"
    )
    print(
        f"exercise 4: relief cost {relief()['cost']:g}; certificate {relief()['check']['certificate']}"
    )
    five, twelve = proposal_sampling(5), proposal_sampling(12)
    print(
        f"exercise 5: N=5 coverage {five['coverage']:.9f}; "
        f"N=12 coverage {twelve['coverage']:.9f}"
    )
    print(
        f"  accepted invalid share {five['invalid_share']:.9f} at either N; "
        "lower pf lowers it"
    )
    spreads = {1: 1.25, 2: 3.75, 3: 10}
    two, one = (normal_row(tiers, 60, 0.9, spreads) for tiers in ([3, 3], [3]))
    exercise6 = chance_plan(monday_queue(), 0.9, spreads)
    print(f"exercise 6: z(0.90) {two['quantile']:.6f}; tier 3 spread 10 minutes")
    print(
        f"  two reviews need {two['required']:.6f}; "
        f"one needs {one['required']:.6f} minutes"
    )
    print(f"  morgan fits 1 tier 3; morning answered {len(exercise6)}")
    from refund_agent.schedule.record import Assignment

    fixture = exercise_five()
    failures = check_plan(
        [Assignment.from_dict(a) for a in fixture["plan"]], monday_queue(), MINUTES
    )
    print(
        f"exercise 7: claimed {fixture['answered']}, solver bound {fixture['solver_bound']}; failures {failures}; independent bound {solve_morning(monday_queue(), MINUTES, BUDGET).bound:g}; reject"
    )
    deadlines = deadline_report()
    print(
        f"exercise 8: infeasible mornings {deadlines['infeasible_mornings']}; answered {deadlines['answered']} / {deadlines['requests']}; remaining {deadlines['remaining']}"
    )
    for proof in deadlines["proofs"]:
        print(
            f"{proof['day']}: due tier 3 minutes {proof['tier3_minutes']}, available {proof['morgan_minutes']}, minimum extra {proof['extra_morgan_minutes']}; missing rule {len(proof['missing_rule'])}; plan check {proof['plan_check']}"
        )
# endregion book:fnd.constrained-optimization.keys-01


def figure_data():
    queue = monday_queue()
    plans = {
        "plain": plain_rule(queue, MINUTES),
        "count_then_wait": solve_morning(queue, MINUTES, BUDGET).plan,
    }
    lab, whole = laboratory(), whole_laboratory()
    corners = [[0, 0], [4 / 2, 0], lab["plan"], [0, 3]]
    return dict(
        laboratory=dict(
            corners=corners,
            corner_values=[float(np.dot([3, 2], p)) for p in corners],
            integer_points=[[x, y] for x in range(4) for y in range(4) if 2 * x + y <= 4],
            relaxation=lab["plan"],
            whole=whole["plan"],
            values=[lab["value"], whole["value"]],
        ),
        monday=dict(
            day="2026-03-23",
            minutes=MINUTES,
            plans={
                name: [
                    dict(
                        **a.to_dict(),
                        tier=next(d.tier for d in queue if d.request_id == a.request_id),
                    )
                    for a in plan
                ]
                for name, plan in plans.items()
            },
            waiting={
                name: [
                    dict(request_id=d.request_id, tier=d.tier)
                    for d in queue
                    if d.request_id not in {a.request_id for a in plan}
                ]
                for name, plan in plans.items()
            },
        ),
    )


def enrichment_figure_data():
    from statistics import NormalDist

    uncertain = uncertain_results()
    row = uncertain["two95"]
    normal = NormalDist(row["mean"], row["spread"])
    minutes = sorted(set(
        np.linspace(row["mean"] - 4 * row["spread"],
                    row["mean"] + 4 * row["spread"], 161).tolist()
        + [60.0, row["required"]]
    ))
    panels = []
    for name, reviews, required in (
        ("robust", 1, 45),
        ("chance-95", 1, normal_row([3], 60, 0.95)["required"]),
        ("scenarios", 2, 60),
    ):
        panels.append(dict(
            stance=name, reviews=reviews, capacity=60,
            normal_review_mean=30, normal_review_spread=7.5,
            minutes_required=required,
            normal_overrun=uncertain["stances"][name]["rows"]["morgan"]["overrun"],
            mean_finish_minutes=[30 * i for i in range(1, reviews + 1)],
            morning_exact_average_answered=uncertain["one" if reviews == 1 else "two"]["exact_average"],
            morning_sample_average_answered=uncertain["one" if reviews == 1 else "two"]["sample_average"],
        ))
    invalid = proposal_sampling(5)["invalid_share"]
    return dict(
        F2=dict(
            N=list(range(1, 21)), coverage=[1 - (2 / 3)**k for k in range(1, 21)],
            invalid_share=[invalid] * 20, valid_probability=1 / 3,
            false_pass_probability=0.1,
            note="Proposal budget N; invalid share is conditional on acceptance, not coverage.",
        ),
        F4=panels,
        F5=dict(
            mean=row["mean"], spread=row["spread"], capacity=60,
            quantile_95=row["required"], minutes=minutes,
            density=[normal.pdf(x) for x in minutes],
            cdf=[normal.cdf(x) for x in minutes],
        ),
        scenario_candidate_rule="All mean-feasible tier/person count plans; oldest representatives.",
        scenario_completion_rule="In each person's order, count a review iff its cumulative end <= capacity.",
        normal_model="Constructed independent, untruncated normals; means 5, 15, 30 and spreads 1.25, 3.75, 7.5 minutes.",
        robust_model="Only tier 3 varies, at most 45 minutes; tiers 1 and 2 stay fixed for this bounded guarantee.",
        seed=15, draws=10000,
    )


def write_data():
    import json
    from pathlib import Path

    directory = Path(__file__).with_name("data")
    for name, data in (
        ("constrained_optimization_figures", figure_data()),
        ("constrained_optimization_enrichment_figures", enrichment_figure_data()),
        ("review_plan_05", exercise_five()),
    ):
        (directory / f"{name}.json").write_text(
            json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
    windows = window_results()
    records = {
        policy: {d["day"]: d["record"].to_dict() for d in result["days"]}
        for policy, result in windows.items()
    }
    records["robust_monday"] = robust_plan().to_dict()
    records["deadline_key"] = deadline_report()
    path = directory.parents[1] / "book_outputs/_fnd.constrained-optimization.solve-records.json"
    path.write_text(json.dumps(records, indent=2, sort_keys=True, allow_nan=False) + "\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-data", action="store_true")
    if parser.parse_args().write_data:
        write_data()
