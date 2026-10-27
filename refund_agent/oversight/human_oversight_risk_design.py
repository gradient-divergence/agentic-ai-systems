"""Signed staffing, role-specific necessary counts and month-by-month acceptance."""

from dataclasses import dataclass
from datetime import datetime
from fractions import Fraction
from itertools import combinations
from math import exp, factorial

from refund_agent.oversight.human_oversight_risk_demand import Stage6Demand
from refund_agent.oversight.human_oversight_risk_simulate import Person, staffing


@dataclass(frozen=True)
class QueueDesign:
    version: str
    people: tuple[Person, ...]
    signed_by: str
    signed_at: datetime
    named_approver: str = "morgan"
    ceiling: Fraction = Fraction(1, 100)
    utilization_target: float = .8

    def __post_init__(self):
        from refund_agent.release.rule import owner_ids

        if (not self.version or self.signed_by not in owner_ids()
                or self.signed_at.utcoffset() is None or self.ceiling != Fraction(1, 100)
                or self.utilization_target != .8 or not self.people
                or len({p.name for p in self.people}) != len(self.people)):
            raise ValueError("design needs its listed signer, version and signed ceilings")

    def as_table(self):
        return (
            ("What triggers escalation", "deadline, refusal, blind disagreement or execution refusal"),
            ("Who approves", f"tier 2 one listed reviewer; tier 3 only {self.named_approver}"),
            ("What context they receive", "proposal, evidence ages, explanation, credential and deadline"),
            ("How long the system waits", "next working day 17:00 UTC; four desk hours per escalation"),
            ("What happens on timeout", "notice and escalation; open; no approval, denial or payment"),
            ("How the decision is logged", "credential-bound action apart from resulting request state"),
            ("How rollback runs", "no refund recall; any compensation needs a new gated action"),
            ("Capacity", ", ".join(f"{p.name} {p.daily_hours * len(p.days)}h/week" for p in self.people)),
            ("Cover", "named approver absent: held, escalated; deputy needs a policy version"),
            ("Measures", "monthly late share, breach-month frequency, waits, load and catches"),
            ("Controller", "recheck 25% to 10%; blind 10% to 5%; staffing request above 0.8"),
            ("Withdrawal", "unbound payment, unreviewed tier 2 decision or clock-written decision"),
            ("Signed by", f"{self.signed_by}; {self.version}; {self.signed_at.isoformat()}"),
        )


def check_design(design: QueueDesign, demand: Stage6Demand, *, weekends=()):
    classes = demand.classes()
    weekly = {row.name: row.rate_per_day * 7 * row.minutes / 60 for row in classes}
    due = {row.name: row.rate_per_day * 55 / 24 * row.minutes / 60
           for row in classes if row.blocking}
    named = [p for p in design.people if p.name == design.named_approver and p.role == "named_approver"]
    reviewers = [p for p in design.people if p.role == "reviewer"]
    owners = [p for p in design.people if p.role == "decision_owner" or p.owner_tasks]
    capacity = sum(p.daily_hours * len(p.days) for p in design.people)
    named_capacity = sum(p.daily_hours * len(p.days) for p in named)
    named_load = sum(weekly[row.name] for row in classes if row.tier == 3)
    reviewer_load = sum(weekly[row.name] for row in classes if row.tier != 3 and row.role != "decision_owner")
    owner_load = sum(weekly[row.name] for row in classes if row.role == "decision_owner")
    pool_capacity = sum(p.daily_hours * len(p.days) for p in reviewers) + sum(
        max(0, p.daily_hours * len(p.days) - named_load - (owner_load if p.owner_tasks else 0))
        for p in named if p.help_reviewers)
    owner_capacity = sum(p.daily_hours * len(p.days) for p in owners)
    failures = []
    for group, load, room in (("named_approver", named_load, named_capacity),
                              ("reviewers", reviewer_load, pool_capacity),
                              ("decision_owner_followups", owner_load, owner_capacity)):
        if load and load >= room:
            failures.append(group + "_unstable")
    monday_named = sum(p.daily_hours for p in named if 0 in p.days)
    monday_reviewers = sum(p.daily_hours for p in reviewers if 0 in p.days)
    monday_owners = sum(p.daily_hours for p in owners if 0 in p.days)
    general_named = sum(p.daily_hours for p in named if 0 in p.days and p.help_reviewers)
    tier3_due = sum(due.get(row.name, 0) for row in classes if row.tier == 3)
    owner_due = sum(due.get(row.name, 0) for row in classes if row.role == "decision_owner")
    general_due = sum(due.values()) - tier3_due - owner_due
    checks = (
        ("tier3_monday", tier3_due, monday_named),
        ("reviewers_monday", general_due, monday_reviewers + max(0, general_named - tier3_due
                                                               - (owner_due if any(p.owner_tasks for p in named) else 0))),
        ("pooled_monday", general_due + tier3_due + owner_due, sum(p.daily_hours for p in design.people if 0 in p.days)),
        ("decision_owner_monday", owner_due, monday_owners),
    )
    for name, load, room in checks:
        if load > room + 1e-10:
            failures.append(name + "_infeasible")
    class_rows = []
    allowed_by_class = {}
    for row in classes:
        if row.tier == 3:
            allowed = named
        elif row.role == "decision_owner":
            allowed = owners
        elif row.name in ("blind", "recheck", "held_denial"):
            allowed = reviewers
        else:
            allowed = reviewers + [p for p in named if p.help_reviewers]
        room = sum(p.daily_hours * len(p.days) for p in allowed)
        allowed_by_class[row.name] = frozenset(p.name for p in allowed)
        if weekly[row.name] and weekly[row.name] >= room:
            failures.append(row.name + "_class_capacity")
        class_rows.append({"class": row.name, "weekly_hours": weekly[row.name],
                           "allowed": tuple(p.name for p in allowed), "weekly_capacity": room})
    # Overlapping roles count one person's hours once, including Morgan's
    # decision-owner work. Check every small staffing subset, not just the pool.
    names = tuple(p.name for p in design.people)
    subsets = [frozenset(group) for n in range(1, len(names) + 1) for group in combinations(names, n)]
    for subset in subsets:
        required = [row.name for row in classes if allowed_by_class[row.name] and allowed_by_class[row.name] <= subset]
        room_week = sum(p.daily_hours * len(p.days) for p in design.people if p.name in subset)
        room_monday = sum(p.daily_hours for p in design.people if p.name in subset and 0 in p.days)
        if sum(weekly[name] for name in required) >= room_week and any(weekly[name] for name in required):
            failures.append("role_subset_week:" + ",".join(sorted(subset)))
        if sum(due.get(name, 0) for name in required) > room_monday + 1e-10:
            failures.append("role_subset_monday:" + ",".join(sorted(subset)))
    observed = []
    for weekend in weekends:
        problems = []
        if weekend["tier3_hours"] > monday_named + 1e-10:
            problems.append("tier3_authorized_capacity")
        if weekend["all_hours"] > sum(p.daily_hours for p in design.people if 0 in p.days) + 1e-10:
            problems.append("pooled_capacity")
        for subset in subsets:
            required = [row.name for row in classes if row.blocking and allowed_by_class[row.name]
                        and allowed_by_class[row.name] <= subset]
            room = sum(p.daily_hours for p in design.people if p.name in subset and 0 in p.days)
            if sum(weekend["hours_by_class"].get(name, 0) for name in required) > room + 1e-10:
                problems.append("role_subset:" + ",".join(sorted(subset)))
        observed.append({**weekend, "failures": tuple(problems)})
    return {"weekly_hours": weekly, "reviewer_hours": reviewer_load, "named_hours": named_load,
            "followup_hours": owner_load, "total_hours": sum(weekly.values()),
            "capacity_hours": capacity, "utilization": sum(weekly.values()) / capacity,
            "reviewer_only_utilization": reviewer_load / sum(p.daily_hours * len(p.days) for p in reviewers) if reviewers else None,
            "monday_hours": sum(due.values()), "monday_capacity": sum(p.daily_hours for p in design.people if 0 in p.days),
            "tier3_monday_hours": tier3_due, "tier3_monday_capacity": monday_named,
            "monday_margin": sum(p.daily_hours for p in design.people if 0 in p.days) - sum(due.values()),
            "classes": tuple(class_rows), "failures": tuple(dict.fromkeys(failures)),
            "observed_weekends": tuple(observed),
            "necessary_counts_only": True}


def accept_months(months, ceiling=Fraction(1, 100)):
    if ceiling != Fraction(1, 100):
        raise ValueError("monthly acceptance must use the signed 1% ceiling")
    rows = tuple(months)
    if not rows:
        raise ValueError("monthly evidence is required")
    failures = []
    for i, row in enumerate(rows):
        late, total = row["late"], row["blocking"]
        if (type(total) is not int or type(late) is not int or total <= 0
                or not 0 <= late <= total):
            raise ValueError("invalid monthly counts")
        if Fraction(late, total) > ceiling or row.get("pending", 0):
            failures.append(i)
    return {"accepted": not failures, "months": len(rows), "breach_months": len(failures),
            "breach_frequency": len(failures) / len(rows), "failed_month_indices": tuple(failures),
            "pooled_share": sum(row["late"] for row in rows) / sum(row["blocking"] for row in rows)}


def poisson_tail(at_least, intensity):
    return max(0.0, 1 - exp(-intensity) * sum(intensity ** k / factorial(k) for k in range(at_least)))


def default_design(name, signed_at):
    return QueueDesign("human-oversight-risk-" + name + "@1", staffing(name), "morgan", signed_at)
