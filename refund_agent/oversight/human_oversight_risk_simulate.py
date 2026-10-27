"""Seeded queue events, calendar staffing, role constraints and finite-window proof.

The cohort consists of the month's arrivals. Service drains for 14 more days to
classify their deadlines. Occupancy and utilization stop at the month's boundary;
unfinished stays are right-censored there. This is not stationary Little's law.
"""

from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from heapq import heappop, heappush
from math import ceil, isfinite
from random import Random
from statistics import mean

from refund_agent import contracts as c
from refund_agent.oversight.human_oversight_risk_demand import Stage6Demand, WorkClass
from refund_agent.oversight.human_oversight_risk_lifecycle import (
    ReviewTicket,
    respond_by,
)


@dataclass(frozen=True)
class Person:
    name: str
    role: str
    days: tuple[int, ...] = (0, 1, 2, 3, 4)
    help_reviewers: bool = True
    absent: frozenset = frozenset()
    daily_hours: float = 8
    owner_tasks: bool = False

    def __post_init__(self):
        if (not self.name or self.role not in ("reviewer", "named_approver", "decision_owner")
                or not self.days or len(set(self.days)) != len(self.days)
                or any(type(d) is not int or d not in range(7) for d in self.days)
                or not isfinite(self.daily_hours) or not 0 < self.daily_hours <= 8):
            raise ValueError("invalid staffing record")


@dataclass(frozen=True)
class QueueArrival:
    review_id: str
    work_class: str
    at: datetime
    first_reviewer: str | None = None
    required_person: str | None = None

    def __post_init__(self):
        c._utc_time(self.at)
        if not self.review_id or not self.work_class:
            raise ValueError("queue arrivals need an identity and class")


@dataclass
class ServedItem:
    arrival: QueueArrival
    work: WorkClass
    minutes: float
    remaining: float
    due_at: datetime
    opened_at: datetime | None = None
    closed_at: datetime | None = None
    reviewer: str | None = None
    queue_length: int = 0
    ticket: ReviewTicket | None = None
    segments: list = field(default_factory=list)


@dataclass
class SimulationResult:
    start: datetime
    end: datetime
    classes: dict
    people: dict
    items: tuple[ServedItem, ...]
    blocking: int
    late_blocking: int
    pending_blocking: int
    monday_count: int
    monday_breaches: int
    max_backlog: int
    max_backlog_at: datetime
    conservation: dict
    weekends: tuple[dict, ...]
    starting_backlog: int
    escalations: int
    notes: tuple[str, ...] = field(default=(
        "cohort=month arrivals; occupancy=calendar boundary, including service",
        "unfinished stays censored at close; deadline outcomes drain 14 days",
        "finite-window conservation; no stationary claim for D1",
    ))

    @property
    def late_share(self):
        return self.late_blocking / self.blocking if self.blocking else 0.0


def staffing(name: str) -> tuple[Person, ...]:
    if name not in ("D1", "D2", "D3", "D4"):
        raise ValueError("unknown staffing")
    people = [Person("alex", "reviewer"),
              Person("morgan", "named_approver", help_reviewers=name != "D1")]
    if name in ("D3", "D4"):
        people.append(Person("riley", "reviewer", days=(0,)))
    if name == "D4":
        # Revised constructed design: the same named approver works Saturday
        # and explicitly reserves time for the decision-owner role he holds.
        people[1] = Person("morgan", "named_approver", days=(0, 1, 2, 3, 4, 5), owner_tasks=True)
    return tuple(people)


def desk_hours_between(start: datetime, end: datetime, working_days=(0, 1, 2, 3, 4)) -> float:
    if end < start:
        raise ValueError("reversed interval")
    total = 0.0
    day = start.replace(hour=0, minute=0, second=0, microsecond=0)
    while day < end:
        if day.weekday() in working_days:
            opening = day + timedelta(hours=9)
            closing = day + timedelta(hours=17)
            total += max(0, (min(end, closing) - max(start, opening)).total_seconds()) / 3600
        day += timedelta(days=1)
    return total


def poisson_arrivals(seed, start, days, classes) -> tuple[QueueArrival, ...]:
    rng = Random(seed)
    result = []
    for row in classes:
        if row.rate_per_day == 0:
            continue
        t, index = 0.0, 0
        while (t := t + rng.expovariate(row.rate_per_day / 86400)) < days * 86400:
            result.append(QueueArrival(f"human-oversight-risk-{seed}-{row.name}-{index}",
                                       row.name, start + timedelta(seconds=t),
                                       first_reviewer="alex" if row.name == "blind" else None))
            index += 1
    return tuple(sorted(result, key=lambda a: (a.at, a.review_id)))


def _eligible(person: Person, job: ServedItem, named_approver: str) -> bool:
    if job.arrival.required_person and person.name != job.arrival.required_person:
        return False
    if job.arrival.first_reviewer == person.name:
        return False
    if job.work.role == "decision_owner":
        return person.role == "decision_owner" or person.owner_tasks
    if job.work.tier == 3:
        return person.role == "named_approver" and person.name == named_approver
    if person.role == "named_approver":
        return person.help_reviewers and job.work.name not in ("blind", "recheck", "held_denial")
    return person.role == "reviewer"


def nearest_rank(values, q):
    if not values:
        return None
    return sorted(values)[max(0, ceil(q * len(values)) - 1)]


def simulate(seed: int, start: datetime, days: int = 30, *, demand=None,
             people=None, classes=None, arrivals=None, initial_backlog=(),
             service_model="exponential", named_approver="morgan") -> SimulationResult:
    from refund_agent.oversight.human_oversight_risk_deadline import on_deadline

    start = c._utc_time(start)
    if type(days) is not int or days < 1 or service_model not in ("fixed", "exponential"):
        raise ValueError("positive days and a named service-time model required")
    end = start + timedelta(days=days)
    drain_end = end + timedelta(days=14)
    people = tuple(staffing("D3") if people is None else people)
    classes = tuple((demand or Stage6Demand()).classes() if classes is None else classes)
    if not people or len({p.name for p in people}) != len(people):
        raise ValueError("staffing needs distinct people")
    by_class = {row.name: row for row in classes}
    if len(by_class) != len(classes):
        raise ValueError("duplicate work class")
    arrivals = poisson_arrivals(seed, start, days, classes) if arrivals is None else tuple(arrivals)
    initial_backlog = tuple(initial_backlog)
    if (any(a.at < start or a.at >= end for a in arrivals)
            or any(a.at >= start for a in initial_backlog)):
        raise ValueError("arrivals/backlog must belong to their declared boundary")
    source = initial_backlog + tuple(arrivals)
    if len({a.review_id for a in source}) != len(source):
        raise ValueError("duplicate review identity")
    if any(a.work_class not in by_class for a in source):
        raise ValueError("arrival has no declared work class")
    rng = Random(seed + 100_000)
    jobs = []
    for a in source:
        work = by_class[a.work_class]
        minutes = work.minutes if service_model == "fixed" else rng.expovariate(1 / work.minutes)
        jobs.append(ServedItem(a, work, minutes, minutes, respond_by(a.at, work.working_days)))
    events, sequence = [], 0

    def schedule(at, rank, kind, payload):
        nonlocal sequence
        sequence += 1
        heappush(events, (at, rank, sequence, kind, payload))

    for i, job in enumerate(jobs):
        schedule(max(start, job.arrival.at), 1, "arrival", i)
        if job.work.blocking:
            schedule(max(start, job.due_at), 2, "deadline", i)
    active, waiting, available = {}, set(), set()
    capacity = {p.name: 0.0 for p in people}
    used = dict(capacity)
    day = start.replace(hour=0, minute=0, second=0, microsecond=0)
    while day < drain_end:
        for p in people:
            if day.weekday() in p.days and day.date() not in p.absent:
                opening = day + timedelta(hours=9)
                closing = opening + timedelta(hours=p.daily_hours)
                if closing > start:
                    schedule(max(start, opening), 3, "opening", p.name)
                    schedule(closing, 4, "closing", p.name)
                    capacity[p.name] += max(0, (min(closing, end) - max(opening, start)).total_seconds()) / 3600
        day += timedelta(days=1)
    counts = {name: 0 for name in by_class}
    area = {name: 0.0 for name in by_class}
    largest = {name: (0, start) for name in by_class}
    previous, max_waiting, max_at, escalation_count = start, 0, start, 0
    person_map = {p.name: p for p in people}

    def dispatch(at):
        # The named approver takes tier 3 first. Ordinary reviewers never do.
        ordered = sorted(available - active.keys(),
                         key=lambda name: (person_map[name].role != "named_approver", name))
        for name in ordered:
            p = person_map[name]
            choices = [i for i in waiting if _eligible(p, jobs[i], named_approver)]
            if not choices:
                continue
            i = min(choices, key=lambda i: (
                0 if jobs[i].work.tier == 3 else 1 if jobs[i].work.blocking else 2,
                jobs[i].arrival.at, jobs[i].arrival.review_id,
            ))
            job = jobs[i]
            if job.opened_at is None:
                job.opened_at, job.reviewer, job.queue_length = at, name, len(waiting) - 1
            waiting.remove(i)
            closing = at.replace(hour=9, minute=0, second=0, microsecond=0) + timedelta(hours=p.daily_hours)
            segment = min(job.remaining, (closing - at).total_seconds() / 60)
            if segment <= 0:
                waiting.add(i)
                continue
            active[name] = i
            stop = at + timedelta(minutes=segment)
            job.segments.append((at, stop, name))
            used[name] += max(0, (min(stop, end) - min(at, end)).total_seconds()) / 3600
            schedule(stop, 0, "service", (name, i, segment))

    while events:
        at, _, _, kind, payload = heappop(events)
        if at > drain_end:
            break
        duration = max(0, (min(at, end) - min(previous, end)).total_seconds()) / 3600
        for name in counts:
            area[name] += counts[name] * duration
        previous = at
        if kind == "arrival":
            i = payload
            job = jobs[i]
            waiting.add(i)
            counts[job.work.name] += 1
            if job.work.blocking:
                path = (named_approver, "decision-owner") if job.work.tier == 3 else (
                    "reviewer", named_approver, "decision-owner")
                job.ticket = ReviewTicket(
                    review_id=job.arrival.review_id, request_id=job.arrival.review_id,
                    requester_id="constructed-requester", work_class=job.work.name,
                    tier=c.Tier(job.work.tier), opened_at=job.arrival.at, due_at=job.due_at,
                    assignee=path[0], path=path, queue_length=len(waiting) - 1,
                )
        elif kind == "service":
            name, i, segment = payload
            job = jobs[i]
            job.remaining = max(0.0, job.remaining - segment)
            del active[name]
            if job.remaining < 1e-8:
                job.closed_at = at
                counts[job.work.name] -= 1
            else:
                waiting.add(i)
        elif kind == "opening":
            available.add(payload)
        elif kind == "closing":
            available.discard(payload)
        elif kind == "deadline":
            job = jobs[payload]
            if job.closed_at is None and job.ticket is not None:
                updated = on_deadline(job.ticket, at)
                escalation_count += len(updated.events) - len(job.ticket.events)
                if updated.step != job.ticket.step:
                    schedule(updated.due_at, 2, "deadline", payload)
                job.ticket = updated
        if at <= end:
            if len(waiting) > max_waiting:
                max_waiting, max_at = len(waiting), at
            queued_counts = Counter(jobs[i].work.name for i in waiting)
            for name in counts:
                n = queued_counts[name]
                if n > largest[name][0]:
                    largest[name] = (n, at)
        dispatch(at)
    for name in counts:
        area[name] += counts[name] * max(0, (end - min(previous, end)).total_seconds()) / 3600
    hours = days * 24
    class_rows, conservation = {}, {}
    cohort = [j for j in jobs if j.arrival.at >= start]
    for name in by_class:
        boundary = [j for j in jobs if j.work.name == name]
        admitted = [j for j in cohort if j.work.name == name]
        closed = [j for j in admitted if j.closed_at is not None]
        calendar = [(j.closed_at - j.arrival.at).total_seconds() / 3600 for j in closed]
        desk_days = tuple(sorted({day for p in people for day in p.days}))
        desk = [desk_hours_between(j.arrival.at, j.closed_at, desk_days) for j in closed]
        stays = [max(0, (min(j.closed_at or end, end) - max(j.arrival.at, start)).total_seconds()) / 3600
                 for j in boundary]
        gap = abs(area[name] - sum(stays)) / max(area[name], sum(stays), 1e-15)
        conservation[name] = {"occupancy_hours": area[name], "clipped_stay_hours": sum(stays),
                              "mean_present": area[name] / hours, "members": len(boundary),
                              "mean_clipped_stay": mean(stays) if stays else 0,
                              "starting_backlog": len(boundary) - len(admitted), "relative_gap": gap}
        class_rows[name] = {
            "arrivals": len(admitted), "completed_by_close": sum(j.closed_at is not None and j.closed_at <= end for j in admitted),
            "completed_after_drain": len(closed), "mean_calendar_hours": mean(calendar) if calendar else None,
            "p95_calendar_hours": nearest_rank(calendar, .95), "max_calendar_hours": max(calendar) if calendar else None,
            "mean_desk_hours": mean(desk) if desk else None, "p95_desk_hours": nearest_rank(desk, .95),
            "late": sum(j.closed_at is None or j.closed_at > j.due_at for j in admitted if j.work.blocking),
            "deadline_breaches": sum(j.closed_at is None or j.closed_at > j.due_at for j in admitted),
            "max_backlog": largest[name][0], "max_backlog_at": largest[name][1],
        }
    blocking = [j for j in cohort if j.work.blocking]
    late = [j for j in blocking if j.closed_at is None or j.closed_at > j.due_at]
    monday_days = {start.date() + timedelta(days=i) for i in range(days)
                   if (start + timedelta(days=i)).weekday() == 0}
    breached = {j.due_at.date() for j in late if j.due_at.date() in monday_days}
    weekends = []
    for monday in sorted(monday_days):
        due = datetime.combine(monday, datetime.min.time(), start.tzinfo) + timedelta(hours=17)
        friday = due - timedelta(days=3)
        midnight = due.replace(hour=0)
        weekend_jobs = [j for j in cohort if j.work.blocking and friday <= j.arrival.at < midnight]
        opening = midnight + timedelta(hours=9)
        due_jobs = [j for j in jobs if j.work.blocking and j.arrival.at < opening and j.due_at <= due
                    and (j.closed_at is None or j.closed_at > opening)]
        remaining = {
            j.arrival.review_id: max(0, j.minutes - sum(
                max(0, (min(stop, opening) - min(began, opening)).total_seconds()) / 60
                for began, stop, _ in j.segments)) for j in due_jobs
        }
        weekends.append({"monday": monday.isoformat(),
                         "tier3_hours": sum(remaining[j.arrival.review_id] for j in due_jobs if j.work.tier == 3) / 60,
                         "all_hours": sum(remaining.values()) / 60,
                         "incoming_weekend_hours": sum(j.minutes for j in weekend_jobs) / 60,
                         "counts": {name: sum(j.work.name == name for j in weekend_jobs) for name in by_class},
                         "hours_by_class": {name: sum(remaining[j.arrival.review_id] for j in due_jobs if j.work.name == name) / 60
                                            for name in by_class}})
    return SimulationResult(start, end, class_rows,
                            {name: {"used_hours": used[name], "available_hours": capacity[name],
                                    "utilization": used[name] / capacity[name] if capacity[name] else None}
                             for name in capacity}, tuple(jobs), len(blocking), len(late),
                            sum(j.closed_at is None for j in blocking), len(monday_days), len(breached),
                            max_waiting, max_at, conservation, tuple(weekends), len(initial_backlog), escalation_count)


def simulate_one(seed, utilization, *, n=100_000, desk=False, mu=2):
    """Continuous FCFS validation or weekday desk at the SAME staffed utilization."""
    if not isfinite(utilization) or not 0 < utilization < 1 or n < 100 or mu <= 0:
        raise ValueError("stable utilization and a long, positive run required")
    rng = Random(seed)
    lam = utilization * mu * (40 / 168 if desk else 1)
    at, finish, durations, occupancy = 0.0, 0.0, [], 0.0
    origin = datetime(2026, 10, 5, tzinfo=timezone.utc)
    begin = 0.0
    for i in range(n):
        at += rng.expovariate(lam)
        service = rng.expovariate(mu)
        if desk:
            remaining = service
            current = origin + timedelta(hours=max(at, finish))
            while remaining > 1e-12:
                opening = current.replace(hour=9, minute=0, second=0, microsecond=0)
                closing = opening + timedelta(hours=8)
                if current.weekday() >= 5 or current >= closing:
                    current = opening + timedelta(days=1)
                    continue
                current = max(current, opening)
                segment = min(remaining, (closing - current).total_seconds() / 3600)
                current += timedelta(hours=segment)
                remaining -= segment
            finish = (current - origin).total_seconds() / 3600
        else:
            finish = max(at, finish) + service
        if i == n // 10:
            begin = at
        if i >= n // 10:
            durations.append(finish - at)
            occupancy += finish - at
    observed = finish - begin  # drained cohort boundary, explicit transient correction
    W = mean(durations)
    L = occupancy / observed
    rate = len(durations) / observed
    return {"W": W, "L": L, "arrival_rate": rate, "tail4": sum(t > 4 for t in durations) / len(durations),
            "relative_gap": abs(L - rate * W) / max(L, 1e-15), "calendar_lambda": lam}


def inattentive_catch(queue_length: int) -> float:
    if queue_length < 0:
        raise ValueError("negative queue length")
    return .9 if queue_length <= 5 else .7 if queue_length <= 15 else .4


def first_empty_after(items, work_class, at):
    """Net backlog drain includes new arrivals, not just the absence cohort."""
    selected = [j for j in items if j.work.name == work_class]
    count = sum(j.arrival.at <= at and (j.closed_at is None or j.closed_at > at) for j in selected)
    if count == 0:
        return at
    events = [(j.arrival.at, 1) for j in selected if j.arrival.at > at]
    events += [(j.closed_at, -1) for j in selected if j.closed_at and j.closed_at > at]
    for when, change in sorted(events):
        count += change
        if count == 0:
            return when
    return None
