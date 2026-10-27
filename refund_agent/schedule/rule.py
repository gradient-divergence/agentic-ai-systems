"""Oldest first, taking the allowed person who can start soonest."""

from datetime import timedelta

from .record import Assignment


def ordered_plan(queue, chosen):
    used, plan = {}, []
    for draft in sorted(queue, key=lambda d: (d.at, d.request_id)):
        person = chosen.get(draft.request_id)
        if person is None:
            continue
        start = draft.plan_at + timedelta(minutes=used.get(person, 0))
        end = start + timedelta(minutes=draft.minutes)
        plan.append(Assignment(draft.request_id, person, start, end))
        used[person] = used.get(person, 0) + draft.minutes
    return plan


def plain_rule(queue, minutes):
    used, chosen = dict.fromkeys(minutes, 0), {}
    for draft in sorted(queue, key=lambda d: (d.at, d.request_id)):
        candidates = [
            p for p in draft.allowed if p in minutes and used[p] + draft.minutes <= minutes[p]
        ]
        if candidates:
            # min retains the reviewer-list order on a tie.
            person = min(candidates, key=lambda p: used[p])
            chosen[draft.request_id] = person
            used[person] += draft.minutes
    return ordered_plan(queue, chosen)
