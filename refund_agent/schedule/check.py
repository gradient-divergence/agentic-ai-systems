"""Check assignments against the input queue, independently of the optimizer."""

from datetime import timedelta


def _assignment_failures(assignment, draft, minutes):
    key, person, failures = assignment.request_id, assignment.person, []
    if person not in draft.allowed:
        failures.append(f"{key}: {person} not allowed at tier {draft.tier}")
    if person not in minutes:
        return failures + [f"{key}: person has no minutes block"]
    if assignment.start.tzinfo is None or assignment.end.tzinfo is None:
        return failures + [f"{key}: timestamps must have a timezone"]
    if assignment.end - assignment.start != timedelta(minutes=draft.minutes):
        failures.append(f"{key}: wrong duration")
    if assignment.start < draft.plan_at or assignment.end > draft.plan_at + timedelta(
        minutes=minutes[person]
    ):
        failures.append(f"{key}: outside person's minutes block")
    return failures


def _block_failures(person, assignments, drafts, minutes):
    assignments.sort(key=lambda a: (a.start, a.request_id))
    failures = []
    if sum(drafts[a.request_id].minutes for a in assignments) > minutes[person]:
        failures.append(f"{person}: total work exceeds minutes")
    if any(b.start < a.end for a, b in zip(assignments, assignments[1:])):
        failures.append(f"{person}: overlap")
    return failures


def check_plan(plan, queue, minutes):
    drafts = {d.request_id: d for d in queue}
    seen, blocks, failures = set(), {}, []
    for assignment in plan:
        key, person = assignment.request_id, assignment.person
        if key in seen:
            failures.append(f"{key}: taken twice")
        seen.add(key)
        draft = drafts.get(key)
        if draft is None:
            failures.append(f"{key}: not in waiting queue")
            continue
        failures.extend(_assignment_failures(assignment, draft, minutes))
        if (
            person in minutes
            and assignment.start.tzinfo is not None
            and assignment.end.tzinfo is not None
        ):
            blocks.setdefault(person, []).append(assignment)
    for person, assignments in blocks.items():
        failures.extend(_block_failures(person, assignments, drafts, minutes))
    return failures
