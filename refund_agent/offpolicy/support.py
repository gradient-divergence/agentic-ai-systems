"""Coverage comes from full applied distributions, never from matching counts."""

from collections import defaultdict


def support_table(records, policy, cell_of=lambda row: row.context):
    groups = defaultdict(list)
    for row in records:
        groups[cell_of(row)].append(row)
    result = []
    for cell, rows in sorted(groups.items()):
        target = policy.probabilities(cell)
        actions = sorted(set(target) | {a for row in rows for a in row.logged})
        for action in actions:
            chances = [row.logged.get(action, 0) if not row.override else 0 for row in rows]
            positives = [p for p in chances if p > 0]
            eligible = all(not row.eligible or action in row.eligible for row in rows)
            result.append(
                dict(
                    cell=cell,
                    offer=action,
                    records=len(rows),
                    positive_share=len(positives) / len(rows),
                    min_positive=min(positives) if positives else None,
                    acts=target.get(action, 0) > 0,
                    eligible=eligible,
                    matches=sum(row.action == action for row in rows),
                )
            )
    return result


def supported(table, policy=None):
    return bool(table) and all(
        not row["acts"] or (row["positive_share"] == 1 and row["eligible"]) for row in table
    )
