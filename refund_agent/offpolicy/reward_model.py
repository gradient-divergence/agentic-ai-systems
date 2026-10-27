"""Add-one-smoothed cell means, fitted only on the learning window."""

from collections import defaultdict
from dataclasses import dataclass

from refund_agent.offers.policies import fingerprint


@dataclass(frozen=True)
class CellModel:
    # cell, action, smoothed prediction, observations
    cells: tuple = ()
    source_fingerprint: str = ""

    def predict(self, context, action):
        return next(
            (mean for cell, offer, mean, _ in self.cells if (cell, offer) == (context, action)), 0.0
        )

    def prediction(self, context, action, *, supported=True):
        count = next(
            (n for cell, offer, _, n in self.cells if (cell, offer) == (context, action)), 0
        )
        return dict(
            value=self.predict(context, action),
            assumption=not supported or count == 0,
            training_records=count,
        )

    @property
    def fingerprint(self):
        return fingerprint(dict(cells=self.cells, source_fingerprint=self.source_fingerprint))


def fit_cells(records):
    records = list(records)
    if any(r.decided_at and not "2025-04-01" <= r.decided_at[:10] < "2026-02-01" for r in records):
        raise ValueError("the reward model may read only learning-window decisions")
    groups = defaultdict(list)
    for row in records:
        if row.reward is None or not row.final:
            raise ValueError("learning outcomes must be final")
        if not row.override:
            groups[(row.context, row.action)].append(row.reward)
    for row in records:
        for action in row.logged:
            groups[(row.context, action)]
    cells = tuple(
        (cell, action, (sum(rewards) + 1) / (len(rewards) + 2), len(rewards))
        for (cell, action), rewards in sorted(groups.items())
    )
    return CellModel(cells, fingerprint([r.record() for r in records]))
