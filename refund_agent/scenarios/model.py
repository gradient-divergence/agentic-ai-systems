"""An add-one autoregressive model fitted only on chapter 4's training window."""

import hashlib
import json
from collections import Counter
from datetime import datetime, timedelta

import numpy as np

from refund_agent.data.generate import generate
from refund_agent.labels import LABEL_POLICY, _label
from refund_agent.scenarios.tokens import FIELDS, VALUES, case_tokens
from refund_agent.split import CUTS, fingerprint, time_split

# Conditional independence is declared, not inferred from the sampled outcome.
PARENTS = ((), (0,), (1,), (), (), (0, 1, 2, 3, 4))


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def training_data(seed=7):
    """Generate the canonical corpus, then discard later rows before feature access.

    Generation constructs the corpus; no frozen set is opened or consulted. Only
    request time/id metadata enters the split. Labels are recomputed on training.
    """
    data = generate(seed)
    training = time_split(data.requests, CUTS, timedelta(days=LABEL_POLICY.window_days))["training"]
    ids = {r.request_id for r in training if r.failure is None}
    rows = [
        (r, o, s)
        for r, o, s in zip(data.requests, data.orders, data.shipments)
        if r.request_id in ids
    ]
    rows.sort(key=lambda row: row[0].at)
    return data.model_copy(
        update={
            "requests": [r for r, _, _ in rows],
            "orders": [o for _, o, _ in rows],
            "shipments": [s for _, _, s in rows],
            "retrievals": [],
            "correct": [_label(r, o, s, data.policy_at(r.at)) for r, o, s in rows],
            "public_scenarios": [],
        }
    )


def _training_manifest(data, rows, labels, seed):
    return {
        "seed": seed,
        "requests": len(rows),
        "first": rows[0][0].at.isoformat(),
        "last": rows[-1][0].at.isoformat(),
        "cut": CUTS[0].isoformat(),
        "policy_versions": sorted({data.policy_at(row[0].at).version for row in rows}),
        "sha256": fingerprint([r for r, _, _ in rows], labels),
        "records_sha256": sha256(
            [[record.model_dump(mode="json") for record in row] for row in rows]
        ),
    }


def _model_fingerprint(counts):
    return sha256(
        {
            "fields": FIELDS,
            "values": VALUES,
            "parents": PARENTS,
            "counts": [
                [
                    {"context": key, "counts": values}
                    for key, values in sorted(table.items(), key=lambda item: canonical(item[0]))
                ]
                for table in counts
            ],
        }
    )


class ScenarioModel:
    training_tokens: tuple[tuple, ...]
    training_times: tuple[datetime, ...]
    training_cases: Counter[tuple]
    counts: list[dict[tuple, list[int]]]
    training_manifest: dict
    fingerprint: str

    @classmethod
    def fit(cls, data=None, seed=7):
        data = training_data(seed) if data is None else data
        training = time_split(data.requests, CUTS, timedelta(days=LABEL_POLICY.window_days))[
            "training"
        ]
        ids = {r.request_id for r in training if r.failure is None}
        rows = [
            (r, o, s)
            for r, o, s in zip(data.requests, data.orders, data.shipments)
            if r.request_id in ids
        ]
        rows.sort(key=lambda row: row[0].at)
        if not rows:
            raise ValueError("no failure-free training requests")
        model = cls()
        labels = [_label(r, o, s, data.policy_at(r.at)) for r, o, s in rows]
        model.training_tokens = tuple(
            case_tokens(row[0], row[1], row[2], label) for row, label in zip(rows, labels)
        )
        model.training_times = tuple(r.at for r, _, _ in rows)
        model.training_cases = Counter(tokens[:5] for tokens in model.training_tokens)
        model.counts = [{} for _ in FIELDS]
        for tokens in model.training_tokens:
            for t, parents in enumerate(PARENTS):
                key = tuple(tokens[i] for i in parents)
                counts = model.counts[t].setdefault(key, [0] * len(VALUES[FIELDS[t]]))
                counts[VALUES[FIELDS[t]].index(tokens[t])] += 1
        model.training_manifest = _training_manifest(data, rows, labels, seed)
        model.fingerprint = _model_fingerprint(model.counts)
        return model

    def conditional(self, t, context):
        t = FIELDS.index(t) if isinstance(t, str) else t
        key = tuple(context[FIELDS[i]] for i in PARENTS[t])
        counts = np.array(self.counts[t].get(key, [0] * len(VALUES[FIELDS[t]])), dtype=float)
        return (counts + 1) / (counts.sum() + len(counts))

    def log_prob(self, tokens):
        if len(tokens) != len(FIELDS):
            raise ValueError("a likelihood needs all six tokens")
        context, total = {}, 0.0
        for t, value in enumerate(tokens):
            total += np.log(self.conditional(t, context)[VALUES[FIELDS[t]].index(value)])
            context[FIELDS[t]] = value
        return float(total)

    def draw(self, rng, tau, n, as_of):
        from refund_agent.contracts import _utc_time

        _utc_time(as_of)  # The date describes use; it is never a model feature.
        if not np.isfinite(tau) or tau <= 0 or type(n) is not int or n < 0:
            raise ValueError("positive finite temperature and nonnegative draw count required")
        draws = []
        for _ in range(n):
            context = {}
            for t, field in enumerate(FIELDS):
                scores = np.log(self.conditional(t, context)) / tau
                weights = np.exp(scores - scores.max())
                index = int(rng.choice(len(weights), p=weights / weights.sum()))
                context[field] = VALUES[field][index]
            draws.append(tuple(context.values()))
        return tuple(draws)
