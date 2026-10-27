"""A deterministic handoff with independent labels, proposals and rejection reasons."""

import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from refund_agent import contracts as c
from refund_agent.data.generate import generate
from refund_agent.labels import LABEL_POLICY
from refund_agent.scenarios.checks import (
    check_policy,
    check_provenance,
    check_schema,
    policy_label,
    review_sample,
)
from refund_agent.scenarios.materialize import ScenarioRecords, SchemaError
from refund_agent.scenarios.model import ScenarioModel, sha256, training_data
from refund_agent.scenarios.tokens import valid_cases
from refund_agent.split import CUTS, freeze, time_split


@dataclass(frozen=True)
class Scenario:
    tokens: tuple
    records: ScenarioRecords
    label: c.Decision
    generator_decision: str | None
    origin: str
    draw: int

    def record(self):
        return {
            "draw": self.draw,
            "tokens": list(self.tokens),
            "request": self.records.request.model_dump(mode="json"),
            "order": self.records.order.model_dump(mode="json"),
            "shipment": self.records.shipment.model_dump(mode="json"),
            "label": self.label.model_dump(mode="json"),
            "generator_decision": self.generator_decision,
            "origin": self.origin,
        }


@dataclass
class ScenarioSet:
    scenarios: tuple[Scenario, ...]
    manifest: dict
    rejected: tuple[dict, ...]
    model: ScenarioModel

    @property
    def fingerprint(self):
        return sha256(self.record())

    def record(self):
        return {
            "manifest": self.manifest,
            "scenarios": [row.record() for row in self.scenarios],
            "rejected": list(self.rejected),
        }

    def to_json(self):
        return (
            json.dumps(
                {**self.record(), "sha256": self.fingerprint},
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        )

    def write(self, path):
        Path(path).write_text(self.to_json(), encoding="utf-8")

    @classmethod
    def read(cls, path):
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        model = ScenarioModel.fit(seed=value["manifest"]["training"]["seed"])
        rows = []
        for row in value["scenarios"]:
            # JSON validation is deliberate: the contracts reject Python date strings.
            records = ScenarioRecords(
                c.Request.model_validate_json(json.dumps(row["request"])),
                c.Order.model_validate_json(json.dumps(row["order"])),
                c.Shipment.model_validate_json(json.dumps(row["shipment"])),
            )
            rows.append(
                Scenario(
                    tuple(row["tokens"]),
                    records,
                    c.Decision.model_validate_json(json.dumps(row["label"])),
                    row["generator_decision"],
                    row["origin"],
                    row["draw"],
                )
            )
        suite = cls(tuple(rows), value["manifest"], tuple(value["rejected"]), model)
        if suite.fingerprint != value["sha256"] or not check_provenance(suite, model):
            raise ValueError("suite fingerprint or provenance differs")
        return suite

    def review_sample(self, k, seed):
        return review_sample(self.scenarios, k, seed)


def _check_draws(draws, as_of, seed, origin, data):
    scenarios, rejected = [], []
    overruled = 0
    for i, tokens in enumerate(draws, 1):
        try:
            records = check_schema(tokens, as_of, seed)
        except SchemaError as error:
            rejected.append({"draw": i, "tokens": list(tokens), "reason": str(error)})
            continue
        if origin == "generated":
            label, agreed = check_policy(tokens, records, data)
            overruled += not agreed
            proposed = tokens[-1]
        else:
            label, proposed = policy_label(records, data), None
            tokens = (*tokens, label.outcome.value)
        scenarios.append(Scenario(tuple(tokens), records, label, proposed, origin, i))
    return scenarios, rejected, overruled


def _scenario_counts(draws, scenarios, rejected, overruled, model):
    distinct = {row.tokens[:5] for row in scenarios}
    return {
        "proposed": len(draws),
        "rejected": len(rejected),
        "overruled": overruled,
        "distinct": len(distinct),
        "unseen_in_training": len(distinct - model.training_cases.keys()),
        "kept": len(scenarios),
    }


def _suite(as_of, tau, n, seed, origin):
    as_of = c._utc_time(as_of)
    corpus = generate(7)
    splits = time_split(corpus.requests, CUTS, timedelta(days=LABEL_POLICY.window_days))
    golden = freeze(corpus, splits["golden"], LABEL_POLICY)
    data = training_data(7)
    model = ScenarioModel.fit(data, seed=7)
    draws = (
        valid_cases()
        if origin == "listed"
        else model.draw(np.random.default_rng(seed), tau, n, as_of)
    )
    scenarios, rejected, overruled = _check_draws(draws, as_of, seed, origin, data)
    counts = _scenario_counts(draws, scenarios, rejected, overruled, model)
    manifest = {
        "as_of": as_of.isoformat(),
        "policy_version": data.policy_at(as_of).version,
        "tau": tau,
        "seed": seed,
        "number_drawn": len(draws),
        "model_sha256": model.fingerprint,
        "training_sha256": model.training_manifest["sha256"],
        "training": model.training_manifest,
        "counts": counts,
        "origin": origin,
        "golden_reads": len(golden.log),
        "human_review": {"status": "pending", "sample_size": min(5, len(scenarios)), "seed": 31},
    }
    suite = ScenarioSet(tuple(scenarios), manifest, tuple(rejected), model)
    if not check_provenance(suite, model):
        raise ValueError("scenario provenance failed")
    return suite


def build_suite(as_of, tau=1, n=200, seed=7):
    return _suite(as_of, tau, n, seed, "generated")


def list_suite(as_of):
    return _suite(as_of, None, len(valid_cases()), 7, "listed")


def regenerate(manifest):
    as_of = datetime.fromisoformat(manifest["as_of"])
    if manifest["origin"] == "listed":
        suite = list_suite(as_of)
    elif manifest["origin"] == "generated":
        suite = build_suite(as_of, manifest["tau"], manifest["number_drawn"], manifest["seed"])
    else:
        raise ValueError("unknown manifest origin")
    if suite.manifest != manifest:
        raise ValueError("regenerated manifest differs")
    return suite
