"""Schema and policy checks; provenance and a sample for a person's later review."""

import numpy as np

from refund_agent.labels import _label
from refund_agent.scenarios.materialize import ScenarioRecords, to_records
from refund_agent.split import CUTS


def check_schema(tokens, as_of, seed):
    return to_records(tokens, as_of, seed)


def policy_label(records: ScenarioRecords, data):
    return _label(*records, data.policy_at(records.request.at))


def check_policy(tokens, records, data):
    label = policy_label(records, data)
    return label, tokens[-1] == label.outcome.value


def check_provenance(suite, model, before=(), after=()):
    manifest = suite.manifest
    required = {
        "as_of",
        "policy_version",
        "tau",
        "seed",
        "number_drawn",
        "model_sha256",
        "training_sha256",
        "training",
        "counts",
        "origin",
        "golden_reads",
        "human_review",
    }
    return (
        required <= manifest.keys()
        and before == after
        and manifest["golden_reads"] == 0
        and all(at < CUTS[0] for at in model.training_times)
        and manifest["training"] == model.training_manifest
        and manifest["training_sha256"] == model.training_manifest["sha256"]
        and manifest["model_sha256"] == model.fingerprint
        and manifest["origin"] in ("listed", "generated")
        and all(
            row.origin == manifest["origin"]
            and row.records.request.at.isoformat() == manifest["as_of"]
            and f"@{manifest['policy_version']}#" in row.label.policy_citation
            for row in suite.scenarios
        )
    )


def review_sample(scenarios, k, seed):
    """Select without replacement. Selection is not a completed human review."""
    if type(k) is not int or not 0 <= k <= len(scenarios):
        raise ValueError("review sample size must fit the kept set")
    indices = np.random.default_rng(seed).choice(len(scenarios), size=k, replace=False)
    return tuple(scenarios[int(i)] for i in indices)
