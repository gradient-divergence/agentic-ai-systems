"""Reference solution for exercise 7, isolated from the canonical six-token fit.

Run test_acceptance.py with GENERATIVE_MODELS_SOLUTION set to this file, plus
test_reference.py and the unchanged tests/refund_agent/test_scenarios.py.
Run this file with --write-report to save both manifests and run-07.json.
The seventh token is independent of the first six, with add-one smoothing on
failure-free training rows. It is a scenario label, never release authority.
"""

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from refund_agent import contracts as c
from refund_agent.data.generate import FAILURES
from refund_agent.scenarios.checks import check_provenance, policy_label
from refund_agent.scenarios.materialize import SchemaError, to_records
from refund_agent.scenarios.model import ScenarioModel, sha256, training_data
from refund_agent.scenarios.suite import Scenario, ScenarioSet

FAILURE_VALUES = ("none", *FAILURES)
AS_OF = datetime(2026, 9, 15, tzinfo=timezone.utc)


class FailureModel(ScenarioModel):
    @classmethod
    def fit(cls, data=None, seed=7):
        model = super().fit(data, seed)
        six_token_fingerprint = model.fingerprint
        model.counts.append({(): [len(model.training_tokens), *([0] * len(FAILURES))]})
        model.fingerprint = sha256(
            {
                "six_token_model": six_token_fingerprint,
                "failure_values": FAILURE_VALUES,
                "failure_parents": [],
                "failure_counts": model.counts[6][()],
            }
        )
        return model

    def conditional(self, t, context):
        if t == 6 or t == "failure":
            counts = np.asarray(self.counts[6][()], dtype=float)
            return (counts + 1) / (counts.sum() + len(counts))
        return super().conditional(t, context)

    def log_prob(self, tokens):
        if len(tokens) != 7 or tokens[6] not in FAILURE_VALUES:
            raise ValueError("a likelihood needs six case tokens and a valid failure")
        return super().log_prob(tokens[:6]) + float(
            np.log(self.conditional(6, {})[FAILURE_VALUES.index(tokens[6])])
        )

    def draw(self, rng, tau, n, as_of):
        # Draw the original six first, preserving their seeded proposals too.
        cases = super().draw(rng, tau, n, as_of)
        scores = np.log(self.conditional(6, {})) / tau
        weights = np.exp(scores - scores.max())
        failures = rng.choice(len(FAILURE_VALUES), size=n, p=weights / weights.sum())
        return tuple((*case, FAILURE_VALUES[int(i)]) for case, i in zip(cases, failures))


def build_suite(as_of, tau, n=200, seed=7):
    as_of = c._utc_time(as_of)
    data = training_data(7)
    model = FailureModel.fit(data, seed=7)
    draws = model.draw(np.random.default_rng(seed), tau, n, as_of)
    scenarios, rejected = [], []
    overruled = 0
    for draw, tokens in enumerate(draws, 1):
        try:
            records = to_records(tokens[:6], as_of, seed)
        except SchemaError as error:
            rejected.append({"draw": draw, "tokens": list(tokens), "reason": str(error)})
            continue
        failure = None if tokens[6] == "none" else tokens[6]
        records = records._replace(request=records.request.model_copy(update={"failure": failure}))
        label = policy_label(records, data)
        proposed = tokens[5]
        overruled += proposed != label.outcome.value
        scenarios.append(Scenario(tokens, records, label, proposed, "generated", draw))
    distinct = {row.tokens[:5] for row in scenarios}
    manifest = {
        "as_of": as_of.isoformat(),
        "policy_version": data.policy_at(as_of).version,
        "tau": tau,
        "seed": seed,
        "number_drawn": len(draws),
        "model_sha256": model.fingerprint,
        "training_sha256": model.training_manifest["sha256"],
        "training": model.training_manifest,
        "failure_values": list(FAILURE_VALUES),
        "counts": {
            "proposed": len(draws),
            "rejected": len(rejected),
            "overruled": overruled,
            "distinct": len(distinct),
            "unseen_in_training": len(distinct - model.training_cases.keys()),
            "kept": len(scenarios),
        },
        "origin": "generated",
        "golden_reads": 0,
        "human_review": {"status": "pending", "sample_size": min(5, len(scenarios)), "seed": 31},
    }
    suite = ScenarioSet(tuple(scenarios), manifest, tuple(rejected), model)
    if not check_provenance(suite, model):
        raise ValueError("scenario provenance failed")
    return suite


def regenerate(manifest):
    suite = build_suite(
        datetime.fromisoformat(manifest["as_of"]),
        manifest["tau"],
        manifest["number_drawn"],
        manifest["seed"],
    )
    if suite.manifest != manifest:
        raise ValueError("regenerated manifest differs")
    return suite


def check_report(suites):
    original = ScenarioModel.fit()
    data = training_data()
    runs = []
    for suite in suites:
        checks = {
            "seven_tokens": all(len(row.tokens) == 7 for row in suite.scenarios),
            "independent_labels": all(
                row.label == policy_label(row.records, data) for row in suite.scenarios
            ),
            "failure_referrals": all(
                row.label.outcome.value == "refer"
                and row.label.reason == row.records.request.failure
                for row in suite.scenarios
                if row.records.request.failure is not None
            ),
            "failure_draws": any(
                row.records.request.failure is not None for row in suite.scenarios
            ),
            "six_token_counts_unchanged": suite.model.counts[:6] == original.counts,
            "training_manifest_unchanged": suite.model.training_manifest
            == original.training_manifest,
            "golden_reads_zero": suite.manifest["golden_reads"] == 0,
            "regenerates_byte_for_byte": regenerate(suite.manifest).to_json().encode()
            == suite.to_json().encode(),
        }
        if not all(checks.values()):
            raise AssertionError(checks)
        runs.append(
            {
                "tau": suite.manifest["tau"],
                "sha256": suite.fingerprint,
                "counts": suite.manifest["counts"],
                "checks": checks,
            }
        )
    return {"exercise": 7, "runs": runs, "human_review": "pending"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-report", action="store_true")
    args = parser.parse_args()
    suites = [build_suite(AS_OF, tau, 200, 7) for tau in (1, 2)]
    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "command": " ".join([sys.executable, *sys.argv]),
        **check_report(suites),
    }
    if args.write_report:
        directory = Path(__file__).parent
        for suite in suites:
            path = directory / f"manifest-tau{suite.manifest['tau']}.json"
            path.write_text(
                json.dumps(suite.manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
        (directory / "run-07.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
    else:
        print(json.dumps(report, indent=2))
