"""Reference solution: extend the constructed policy-change regression suite."""

from dataclasses import replace
from datetime import datetime, timezone

from demos.adaptation_demo import build_experiment
from demos.ml_data_evaluation_demo import wilson_interval
from refund_agent.adaptation.arms import dated_index
from refund_agent.adaptation.experiment import agreement, forbidden
from refund_agent.adaptation.suite import constructed_requests


def build(directory, seed=7):
    report = build_experiment(seed)
    latest = report["data"].policies[-1]
    policy = latest.model_copy(
        update={
            "version": "2026-11-01",
            "effective_at": datetime(2026, 11, 1, tzinfo=timezone.utc),
            "return_days": 21,
            "text": latest.text.replace("2026-09-01", "2026-11-01").replace(
                "within 45 days", "within 21 days"
            ),
        }
    )
    future = constructed_requests(policy, 20, seed)
    index = dated_index(report["data"].policies + [policy], report["encoder"])
    extended = report["suite"].extend(future, index)
    from refund_agent.adaptation.experiment import run_arm

    prompt = replace(report["arms"]["prompt"], index=index)
    adapter = replace(report["arms"]["adapter"], index=index)
    prompt_rows = run_arm(prompt, extended.data, future.requests)
    adapter_rows = run_arm(adapter, extended.data, future.requests)
    ids = [
        r.request_id for r, label in zip(adapter_rows, future.correct) if forbidden(r.reply, label)
    ]
    training = [r.request_id for r in report["training_drafts"]]
    return dict(
        policy=policy,
        requests=future.requests,
        prompt_correct=sum(agreement(prompt_rows, future.correct)),
        forbidden_ids=ids,
        forbidden_interval=wilson_interval(len(ids), len(future.requests)),
        before_fingerprint=report["suite"].fingerprint,
        after_fingerprint=extended.fingerprint,
        before_manifest=report["golden"].manifest,
        after_manifest=extended.golden_manifest,
        training_ids_before=training,
        training_ids_after=training.copy(),
        policy_change_ids=extended.policy_change_ids,
        suite_request_ids=[r.request_id for r in extended.requests],
    )


if __name__ == "__main__":
    import json
    import tempfile

    with tempfile.TemporaryDirectory() as directory:
        report = build(directory)
        print(
            json.dumps(
                {
                    k: report[k]
                    for k in (
                        "prompt_correct",
                        "forbidden_ids",
                        "forbidden_interval",
                        "before_fingerprint",
                        "after_fingerprint",
                    )
                },
                indent=2,
            )
        )
