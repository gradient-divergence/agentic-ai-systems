"""Reference solution: a parent-bound September slice and per-tier intervals."""

from datetime import datetime, timedelta, timezone

from demos import ml_data_evaluation_demo as demo
from refund_agent.data.generate import generate
from refund_agent.labels import LABEL_POLICY
from refund_agent.split import CUTS, freeze, time_split


def build(directory, seed=7):
    data = generate(seed)
    windows = time_split(data.requests, CUTS, timedelta(days=LABEL_POLICY.window_days))
    parent = freeze(data, windows["golden"], LABEL_POLICY)
    parent.persist(directory)
    parent.consult("parent report", "baseline and classifier")
    rows = [r for r in parent.requests if r.at >= datetime(2026, 9, 1, tzinfo=timezone.utc)]
    child = freeze(data, rows, LABEL_POLICY, name="golden-september", parent=parent)
    read = child.consult("September report", "per-tier baseline errors, recall and precision")
    asked = [r for r, _ in read]
    training = windows["training"]
    labels = {d.request_id: d for d in data.correct}
    flags = demo.fit_and_predict(
        demo.decision_time_inputs(data, training),
        [demo.payment_forbidden(labels[r.request_id]) for r in training],
        demo.decision_time_inputs(data, asked),
        demo.INPUTS,
    )
    drafts = demo.baseline_decisions(data, asked)
    counted = demo.tier_counts(read, drafts, dict(zip((r.request_id for r in asked), flags)))

    def rate(k, n):
        return (k / n, demo.wilson_interval(k, n)) if n else (None, None)

    report = [
        dict(
            tier=tier,
            requests=n,
            errors=errors,
            tp=tp,
            fp=fp,
            fn=fn,
            error_rate=rate(errors, n),
            recall=rate(tp, tp + fn),
            precision=rate(tp, tp + fp),
        )
        for tier, n, errors, tp, fp, fn in counted
    ]
    return {"parent": parent, "slice": child, "report": report}


if __name__ == "__main__":
    import json
    import tempfile

    with tempfile.TemporaryDirectory() as directory:
        result = build(directory)
        print(
            json.dumps(
                {
                    "manifest": result["slice"].manifest,
                    "report": result["report"],
                    "log": result["slice"].log,
                },
                indent=2,
            )
        )
