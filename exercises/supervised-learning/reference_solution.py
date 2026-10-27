"""Reference solution: fit, map, then consult August once and judge its gate."""

import numpy as np

from demos import supervised_learning_demo as demo


def build(directory, seed=7):
    rows = [r for r in demo.generate_history(seed=seed) if r.paid]
    windows = demo.windows(rows)
    golden = demo.frozen(windows.pop("test"), "history-august-exercise")
    golden.persist(directory)
    Xt, yt, _ = demo.arrays(windows["training"])
    Xv, yv, _ = demo.arrays(windows["validation"])
    model = demo.boosted(class_weight="balanced").fit(Xt, yt)
    events = ["fit training"]
    mapping, _, _ = demo.platt(demo.risk(model, Xv), yv)
    events.append("fit validation map")
    X, y, amount = demo.consulted(golden, "exercise gate after fitting the validation map")
    events.append("read August")
    score = mapping(demo.risk(model, X))
    threshold = round(demo.REVIEW / 49.99, 3)
    pay = demo.gate(score, threshold=threshold)
    cost = float(demo.realized_cost(y, amount, pay).mean())
    placeholder = float(demo.realized_cost(y, amount, np.ones(len(y), dtype=bool)).mean())
    return dict(
        golden=golden,
        events=events,
        threshold=threshold,
        requests=len(y),
        calibration_error=demo.calibration_error(demo.reliability(y, score)),
        referred=int((~pay).sum()),
        cost=cost,
        placeholder_cost=placeholder,
        saving=placeholder - cost,
        actions=["pay" if p else "refer" for p in pay],
        judgment=f"Propose a measured rehearsal, not payment authority: saving {placeholder - cost:.6f} dollars per request assumes accurate review.",
        measurement="Measure reviewers' false-denial rate; reverse the proposal when expected review cost plus wrong-denial harm exceeds paying.",
    )


if __name__ == "__main__":
    import json
    import tempfile

    with tempfile.TemporaryDirectory() as directory:
        report = build(directory)
        print(
            json.dumps(
                {k: v for k, v in report.items() if k not in ("golden", "actions")}, indent=2
            )
        )
