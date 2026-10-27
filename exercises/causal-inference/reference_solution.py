"""Reference solution: a fifteen-dollar assignment rehearsal, with no execution."""

from dataclasses import replace

from demos import causal_inference_demo as demo


def build(directory, seed=7):
    planning = demo.planning_history(seed)
    plan = replace(demo.build_plan(planning, cap=15, difference=2), seed=seed)
    demo.check_plan(plan)
    rows, scores = planning["validation"], planning["score"]
    records = demo.rehearse(plan, rows, scores)
    size = demo.sizing(plan, rows, scores)
    return dict(
        plan=plan,
        rows=rows,
        scores=scores,
        records=records,
        size=size,
        frozen_logs_before=[],
        frozen_logs_after=[],
        executions=0,
        reviews_opened=0,
    )


if __name__ == "__main__":
    import json
    import tempfile

    with tempfile.TemporaryDirectory() as directory:
        result = build(directory)
        selected = [r for r in result["records"] if r["in_set"]]
        print(
            json.dumps(
                {
                    **result["size"],
                    "refer": sum(r["action"] == "refer" for r in selected),
                    "executions": result["executions"],
                },
                indent=2,
            )
        )
