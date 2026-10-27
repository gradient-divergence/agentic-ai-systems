"""Reference solution: forty fixed search trials and one open-pool comparison."""

from demos import delegated_model_demo as demo


def build(directory, seed=7):
    window, golden = demo.prepare("open")
    built = demo.build(window, budget=40, seed=seed)
    record, _ = demo.evaluate(built, golden, directory, replay=False)
    return dict(
        built=built,
        record=record,
        authority="selection only",
        defense="The record selects a score under its prespecified checks and paired cost rule. Its weakest assumption is the paid-only outcome sample: labels from refused requests are missing. Next collect a randomized, reviewed sample with matured outcomes and measure reviewer errors. A lower cost on this spent pool grants neither promotion nor permission to pay; those need the calibration record, a signed threshold experiment and the later release gates.",
    )


if __name__ == "__main__":
    import json
    import tempfile

    with tempfile.TemporaryDirectory() as directory:
        result = build(directory)
        print(json.dumps(result["record"].payload(), indent=2))
        print(result["defense"])
