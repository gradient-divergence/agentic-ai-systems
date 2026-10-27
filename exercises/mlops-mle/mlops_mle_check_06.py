"""Check the exercise's stipulated evidence; never issue runtime admission.

The exercise supplies owner-approved prerequisites as a premise. This command
tests which additional evidence a table asks for; it neither signs a promotion
record nor writes the runtime registry. The production command additionally
rebuilds the exact candidate and validates its journaled release evidence.
"""

import argparse
import importlib.util
import json
from pathlib import Path

from refund_agent.promotion.mlops_mle_bundle import VersionBundle
from refund_agent.promotion.mlops_mle_diff import diff


def check(candidate, incumbent, evidence, required):
    changed = diff(incumbent, candidate)
    needs = set(required(changed))
    met = evidence["by_bundle"][candidate.fingerprint]
    unmet = sorted(name for name in needs if name not in met)
    return {
        "scenario": evidence["scenario"],
        "verdict": "held" if unmet else "promoted",
        "unmet": unmet,
        "authority": "none; exercise premises only",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--requirements", type=Path, default=Path(__file__).with_name("requirements_06.py")
    )
    parser.add_argument(
        "--candidate", type=Path, default=Path(__file__).with_name("candidate_06.json")
    )
    args = parser.parse_args(argv)
    root = Path(__file__).parent
    spec = importlib.util.spec_from_file_location(
        "mlops_mle_exercise_requirements", args.requirements
    )
    if spec is None or spec.loader is None:
        raise ValueError("requirements module unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    candidate = VersionBundle.from_record(json.loads(args.candidate.read_text()))
    incumbent = VersionBundle.from_record(json.loads((root / "incumbent-05.json").read_text()))
    evidence = json.loads((root / "mlops_mle_evidence_06.json").read_text())
    result = check(candidate, incumbent, evidence, module.required)
    print(json.dumps(result, sort_keys=True))
    return int(bool(result["unmet"]))


if __name__ == "__main__":
    raise SystemExit(main())
