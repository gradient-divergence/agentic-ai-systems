"""Reference solution for Exercise 6's fixed, supported-cell Candidate C."""

import json
from pathlib import Path

from refund_agent.offers.notices import OFFERS
from refund_agent.offpolicy.policies import Policy


def candidate_c(frozen_at):
    specification = json.loads(Path(__file__).with_name("candidate_c.json").read_text())
    return Policy.deterministic(
        specification["version"], specification["choices"], OFFERS, frozen_at=frozen_at
    )


def evaluate_candidate_c(output_dir):
    from demos.practical_rl_demo import evaluate_candidate_c as run_evaluation

    return run_evaluation(output_dir, candidate_c)


if __name__ == "__main__":
    print(evaluate_candidate_c("candidate-c-output").render())
