"""Exercise 6: construct Candidate C before the evaluator reads outcomes.

Complete candidate_c, then run the acceptance test against this submission.
The provided evaluator writes the pre-read rule, record and rendered output.
"""

def candidate_c(frozen_at):
    """Return the fixed Policy specified by candidate_c.json, including frozen_at."""
    raise NotImplementedError("construct Candidate C's fixed supported-cell policy")


def evaluate_candidate_c(output_dir):
    from demos.practical_rl_demo import evaluate_candidate_c as run_evaluation

    return run_evaluation(output_dir, candidate_c)


if __name__ == "__main__":
    print(evaluate_candidate_c("candidate-c-output").render())
