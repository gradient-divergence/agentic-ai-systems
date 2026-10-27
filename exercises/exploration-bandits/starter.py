"""Exercise 6: add the amount band to the reason before replaying both caps.

Run the acceptance tests with EXPLORATION_BANDITS_SOLUTION pointing to this file.
The unfinished starter deliberately fails; these tests are outside the default suite.
"""

from demos.exploration_bandits_demo import exercise_six_runs as run_replay


def amount_context(row):
    """Return reason|under25 below 2500 cents, otherwise reason|25plus."""
    raise NotImplementedError("add the decision-time amount band beside the reason")


def exercise_six_runs(seed=7):
    return run_replay(seed, context_key=amount_context)
