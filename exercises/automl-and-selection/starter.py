"""Complete the chapter's build, then select this file in the acceptance tests."""

from demos.delegated_model_demo import commission_for, prepare, random_search


def build(directory, seed=7):
    """Return the evidence fields named in test_acceptance.py; preserve source data."""
    raise NotImplementedError("complete the build and its report")


# The search call and its fixed budget are provided; complete the checks,
# sole frozen-pool read, selection record and written defense around it.


def search(seed=7):
    window, golden = prepare("open")
    commission = commission_for(window, budget=40, seed=seed)
    events = [{"event": "commission", "sha256": commission.sha256}]
    searched, trials, winner = random_search(window, commission, events)
    return window, golden, commission, events, searched, trials, winner
