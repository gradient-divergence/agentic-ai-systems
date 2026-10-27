"""An expected extra-contact budget and the support left by an exposure cap."""

from demos.exploration_bandits_demo import cap_from_budget as cap_from_budget


def propensity_floor(cap, epsilon=1.0, offers=3):
    if not 0 <= cap <= 1 or not 0 <= epsilon <= 1 or type(offers) is not int or offers < 1:
        raise ValueError("valid cap, epsilon and positive offer count required")
    return cap * epsilon / offers
