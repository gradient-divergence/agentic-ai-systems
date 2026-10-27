"""Exercise 6: implement the base-rate guard before adding labels or refitting."""

from demos.supervised_learning_demo import boosted


def self_train(X, y, pool, factory=boosted, confidence=0.9, rounds=3):
    raise NotImplementedError("add the class-share guard before each refit")
