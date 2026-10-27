"""Constructed agent pipeline: one label-built statistic, and its honest refit.

The bad version includes each row's outcome and validation outcomes in its input.
The clean version fits on training only and leaves a training row's own label out.
Neither version has a handle to a frozen month.
"""

from collections import defaultdict

import numpy as np
from sklearn.linear_model import LogisticRegression

from demos.supervised_learning_demo import arrays
from refund_agent.selection import fingerprint

GROUP_INPUTS = ("reason", "prior_refund_count", "delivery_age_days", "amount_cents")


def group(row):
    return (row.reason, row.prior_refund_count > 0, row.delivery_age_days, row.amount_cents // 500)


class GroupRisk:
    """A label-built group rate followed by a fitted logistic regression."""

    def __init__(self, clean=False, C=1.0, seed=7):
        self.clean, self.C, self.seed = clean, C, seed

    def fit(self, training, validation):
        source = training if self.clean else training + validation
        self.counts = defaultdict(lambda: [0, 0])
        self.labels = {}
        for row in source:
            cell = self.counts[group(row)]
            cell[0] += row.found_false_30d
            cell[1] += 1
            self.labels[row.request_id] = row.found_false_30d
        self.prior = float(arrays(training)[1].mean())
        self.training_count = len(training)
        self.training_wrong = sum(r.found_false_30d for r in training)
        self.source_ids = tuple(r.request_id for r in source)
        X = self.features(training, leave_out=self.clean)
        self.model = LogisticRegression(C=self.C, max_iter=1000, random_state=self.seed)
        self.model.fit(X, arrays(training)[1])
        return self

    def features(self, rows, leave_out=False):
        values = []
        for row in rows:
            wrong, n = self.counts.get(group(row), (0, 0))
            prior = self.prior
            if leave_out and row.request_id in self.labels:
                wrong -= self.labels[row.request_id]
                n -= 1
                prior = (self.training_wrong - self.labels[row.request_id]) / (
                    self.training_count - 1
                )
            values.append((wrong + prior) / (n + 1))
        return np.array(values).reshape(-1, 1)

    def predict(self, rows):
        # For a seen training row the honest version retains leave-one-out encoding.
        return self.model.predict_proba(self.features(rows, leave_out=self.clean))[:, 1]

    def provenance(self):
        return {
            "inputs": list(GROUP_INPUTS) + ([] if self.clean else ["found_false_30d"]),
            "fit_rows": len(self.source_ids),
            "row_sha256": fingerprint(self.source_ids),
            "own_label_left_out": self.clean,
            "statistics_sha256": fingerprint(
                {
                    "cells": sorted((list(key), value) for key, value in self.counts.items()),
                    "prior": self.prior,
                }
            ),
        }
