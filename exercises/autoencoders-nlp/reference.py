"""Reference solution for testing exercise 6's learner harness.

Run test_acceptance.py with AUTOENCODERS_NLP_SOLUTION set to this file.
"""

import runpy
from datetime import datetime, timedelta, timezone
from pathlib import Path

from refund_agent.retrieval.clauses import policy_clauses
from refund_agent.retrieval.encoders import CountEncoder
from refund_agent.retrieval.index import PolicyIndex
from refund_agent.retrieval.questions import Question

CHANGE = datetime(2026, 10, 1, tzinfo=timezone.utc)
print_hits = runpy.run_path(str(Path(__file__).with_name("starter.py")))["print_hits"]


def extend_policies(policies):
    latest = policies[-1]
    sixth = latest.model_copy(
        update={
            "version": "2026-10-01",
            "effective_at": CHANGE,
            "return_days": 60,
            "text": latest.text.replace("2026-09-01", "2026-10-01").replace(
                "within 45 days", "within 60 days"
            ),
        }
    )
    return [*policies, sixth]


def boundary_questions():
    return [
        Question(
            f"exercise-6-{i}",
            "eligibility",
            "damaged 55 days after delivery",
            CHANGE + timedelta(seconds=i - 10),
            f"refund_cap_v4@{'2026-09-01' if i < 10 else '2026-10-01'}#eligibility",
        )
        for i in range(20)
    ]


def build_index(policies, observed_at, rarity=False):
    texts = [clause.text for clause in policy_clauses(policies, observed_at)]
    encoder = CountEncoder(rarity=rarity).fit(texts)
    return PolicyIndex.build(policies, observed_at, encoder)
