"""A complete register copy, trusted only within the register's notice promise."""

from datetime import timedelta

from refund_agent.contracts import _utc_time
from refund_agent.memory.register import project
from refund_agent.retrieval.encoders import CountEncoder
from refund_agent.retrieval.index import PolicyIndex, Retrieval

NOTICE = timedelta(
    days=7
)  # Constructed: announcements recorded at the same instant, seven days early.
MONTHLY_READ_DAY = 20  # Constructed demonstration schedule, at 00:00 UTC.


class PolicyMemory:
    def __init__(self, register, at, notice=NOTICE):
        if notice.total_seconds() < 0:
            raise ValueError("notice must be nonnegative")
        self.register, self.notice = register, notice
        self.refresh(at)

    def refresh(self, at):
        snapshot = self.register.snapshot(at)
        policies = project(snapshot, snapshot.at, live=True)
        if not policies:
            raise ValueError("no policy recorded")
        encoder = CountEncoder().fit([p.text for p in policies])
        self.index = PolicyIndex.build(policies, snapshot.at, encoder)
        self.snapshot = snapshot
        self.observed_at = snapshot.at

    def retrieve(self, question, at, k):
        if type(k) is not int or k < 1:
            raise ValueError("k must be a positive integer")
        age = _utc_time(at) - self.observed_at
        if age < timedelta(0):
            return Retrieval(refusal="future_policy")
        if age > self.notice:
            return Retrieval(refusal="beyond_notice")
        return self.index.retrieve(question, at, k)
