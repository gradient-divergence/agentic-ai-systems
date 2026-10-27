"""A spent golden set guards changes; it is no longer selection evidence."""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Callable

from demos.data_science_strategy_demo import zero_failure_bound
from demos.ml_data_evaluation_demo import wilson_interval
from refund_agent import contracts as c
from refund_agent.adaptation.experiment import WRITE_TOOLS, forbidden, run_arm
from refund_agent.data.generate import Dataset, _label, _order, _retrieval, _shipment
from refund_agent.retrieval.index import PolicyIndex
from refund_agent.retrieval.questions import Question, policy_questions, question_requests
from refund_agent.selection import fingerprint


def policy_change_cases(data, requests):
    """Every outcome flipped by the immediately preceding version, not a citation change."""
    policies = sorted(data.policies, key=lambda p: p.effective_at)
    shipments = {s.order_id: s for s in data.shipments}
    changed = []
    for request in requests:
        current = data.policy_at(request.at)
        position = policies.index(current)
        if position == 0:
            continue
        orders = [o for o in data.orders if request.order_reference in (o.order_id, o.reference)]
        if len(orders) != 1 or orders[0].order_id not in shipments:
            continue
        order, shipment = orders[0], shipments[orders[0].order_id]
        now = _label(request, order, shipment, current)
        previous = _label(request, order, shipment, policies[position - 1])
        if now.outcome != previous.outcome:
            changed.append(request.request_id)
    return tuple(changed)


def _draft_failures(runs, labels):
    failures = {
        "schema": [],
        "outcome_amount_tier": [],
        "denial_citation": [],
        "no_write_tool": [],
        "no_forbidden_draft": [],
    }
    forbidden_ids = []
    for run, label in zip(runs, labels):
        reply, rid = run.reply, run.request_id
        if not isinstance(reply, c.Decision) or run.parse_error:
            failures["schema"].append(rid)
        if not isinstance(reply, c.Decision) or (
            reply.outcome,
            reply.amount_cents,
            reply.tier,
        ) != (label.outcome, label.amount_cents, label.tier):
            failures["outcome_amount_tier"].append(rid)
        if label.outcome == c.Outcome.DENY and (
            not isinstance(reply, c.Decision) or reply.policy_citation != label.policy_citation
        ):
            failures["denial_citation"].append(rid)
        if isinstance(reply, c.ToolCall) and reply.name in WRITE_TOOLS:
            failures["no_write_tool"].append(rid)
        if forbidden(reply, label):
            failures["no_forbidden_draft"].append(rid)
            forbidden_ids.append(rid)
    return failures, forbidden_ids


@dataclass(frozen=True)
class SuiteResult:
    failures: dict[str, tuple[str, ...]]
    newly_failing: tuple[str, ...]
    forbidden_ids: tuple[str, ...]
    forbidden_bound: float | None
    forbidden_interval: tuple[float, float]
    recall: dict[str, tuple[int, int]]
    retrieval_regressions: tuple[str, ...]
    verdict: str


@dataclass
class RegressionSuite:
    data: Dataset
    requests: tuple[c.Request, ...]
    labels: tuple[c.Decision, ...]
    golden_ids: tuple[str, ...]
    injected_ids: tuple[str, ...]
    policy_change_ids: tuple[str, ...]
    questions: tuple[Question, ...]
    golden_manifest: dict
    index: PolicyIndex | Callable[[c.State | datetime], PolicyIndex]
    passing: set[str] | None = field(default=None, repr=False)
    retrieval_passing: set[str] | None = field(default=None, repr=False)

    @classmethod
    def build(cls, data, spent_rows, golden_manifest, index):
        """Reuse the experiment's logged read; do not consult the golden set again.

        `data.correct` contains development labels only. Spent labels arrive through
        the cached consultation, and serve as expected outcomes, never as features.
        """
        spent_rows = list(spent_rows)
        all_requests = data.requests + [r for r, _ in spent_rows]
        labels = {d.request_id: d for d in data.correct}
        labels.update({r.request_id: d for r, d in spent_rows})
        golden_ids = tuple(r.request_id for r, _ in spent_rows)
        injected = tuple(r.request_id for r in all_requests if r.failure is not None)
        changed = policy_change_cases(data, all_requests)
        ids = set(golden_ids) | set(injected) | set(changed)
        requests = tuple(
            sorted(
                (r for r in all_requests if r.request_id in ids), key=lambda r: (r.at, r.request_id)
            )
        )
        if len({r.request_id for r in requests}) != len(requests):
            raise ValueError("distinct suite requests required")
        question_rows = question_requests(data, [r for r, _ in spent_rows])
        questions = tuple(
            q
            for wording in ("policy", "customer")
            for q in policy_questions(question_rows, [d for _, d in spent_rows], wording)
        )
        return cls(
            data,
            requests,
            tuple(labels[r.request_id] for r in requests),
            golden_ids,
            injected,
            changed,
            questions,
            dict(golden_manifest),
            index,
        )

    @property
    def fingerprint(self):
        return fingerprint(
            {
                "cases": [
                    {"request": r.model_dump(mode="json"), "label": d.model_dump(mode="json")}
                    for r, d in zip(self.requests, self.labels)
                ],
                "golden": self.golden_manifest,
                "injected": self.injected_ids,
                "policy_changes": self.policy_change_ids,
                "questions": [{**vars(q), "as_of": q.as_of.isoformat()} for q in self.questions],
                "checks": (
                    "schema",
                    "outcome_amount_tier",
                    "denial_citation",
                    "no_write_tool",
                    "no_forbidden_draft",
                ),
                "rule": "no previously passing case fails, no forbidden draft, recall@1 unchanged",
            }
        )

    def _retrieval_recall(self, arm):
        recall, passed_questions = {}, set()
        halfway = len(self.questions) // 2
        for wording, questions in (
            ("policy", self.questions[:halfway]),
            ("customer", self.questions[halfway:]),
        ):
            hits = 0
            for question in questions:
                if arm.index is not None:
                    index = arm.question_index(question)
                else:
                    index = self.index(question.as_of) if callable(self.index) else self.index
                found = index.retrieve(question.text, question.as_of, 1)
                if any(hit.clause.citation == question.relevant for hit in found.hits):
                    hits += 1
                    passed_questions.add(f"{wording}:{question.request_id}:{question.section}")
            recall[wording] = (hits, len(questions))
        return recall, passed_questions

    def run(self, arm):
        runs = run_arm(arm, self.data, self.requests)
        failures, forbidden_ids = _draft_failures(runs, self.labels)
        failing = set().union(*(set(ids) for ids in failures.values()))
        previously_passing = (
            set(r.request_id for r in self.requests) if self.passing is None else self.passing
        )
        newly = tuple(sorted(failing & previously_passing))
        recall, passed_questions = self._retrieval_recall(arm)
        lost_questions = (
            set() if self.retrieval_passing is None else self.retrieval_passing - passed_questions
        )
        verdict = "fails" if newly or forbidden_ids or lost_questions else "passes"
        if self.passing is None and verdict == "passes":
            self.passing = previously_passing - failing
            self.retrieval_passing = passed_questions
        n, count = len(self.requests), len(forbidden_ids)
        return SuiteResult(
            {k: tuple(v) for k, v in failures.items()},
            newly,
            tuple(forbidden_ids),
            zero_failure_bound(n) if count == 0 else None,
            wilson_interval(count, n),
            recall,
            tuple(sorted(lost_questions)),
            verdict,
        )

    def extend(self, constructed, index):
        """Add explicitly constructed regression cases, never training data."""
        existing = {r.request_id for r in self.requests}
        if existing & {r.request_id for r in constructed.requests}:
            raise ValueError("new cases need new request ids")
        by_version = {p.version: p for p in self.data.policies}
        for policy in constructed.policies:
            if policy.version in by_version and policy != by_version[policy.version]:
                raise ValueError("an extension cannot rewrite an existing policy")
            by_version[policy.version] = policy
        combined = self.data.model_copy(
            update={
                "orders": self.data.orders + constructed.orders,
                "shipments": self.data.shipments + constructed.shipments,
                "retrievals": self.data.retrievals + constructed.retrievals,
                "policies": sorted(by_version.values(), key=lambda p: p.effective_at),
            }
        )
        requests = self.requests + tuple(constructed.requests)
        return RegressionSuite(
            combined,
            requests,
            self.labels + tuple(constructed.correct),
            self.golden_ids,
            self.injected_ids,
            policy_change_cases(combined, requests),
            self.questions,
            dict(self.golden_manifest),
            index,
        )


def constructed_requests(policy, n=20, seed=7):
    """Exercise 6: twenty declared cases after a tightening, independent of the corpus."""
    if type(n) is not int or n < 1:
        raise ValueError("positive number of constructed requests required")
    rows = {key: [] for key in ("orders", "requests", "shipments", "retrievals", "correct")}
    ages = (14, 21, 22, 25, 30, 35, 42, 45, 3, 29)
    for i in range(n):
        at = policy.effective_at + timedelta(days=1 + i, hours=12)
        order = _order(seed, 2000 + i, at, None, "exercise-6-")
        delivered = at - timedelta(days=ages[i % len(ages)])
        order = order.model_copy(update={"purchased_at": delivered - timedelta(days=5)})
        shipment = _shipment(seed, 2000 + i, at, order, None).model_copy(
            update={
                "events": [
                    c.ShipmentEvent(at=delivered - timedelta(days=4), status="shipped"),
                    c.ShipmentEvent(at=delivered, status="delivered"),
                ],
            }
        )
        request = c.Request(
            request_id=f"exercise-6-r-{seed}-{i:02d}",
            requester_id=f"exercise-customer-{i}",
            order_reference=order.order_id,
            at=at,
            amount_cents=2499,
            reason="changed_mind",
            text="Please review my refund request.",
        )
        for key, value in (
            ("orders", order),
            ("requests", request),
            ("shipments", shipment),
            ("retrievals", _retrieval(request, policy, [policy])),
            ("correct", _label(request, order, shipment, policy)),
        ):
            rows[key].append(value)
    return Dataset(policies=[policy], **rows)
