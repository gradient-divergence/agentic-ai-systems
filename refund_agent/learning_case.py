"""Written learning-fit cases beside the refund agent; no action authority is added."""

from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class FitRow(BaseModel):
    model_config = {"extra": "forbid", "frozen": True}
    question: Literal["sequential", "feedback", "choice", "safety", "value"]
    answer: Literal["yes", "no", "unknown"]
    evidence: str
    source: str

    @field_validator("evidence", "source")
    @classmethod
    def evidenced(cls, text):
        if not text.strip():
            raise ValueError("every answer needs evidence and a source")
        return text.strip()


class LearningCase(BaseModel):
    model_config = {"extra": "forbid", "frozen": True}
    decision: str = Field(min_length=1)
    rows: tuple[FitRow, ...] = Field(min_length=5, max_length=5)
    learned: tuple[str, ...]
    coded: tuple[str, ...]
    forbidden: tuple[str, ...]
    next_steps: tuple[str, ...]
    revisit_when: tuple[str, ...]
    owner: str = Field(min_length=1)
    verdict: Literal["learn", "not yet"] = "not yet"
    version: str = "1"
    date: str = "2026-10-10"

    @model_validator(mode="after")
    def checked_verdict(self):
        if len({row.question for row in self.rows}) != 5:
            raise ValueError("the five fit questions must be distinct")
        expected = verdict(self.rows)
        if "verdict" in self.model_fields_set and self.verdict != expected:
            raise ValueError("verdict contradicts the five answers")
        object.__setattr__(self, "verdict", expected)
        return self

    def render(self):
        lines = [
            f"{self.decision} (version {self.version}, {self.date})",
            "question | answer | evidence | source",
            "--- | --- | --- | ---",
        ]
        lines.extend(
            f"| {r.question} | {r.answer} | {r.evidence} | {r.source} |" for r in self.rows
        )
        for name in ("learned", "coded", "forbidden", "next_steps", "revisit_when"):
            lines.append(f"{name}: " + "; ".join(getattr(self, name)))
        lines.append(
            f"owner: {self.owner}; verdict: {self.verdict}; stage 3, autonomy unchanged: drafts only, a person executes"
        )
        return "\n".join(lines)


# region book:fnd.rl-foundations.case-01 output
if __name__ == "__main__":
    from refund_agent import learning_case


def verdict(rows):
    return (
        "learn"
        if len(rows) == 5 and all(row.answer == "yes" for row in rows)
        else "not yet"
    )


if __name__ == "__main__":
    for case in learning_case.stage3_cases():
        fields = {**case.model_dump(), "verdict": verdict(case.rows)}
        print(learning_case.LearningCase.model_validate(fields).render())
        print()
# endregion book:fnd.rl-foundations.case-01


_REFUND_ROWS = (
    FitRow(
        question="sequential",
        answer="no",
        evidence="Each request has its own customer; the generator does not condition a request on earlier desk decisions.",
        source="companion/refund_agent/data/history.py:147; 178",
    ),
    FitRow(
        question="feedback",
        answer="no",
        evidence="Only policy-payable tier 1 claims have a paid-only 30-day outcome; denied and referred outcomes are unobserved.",
        source="companion/refund_agent/data/history.py:58",
    ),
    FitRow(
        question="choice",
        answer="no",
        evidence="The fictional refund policy labels the permitted answer; a learner cannot choose to alter eligibility.",
        source="companion/refund_agent/data/generate.py:165",
    ),
    FitRow(
        question="safety",
        answer="no",
        evidence="The history supplies no evidence of the cost of changing the desk's decision on a customer.",
        source="companion/refund_agent/data/history.py:156; 137",
    ),
    FitRow(
        question="value",
        answer="no",
        evidence="The expected-cost score and fixed tier threshold already support an offline one-request comparison, without a sequential learner.",
        source="companion/refund_agent/model/history_baseline.py:91; 30",
    ),
)

_WAITING_ROWS = (
    FitRow(
        question="sequential",
        answer="yes",
        evidence="Waiting changes the next day's evidence and incurs a waiting cost; paying ends this constructed claim.",
        source="companion/demos/rl_foundations_demo.py:parcel_mdp",
    ),
    FitRow(
        question="feedback",
        answer="unknown",
        evidence="Current history stops reading the carrier at the request; it contains no post-request carrier path or customer's reaction to waiting.",
        source="companion/refund_agent/data/history.py:122",
    ),
    FitRow(
        question="choice",
        answer="unknown",
        evidence="Waiting is outside the current three outcomes; only the policy owner can decide when such a claim is payable.",
        source="companion/refund_agent/data/generate.py:165",
    ),
    FitRow(
        question="safety",
        answer="unknown",
        evidence="The price of a customer's waiting day is unmeasured; constructed costs cannot establish that waiting is safe.",
        source="companion/book_outputs/_fnd.rl-foundations.model-01.txt:2",
    ),
)

_THRESHOLD_ROWS = (
    FitRow(
        question="sequential",
        answer="no",
        evidence="A threshold changes the current disposition, not the next generated request.",
        source="companion/refund_agent/data/history.py:137",
    ),
    FitRow(
        question="feedback",
        answer="no",
        evidence="Paid-only outcomes do not establish denied or referred customers' outcomes.",
        source="companion/refund_agent/data/history.py:58",
    ),
    FitRow(
        question="choice",
        answer="unknown",
        evidence="The fixed threshold is part of the owner's rule and no authority to learn it is given.",
        source="companion/refund_agent/model/history_baseline.py:30",
    ),
    FitRow(
        question="safety",
        answer="unknown",
        evidence="No approved threshold experiment or reaction to changed decisions exists in the constructed history.",
        source="companion/refund_agent/data/history.py:156",
    ),
    FitRow(
        question="value",
        answer="unknown",
        evidence="This chapter compares sequential learning with planning, not alternative tier thresholds.",
        source="companion/book_outputs/_fnd.rl-foundations.learn-01.txt:9",
    ),
)


def _refund_case():
    return LearningCase(
        decision="Should the refund agent learn approve, deny or refer?",
        rows=_REFUND_ROWS,
        learned=("No sequential policy; the chapter 5 risk estimate is fitted offline.",),
        coded=(
            "Eligibility rules, fixed tier threshold, audit and person approval; the chapter 5 expected-cost comparison stays an offline proposal.",
        ),
        forbidden=(
            "Changing eligibility, approvals or execution authority through a learned value.",
        ),
        next_steps=(
            "Retain the existing fixed rule; treat one-request scores as supervised predictions.",
        ),
        revisit_when=(
            "A repeated-customer decision is shown to change future state, alternatives have supported feedback, and a permitted safe choice beats the fixed baseline.",
        ),
        owner="refund_cap_v4 policy owner",
    )


def _waiting_case(rows):
    return LearningCase(
        decision="Should the refund agent learn when a late-parcel claim is payable?",
        rows=rows,
        learned=(
            "Carrier probabilities could be estimated offline after follow-up logging; no live sequential policy.",
        ),
        coded=(
            "The finite model, backward planning, a fixed stopping rule if the policy owner adopts it.",
        ),
        forbidden=(
            "A fourth agent outcome, changing refund_cap_v4 without its owner, live exploration or new payment authority.",
        ),
        next_steps=(
            "Propose continued carrier reads after a not-delivered claim; estimate the model offline.",
            "Have the policy owner set the waiting price or authorize a capped test.",
            "If adopted, deploy a fixed rule and judge it on supported logs.",
        ),
        revisit_when=(
            "Post-request carrier paths exist; the owner permits waiting and its price is justified.",
            "A permitted policy beats the planned baseline under stated support and an agreed exposure cap.",
        ),
        owner="refund_cap_v4 policy owner",
    )


def _waiting_rows(result):
    summary, exact = result["summary"], result["exact"]
    return (
        *_WAITING_ROWS,
        FitRow(
            question="value",
            answer="no",
            evidence=f"Exploring at planned values costs {exact['extra_cost']:.6f} extra per claim; planning from 200 followed claims costs {summary['log_gap_mean']:.6f} extra on average in 200 simulated logs.",
            source="companion/book_outputs/_fnd.rl-foundations.learn-01.txt:5; 9",
        ),
    )


def stage3_cases():
    from demos.rl_foundations_demo import learning_experiment

    return [_refund_case(), _waiting_case(_waiting_rows(learning_experiment()))]


def threshold_case():
    """Instructor reference for exercise 6, citing source without reading a dataset."""
    return LearningCase(
        decision="Should the refund agent learn its tier 1 threshold from outcomes?",
        rows=_THRESHOLD_ROWS,
        learned=("A supervised risk score, separately from the permission threshold.",),
        coded=("The current fixed threshold, eligibility checks and approval boundary.",),
        forbidden=("Changing the policy or executing a payment through a fitted score.",),
        next_steps=(
            "Ask the owner to specify the decision cost and an authorized evaluation before considering a threshold change.",
        ),
        revisit_when=(
            "The owner authorizes the choice, feedback supports it, and a safe candidate beats the current rule.",
        ),
        owner="refund_cap_v4 policy owner",
    )
