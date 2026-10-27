"""Declared answers and an offline risk signal, never authority to execute a refund."""

import math
from typing import Literal

from pydantic import Field, model_validator

from refund_agent.contracts import Record
from refund_agent.data.history import _false_probability


class TypedQuestion(Record):
    name: str = Field(min_length=1)
    kind: Literal["choice", "levels", "true_false"]
    answers: tuple[str, ...]
    cannot_tell: bool = Field(default=False, strict=True)

    @model_validator(mode="after")
    def declared_type(self):
        if len(self.answers) < 2 or len(set(self.answers)) != len(self.answers):
            raise ValueError("declare at least two distinct answers")
        if any(not answer.strip() or answer == "cannot_tell" for answer in self.answers):
            raise ValueError("name each answer; declare cannot_tell separately")
        if self.kind == "true_false" and len(self.answers) != 2:
            raise ValueError("true_false requires exactly two answers")
        return self


class TypedAnswer(Record):
    question: TypedQuestion
    probabilities: dict[str, float] | None = None
    cannot_tell: bool = Field(default=False, strict=True)
    model_version: str = Field(min_length=1)

    @model_validator(mode="after")
    def valid_answer(self):
        if self.cannot_tell:
            if not self.question.cannot_tell:
                raise ValueError("cannot_tell was not declared by the question")
            if self.probabilities is not None:
                raise ValueError("a refusal carries no probability vector")
        else:
            if self.probabilities is None or set(self.probabilities) != set(self.question.answers):
                raise ValueError("probabilities must cover exactly the declared answers")
            values = tuple(self.probabilities.values())
            if any(not math.isfinite(p) or p < 0 or p > 1 for p in values):
                raise ValueError("probabilities must be finite and between zero and one")
            if abs(math.fsum(values) - 1) > 1e-9:
                raise ValueError("probabilities must sum to one within 1e-9")
        return self


RISK_QUESTION = TypedQuestion(
    name="Is paying this tier 1 refund a wrong payment?",
    kind="true_false",
    answers=("sound", "wrong"),
    cannot_tell=True,
)

PAY_EDGE, THIRD_EDGE = .080, .80
ACTION_MAP = {
    "pay": "pay_if_gate_allows",
    "review": "refer",
    "third": "refer",
    "cannot_tell": "refer",
}


def band(p):
    """Cost ties go to review. The third band begins strictly above 0.80."""
    if not math.isfinite(p) or not 0 <= p <= 1:
        raise ValueError("risk must be a probability")
    return "pay" if p < PAY_EDGE else "third" if p > THIRD_EDGE else "review"


def risk_answer(p, amount_cents, model_version):
    """Amounts of $50 or more lie outside the training population."""
    if type(amount_cents) is not int or amount_cents <= 0:
        raise ValueError("amount must be positive integer cents")
    return TypedAnswer(
        question=RISK_QUESTION,
        probabilities=None if amount_cents >= 5000 else {"sound": 1 - p, "wrong": p},
        cannot_tell=amount_cents >= 5000,
        model_version=model_version,
    )


class RiskSignalRecord(Record):
    """Question, bands, action map, retest, conformal check and their versions."""

    question: TypedQuestion
    model_version: str
    model_sha256: str
    class_weighted_version: str
    cost_matrix_version: str
    signer_status: Literal["illustrative, unsigned"]
    band_edges: tuple[float, float]
    action_map: dict[str, str]
    seed: int
    history_rows: int
    population: str
    calibration: dict
    threshold_sha256: str
    retest: dict
    journals: dict
    state: str = "offline proposal; drafts only, a person executes; gate uses placeholder risk"

    @model_validator(mode="after")
    def no_denial(self):
        if self.band_edges != (PAY_EDGE, THIRD_EDGE) or self.action_map != ACTION_MAP:
            raise ValueError("the risk signal must preserve the declared bands and action map")
        return self


class ScriptedVendorModel:
    """A constructed stand-in with two planted faults, unrelated to any product."""

    version = "scripted-overconfident-incoherent-v1"

    def predict(self, rows, negated=False):
        probabilities = []
        for row in rows:
            mu = _false_probability(row)
            p = mu * mu / (mu * mu + (1 - mu) ** 2)  # logistic of twice the log-odds
            probabilities.append(min(1.0, 1 - p + .1) if negated else p)
        return probabilities

    def answer(self, row, negated=False):
        p = self.predict([row], negated=negated)[0]
        question = RISK_QUESTION if not negated else TypedQuestion(
            name="Is paying this tier 1 refund a sound payment?",
            kind="true_false", answers=("wrong", "sound"), cannot_tell=True,
        )
        return TypedAnswer(
            question=question,
            probabilities=None if row.amount_cents >= 5000 else dict(zip(question.answers, (1 - p, p))),
            cannot_tell=row.amount_cents >= 5000,
            model_version=self.version,
        )
