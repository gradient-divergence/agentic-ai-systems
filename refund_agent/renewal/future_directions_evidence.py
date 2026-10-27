"""Paired generated evidence, with independent capacity and release gates."""

from dataclasses import replace
from datetime import datetime, timedelta

import numpy as np
from pydantic import Field

from demos.supervised_learning_demo import bootstrap, wilson_interval
from refund_agent.contracts import Record
from refund_agent.data.history import END_AT
from refund_agent.oversight.human_oversight_risk_demand import Stage6Demand
from refund_agent.oversight.human_oversight_risk_design import QueueDesign
from refund_agent.oversight.human_oversight_risk_simulate import QueueArrival, simulate
from refund_agent.renewal.future_directions_requirement import validate
from refund_agent.renewal.future_directions_windows import candidate_decision, validate_window
from refund_agent.schedule.inputs import REVIEW_MINUTES
from refund_agent.selection import decide as cost_decide
from refund_agent.selection import fingerprint


class Check(Record):
    name: str
    passed: bool
    detail: str


class Evidence(Record):
    requirement_fingerprint: str
    candidate_bundle_fingerprint: str | None
    candidate_fingerprint: str | None
    score_fingerprint: str
    window_fingerprint: str
    window_id: str
    paid: int
    unique: int
    all_matured: bool
    selected: bool
    wrong: int
    prediction: float
    interval: tuple[float, float]
    difference: tuple[float, float, float]
    referrals: int
    added_minutes: int
    wrongful_denials: tuple[int, int] = (0, 0)
    checks: tuple[Check, ...]
    first_day: float
    last_day: float
    closed_day: float
    day: float
    threshold_band: dict = Field(default_factory=dict)
    first_decision_fingerprint: str | None = None
    fit_window_fingerprint: str | None = None

    @property
    def fingerprint(self):
        return fingerprint(self.model_dump(mode="json"))


def evaluate(
    rows,
    candidate,
    incumbent,
    req,
    day: datetime,
    *,
    window,
    expected_start,
    admitted_targets=(),
    check_design=None,
    capacity_design=None,
    capacity_demand=None,
    capacity_arrivals=None,
    release_record=None,
    suite=None,
    journal=None,
    candidate_bundle_fingerprint=None,
    first_decision=None,
    first_evidence=None,
    fit_window=None,
    original_score=None,
):
    validate_window(
        window,
        rows,
        paid=req.paid_per_window,
        evaluated_at=day,
        expected_start=expected_start,
        seed=req.seed,
    )
    if incumbent != req.incumbent_candidate_fingerprint:
        raise ValueError("evidence is bound to another incumbent or risk component")
    if first_decision is None:
        if candidate.fingerprint != req.score_fingerprint:
            raise ValueError("risk component differs from the signed first candidate")
        candidate_fingerprint = req.candidate_fingerprint
        if candidate_bundle_fingerprint is None:
            candidate_bundle_fingerprint = req.candidate_bundle_fingerprint
        if candidate_bundle_fingerprint != req.candidate_bundle_fingerprint:
            raise ValueError("candidate bundle differs from the signed requirement")
    else:
        from refund_agent.renewal.future_directions_candidate import recalibrate

        if (
            first_decision.verdict != "rebuild"
            or first_decision.reasons != ("calibration",)
            or (
                first_decision.requirement_fingerprint != req.fingerprint
                or expected_start != END_AT + timedelta(days=first_decision.day)
                or original_score is None
                or original_score.fingerprint != req.score_fingerprint
                or fit_window is None
                or first_evidence is None
                or first_decision.evidence_fingerprint != first_evidence.fingerprint
                or first_evidence.window_fingerprint != fit_window.fingerprint
                or first_evidence.requirement_fingerprint != req.fingerprint
            )
        ):
            raise ValueError("rebuild needs the first calibration-only decision and its fit window")
        first_rows = [r for r in fit_window.rows if r.paid]
        validate_window(
            fit_window,
            first_rows,
            paid=req.paid_per_window,
            evaluated_at=END_AT + timedelta(days=first_decision.day),
            expected_start=END_AT,
            seed=req.seed,
        )
        rebuilt = recalibrate(
            original_score,
            first_rows,
            evaluated_at=fit_window.mature_at,
        )
        if rebuilt.fingerprint != candidate.fingerprint:
            raise ValueError("rebuild was not fit on the first window alone")
        if suite is None:
            raise ValueError("rebuilt candidate suite missing; no substitute bundle identity")
        from refund_agent.release.candidate import build_candidate

        candidate_fingerprint = build_candidate(suite).fingerprint
        if candidate_bundle_fingerprint is None:
            raise ValueError("rebuilt version-bundle identity missing")
        selected = [r for r in suite.components if r["name"] == "future-directions.risk"]
        if len(selected) != 1 or selected[0]["settings"]["fingerprint"] != candidate.fingerprint:
            raise ValueError("rebuilt candidate suite selects another risk component")
    probabilities = candidate.predict(rows)
    y = np.asarray([r.found_false_30d for r in rows], dtype=int)
    observed = int(y.sum())
    interval = tuple(wilson_interval(observed, len(rows)))
    prediction = float(probabilities.mean())
    pay = np.asarray(
        [candidate_decision(r, float(p)) == "pay" for r, p in zip(rows, probabilities)]
    )
    base_cost = y * np.asarray([r.amount_cents for r in rows]) / 100
    candidate_cost = np.where(pay, base_cost, req.review_cents / 100)
    difference = bootstrap(candidate_cost - base_cost, seed=req.seed)
    referrals = [r for r, paying in zip(rows, pay) if not paying]
    added = tuple(
        {
            "request_id": r.request_id,
            "at": r.at,
            "class": "referral",
            "review_minutes": REVIEW_MINUTES[1],
            "blocking": True,
        }
        for r in referrals
    )
    checks = [
        Check(
            name="calibration",
            passed=interval[0] <= prediction <= interval[1],
            detail="inclusive Wilson; calibration in the large",
        ),
        Check(
            name="cost",
            passed=cost_decide(True, difference[0], difference[1:]) == "accepted",
            detail="95% percentile bootstrap upper end; review assumed always right",
        ),
        Check(name="wrongful_denials", passed=True, detail="0 versus 0; neither rule denies"),
    ]
    if check_design is None:
        checks.append(
            Check(name="capacity", passed=False, detail="Human Oversight and Risk Management check_design missing")
        )
    elif capacity_design is None or capacity_demand is None or capacity_arrivals is None:
        checks.append(
            Check(name="capacity", passed=False, detail="Human Oversight and Risk Management design, full demand or arrivals missing")
        )
    else:
        try:
            if not isinstance(capacity_design, QueueDesign) or not isinstance(capacity_demand, Stage6Demand):
                raise TypeError("capacity requires chapter 37 QueueDesign and Stage6Demand")
            elapsed = (window.rows[-1].at - window.start).total_seconds() / 86400
            demand = replace(
                capacity_demand,
                referral_share=capacity_demand.referral_share
                + len(added) / elapsed * capacity_demand.month_days / capacity_demand.tier1_monthly,
            )
            classes = {work.name: work for work in demand.classes()}
            arrivals = tuple(capacity_arrivals) + added
            for arrival in arrivals:
                work = classes[arrival["class"]]
                if arrival["review_minutes"] != work.minutes or arrival["blocking"] != work.blocking:
                    raise ValueError("arrival differs from chapter 37's work class")
            queue = simulate(
                req.seed, window.start, int(elapsed) + 1, demand=demand,
                people=capacity_design.people, named_approver=capacity_design.named_approver,
                service_model="fixed", arrivals=tuple(
                    QueueArrival(review_id=r["request_id"], work_class=r["class"], at=r["at"])
                    for r in arrivals
                ),
            )
            result = check_design(capacity_design, demand, weekends=queue.weekends)
            if result["necessary_counts_only"] is not True or not isinstance(result["failures"], (tuple, list)):
                raise ValueError("check_design must return chapter 37's necessary counts")
            failures = list(result["failures"])
            for weekend in result["observed_weekends"]:
                failures.extend(weekend["monday"] + ":" + reason for reason in weekend["failures"])
            if result["utilization"] > capacity_design.utilization_target:
                failures.append("weekly utilization exceeds 0.8")
            checks.append(Check(
                name="capacity", passed=not failures and result["utilization"] <= capacity_design.utilization_target,
                detail="necessary counts; " + (", ".join(failures) if failures else "fit"),
            ))
        except Exception as error:
            checks.append(
                Check(name="capacity", passed=False, detail="capacity refused: " + str(error))
            )
    try:
        validate(req, admitted_targets)
        checks.append(
            Check(name="incumbent_admission", passed=True, detail="stage-6 target admitted")
        )
    except ValueError as error:
        checks.append(Check(name="incumbent_admission", passed=False, detail=str(error)))
    if release_record is None or suite is None or journal is None:
        checks.append(Check(name="release", passed=False, detail="signed release record missing"))
    else:
        from refund_agent.release.candidate import build_candidate
        from refund_agent.shadow.__main__ import validate_release

        release_candidate = build_candidate(suite)
        try:
            if release_candidate.fingerprint != candidate_fingerprint:
                raise ValueError("release candidate differs from requirement")
            validate_release(release_record, release_candidate, suite, journal)
            if datetime.fromisoformat(release_record["signature"]["at"]) > day:
                raise ValueError("release signed after evidence evaluation")
            checks.append(Check(name="release", passed=True, detail=release_record["record_id"]))
        except (ValueError, KeyError, TypeError) as error:
            checks.append(Check(name="release", passed=False, detail=str(error)))
    band = (probabilities >= 0.060) & (probabilities <= 0.100)
    count = int(band.sum())
    band_result = {
        "range": [0.060, 0.100],
        "paid": count,
        "wrong": int(y[band].sum()),
        "mean_prediction": float(probabilities[band].mean()) if count else None,
        "interval": list(wilson_interval(int(y[band].sum()), count)) if count else None,
    }
    return Evidence(
        requirement_fingerprint=req.fingerprint,
        candidate_bundle_fingerprint=candidate_bundle_fingerprint,
        candidate_fingerprint=candidate_fingerprint,
        score_fingerprint=candidate.fingerprint,
        window_fingerprint=window.fingerprint,
        window_id=window.window_id,
        paid=len(rows),
        unique=len({r.request_id for r in rows}),
        all_matured=True,
        selected=True,
        wrong=observed,
        prediction=prediction,
        interval=interval,
        difference=difference,
        referrals=len(referrals),
        added_minutes=len(referrals) * REVIEW_MINUTES[1],
        checks=tuple(checks),
        first_day=(rows[0].at - END_AT).total_seconds() / 86400,
        last_day=(rows[-1].at - END_AT).total_seconds() / 86400,
        closed_day=(window.mature_at - END_AT).total_seconds() / 86400,
        day=(day - END_AT).total_seconds() / 86400,
        threshold_band=band_result,
        first_decision_fingerprint=None if first_decision is None else first_decision.fingerprint,
        fit_window_fingerprint=None if fit_window is None else fit_window.fingerprint,
    )
