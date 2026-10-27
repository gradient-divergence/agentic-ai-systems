"""Run the offline pilot and write deterministic scorecards, audits, and traces."""

import argparse
from pathlib import Path

from refund_agent.agent import Agent
from refund_agent.baseline import Scorecard, baseline, score_decisions
from refund_agent.contracts import AuditRecord, Record, Trace, write_json
from refund_agent.data.generate import Dataset, generate, instructor_data, write_data
from refund_agent.evaluation import (
    PaymentScorecard,
    SafetyScorecard,
    positive_evaluation,
    safety_evaluation,
)
from refund_agent.tools import Tools


class RunScorecard(Record):
    seed: int
    baseline: Scorecard
    agent: Scorecard
    safety: SafetyScorecard
    gate_removed: SafetyScorecard
    positive_payments: PaymentScorecard
    instructor: Scorecard | None = None


class AuditLog(Record):
    records: list[AuditRecord]


def _arguments(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--requests", type=int, help="score first N requests")
    parser.add_argument("--out", type=Path, default=Path("refund-output"))
    parser.add_argument("--instructor-seed", type=int)
    parser.add_argument("--score-instructor", action="store_true")
    args = parser.parse_args(argv)
    if args.requests is not None and not 0 <= args.requests <= 600:
        parser.error("--requests must be between 0 and 600")
    if args.score_instructor and args.instructor_seed is None:
        parser.error("--score-instructor requires --instructor-seed")
    return args


def _run(data: Dataset, count: int | None) -> tuple[Scorecard, Scorecard, list[Trace], Tools]:
    requests = (
        data.requests[:count]
        if count is not None
        else [r for r in data.requests if r.request_id in data.public_scenarios]
    )
    ids = {r.request_id for r in requests}
    correct = [d for d in data.correct if d.request_id in ids]
    drafts = score_decisions(baseline(data, requests), correct)
    tools = Tools(data)
    traces = [Agent(tools).run(request) for request in requests]
    return drafts, score_decisions([t.decision for t in traces], correct), traces, tools


def _write_run(out: Path, traces: list[Trace], tools: Tools) -> None:
    for trace in traces:
        trace.write(out / "traces" / f"{trace.decision.request_id}.json")
    write_json(out / "audits.json", AuditLog(records=list(tools.audits.values())))


def main(argv: list[str] | None = None) -> int:
    args = _arguments(argv)
    data = generate(args.seed, instructor_seed=args.instructor_seed)
    drafts, scored, traces, tools = _run(data, args.requests)
    _write_run(args.out, traces, tools)
    safety, ungated = safety_evaluation()
    positive = positive_evaluation()
    held_score = _instructor_run(data, args)
    scores = RunScorecard(
        seed=args.seed,
        baseline=drafts,
        agent=scored,
        safety=safety,
        gate_removed=ungated,
        positive_payments=positive,
        instructor=held_score,
    )
    write_json(args.out / "scorecard.json", scores)
    for name in ("baseline", "agent", "safety", "gate_removed", "positive_payments"):
        print(getattr(scores, name).line(name))
    if held_score is not None:
        print(held_score.line("instructor"))
    return 0


def _instructor_run(data: Dataset, args: argparse.Namespace) -> Scorecard | None:
    if args.instructor_seed is None:
        return None
    held = instructor_data(data)
    write_data(held, args.out / "instructor")
    if not args.score_instructor:
        return None
    _, score, traces, tools = _run(held, len(held.requests))
    _write_run(args.out / "instructor", traces, tools)
    return score


if __name__ == "__main__":
    raise SystemExit(main())
