"""One entry point admits only a signed release; shadow changes no external store."""

import json
from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from refund_agent.evaluation import _positive_data


@pytest.mark.parametrize("runtime", ["own", "second"])
@pytest.mark.parametrize(
    "verifier,reason",
    [
        ("independent", "verifier_objection:words_contradict_record"),
        ("timeout", "verifier_timeout"),
        ("void", "verifier_verdict_void"),
    ],
)
def test_same_proposal_bound_verifier_and_referral_effects_everywhere(runtime, verifier, reason):
    from refund_agent.release.candidate import build_candidate, build_tools, run_request
    from refund_agent.release.suite import build_suite

    data = _positive_data()
    request = data.requests[0].model_copy(update={"text": "The parcel never arrived."})
    data = data.model_copy(update={"requests": [request]})
    candidate = build_candidate(build_suite(trials=1), runtime=runtime, verifier=verifier)
    results = []
    for mode in ("execute", "shadow"):
        tools = build_tools(data, candidate, mode)
        trace = run_request(candidate, tools, request)
        assert trace.decision.reason == reason and not trace.decision.executed
        assert not tools.payments
        reviews = tools.shadow_reviews if mode == "shadow" else tools.reviews
        assert len(reviews) == 1
        results.append(
            (
                trace.decision.model_dump(),
                [v.model_dump(mode="json") for v in tools.verdict_log.entries],
                [r.proposal.proposal_id for r in reviews.values()],
            )
        )
    assert results[0] == results[1]


@pytest.mark.parametrize("runtime", ["own", "second"])
def test_shadow_keeps_real_ledger_and_reviews_byte_identical_and_runs_gate(runtime):
    from refund_agent.release.candidate import build_candidate, build_tools, run_request
    from refund_agent.release.suite import build_suite

    data = _positive_data()
    candidate = build_candidate(build_suite(trials=1), runtime=runtime)
    tools = build_tools(data, candidate, "shadow")
    before = (json.dumps(tools.ledger, sort_keys=True), json.dumps(tools.reviews, sort_keys=True))
    for request in data.requests:
        run_request(candidate, tools, request)
    assert (
        json.dumps(tools.ledger, sort_keys=True),
        json.dumps(tools.reviews, sort_keys=True),
    ) == before
    assert not tools.payments
    assert len(tools.shadow_payments) == 3 and len(tools.shadow_reviews) == 2
    assert tools.gate_calls == 3
    assert sum(r["kind"] == "payment" for r in tools.shadow_log) == 3
    denied = data.requests[0].model_copy(update={"request_id": "bad", "amount_cents": 4000})
    trace = run_request(candidate, tools, denied)
    assert not trace.decision.executed


@pytest.mark.parametrize(
    "mutation",
    ["missing", "failed", "unsigned", "candidate", "suite", "past-failure", "forged-signature"],
)
def test_shadow_refuses_invalid_record_without_starting(tmp_path, mutation):
    from refund_agent.release.candidate import build_candidate
    from refund_agent.release.journal import Journal
    from refund_agent.release.rule import evaluate_release, sign
    from refund_agent.release.suite import build_suite
    from refund_agent.shadow.__main__ import start

    suite = replace(
        build_suite(trials=1),
        cases=tuple(c for c in build_suite(trials=1).cases if c.part == "public"),
    )
    candidate = build_candidate(suite)
    journal = Journal(tmp_path / "journal.jsonl")
    record = evaluate_release(candidate, suite, journal, trials=1)
    if mutation != "unsigned":
        record = sign(record, journal, "morgan", "2026-10-12T12:00:00+00:00")
    if mutation == "failed":
        record["decision"] = "fail"
    if mutation in ("candidate", "suite"):
        record[mutation + "_fingerprint"] = "different"
    if mutation == "past-failure":
        journal.append(
            {
                **record,
                "decision": "fail",
                "failed_cases": ["planted"],
                "signature": None,
                "record_id": "earlier-failed",
            }
        )
    if mutation == "forged-signature":
        record["signature"]["at"] = "2026-10-13T12:00:00+00:00"
    path = tmp_path / "release.json"
    if mutation != "missing":
        path.write_text(json.dumps(record))
    with pytest.raises((ValueError, FileNotFoundError)):
        start(
            path, candidate=candidate, suite=suite, journal=journal, output=tmp_path / "shadow.json"
        )
    assert not (tmp_path / "shadow.json").exists()


def test_offers_use_real_record_and_both_constructed_decisions():
    from refund_agent.offpolicy.record import EvaluationRecord
    from refund_agent.shadow.offers import load_offers, select_offers

    real = load_offers()
    assert real.name == "fixed" and {c["decision"] for c in real.candidates.values()} == {"hold"}
    assert real.access["purpose"] == "read chapter 21 evaluation record; no outcomes read"
    constructed = EvaluationRecord(
        {
            "policies": {
                "A": {
                    "fingerprint": "a",
                    "decision": "hold",
                    "reasons": ["bad"],
                    "policy": real.policy.record(),
                },
                "B": {
                    "fingerprint": real.policy.fingerprint,
                    "decision": "eligible for the shadow run",
                    "reasons": [],
                    "policy": real.policy.record(),
                },
                "fixed": {"fingerprint": real.policy.fingerprint, "policy": real.policy.record()},
            },
            "limits": ["constructed"],
        }
    )
    assert select_offers(constructed).name == "B"
    constructed.data["policies"]["B"]["decision"] = "hold"
    assert select_offers(constructed).name == "fixed"


def test_live_generator_freezes_and_fingerprints_at_start():
    from refund_agent.data.live import generate_live

    start = datetime(2026, 10, 1, tzinfo=timezone.utc)
    first = generate_live(7, start, 1, 120)
    second = generate_live(7, start, 1, 120)
    assert first.fingerprint == second.fingerprint
    assert len(first.data.requests) == 120
    assert all(start <= r.at < start + timedelta(days=1) for r in first.data.requests)
    assert {p.version for p in first.data.policies} == {"2026-09-01"}
    assert first.manifest["frozen_at"] == start.isoformat()


def test_d6_uses_paid_not_found_false_only_and_requires_299():
    from refund_agent.shadow.compare import shadow_to_canary

    rows = [
        {
            "request_id": f"d6-{i}",
            "person_saw_shadow": False,
            "tier": 1,
            "agent": "deny",
            "baseline": "deny",
            "person_paid": True,
            "found_false_30d": 0,
            "label_final_at": "2026-10-31T00:00:00+00:00",
            "request_at": "2026-10-01T00:00:00+00:00",
            "payment_at": "2026-10-01T00:00:00+00:00",
            "forbidden_payment": False,
        }
        for i in range(299)
    ]
    record = shadow_to_canary(rows, at="2026-10-31T00:00:00+00:00")
    assert record["wrongful_denials"] == record["baseline_wrongful_denials"] == 299
    assert record["measures_pass"] and not record["provisional"]
    rows[0]["baseline"] = "approve"
    assert not shadow_to_canary(rows, at="2026-10-31T00:00:00+00:00")["measures_pass"]
    rows[0]["found_false_30d"] = 1
    assert shadow_to_canary(rows, at="2026-10-31T00:00:00+00:00")["measures_pass"]
    assert not shadow_to_canary(rows[:120], at="2026-10-31T00:00:00+00:00")["window_complete"]
    assert not shadow_to_canary(rows, at="2026-10-12T00:00:00+00:00")["window_complete"]


@pytest.mark.parametrize("fault", ["duplicate", "early-label", "shown-reference"])
def test_d6_refuses_invalid_request_window(fault):
    from refund_agent.shadow.compare import shadow_to_canary

    rows = [
        {
            "request_id": f"window-{i}",
            "request_at": "2026-10-01T00:00:00+00:00",
            "payment_at": "2026-10-01T00:00:00+00:00",
            "person_saw_shadow": False,
            "tier": 1,
            "agent": "deny",
            "baseline": "deny",
            "person_paid": True,
            "found_false_30d": 0,
            "label_final_at": "2026-10-31T00:00:00+00:00",
            "forbidden_payment": False,
        }
        for i in range(299)
    ]
    if fault == "duplicate":
        rows[1]["request_id"] = rows[0]["request_id"]
    elif fault == "early-label":
        rows[0]["label_final_at"] = rows[0]["request_at"]
    else:
        rows[0]["person_saw_shadow"] = True
    with pytest.raises(ValueError):
        shadow_to_canary(rows, at="2026-10-31T00:00:00+00:00")


def test_both_constructed_offer_records_produce_their_notice_columns():
    from refund_agent.data.live import generate_live
    from refund_agent.offpolicy.policies import Policy
    from refund_agent.offpolicy.record import EvaluationRecord
    from refund_agent.release.candidate import build_candidate
    from refund_agent.release.suite import build_suite
    from refund_agent.shadow.compare import compare
    from refund_agent.shadow.offers import load_offers, select_offers

    fixed = load_offers().policy
    dated = Policy.deterministic(
        "constructed-dated",
        {reason: "dated" for reason in ("damaged", "changed_mind", "not_delivered")},
        ("standard", "dated", "referenced"),
    )
    record = EvaluationRecord(
        {
            "policies": {
                "fixed": {"fingerprint": fixed.fingerprint, "policy": fixed.record()},
                "B": {
                    "fingerprint": dated.fingerprint,
                    "decision": "hold",
                    "reasons": ["constructed hold"],
                    "policy": dated.record(),
                },
            },
            "limits": ["constructed record"],
        }
    )
    stream = generate_live(7, datetime(2026, 10, 1, tzinfo=timezone.utc), 1, 120)
    for decision, expected in [("hold", "standard"), ("eligible for the shadow run", "dated")]:
        record.data["policies"]["B"]["decision"] = decision
        record.data["policies"]["B"]["reasons"] = [] if decision != "hold" else ["constructed hold"]
        candidate = build_candidate(build_suite(trials=1), offers=select_offers(record))
        result = compare(stream, candidate)
        payments = [r for r in result["rows"] if r["would_pay"]]
        assert payments and all(
            r["offer"] == expected and r["fixed_offer"] == "standard" for r in payments
        )
        assert result["notices_sent"] == result["offer_outcomes_observed"] == 0
        assert result["adjudication_pending"] == sum(result["disagreements"].values())


def test_changed_live_stream_is_refused_before_shadow():
    from refund_agent.data.live import generate_live
    from refund_agent.release.candidate import build_candidate
    from refund_agent.release.suite import build_suite
    from refund_agent.shadow.compare import compare

    stream = generate_live(7, datetime(2026, 10, 1, tzinfo=timezone.utc), 1, 120)
    stream.data.requests.pop()
    with pytest.raises(ValueError, match="frozen live stream"):
        compare(stream, build_candidate(build_suite(trials=1)))


def test_shadow_audits_describe_actual_zero_payment_effects():
    from refund_agent.release.candidate import build_candidate, build_tools, run_request
    from refund_agent.release.suite import build_suite

    data = _positive_data()
    candidate = build_candidate(build_suite(trials=1))
    tools = build_tools(data, candidate, "shadow")
    trace = run_request(candidate, tools, data.requests[0])
    assert trace.decision.executed  # Virtual receipt: the agent would pay.
    assert not tools.payments
    assert all(
        not a.money_moved and a.paid_amount_cents == 0 and a.receipt_id is None
        for a in tools.audits.values()
    )
    assert any(row.get("would_execute") for row in tools.shadow_log)


def test_delayed_payment_requires_thirty_days_from_payment():
    from refund_agent.shadow.compare import shadow_to_canary

    rows = [
        {
            "request_id": f"delayed-{i}",
            "request_at": "2026-10-01T00:00:00+00:00",
            "payment_at": "2026-10-10T00:00:00+00:00",
            "label_final_at": "2026-10-31T00:00:00+00:00",
            "person_saw_shadow": False,
            "tier": 1,
            "agent": "deny",
            "baseline": "deny",
            "person_paid": True,
            "found_false_30d": 0,
            "forbidden_payment": False,
        }
        for i in range(299)
    ]
    with pytest.raises(ValueError, match="payment"):
        shadow_to_canary(rows, at="2026-10-31T00:00:00+00:00")
    for row in rows:
        row["label_final_at"] = "2026-11-09T00:00:00+00:00"
    assert not shadow_to_canary(rows, at="2026-10-31T00:00:00+00:00")["window_complete"]
    assert shadow_to_canary(rows, at="2026-11-09T00:00:00+00:00")["window_complete"]
