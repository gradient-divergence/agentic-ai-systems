"""The extra check may refuse a payment; it cannot confer payment authority."""

import subprocess
import sys
from collections import defaultdict
from datetime import timedelta
from pathlib import Path

import pytest

from demos import regions as _regions
from refund_agent import contracts as c
from refund_agent.agent import Agent
from refund_agent.tools import Tools

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def fixture():
    from refund_agent.verifier.fixture import build_fixture

    return build_fixture(7)


@pytest.fixture(scope="module")
def experiment():
    from refund_agent.verifier.experiment import run_experiment

    return run_experiment(7)


def test_fixture_is_seeded_frozen_and_has_no_runtime_labels(fixture):
    from refund_agent.verifier.fixture import build_fixture

    assert fixture.manifest == build_fixture(7).manifest
    assert fixture.manifest != build_fixture(8).manifest
    assert fixture.truth.intact()
    assert len(fixture.data.requests) == 400
    assert fixture.data.correct == [] and fixture.data.instructor_scenarios == []
    assert all(r.amount_cents == 2499 and r.failure is None for r in fixture.data.requests)
    assert not any("ignore" in r.text.casefold() for r in fixture.data.requests)


def test_purchase_dates_cannot_reconstruct_hidden_dates_or_false_date_group():
    from refund_agent.verifier.fixture import AT, _case, build_fixture

    fixture = build_fixture(7)
    truth = fixture.truth.consult("isolation regression", "check purchase-only shortcuts")
    dates, memberships = defaultdict(set), defaultdict(set)
    false = set(fixture.groups["false_date"])
    orders = {order.order_id: order for order in fixture.data.orders}
    for request, row in zip(fixture.data.requests, truth):
        purchase = orders[request.order_reference].purchased_at
        if row.true_delivery_at is not None:
            dates[purchase].add(row.true_delivery_at)
        memberships[purchase].add(request.request_id in false)
    # Identical purchase inputs require different answers for both hidden targets.
    assert any(len(values) > 1 for values in dates.values())
    assert all(
        memberships[orders[request.order_reference].purchased_at] == {False, True}
        for request in fixture.data.requests
        if request.request_id in false
    )
    policy = fixture.data.policy_at(AT)
    plain = _case(7, 0, "plain", policy)
    false_date = _case(7, 0, "false_date", policy)
    assert plain[0].purchased_at == false_date[0].purchased_at
    assert plain[4].true_delivery_at != false_date[4].true_delivery_at


@pytest.mark.parametrize("comparison", ["authority_count", "changed_decisions"])
@pytest.mark.parametrize(
    "kind", ["empty", "missing_baseline", "unequal", "duplicate", "different", "reordered"]
)
def test_comparisons_reject_unmatched_request_ids(experiment, comparison, kind):
    from refund_agent.verifier import experiment as module

    rows = experiment["runs"]["single"]
    baseline, challenger = rows[:2], rows[:2]
    if kind == "empty":
        baseline, challenger = [], []
    elif kind == "missing_baseline":
        baseline, challenger = [], rows[:1]
    elif kind == "unequal":
        challenger = rows[:1]
    elif kind == "duplicate":
        baseline = challenger = [rows[0], rows[0]]
    elif kind == "different":
        challenger = rows[1:3]
    else:
        challenger = list(reversed(challenger))
    with pytest.raises(ValueError, match="request IDs"):
        getattr(module, comparison)(baseline, challenger)


@pytest.mark.parametrize(
    "kind",
    [
        "empty",
        "missing_baseline",
        "unequal",
        "duplicate",
        "different",
        "reordered",
        "missing_truth",
        "duplicate_truth",
        "different_truth",
        "reordered_truth",
    ],
)
def test_saving_rejects_unmatched_request_ids(experiment, kind):
    from refund_agent.verifier.experiment import saving
    from refund_agent.verifier.fixture import build_fixture

    rows = experiment["runs"]["single"]
    labels = build_fixture(7).truth.consult("cost regression", "check matched request IDs")
    baseline, challenger, truth = rows[:2], rows[:2], labels[:2]
    if kind == "empty":
        baseline, challenger, truth = [], [], []
    elif kind == "missing_baseline":
        baseline = []
    elif kind == "unequal":
        challenger = rows[:1]
    elif kind == "duplicate":
        baseline = challenger = [rows[0], rows[0]]
        truth = [labels[0], labels[0]]
    elif kind == "different":
        challenger = rows[1:3]
    elif kind == "reordered":
        challenger = list(reversed(challenger))
    elif kind == "missing_truth":
        truth = []
    elif kind == "duplicate_truth":
        truth = [labels[0], labels[0]]
    elif kind == "different_truth":
        truth = labels[1:3]
    else:
        truth = list(reversed(truth))
    with pytest.raises(ValueError, match="request IDs"):
        saving(baseline, challenger, truth)


def test_arms_preserve_authority_and_the_agreeing_arm_changes_nothing(experiment):
    assert experiment["arms"]["single"]["correct"] == 360
    assert experiment["arms"]["agreeing"]["correct"] == 360
    assert experiment["arms"]["independent"]["correct"] == 376
    for arm in ("agreeing", "independent"):
        assert experiment["arms"][arm]["additional_payments"] == 0
        assert experiment["public"][arm]["additional_payments"] == 0
    assert experiment["arms"]["agreeing"]["changed_decisions"] == 0
    assert experiment["public"]["agreeing"]["changed_decisions"] == 0
    assert experiment["public"]["independent"]["changed_decisions"] == 0


def test_inline_check_matches_the_independent_arm_without_verdicts_or_authority(experiment):
    from refund_agent.verifier import experiment as module

    assert callable(getattr(module, "run_diagnostic", None)), "the inline diagnostic is missing"
    diagnostic = module.run_diagnostic(7)
    arm = diagnostic["arm"]
    assert arm["correct"] == 376 and arm["payments"] == 368 and arm["wrong_payments"] == 16
    assert arm["referrals"] == arm["review_items"] == arm["changed_decisions"] == 32
    assert arm["additional_payments"] == arm["independent_differences"] == 0
    assert arm["verification_calls"] == arm["verdict_records"] == 0
    assert arm["checks"] == 400 and arm["words_per_check"] == pytest.approx(80.08)
    for inline, delegated in zip(diagnostic["runs"]["in_agent"], experiment["runs"]["independent"]):
        assert inline.trace.decision == delegated.trace.decision
        assert inline.trace.receipt == delegated.trace.receipt
        assert inline.audits == delegated.audits and inline.reviews == delegated.reviews
        assert inline.verdicts == () and inline.verification_calls == 0
    assert diagnostic["events"] == [
        "single finished",
        "independent finished",
        "in_agent finished",
        "truth consulted",
    ]
    assert len(diagnostic["fixture"].truth.log) == 1
    assert len(experiment["fixture"].truth.log) == 1
    assert set(experiment["commission"]["challengers"]) == {"agreeing", "independent"}


def test_inline_check_runs_only_after_the_gate_and_makes_no_verifier_call(fixture, monkeypatch):
    from refund_agent.verifier import experiment as module
    from refund_agent.verifier.independent import IndependentVerifier

    assert hasattr(module, "InAgentTools"), "the in-agent tools check is missing"

    def forbidden(*args, **kwargs):
        pytest.fail("the inline arm invoked a verifier")

    monkeypatch.setattr(IndependentVerifier, "verify", forbidden)
    request = fixture.data.requests[0]
    tools = module.InAgentTools(module.scoped_data(fixture.data, request))
    Agent(tools).run(request)
    assert tools.checks == 1 and tools.verification_calls == 0
    tools = module.InAgentTools(module.scoped_data(fixture.data, request))
    denied = request.model_copy(update={"amount_cents": 2500})
    Agent(tools).run(denied)
    assert tools.checks == 0 and not tools.payments


@pytest.mark.parametrize(
    "error,reason",
    [
        (TimeoutError, "verifier_timeout"),
        (ValueError, "verifier_verdict_void"),
        (RuntimeError, "verifier_error"),
    ],
)
def test_inline_read_failures_keep_the_delegated_refusal_and_review(
    fixture, monkeypatch, error, reason
):
    from refund_agent.verifier import experiment as module

    def failure(*args, **kwargs):
        raise error("failed evidence read")

    class FailingVerifier:
        verify = failure

    request = fixture.data.requests[0]
    delegated = module.run_case(fixture.data, request, FailingVerifier())
    monkeypatch.setattr(module, "independent_check", failure)
    tools = module.InAgentTools(module.scoped_data(fixture.data, request))
    inline = Agent(tools).run(request)
    assert inline.decision == delegated.trace.decision
    assert inline.decision.reason == reason and inline.receipt is None
    assert tuple(tools.audits.values()) == delegated.audits
    assert len(tools.reviews) == delegated.reviews == 1
    assert tools.verification_calls == 0


def test_objections_match_constructed_groups_and_false_dates_stay_missed(experiment):
    rows = experiment["runs"]["independent"]
    groups = experiment["fixture"].groups
    by_id = {row.trace.decision.request_id: row for row in rows}
    for group, outcome in (
        ("contradiction", "object"),
        ("phrase_trip", "object"),
        ("false_date", "agree"),
        ("plain", "agree"),
    ):
        assert (
            len(groups[group])
            == {"contradiction": 24, "phrase_trip": 8, "false_date": 16, "plain": 352}[group]
        )
        for rid in groups[group]:
            row = by_id[rid]
            assert len(row.verdicts) == 1 and row.verdicts[0].answer == outcome
            assert row.trace.decision.outcome == (
                c.Outcome.REFER if outcome == "object" else c.Outcome.APPROVE
            )
            if outcome == "object":
                assert row.trace.receipt is None and row.reviews == 1
                assert any(
                    a.decision.reason == "verifier_objection:words_contradict_record"
                    and not a.money_moved
                    for a in row.audits
                )


def test_read_only_view_has_no_write_tools_or_shared_mutable_records(fixture):
    from refund_agent.verifier.record import ReadOnlyTools

    request = fixture.data.requests[0]
    reads = ReadOnlyTools(fixture.data, request.at)
    assert reads.names == ("order.lookup", "shipment.status", "policy.retrieve")
    for name in ("refund.propose", "approval.request", "refund.execute", "audit.append"):
        with pytest.raises(ValueError, match="read-only"):
            reads.call(c.ToolCall(name=name), c.State(request=request, trace_id="read-only"))
    state = c.State(request=request, trace_id="read-only")
    assert reads.call(c.ToolCall(name="order.lookup"), state).ok
    assert state.order is not None
    state.order.items.clear()
    assert fixture.data.orders[0].items


def test_truth_is_consulted_once_after_all_arms_and_is_journaled(experiment, tmp_path):
    from refund_agent.verifier.fixture import build_fixture

    assert experiment["events"] == [
        "single finished",
        "agreeing finished",
        "independent finished",
        "truth consulted",
    ]
    log = experiment["fixture"].truth.log
    assert len(log) == 1 and log[0]["reader"] == "matched experiment"
    truth = build_fixture(7).truth
    truth.persist(tmp_path)
    truth.consult("test", "score completed arms")
    first = next(tmp_path.glob("*.access.jsonl")).read_bytes()
    truth.consult("audit", "inspect score")
    assert next(tmp_path.glob("*.access.jsonl")).read_bytes().startswith(first)
    assert len(truth.log) == 2 and truth.intact()
    resumed = build_fixture(7).truth
    resumed.persist(tmp_path)
    assert len(resumed.log) == 2
    resumed.consult("resumed audit", "inspect score without erasing prior reads")
    assert len(resumed.log) == 3


def test_no_arm_reads_the_truth_or_a_frozen_set(fixture, monkeypatch):
    from refund_agent.split import GoldenSet
    from refund_agent.verifier.agreeing import AgreeingVerifier
    from refund_agent.verifier.experiment import run_arm
    from refund_agent.verifier.independent import IndependentVerifier

    def forbidden(*args, **kwargs):
        pytest.fail("an arm read a sealed label or truth table")

    monkeypatch.setattr(GoldenSet, "consult", forbidden)
    for verifier in (None, AgreeingVerifier(), IndependentVerifier()):
        assert len(run_arm(fixture.data, verifier)) == 400
    assert fixture.truth.log == ()


def test_paired_rule_saving_and_observed_covariance(experiment):
    arm = experiment["arms"]["independent"]
    assert (arm["paired"].n_plus, arm["paired"].n_minus) == (24, 8)
    assert arm["paired"].lead == 0.04
    assert arm["paired"].interval == pytest.approx((0.00862, 0.07138), abs=1e-5)
    assert arm["paired"].exact_chance == pytest.approx(0.00700036669)
    assert arm["decision"] == "accepted"
    assert arm["saving"]["mean_dollars"] == pytest.approx(1.1794)
    assert arm["saving"]["interval_dollars"] == pytest.approx((0.61499, 1.74381), abs=1e-5)
    assert arm["errors"]["counts"] == {"00": 352, "01": 8, "10": 24, "11": 16}
    assert arm["errors"]["covariance"] == pytest.approx(0.034)
    assert experiment["arms"]["agreeing"]["decision"] == "not adopted"


def test_five_to_none_is_not_adopted():
    from refund_agent.adaptation.experiment import decide, paired

    card = paired([False] * 5 + [True] * 195, [True] * 200, alpha=0.025)
    assert card.exact_chance == 0.0625
    assert decide(True, card, alpha=0.025) == "not adopted"


@pytest.mark.parametrize("mixed", [False, True])
@pytest.mark.parametrize(
    "mode,reason",
    [
        ("contradiction", "verifier_objection:words_contradict_record"),
        ("timeout", "verifier_timeout"),
        ("void", "verifier_verdict_void"),
        ("policy", "verifier_verdict_void"),
    ],
)
def test_failure_refers_and_preserves_audit_and_tools_composition(fixture, mixed, mode, reason):
    from refund_agent.verifier.experiment import scoped_data
    from refund_agent.verifier.independent import IndependentVerifier, VerifiedTools

    class ShadowWrites(Tools):
        def _commit(self, state, key, amount, approval):
            raise AssertionError("a failed verification reached the payment write")

    class Composed(VerifiedTools, ShadowWrites):
        pass

    class Fault(IndependentVerifier):
        def verify(self, proposal, reads, *, deadline):
            if mode == "timeout":
                raise TimeoutError("deadline")
            verdict = super().verify(proposal, reads, deadline=deadline)
            update = (
                {"proposal_id": "earlier-proposal"}
                if mode == "void"
                else {"policy_version": "2026-03-01"}
            )
            return verdict.model_copy(update=update) if mode in ("void", "policy") else verdict

    rid = (
        fixture.groups["contradiction"][0]
        if mode == "contradiction"
        else fixture.groups["plain"][0]
    )
    request = next(r for r in fixture.data.requests if r.request_id == rid)
    cls = Composed if mixed else VerifiedTools
    tools = cls(scoped_data(fixture.data, request), Fault(), deadline_seconds=1)
    trace = Agent(tools).run(request)
    assert trace.decision.outcome == c.Outcome.REFER and trace.decision.reason == reason
    assert not tools.payments and len(tools.reviews) == 1
    assert all(not a.money_moved for a in tools.audits.values())
    assert any(a.decision.reason == reason for a in tools.audits.values())


def test_simulated_response_past_deadline_refers(fixture):
    from refund_agent.verifier.experiment import scoped_data
    from refund_agent.verifier.independent import IndependentVerifier, VerifiedTools

    request = fixture.data.requests[0]
    tools = VerifiedTools(
        scoped_data(fixture.data, request),
        IndependentVerifier(latency_seconds=2),
        deadline_seconds=1,
    )
    trace = Agent(tools).run(request)
    assert trace.decision.reason == "verifier_timeout" and trace.receipt is None


def test_changed_shipment_voids_an_earlier_verdict(fixture):
    from refund_agent.verifier.experiment import scoped_data
    from refund_agent.verifier.independent import IndependentVerifier, VerifiedTools

    rid = fixture.groups["plain"][0]
    request = next(r for r in fixture.data.requests if r.request_id == rid)
    data = scoped_data(fixture.data, request)
    state = c.State(request=request, trace_id="late-verdict", execution_at=request.at)
    tools = VerifiedTools(data, IndependentVerifier())
    for name in ("order.lookup", "shipment.status", "policy.retrieve", "refund.propose"):
        arguments = (
            c.ToolArguments(amount_cents=2499, reason=request.reason)
            if name == "refund.propose"
            else c.ToolArguments()
        )
        assert tools.call(c.ToolCall(name=name, arguments=arguments), state).ok
    assert tools._authorize(state, 2499, None) is None
    prior = tools.verdict_log.entries
    old = data.shipments[0]
    changed = old.model_copy(
        update={
            "events": [
                old.events[0].model_copy(update={"at": old.events[0].at - timedelta(days=1)})
            ]
        }
    )
    tools.data = data.model_copy(update={"shipments": [changed]})
    result = tools.call(
        c.ToolCall(
            name="refund.execute",
            arguments=c.ToolArguments(amount_cents=2499, idempotency_key="late"),
        ),
        state,
    )
    assert not result.ok and result.reason == "verifier_verdict_void"
    assert not tools.payments and tools.verdict_log.entries == prior


def test_verdict_log_is_append_only_and_verdict_is_frozen(experiment):
    from refund_agent.verifier.record import VerdictLog

    verdict = experiment["runs"]["independent"][0].verdicts[0]
    log = VerdictLog()
    log.append(verdict)
    snapshot = log.entries
    with pytest.raises((ValueError, AttributeError)):
        verdict.answer = "object"
    with pytest.raises(ValueError, match="one verdict"):
        log.append(verdict)
    assert log.entries == snapshot and isinstance(snapshot, tuple)


def test_gate_refuses_before_asking_any_verifier(corpus):
    from refund_agent.verifier.experiment import scoped_data
    from refund_agent.verifier.independent import IndependentVerifier, VerifiedTools

    class MustNotBeAsked(IndependentVerifier):
        def verify(self, *args, **kwargs):
            raise AssertionError("the gate was bypassed")

    for request in corpus.requests[:5]:
        tools = VerifiedTools(scoped_data(corpus, request), MustNotBeAsked())
        trace = Agent(tools).run(request)
        assert trace.receipt is None and tools.verdict_log.entries == ()


def test_already_reviewed_tier2_payment_does_not_gain_an_agent_verifier(fixture):
    from test_stage1 import approval_for

    from refund_agent.verifier.experiment import scoped_data
    from refund_agent.verifier.independent import IndependentVerifier, VerifiedTools

    class MustNotBeAsked(IndependentVerifier):
        def verify(self, *args, **kwargs):
            raise AssertionError("the chapter 27 check is tier 1 only")

    rid = fixture.groups["plain"][0]
    request = next(r for r in fixture.data.requests if r.request_id == rid).model_copy(
        update={"amount_cents": 5000}
    )
    data = scoped_data(fixture.data, request)
    order = data.orders[0].model_copy(
        update={
            "paid_cents": 5000,
            "items": [c.Item(name="physical desk lamp", quantity=1, unit_cents=5000)],
        }
    )
    tools = VerifiedTools(data.model_copy(update={"orders": [order]}), MustNotBeAsked())
    first = Agent(tools).run(request)
    assert first.decision.outcome == c.Outcome.REFER
    assert tools.record_approval(approval_for(first.steps[-1].state)).ok
    second = Agent(tools).run(request)
    assert second.receipt is not None and second.receipt.amount_cents == 5000
    assert tools.verification_calls == 0 and tools.verdict_log.entries == ()


def test_public_tier3_denials_still_require_a_named_approver(experiment):
    assert experiment["public"]["single"]["correct"] == 26
    for name in ("single", "agreeing", "independent"):
        assert experiment["public"][name]["tier3_referrals"] == 15


@pytest.mark.parametrize(
    "name",
    [
        "agentic_loop_demo.py",
        "perception_tools_demo.py",
        "reasoning_planning_demo.py",
        "refund_gate_demo.py",
    ],
)
def test_earlier_demo_behavior_matches_committed_outputs(name):
    # Refactors may change source; earlier observable results must stay committed.
    tools = vars(_regions)
    source = ROOT / "demos" / name
    regions = {
        s.snippet_id: s for s in tools["extract_snippets_from_text"](source.read_text(), source)
    }
    fresh = tools["capture_outputs"](regions, sys.executable, timeout=300)
    assert fresh, f"no outputs checked for {name}"
    prefix = subprocess.check_output(
        ["git", "rev-parse", "--show-prefix"], cwd=ROOT, text=True
    ).strip()
    for snippet_id, stdout in fresh.items():
        path = ROOT / "book_outputs" / f"_{snippet_id}.txt"
        committed = subprocess.check_output(
            ["git", "show", f"HEAD:{prefix}book_outputs/_{snippet_id}.txt"], cwd=ROOT
        )
        assert stdout.encode() == path.read_bytes()
        if snippet_id in {
            "fnd.perception-tools.freshness-01",
            "fnd.perception-tools.contracts-01",
            "fnd.perception-tools.execute-01",
        }:
            import ast

            assert ast.dump(ast.parse(stdout)) == ast.dump(ast.parse(committed))
            assert max(map(len, stdout.splitlines())) <= 79
        elif snippet_id == "fnd.perception-tools.versions-01":
            assert stdout.encode().split() == committed.split()
            assert max(map(len, stdout.splitlines())) <= 79
        else:
            assert path.read_bytes() == committed
