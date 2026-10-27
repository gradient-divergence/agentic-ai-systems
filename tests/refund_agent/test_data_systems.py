"""Behavior proof for chapter 30's decision-bound data contracts."""

import json
import math
from datetime import timedelta
from pathlib import Path

import pytest
from scipy.stats import poisson

from refund_agent import contracts as c
from refund_agent.agent import Agent
from refund_agent.data.generate import generate
from refund_agent.evaluation import _positive_data
from refund_agent.model import ScriptedModel
from refund_agent.release.candidate import build_candidate, build_tools, run_request
from refund_agent.release.suite import build_suite
from refund_agent.rules import sources
from refund_agent.sources import ContractedTools
from refund_agent.sources.data_systems_budget import (
    budget_hours,
    rate_estimate,
    rate_upper,
    stale_chance,
)
from refund_agent.sources.data_systems_check import contract_problem
from refund_agent.sources.data_systems_contract import SourceContract
from refund_agent.sources.data_systems_lineage import LineageRecord, RecordStore, rebuild
from refund_agent.sources.data_systems_registry import Registry, default_registry
from refund_agent.sources.data_systems_release import SELECTOR
from refund_agent.sources.data_systems_releases import add_status, dollars, rename_delivery
from refund_agent.split import CUTS
from refund_agent.tools import Tools


@pytest.fixture(scope="module")
def candidate():
    return build_candidate(build_suite(components=[SELECTOR]))


def run_data(data, candidate, contracted=True, requests=None):
    if contracted:
        tools = build_tools(data, candidate, "execute")
        traces = [run_request(candidate, tools, r) for r in (requests or data.requests)]
    else:
        tools = Tools(data)
        traces = [Agent(tools).run(r) for r in (requests or data.requests)]
    return tools, traces


def ready(tools, request):
    state = c.State(request=request, trace_id=f"trace-{request.request_id}")
    for name in ("order.lookup", "shipment.status", "policy.retrieve"):
        assert tools.call(c.ToolCall(name=name), state).ok
    return state


def test_data_systems_rename_refers_instead_of_paying(candidate):
    data = generate(7, orders=40, requests=40)
    renamed = data.model_copy(
        update={
            "shipments": [
                s.model_copy(
                    update={
                        "events": [
                            e.model_copy(
                                update={
                                    "status": "delivered_to_door"
                                    if e.status == "delivered"
                                    else e.status
                                }
                            )
                            for e in s.events
                        ]
                    }
                )
                for s in data.shipments
            ]
        }
    )
    request = next(r for r in data.requests if r.request_id == "r-7-0029")
    tools = ContractedTools(renamed, candidate_record=candidate.record())
    trace = Agent(tools).run(request)
    assert trace.decision.reason == "contract_vocabulary:shipment.status"
    assert trace.decision.outcome == c.Outcome.REFER
    assert not tools.payments


def test_data_systems_public_decisions_and_failures_unchanged(candidate):
    data = generate(7, orders=40, requests=40)
    _, before = run_data(data, candidate, False)
    tools, after = run_data(data, candidate)
    assert [t.decision for t in before] == [t.decision for t in after]
    assert len(tools.lineage) == 40
    for request in data.requests[1:5]:
        state = ready(tools, request)
        order, shipment, _ = sources(data, state)
        for source, record in (("order", order), ("shipment", shipment), ("policy", state.policy)):
            # The base owns the spec's age limit; only a tighter contract
            # budget would add a reason. Use the actual injected ages.
            assert contract_problem(source, record, f"{source}@1", state.freshness[source]) is None


def test_data_systems_training_validation_exactly_five_changes(candidate):
    data = generate().model_copy(update={"correct": []})
    requests = [r for r in data.requests if r.at < CUTS[1]]
    _, before = run_data(data, candidate, False, requests)
    _, after = run_data(data, candidate, True, requests)
    changed = [b.decision.request_id for a, b in zip(before, after) if a.decision != b.decision]
    assert changed == ["r-7-0074", "r-7-0084", "r-7-0127", "r-7-0181", "r-7-0506"]
    assert all(
        b.decision.reason == "contract_invariant:shipment.not_after_observation"
        for a, b in zip(before, after)
        if a.decision != b.decision
    )
    bad = [s for s in data.shipments if any(e.at > s.observed_at for e in s.events)]
    assert len(bad) == 23
    assert {
        int((e.at - s.observed_at).total_seconds())
        for s in bad
        for e in s.events
        if e.at > s.observed_at
    } == {120}
    assert not set(changed) & set(data.public_scenarios)


def test_data_systems_rename_absolute_payments_and_referrals(candidate):
    data = rename_delivery(generate(7, orders=40, requests=40))
    bare, bare_traces = run_data(data, candidate, False)
    wrongful = {"r-7-0014", "r-7-0029", "r-7-0035"}
    assert {p.request.request_id for p in bare.payments} == wrongful | {"r-7-0008"}
    assert sum(p.amount_cents for p in bare.payments if p.request.request_id in wrongful) == 9997
    tools, traces = run_data(data, candidate)
    # The brief's "nothing paid" misses the legitimate no-delivery case r-7-0008.
    assert {p.request.request_id for p in tools.payments} == {"r-7-0008"}
    assert sum(p.amount_cents for p in tools.payments) == 4999
    assert sum(t.decision.reason == "contract_vocabulary:shipment.status" for t in traces) == 26
    delivered_ids = {
        s.order_id
        for s in generate(7, orders=40, requests=40).shipments
        if any(e.status == "delivered" for e in s.events)
    }
    higher = [
        r
        for r in data.requests
        if r.order_reference in delivered_ids
        and r.reason == "not_delivered"
        and r.amount_cents >= 5000
    ]
    drafts = {r.request_id: bare.assess(ready(bare, r)) for r in higher}
    assert len(drafts) == 6
    assert sum(d.outcome == c.Outcome.APPROVE for d in drafts.values()) == 3
    assert {rid for rid, d in drafts.items() if d.outcome == c.Outcome.DENY} == {
        "r-7-0020",
        "r-7-0026",
        "r-7-0038",
    }
    assert all(
        t.decision.outcome == c.Outcome.REFER
        for t in bare_traces
        if t.decision.tier == c.Tier.NAMED_APPROVER
    )


def test_data_systems_added_harmless_status_cost(candidate):
    data = generate(7, orders=40, requests=40)
    _, original = run_data(data, candidate, False)
    _, harmless = run_data(add_status(data), candidate, False)
    assert [t.decision for t in harmless] == [t.decision for t in original]
    _, traces = run_data(add_status(data), candidate)
    assert sum(t.decision.reason == "contract_vocabulary:shipment.status" for t in traces) == 26


def test_data_systems_denial_checks_broken_order(candidate):
    from demos.modern_ai_stack_demo import REQUEST, load, releases

    data = load(releases["in dollars"])
    request = next(r for r in data.requests if r.request_id == REQUEST)
    plain = Tools(data)
    assert plain.assess(ready(plain, request)).outcome == c.Outcome.DENY
    tools = build_tools(data, candidate, "execute")
    assessed = tools.assess(ready(tools, request))
    assert assessed.outcome == c.Outcome.REFER
    assert assessed.reason == "contract_invariant:order.paid_equals_items"


def test_data_systems_announced_dollars_and_silent_limit():
    data = generate(7, orders=40, requests=40)
    original = data.model_dump_json()
    announced, silent = dollars(data, True), dollars(data, False)
    for release in (announced, silent):
        for order in release.data.orders:
            problem = contract_problem("order", order, release.served_versions["order"], 60)
            assert problem == ("contract_version_unknown:order" if release is announced else None)
    assert data.model_dump_json() == original
    # A plausible silent code reinterpretation is also invisible to single-record checks.
    assert contract_problem("shipment", data.shipments[7], "shipment@1", 120) is None


@pytest.mark.parametrize("n", [0, 1, 5, 48])
def test_data_systems_exact_poisson_bound(n):
    upper = rate_upper(n, 2400)
    assert poisson.cdf(n, upper * 2400) == pytest.approx(0.05, abs=1e-13)
    assert rate_upper(n, 1200) == pytest.approx(upper * 2)
    assert rate_upper(n + 1, 2400) > upper


def test_data_systems_budget_numbers_and_boundaries():
    assert rate_upper(0, 6000) == pytest.approx(-math.log(0.05) / 6000)
    upper = rate_upper(48, 2400)
    assert rate_estimate(48, 2400) == 0.02
    assert upper == pytest.approx(0.025439, abs=1e-6)
    assert stale_chance(0.02, 0.5) == pytest.approx(0.00995, abs=1e-6)
    assert stale_chance(upper, 0.5) == pytest.approx(0.012639, abs=1e-6)
    assert 60 * budget_hours(0.02, 0.01) == pytest.approx(30.15, abs=0.01)
    assert 60 * budget_hours(upper, 0.01) == pytest.approx(23.70, abs=0.01)
    assert stale_chance(upper, budget_hours(upper, 0.01)) == pytest.approx(0.01)
    raw = default_registry().contract("shipment").model_dump()
    raw["budget"]["limit_seconds"] = 1801
    with pytest.raises(ValueError, match="exceeds"):
        SourceContract.model_validate(raw)
    with pytest.raises(ValueError):
        rate_upper(-1, 100)
    with pytest.raises(ValueError):
        rate_upper(0, 0)


@pytest.mark.parametrize("problem", ["owner", "approval", "budget"])
def test_data_systems_source_without_authority_refers(problem, candidate):
    registry = default_registry()
    entries = list(registry.entries)
    raw = entries[1].model_dump()
    if problem == "owner":
        raw["contract"]["owner_role"] = None
        reason = "contract_owner_missing:shipment"
    elif problem == "approval":
        raw["contract"]["budget"]["approver_role"] = None
        reason = "contract_budget_unapproved:shipment"
    else:
        raw["contract"]["budget"]["limit_seconds"] = 119
        reason = "contract_budget:shipment"
    entries[1] = type(entries[1]).model_validate(raw)
    registry = Registry(tuple(entries), registry.pins)
    data = generate(7, orders=40, requests=40)
    tools = ContractedTools(
        data, registry=registry, code_version=candidate.fingerprint,
        candidate_record=candidate.record(),
    )
    for request in data.requests[5:]:
        assert Agent(tools).run(request).decision.reason == reason


def test_data_systems_gate_rereads_contract_before_payment(candidate):
    data = generate(7, orders=40, requests=40)
    request = data.requests[7]
    tools = build_tools(data, candidate, "execute")
    state = ready(tools, request)
    assert tools.assess(state).outcome == c.Outcome.APPROVE
    assert tools.call(
        c.ToolCall(
            name="refund.propose",
            arguments=c.ToolArguments(amount_cents=request.amount_cents, reason=request.reason),
        ),
        state,
    ).ok
    tools.data = rename_delivery(data)
    result = tools.call(
        c.ToolCall(
            name="refund.execute",
            arguments=c.ToolArguments(
                amount_cents=request.amount_cents, idempotency_key="data-systems-reread"
            ),
        ),
        state,
    )
    assert result.reason == "contract_vocabulary:shipment.status"
    assert not tools.payments


@pytest.mark.parametrize("renamed", [False, True])
def test_data_systems_every_public_decision_rebuilds(renamed, candidate):
    data = generate(7, orders=40, requests=40)
    tools, traces = run_data(rename_delivery(data) if renamed else data, candidate)
    assert len(tools.lineage) == 40
    for row in tools.lineage:
        assert rebuild(row, tools.record_store)["equal"], row.request.request_id
        assert row.code_version == candidate.fingerprint
        assert row.bundle is None
    before = tools.record_store.export()
    row = next(r for r in tools.lineage if r.request.request_id == "r-7-0007")
    assert row.model_proposal is not None
    assert not row.approvals
    assert [ref.age_seconds for ref in row.inputs[:3]] == [60.0, 120.0, 3600.0]
    assert tools.record_store.export() == before


def test_data_systems_store_never_overwrites_and_detects_tampering():
    store = RecordStore()
    record = {"value": 1}
    key = store.put(record)
    record["value"] = 2
    assert store.get(key) == {"value": 1}
    assert store.put({"value": 1}) == key
    changed = store.get(key)
    changed["value"] = 9
    assert store.get(key) == {"value": 1}
    store._records[key] = json.dumps(changed)
    with pytest.raises(ValueError, match="fingerprint"):
        store.get(key)


def test_data_systems_exercise_lineage_fails_for_observation_time():
    path = Path(__file__).resolve().parents[2] / "exercises/data-systems/lineage-05.json"
    fixture = json.loads(path.read_text())
    lineage = LineageRecord.model_validate_json(json.dumps(fixture["lineage"]))
    store = RecordStore()
    for key, record in fixture["records"].items():
        assert store.put(record) == key
    with pytest.raises(ValueError, match="observed after decision"):
        rebuild(lineage, store)


@pytest.mark.parametrize("source,kind", [
    ("order", "orders"), ("shipment", "shipments"), ("policy", "retrievals"),
])
@pytest.mark.parametrize("runtime", ["own", "second"])
@pytest.mark.parametrize("mode", ["execute", "shadow"])
def test_data_systems_future_source_referral_rebuilds(source, kind, runtime, mode):
    data = generate(7, orders=40, requests=40)
    request = data.requests[7]
    records = getattr(data, kind)
    original = next(
        r for r in records
        if (r.request_id == request.request_id if kind == "retrievals"
            else r.order_id == request.order_reference)
    )
    future = request.at + timedelta(seconds=60)
    data = data.model_copy(update={kind: [
        r.model_copy(update={"observed_at": future}) if r == original else r
        for r in records
    ]})
    candidate = build_candidate(build_suite(components=[SELECTOR]), runtime=runtime)
    tools = build_tools(data, candidate, mode)
    trace = run_request(candidate, tools, request)
    assert trace.decision.outcome == c.Outcome.REFER
    assert trace.decision.reason == f"future_{source}"
    before = tools.record_store.export()
    assert rebuild(tools.lineage[0], tools.record_store)["equal"]
    ref = next(r for r in tools.lineage[0].inputs if r.record_kind == kind)
    assert ref.observed_at == future and ref.age_seconds == -60
    assert ref.captured_at == request.at
    assert tools.record_store.export() == before
    assert not tools.payments
    bad = ref.model_copy(update={"captured_at": future})
    inputs = tuple(bad if r == ref else r for r in tools.lineage[0].inputs)
    with pytest.raises(ValueError, match="captured after decision"):
        rebuild(tools.lineage[0].model_copy(update={"inputs": inputs}), tools.record_store)


def test_data_systems_registry_records_completeness_without_claiming_a_check():
    contract = default_registry().contract("shipment")
    history = next(f for f in contract.fields if f.field == "events")
    assert history.meaning == "complete: every scan the carrier made has been sent"
    assert "completeness" not in contract.invariants
    data = generate(7, orders=40, requests=40)
    assert contract_problem("shipment", data.shipments[8], "shipment@1", 120) is None


def test_data_systems_correction_changes_recompute_not_payment(candidate):
    data = generate(7, orders=40, requests=40)
    tools, _ = run_data(data, candidate)
    row = next(r for r in tools.lineage if r.request.request_id == "r-7-0007")
    ref = next(r for r in row.inputs if r.source == "shipment")
    before = tools.record_store.get(ref.fingerprint)
    paid = tools.ledger
    corrected = dict(before)
    corrected["events"] = [e for e in before["events"] if e["status"] == "shipped"]
    key = tools.record_store.put(corrected)
    inputs = tuple(r.model_copy(update={"fingerprint": key}) if r == ref else r for r in row.inputs)
    replay = rebuild(row.model_copy(update={"inputs": inputs}), tools.record_store)
    assert not replay["equal"]
    assert replay["decision"]["reason"] == "delivery_unknown"
    assert tools.record_store.get(ref.fingerprint) == before
    assert tools.ledger == paid
    assert row.decision.executed
    expected = tuple(
        r
        for r in tools.lineage
        if any(i.used and i.fingerprint == ref.fingerprint for i in r.inputs)
    )
    assert tools.record_store.decisions_using(ref.fingerprint) == expected
    future_policy = next(
        i
        for i in row.inputs
        if i.record_kind == "policies"
        and tools.record_store.get(i.fingerprint)["version"] == "2026-09-01"
    )
    assert not future_policy.used
    assert row not in tools.record_store.decisions_using(future_policy.fingerprint)


def test_data_systems_replay_uses_recorded_model_proposal(candidate):
    data = generate(7, orders=40, requests=40)
    request = data.requests[7]
    tools = build_tools(data, candidate, "execute")
    state = ready(tools, request)
    wrong = tools.assess(state).model_copy(update={"outcome": c.Outcome.DENY})
    model = ScriptedModel(replies={(request.request_id, c.Phase.ASSESS): wrong})
    trace = run_request(candidate, tools, request, model=model)
    row = tools.lineage[0]
    assert trace.decision.reason == "invalid_decision"
    assert row.model_proposal == wrong
    assert rebuild(row, tools.record_store)["equal"]
    assert row.model_proposal != tools.assess(state)


def test_data_systems_direct_denial_append_checks_contract(candidate):
    from refund_agent.rules import decision

    data = rename_delivery(generate(7, orders=40, requests=40))
    request = data.requests[29]
    tools = build_tools(data, candidate, "execute")
    state = ready(tools, request)
    state.decision = decision(state, c.Outcome.DENY, "already_delivered")
    assert tools.record_decision(state).ok
    assert state.decision.outcome == c.Outcome.REFER
    assert state.decision.reason == "contract_vocabulary:shipment.status"
    assert all(a.decision.outcome == c.Outcome.REFER for a in tools.audits.values())
    assert not tools.payments


@pytest.mark.parametrize("path", ["record_decision", "audit.append"])
@pytest.mark.parametrize("mode", ["execute", "shadow"])
def test_data_systems_direct_named_denial_opens_review(candidate, path, mode):
    data = generate(7, orders=40, requests=40)
    request = next(r for r in data.requests if r.request_id == "r-7-0006")
    tools = build_tools(data, candidate, mode)
    state = ready(tools, request)
    state.decision = tools.assess(state)
    assert state.decision.outcome == c.Outcome.DENY
    assert request.amount_cents == 120000
    result = (
        tools.record_decision(state)
        if path == "record_decision"
        else tools.call(c.ToolCall(name=path), state)
    )
    assert result.ok
    assert state.decision.outcome == c.Outcome.REFER
    assert state.decision.reason == "amount_over_policy"
    reviews = tools.shadow_reviews if mode == "shadow" else tools.reviews
    assert len(reviews) == 1
    review = next(iter(reviews.values()))
    assert review.open and review.required_role == "named_approver"
    assert review.proposal.request == request
    assert all(a.decision.outcome == c.Outcome.REFER for a in tools.audits.values())
    assert not tools.payments and not tools._approvals


@pytest.mark.parametrize("runtime", ["own", "second"])
@pytest.mark.parametrize("mode", ["execute", "shadow"])
@pytest.mark.parametrize(
    "verifier,max_steps,reason",
    [
        ("timeout", None, "verifier_timeout"),
        ("independent", 3, "step_budget_exhausted"),
        ("independent", 1, "step_budget_exhausted"),
    ],
)
def test_data_systems_rebuild_restores_candidate(runtime, mode, verifier, max_steps, reason):
    data = generate(7, orders=40, requests=40)
    request = data.requests[7]
    candidate = build_candidate(
        build_suite(components=[SELECTOR]), runtime=runtime, verifier=verifier
    )
    tools = build_tools(data, candidate, mode)
    trace = run_request(candidate, tools, request, max_steps=max_steps)
    assert trace.decision.reason == reason
    row = tools.lineage[0]
    assert rebuild(row, tools.record_store)["equal"]
    assert row.candidate == candidate.record()
    assert row.max_steps == (max_steps or 12)
    assert row.candidate["runtime"]["identity"] == runtime


@pytest.mark.parametrize("runtime", ["own", "second"])
@pytest.mark.parametrize("failure", ["decision", "arguments", "error", "type", "nan"])
def test_data_systems_recording_preserves_model_failure(runtime, failure):
    data = generate(7, orders=40, requests=40)
    request = data.requests[7]

    class FailedModel(ScriptedModel):
        def reply(self, state):
            if state.phase != c.Phase.ASSESS:
                return super().reply(state)
            if failure == "error":
                raise TimeoutError("model unavailable")
            if failure == "type":
                return "not a model reply"
            if failure == "arguments":
                return c.ToolCall.model_construct(
                    name="refund.propose",
                    arguments=c.ToolArguments.model_construct(amount_cents=-1),
                )
            return (
                super()
                .reply(state)
                .model_copy(update={"amount_cents": float("nan") if failure == "nan" else -1})
            )

    traces = []
    for stage in (4, 5):
        candidate = build_candidate(
            build_suite(components=[SELECTOR] if stage == 5 else ()),
            runtime=runtime,
        )
        tools = build_tools(data, candidate, "execute")
        traces.append(run_request(candidate, tools, request, model=FailedModel()))
    assert traces[0] == traces[1]
    row = tools.lineage[0]
    assert len(row.model_exchanges) == 3
    assert len(row.model_errors) == int(failure == "error")
    assert len(row.rejected_replies) == int(failure != "error")
    assert rebuild(row, tools.record_store)["equal"]


def test_data_systems_recovered_model_does_not_replay_prior_error(candidate):
    data = generate(7, orders=40, requests=40)
    tools = build_tools(data, candidate, "execute")

    class UnavailableModel:
        def reply(self, state):
            raise TimeoutError()

    assert (
        run_request(candidate, tools, data.requests[7], model=UnavailableModel()).decision.reason
        == "model_error"
    )
    trace = run_request(candidate, tools, data.requests[7])
    assert trace.decision.executed
    row = tools.lineage[-1]
    assert not row.model_errors
    assert rebuild(row, tools.record_store)["equal"]


def test_data_systems_rebuild_uses_effective_candidate_budget():
    from refund_agent.runtimes.adapter import AdapterSettings

    data = generate(7, orders=40, requests=40)
    candidate = build_candidate(
        build_suite(components=[SELECTOR]),
        runtime="second",
        settings=AdapterSettings(max_steps=3),
    )
    tools = build_tools(data, candidate, "execute")
    trace = run_request(candidate, tools, data.requests[7], max_steps=20)
    assert trace.decision.reason == "step_budget_exhausted"
    row = tools.lineage[0]
    assert row.max_steps == 3
    assert rebuild(row, tools.record_store)["equal"]


def test_data_systems_delivered_only_fixture_and_absolute_must_pay(candidate):
    data = _positive_data()
    shipment = next(s for s in data.shipments if s.order_id == "delivered")
    assert len(shipment.events) == 1
    assert contract_problem("shipment", shipment, "shipment@1", 0) is None
    from refund_agent.release.suite import run_loop_case

    cases = [case for case in build_suite(components=[SELECTOR]).cases if case.part == "positive"]
    results = [run_loop_case(case, candidate) for case in cases]
    assert all(not r["failed"] for r in results)
    assert len(cases) == 5
    assert sum(r["decision"]["amount_cents"] for r in results if r["decision"]["executed"]) == 62598
