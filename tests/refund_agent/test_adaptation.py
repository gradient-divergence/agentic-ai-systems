"""Adaptation contracts, paired decisions and regression checks beside the loop."""

import hashlib
import json
from dataclasses import replace
from datetime import timedelta
from pathlib import Path

import numpy as np
import pytest

from refund_agent import contracts as c
from refund_agent.adaptation.arms import fit_adapter_terms, prompt_arm
from refund_agent.adaptation.drafter import ScriptedDrafter
from refund_agent.adaptation.experiment import (
    AdaptationCommission,
    agreement,
    decide,
    forbidden,
    paired,
    read_evidence,
    run_arm,
)
from refund_agent.adaptation.lora import LowRankAdapter, adapter_bytes, fit, trainable_numbers
from refund_agent.adaptation.prompt import DraftFormatError, StructuredPrompt, parse_draft
from refund_agent.baseline import baseline
from refund_agent.data.generate import generate


@pytest.fixture(scope="module")
def data():
    return generate(7)


@pytest.fixture(scope="module")
def report():
    from demos.adaptation_demo import build_experiment

    return build_experiment(7)


def test_prompt_arm_reproduces_baseline_on_all_600_requests(data):
    runs = run_arm(prompt_arm(), data, data.requests)
    assert [row.reply for row in runs] == baseline(data)


def test_each_arm_sees_the_same_request_and_evidence(report):
    runs = report["runs"]
    reference = runs["prompt"]
    for name in ("retrieval", "adapter"):
        assert [r.request_id for r in runs[name]] == [r.request_id for r in reference]
        assert [r.evidence_sha256 for r in runs[name]] == [r.evidence_sha256 for r in reference]
    assert report["correct"] == {"prompt": 121, "retrieval": 121, "adapter": 83}
    assert report["july_correct"] == {"prompt": 81, "retrieval": 81, "adapter": 25}
    assert all(not row.reply.executed for arm in runs.values() for row in arm)


def test_probe_precedes_commission_and_experiment_is_the_second_access(report):
    assert len(report["access_log"]) == 2
    probe, experiment = report["access_log"]
    assert probe["entry"] == 1 and probe["date"] == "2026-10-10"
    assert probe["brief"].endswith("prompting-finetuning-rag-brief.md")
    assert experiment["entry"] == 2
    events = report["events"]
    assert [row["event"] for row in events] == ["access", "commission", "access"]
    assert events[1]["sha256"] == report["commission"].sha256
    assert report["commission"].intact()
    assert report["golden"].intact()


def test_no_golden_label_or_request_enters_fitting_or_settings(report, monkeypatch):
    from demos import adaptation_demo as demo
    from refund_agent.retrieval.encoders import CountEncoder
    from refund_agent.split import GoldenSet

    held_ids = {r.request_id for r in report["golden"].requests}
    original_fit = demo.fit_adapter_terms
    original_run = demo.run_arm
    original_consult = GoldenSet.consult
    consulted = []
    fits = []

    def guarded_fit(drafts, *args, **kwargs):
        assert not consulted
        ids = {d.request_id for d in drafts}
        assert not ids & held_ids
        fits.append(ids)
        return original_fit(drafts, *args, **kwargs)

    def guarded_run(arm, data, requests, *args, **kwargs):
        ids = {r.request_id for r in requests}
        if not consulted:
            assert not ids & held_ids
        return original_run(arm, data, requests, *args, **kwargs)

    def consult(self, reader, purpose):
        assert self.log[0]["reader"] == "brief-writer probe"
        consulted.append(reader)
        return original_consult(self, reader, purpose)

    original_encoder_fit = CountEncoder.fit

    def encoder_fit(self, texts):
        assert not consulted
        return original_encoder_fit(self, texts)

    monkeypatch.setattr(demo, "fit_adapter_terms", guarded_fit)
    monkeypatch.setattr(demo, "run_arm", guarded_run)
    monkeypatch.setattr(CountEncoder, "fit", encoder_fit)
    monkeypatch.setattr(GoldenSet, "consult", consult)
    result = demo.build_experiment(7)
    assert len(consulted) == 1
    assert [len(ids) for ids in fits] == [398, 479]
    assert result["drafts_needed"]["2026-09-01"]["available"] == 0


def test_adapter_uses_the_latest_version_cited_by_training_drafts(report):
    from refund_agent.adaptation.arms import adapter_arm

    assert report["june_terms"].version == "2026-05-01"
    assert report["adapter_terms"].version == "2026-07-01"
    assert fit_adapter_terms(report["training_drafts"]).version == "2026-07-01"
    assert len(report["training_drafts"]) == 479
    arm = adapter_arm(report["training_drafts"])
    assert arm.adapter_terms == report["adapter_terms"]
    assert set(arm.training_ids) == {d.request_id for d in report["training_drafts"]}


def test_unlogged_frozen_rows_are_unavailable_to_the_experiment(monkeypatch):
    from demos import adaptation_demo as demo

    original = demo.freeze

    class GuardedSet:
        def __init__(self, held):
            self._held = held
            self.manifest, self.log = held.manifest, held.log

        def consult(self, reader, purpose):
            return self._held.consult(reader, purpose)

        def intact(self):
            return self._held.intact()

        def __getattr__(self, name):
            raise AssertionError(f"unlogged frozen access: {name}")

    monkeypatch.setattr(demo, "freeze", lambda *args: GuardedSet(original(*args)))
    result = demo.build_experiment(7)
    assert result["correct"] == {"prompt": 121, "retrieval": 121, "adapter": 83}
    assert len(result["access_log"]) == 2


def test_generated_corpus_policy_text_and_default_model_are_unchanged(data):
    from refund_agent.agent import Agent
    from refund_agent.model import ScriptedModel
    from refund_agent.selection import fingerprint as record_fingerprint
    from refund_agent.split import fingerprint as data_fingerprint
    from refund_agent.tools import Tools

    assert len(data.requests) == len(data.correct) == 600
    assert (
        data_fingerprint(data.requests, data.correct)
        == "4cc4890a2223837c6e07fd0a928e7ce008c8f349a357caa4ac1ec0352abd56b2"
    )
    assert (
        record_fingerprint([p.model_dump(mode="json") for p in data.policies])
        == "234beda6fea776666e17aed6c94ae526f414c9b19dee7cd24c37329b7ce5c56b"
    )
    assert type(Agent(Tools(data)).model) is ScriptedModel
    root = Path(__file__).resolve().parents[2] / "refund_agent"
    for path, expected in (
        (
            root / "agent/__init__.py",
            "59b6aa9e3f0ad10618e0d259883dd9a253db538bdac689aee931ad95106ee54a",
        ),
        (
            root / "model/__init__.py",
            "2d6ce2b1f064c1fbe659af6ddeb0e42081dcff1ff9b6a1da68a16b9f70737f16",
        ),
    ):
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected


def test_drafter_needs_policy_in_context_or_adapter(data):
    request = next(r for r in data.requests if r.request_id == "r-7-0484")
    state = read_evidence(data, [request])[0]
    client = ScriptedDrafter(data=data)
    assert client.reply(state).reason == "missing_policy"
    assert client.reply(state).outcome == c.Outcome.REFER
    assert client.reply(state).policy_citation is None


def test_prompt_sections_are_ordered_versioned_and_within_their_shares(data):
    request = next(r for r in data.requests if r.request_id == "r-7-0484")
    state = read_evidence(data, [request])[0]
    prompt = StructuredPrompt()
    rendered = prompt.render(state, state.policy.text)
    assert [s.name for s in rendered.sections] == [
        "Instructions",
        "Output schema",
        "Two worked drafts",
        "Policy",
        "Order",
        "Shipment events",
        "Request",
    ]
    assert rendered.within_budget()
    assert "within 45 days" in rendered.text
    assert prompt.fingerprint == StructuredPrompt().fingerprint
    assert replace(prompt, instructions="draft differently").fingerprint != prompt.fingerprint


@pytest.mark.parametrize(
    "change",
    [
        {"outcome": "pay"},
        {"amount_cents": 4999.0},
        {"amount_cents": True},
        {"amount_cents": "4999"},
        {"amount_cents": 0},
        {"tier": 4},
        {"tier": True},
        {"reason": ""},
        {"citation": "refund_cap_v4#eligibility"},
        {"citation": "refund_cap_v4@2026-02-30#eligibility"},
        {"citation": "refund_cap_v4@2026-09-01#eligibility trailing"},
        {"tool": "refund.execute"},
        {"citation": None},
    ],
)
def test_parse_draft_refuses_malformed_fields(data, change):
    request = data.requests[22]
    row = {
        "outcome": "approve",
        "amount_cents": request.amount_cents,
        "tier": 1,
        "reason": "eligible",
        "citation": "refund_cap_v4@2026-09-01#eligibility",
    }
    with pytest.raises(DraftFormatError):
        parse_draft(json.dumps({**row, **change}), request)


@pytest.mark.parametrize("text", ["not json", "[]", "{}", '{"outcome": "refer"}'])
def test_parse_draft_requires_an_object_and_every_field(data, text):
    with pytest.raises(DraftFormatError):
        parse_draft(text, data.requests[22])


def test_paired_counts_interval_and_exact_check():
    base = np.array([1] * 166 + [0] * 14 + [1] * 4 + [0] * 16, dtype=bool)
    challenger = np.array([1] * 180 + [0] * 20, dtype=bool)
    card = paired(base, challenger)
    assert (card.n_plus, card.n_minus, card.n) == (14, 4, 200)
    assert card.lead == pytest.approx(0.05)
    assert card.standard_error == pytest.approx(np.sqrt(18 - 100 / 200) / 200)
    assert card.interval == pytest.approx((0.0090038, 0.0909962), abs=1e-6)
    assert card.exact_chance == pytest.approx(0.0308837890625)
    reverse = paired(challenger, base)
    assert reverse.lead == -card.lead
    assert reverse.interval == pytest.approx((-card.interval[1], -card.interval[0]))
    tied = paired(base, base)
    assert tied.interval == (0, 0) and tied.exact_chance == 1


@pytest.mark.parametrize("wins, verdict", [(5, "not adopted"), (6, "not adopted"), (7, "accepted")])
def test_decide_requires_the_interval_and_the_exact_chance(wins, verdict):
    base = np.zeros(121, dtype=bool)
    challenger = base.copy()
    challenger[:wins] = True
    commission = AdaptationCommission()
    card = paired(base, challenger, alpha=commission.split_alpha)
    assert card.interval[0] > 0
    assert decide(True, card, commission.split_alpha) == verdict
    assert decide(False, card, commission.split_alpha) == "sent back"
    if wins == 5:
        assert card.exact_chance == 0.0625


def test_a_draft_over_its_share_makes_its_arm_ineligible(report):
    from refund_agent.adaptation.experiment import selection_record
    from refund_agent.adaptation.prompt import DRAFT_SHARE, count_words

    class LongReplyClient(ScriptedDrafter):
        """The scripted decision, well formed, with a reason longer than the draft's share."""

        def reply(self, state):
            chosen = super().reply(state)
            return chosen.model_copy(update={"reason": " ".join([chosen.reason] * DRAFT_SHARE)})

    requests = [r for r, _ in report["spent_rows"]]
    labels = [d for _, d in report["spent_rows"]]
    verbose = replace(report["arms"]["retrieval"], client_factory=LongReplyClient)
    runs = {**report["runs"], "retrieval": run_arm(verbose, report["data"], requests)}
    rows = runs["retrieval"]
    assert all(r.parse_error is None and r.rendered.within_budget() for r in rows)
    assert min(count_words(r.text) for r in rows) > DRAFT_SHARE
    assert agreement(rows, labels) == report["agreements"]["retrieval"]
    record = selection_record(
        report["commission"], report["golden"], report["events"], runs, labels, {}
    )
    assert record.checks["eligible"] == {"prompt": True, "retrieval": False, "adapter": True}
    assert record.decisions["retrieval"] == "sent back"
    assert record.selected == "prompt"
    assert report["record"].checks["eligible"] == {
        "prompt": True,
        "retrieval": True,
        "adapter": True,
    }


def test_agreement_keeps_denial_citations_in_the_match_rule(data):
    label = next(d for d in data.correct if d.outcome == c.Outcome.DENY)
    stale = label.model_copy(update={"policy_citation": "refund_cap_v4@2026-01-20#cap"})
    assert agreement([label], [label]) == [True]
    assert agreement([stale], [label]) == [False]
    with pytest.raises(ValueError):
        agreement([], [label])


def test_adapter_counts_zero_start_merge_rank_and_gradients():
    from demos.appendix_math_demo import central_difference

    rng = np.random.default_rng(7)
    w0, x = rng.normal(size=(4, 3)), rng.normal(size=(5, 4))
    adapter = LowRankAdapter(w0, 2, 8, 7)
    assert trainable_numbers(4, 3, 2) == adapter.trainable_numbers == 14
    assert adapter_bytes(4, 3, 2, 2) == 28
    np.testing.assert_allclose(adapter.forward(x), x @ w0, atol=1e-12)
    down_initial, up_initial = adapter.gradients(np.ones_like(w0))
    np.testing.assert_array_equal(down_initial, np.zeros_like(adapter.down))
    assert np.linalg.norm(up_initial) > 0
    adapter.up[:] = rng.normal(size=adapter.up.shape)
    np.testing.assert_allclose(adapter.forward(x), x @ adapter.merged(), atol=1e-12)
    assert np.linalg.matrix_rank(adapter.merged() - w0) <= 2
    target = rng.normal(size=(5, 3))
    g = 2 * x.T @ (adapter.forward(x) - target) / target.size
    down_g, up_g = adapter.gradients(g)
    parameters = np.r_[adapter.down.ravel(), adapter.up.ravel()]

    def loss(z):
        d, u = z[:8].reshape(4, 2), z[8:].reshape(2, 3)
        return np.mean((x @ (w0 + 4 * d @ u) - target) ** 2)

    numerical = central_difference(loss, parameters)
    np.testing.assert_allclose(np.r_[down_g.ravel(), up_g.ravel()], numerical, rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(adapter.w0, w0)


def test_toy_training_never_reads_held_out_examples_and_is_deterministic():
    from demos.adaptation_demo import lowrank_experiment

    first, second = lowrank_experiment(7), lowrank_experiment(7)
    assert first == second
    rows = first["results"]
    assert rows[0]["test_mse"] >= 2 * first["noise_floor"]
    full = rows[-1]["test_mse"]
    assert all(abs(row["test_mse"] / full - 1) <= 0.1 for row in rows[1:-1])
    x, y, w0 = np.eye(2), np.eye(2), np.zeros((2, 2))
    assert fit(x, y, w0, rank=None, steps=1, step_size=0.1).shape == (2, 2)


def test_toy_fit_receives_only_training_arrays_with_the_same_steps(monkeypatch):
    from demos import adaptation_demo as demo

    original, calls = demo.fit, []

    def guarded(x, y, w0, rank, **kwargs):
        assert x.shape == y.shape == (768, 16)
        assert kwargs["steps"] == 4000 and kwargs["step_size"] == 0.1
        calls.append((x, y, rank))
        return original(x, y, w0, rank, **kwargs)

    monkeypatch.setattr(demo, "fit", guarded)
    demo.lowrank_experiment(7)
    assert [rank for _, _, rank in calls] == [1, 2, 4, 8, None]
    assert all(x is calls[0][0] and y is calls[0][1] for x, y, _ in calls)


def test_suite_passes_prompt_and_fails_the_exact_adapter_cases(report):
    suite = report["suite"]
    assert len(suite.requests) == 141
    assert len(suite.questions) == 484
    prompt, adapter = report["suite_runs"]["prompt"], report["suite_runs"]["adapter"]
    assert prompt.verdict == "passes"
    assert adapter.verdict == "fails"
    wrong = {
        r.request_id
        for r, ok in zip(report["runs"]["adapter"], report["agreements"]["adapter"])
        if not ok
    }
    assert set(adapter.newly_failing) == wrong and len(wrong) == 38
    assert sum(r in wrong for r in suite.policy_change_ids if r in suite.golden_ids) == 5
    assert prompt.recall == {"policy": (242, 242), "customer": (121, 242)}
    assert suite.fingerprint == report["suite"].fingerprint
    stripped = prompt_arm(replace(StructuredPrompt(), include_policy=False))
    assert suite.run(stripped).verdict == "fails"


def test_suite_fingerprint_ignores_runs_and_detects_changed_cases(report):
    from demos.adaptation_demo import build_experiment

    suite = build_experiment(7)["suite"]
    before = suite.fingerprint
    suite.run(report["arms"]["prompt"])
    assert suite.fingerprint == before == report["suite"].fingerprint
    changed = replace(
        suite, labels=(suite.labels[0].model_copy(update={"reason": "changed"}),) + suite.labels[1:]
    )
    assert changed.fingerprint != before


def test_suite_detects_retrieval_regression_and_write_tool_output(report):
    from refund_agent.retrieval.index import PolicyIndex

    suite = report["suite"]

    def stale(at):
        return PolicyIndex.build(
            report["data"].policies, at - timedelta(days=31), report["encoder"]
        )

    arm = replace(report["arms"]["prompt"], index=stale)
    result = suite.run(arm)
    assert result.verdict == "fails" and len(result.retrieval_regressions) == 363

    class WritingClient:
        def __init__(self, *args, **kwargs):
            pass

        def reply(self, state):
            return c.ToolCall(name="refund.execute")

    writing = replace(report["arms"]["prompt"], client_factory=WritingClient)
    result = suite.run(writing)
    assert result.verdict == "fails"
    assert len(result.failures["schema"]) == len(result.forbidden_ids) == 141
    assert len(result.failures["no_write_tool"]) == 141


def test_write_tools_are_forbidden_even_if_the_label_refers(data):
    assert forbidden(c.ToolCall(name="refund.execute"), data.correct[0])
    assert forbidden(c.ToolCall(name="payment.send"), data.correct[0])


def test_november_exercise_lists_forbidden_drafts_and_preserves_the_golden_set(report):
    from demos.adaptation_demo import exercise_acceptance

    before = dict(report["golden"].manifest)
    exercise = exercise_acceptance(report)
    assert exercise["requests"] == exercise["prompt_correct"] == 20
    assert exercise["prompt_verdict"] == "passes"
    assert exercise["adapter_verdict"] == "fails"
    assert len(exercise["forbidden_ids"]) > 0
    assert exercise["fingerprint_changed"] and exercise["golden_unchanged"]
    assert report["golden"].manifest == before
    assert not set(exercise["request_ids"]) & {d.request_id for d in report["training_drafts"]}
