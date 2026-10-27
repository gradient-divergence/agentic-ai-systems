"""A runtime changes mechanisms without changing the agent's authority or record."""

import ast
import importlib
from dataclasses import replace
from pathlib import Path

import pytest

from refund_agent.data.generate import generate
from refund_agent.tools import Tools


def runtime():
    return importlib.import_module("refund_agent.runtimes.swap")


def test_changed_call_arguments_hold_the_swap():
    from refund_agent.runtimes.adapter import AdapterSettings, RefundAdapter
    from refund_agent.runtimes.runner import Call

    class ChangedArgument(RefundAdapter):
        def _dispatch(self, state, call):
            tool_call = call.arguments
            if call.name == "order.lookup":
                arguments = tool_call.arguments.model_copy(update={"reason": "changed"})
                tool_call = tool_call.model_copy(update={"arguments": arguments})
            return super()._dispatch(state, Call(call.name, tool_call))

    swap = runtime()
    case = next(c for c in swap.build_cases() if c.case_id == "public-r-7-0007")
    result = swap.compare([case], AdapterSettings(), adapter_factory=ChangedArgument)
    assert result["decision"] == "held"
    assert result["cases"][0]["different_fields"] == ["tool_calls"]
    assert result["money_differences"] == result["unauthorized_cents"] == 0


@pytest.mark.parametrize(
    "binding", ["receipt_proposal", "proposal_id", "reviewer", "policy_version"]
)
def test_changed_approval_bindings_hold_the_swap(binding):
    from refund_agent.runtimes.adapter import AdapterSettings, RefundAdapter

    class ChangedBinding(RefundAdapter):
        def run(self, request, **kwargs):
            trace = super().run(request, **kwargs)
            receipts = self.tools.payments
            if receipts:
                receipt = receipts[0]
                if binding == "receipt_proposal":
                    receipt = receipt.model_copy(update={"proposal_id": "changed"})
                else:
                    approval = receipt.approvals[0].model_copy(update={binding: "changed"})
                    receipt = receipt.model_copy(update={"approvals": (approval,)})
                self.tools.payments = [receipt]
            return trace

    swap = runtime()
    case = next(c for c in swap.build_cases() if c.case_id == "positive-5000")
    result = swap.compare([case], AdapterSettings(), adapter_factory=ChangedBinding)
    assert result["decision"] == "held"
    assert result["cases"][0]["different_fields"] == ["ledger_receipts"]
    assert result["money_differences"] == 1
    assert result["unauthorized_cents"] == 0


@pytest.mark.parametrize(
    "reply_type", ["none", "text", "integer", "mapping", "two-reads", "tuple", "subclass"]
)
def test_malformed_reply_types_keep_the_own_loop_failure(reply_type, monkeypatch):
    from refund_agent import contracts as c
    from refund_agent.model import ScriptedModel
    from refund_agent.runtimes.adapter import AdapterSettings

    class DerivedCall(c.ToolCall):
        pass

    replies = {
        "none": None,
        "text": "order.lookup",
        "integer": 1,
        "mapping": {"name": "order.lookup"},
        "two-reads": [c.ToolCall(name="order.lookup"), c.ToolCall(name="policy.retrieve")],
        "tuple": (c.ToolCall(name="order.lookup"),),
        "subclass": DerivedCall(name="order.lookup"),
    }

    class MalformedReply(ScriptedModel):
        def reply(self, state):
            return replies[reply_type]

    swap = runtime()
    monkeypatch.setattr(swap.SwapCase, "model", lambda self: MalformedReply())
    case = next(c for c in swap.build_cases() if c.case_id == "public-r-7-0007")
    result = swap.compare([case], AdapterSettings())
    assert result["different_cases"] == 0
    for side in ("own", "second"):
        seen = result["cases"][0][side]
        assert seen["outcome"] == "refer" and seen["reason"] == "model_error"
        assert seen["failure_detail"] == "model_error:order:TypeError"
        assert not seen["ledger_receipts"]


def test_an_empty_suite_is_held_for_missing_coverage():
    from refund_agent.runtimes.adapter import AdapterSettings

    result = runtime().compare([], AdapterSettings())
    assert result["total_cases"] == 0
    assert result["decision"] == "held"
    assert result["reasons"] == ["missing coverage: suite is empty"]


def test_full_adapter_matches_every_case():
    record = runtime().build_record(seed=7)
    full = record["configurations"][-1]
    assert full["different_cases"] == full["money_differences"] == 0
    assert full["decision"] == "eligible for the shadow run"
    assert all(not row["different_fields"] for row in full["cases"])
    assert {row["kind"] for row in full["cases"]} == {
        "public",
        "refusal",
        "positive",
        "fault",
        "resume",
    }


def test_defaults_change_money_but_never_breach_policy():
    record = runtime().build_record()
    rows = record["configurations"]
    assert rows[0]["money_differences"] > 0
    assert all(row["unauthorized_cents"] == 0 for row in rows)
    assert all(row["decision"] == "held" for row in rows[:-1])
    for row in rows:
        for case in row["cases"]:
            for side in ("own", "second"):
                paid = case[side]["ledger_receipts"]
                assert len(paid) <= 1
                assert sum(r["amount_cents"] for r in paid) <= case["allowed_cents"]


def test_timeout_after_payment_leaves_one_receipt_under_each_configuration():
    record = runtime().build_record()
    for row in record["configurations"]:
        case = next(r for r in row["cases"] if r["case_id"] == "execute-after")
        assert len(case["own"]["ledger_receipts"]) == 1
        expected = 0 if row["name"] == "runtime defaults" else 1
        assert len(case["second"]["ledger_receipts"]) == expected
        assert sum(r["amount_cents"] for r in case["second"]["ledger_receipts"]) == expected * 2499


def test_retry_before_payment_changes_a_referral_to_one_payment():
    rows = runtime().build_record()["configurations"]
    retry = next(row for row in rows if row["name"] == "referral ending")
    case = next(r for r in retry["cases"] if r["case_id"] == "execute-before")
    assert case["own"]["failure_detail"] == "tool_error:refund.execute:TimeoutError"
    assert case["own"]["outcome"] == "refer" and not case["own"]["executed"]
    assert case["second"]["executed"]
    assert [r["amount_cents"] for r in case["second"]["ledger_receipts"]] == [2499]
    assert sum(call["name"] == "refund.execute" for call in case["second"]["tool_calls"]) == 2


def test_resumes_cross_both_runtimes_and_pay_once():
    cases = runtime().build_record()["configurations"][-1]["cases"]
    resumes = [row for row in cases if row["kind"] == "resume"]
    assert {row["case_id"] for row in resumes} == {"resume-own-to-second", "resume-second-to-own"}
    for row in resumes:
        assert not row["different_fields"]
        for side in ("own", "second"):
            assert row[side]["executed"]
            assert len(row[side]["ledger_receipts"]) == 1
            assert len(row[side]["ledger_receipts"][0]["approvals"]) == 1


def test_denials_above_500_need_the_named_approver_in_every_configuration():
    supplied = []

    def factory(data):
        tools = Tools(data)
        supplied.append(tools)
        return tools

    record = runtime().build_record(tools_factory=factory)
    for config in record["configurations"]:
        denied = [r for r in config["cases"] if r["kind"] == "public" and r["named_denial"]]
        assert len(denied) == 14
        for row in denied:
            for side in ("own", "second"):
                observed = row[side]
                assert observed["outcome"] == "refer" and observed["tier"] == 3
                assert not observed["executed"] and not observed["ledger_receipts"]
    for tools in supplied:
        if (
            tools.data.correct
            and tools.data.correct[0].tier == 3
            and tools.data.correct[0].outcome.value == "deny"
        ):
            assert [r.required_role for r in tools.reviews.values()] == ["named_approver"]
            assert not tools.payments


def test_supplied_tools_authorization_runs_after_the_gate():
    class ExtraCheck(Tools):
        def _authorize(self, state, amount, approval):
            problem = super()._authorize(state, amount, approval)
            return problem or (
                "independent_objection" if state.request.request_id == "r-7-0007" else None
            )

    record = runtime().build_record(tools_factory=ExtraCheck)
    full = record["configurations"][-1]
    assert full["different_cases"] == 0
    case = next(r for r in full["cases"] if r["case_id"] == "public-r-7-0007")
    for side in ("own", "second"):
        assert case[side]["reason"] == "independent_objection"
        assert not case[side]["executed"]
        assert case[side]["audits"][-1]["decision"]["reason"] == "independent_objection"


@pytest.mark.parametrize(
    "setting", ["end_at_audit", "one_tool_per_phase", "refer_on_failure", "no_retry"]
)
def test_removing_one_semantic_is_caught(setting):
    swap = runtime()
    from refund_agent.runtimes.adapter import AdapterSettings

    row = swap.compare(swap.build_cases(), replace(AdapterSettings(), **{setting: False}))
    assert row["different_cases"] > 0


def test_six_turns_stop_every_public_payment_with_an_audit():
    swap = runtime()
    from refund_agent.runtimes.adapter import AdapterSettings

    cases = [case for case in swap.build_cases() if case.kind == "public"]
    row = swap.compare(cases, replace(AdapterSettings(), max_steps=6))
    paid = [r for r in row["cases"] if r["own"]["executed"]]
    assert len(paid) == 4
    assert all(not r["second"]["executed"] for r in paid)
    assert all(r["second"]["reason"] == "step_budget_exhausted" for r in paid)
    assert all(r["second"]["tool_calls"][-1]["name"] == "audit.append" for r in paid)


def test_seeded_runs_are_deterministic_and_read_no_frozen_data(monkeypatch):
    import refund_agent.split as split

    def forbidden(*args, **kwargs):
        pytest.fail("chapter 28 must not open a frozen or held-out set")

    monkeypatch.setattr(split, "freeze", forbidden)
    original = Path.open

    def checked(path, *args, **kwargs):
        assert not {"evaluation", "cap-zero", "golden", "frozen"}.intersection(path.parts)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", checked)
    first = runtime().build_record(seed=7)
    assert first == runtime().build_record(seed=7)
    assert first["data_access"] == {"frozen_reads": 0, "held_out_reads": 0}


def test_runner_imports_no_refund_logic_and_core_imports_no_runtime():
    root = Path(__file__).resolve().parents[2] / "refund_agent"
    runner = (root / "runtimes/runner.py").read_text()
    assert "refund" not in runner.casefold()
    for path in (root / "agent/__init__.py", root / "tools.py", root / "rules.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert "runtimes" not in (node.module or "")
            elif isinstance(node, ast.Import):
                assert all("runtimes" not in alias.name for alias in node.names)


def test_generic_runner_keeps_session_spans_and_retries():
    from refund_agent.runtimes.runner import Call, Reply, Runner, RuntimeSettings

    attempts = []

    def operation(context, call):
        attempts.append(call.name)
        if len(attempts) == 1:
            raise TimeoutError("lost response")
        return 17

    def model(context, session):
        return Reply(calls=[Call("read")]) if not attempts else Reply(value="done")

    runner = Runner(RuntimeSettings())
    session = runner.run(None, model, {"read": operation})
    assert attempts == ["read", "read"]
    assert [span["status"] for span in session.spans] == ["TimeoutError", "ok"]
    assert session.messages[-1]["role"] == "assistant"
    assert session.stop_reason == "no_calls"


def test_a_failed_no_call_turn_returns_the_error_to_the_model():
    from refund_agent.runtimes.runner import Reply, Runner

    seen = []

    def model(context, session):
        return Reply(value="done" if session.messages else "first")

    def before(context, reply):
        seen.append(reply.value)
        if reply.value == "first":
            raise TimeoutError("hook failed")
        return []

    session = Runner().run(None, model, {}, before=before)
    assert seen == ["first", "done"]
    assert session.turns == 2 and session.stop_reason == "no_calls"
    assert session.messages[1] == {"role": "error", "error": "TimeoutError"}


def test_reserved_audit_has_a_runtime_span_and_counts_as_one_turn():
    from refund_agent.runtimes.adapter import RefundAdapter
    from refund_agent.runtimes.faults import FaultTools

    case = next(case for case in runtime().build_cases() if case.case_id == "public-r-7-0007")
    tools = FaultTools(Tools(case.data))
    adapter = RefundAdapter(tools)
    trace = adapter.run(case.request)
    assert [span["tool"] for span in adapter.session.spans] == [
        call["name"] for call in tools.calls
    ]
    assert adapter.session.spans[-1]["tool"] == "audit.append"
    assert adapter.session.turns == len(trace.steps)


@pytest.mark.parametrize("budget", [0, -1, True, 1.5])
def test_adapter_rejects_invalid_step_budgets(budget):
    from refund_agent.runtimes.adapter import AdapterSettings, RefundAdapter

    with pytest.raises(ValueError):
        RefundAdapter(
            Tools(generate(7, orders=40, requests=40)), settings=AdapterSettings(max_steps=budget)
        )
