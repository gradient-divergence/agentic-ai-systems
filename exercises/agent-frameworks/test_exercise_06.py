"""Run with AGENT_FRAMEWORKS_SOLUTION pointing to the reader's mount function."""

import importlib.util
import os
import threading
from pathlib import Path

import pytest
from runner_c import RunnerC

from refund_agent import contracts as c
from refund_agent.model import ScriptedModel
from refund_agent.runtimes.adapter import AdapterSettings
from refund_agent.runtimes.faults import FaultTools
from refund_agent.runtimes.runner import Call, Reply
from refund_agent.runtimes.swap import build_cases, compare
from refund_agent.tools import Tools


@pytest.fixture(scope="module")
def solution():
    selected = os.environ.get("AGENT_FRAMEWORKS_SOLUTION")
    assert selected, "set AGENT_FRAMEWORKS_SOLUTION to the student file or reference.py explicitly"
    path = Path(selected)
    assert path.is_file(), "provide the exercise's mount function"
    spec = importlib.util.spec_from_file_location("agent_frameworks_solution", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert callable(getattr(module, "mount", None)), (
        "provide mount(tools, model=None, settings=None)"
    )
    return module


def test_third_runtime_matches_the_entire_suite_including_resumes(solution, monkeypatch):
    mounted, invoked = [], []
    original = RunnerC.run

    def observed_run(runner, *args, **kwargs):
        invoked.append(runner)
        return original(runner, *args, **kwargs)

    def checked_mount(tools, model=None, settings=None):
        adapter = solution.mount(tools, model, settings=settings)
        assert isinstance(adapter.runner, RunnerC), "mount must use the third runtime"
        mounted.append(adapter.runner)
        return adapter

    monkeypatch.setattr(RunnerC, "run", observed_run)
    result = compare(build_cases(), AdapterSettings(), adapter_factory=checked_mount)
    assert invoked == mounted  # Every mounted runtime actually drives its refund run.
    assert result["different_cases"] == result["money_differences"] == 0
    assert result["unauthorized_cents"] == 0
    assert all(len(row["second"]["ledger_receipts"]) <= 1 for row in result["cases"])


def test_a_two_read_reply_is_refused_before_parallel_dispatch(solution, monkeypatch):
    class TwoReads(ScriptedModel):
        def reply(self, state):
            return [c.ToolCall(name="order.lookup"), c.ToolCall(name="policy.retrieve")]

    case = next(case for case in build_cases() if case.case_id == "public-r-7-0007")
    tools = FaultTools(Tools(case.data))
    adapter = solution.mount(tools, TwoReads(), settings=AdapterSettings())
    batches = []
    dispatch = RunnerC._dispatch

    def checked_dispatch(runner, context, reply, registry, session, fatal_errors):
        if len(reply.calls) > 1:
            batches.append(reply.calls)
            # The wrong adapter would deadlock in the agent's locked tools boundary.
            raise AssertionError("the adapter must refuse multiple reads before dispatch")
        return dispatch(runner, context, reply, registry, session, fatal_errors)

    monkeypatch.setattr(RunnerC, "_dispatch", checked_dispatch)
    trace = adapter.run(case.request)
    assert not batches, "a malformed reply must never enter the parallel path"
    assert trace.decision.outcome == c.Outcome.REFER
    assert trace.decision.reason == "model_error"
    assert trace.failure_detail == "model_error:order:TypeError"
    assert [call["name"] for call in tools.calls] == ["audit.append"]
    assert not tools.payments
    assert [span["tool"] for span in adapter.session.spans] == ["audit.append"]


def test_third_runtime_really_runs_multiple_reads_in_parallel():
    barrier = threading.Barrier(2, timeout=5)
    done = []

    def read(context, call):
        barrier.wait()
        done.append(call.name)
        return call.name

    def model(context, session):
        return Reply([Call("left"), Call("right")]) if not done else Reply(value="done")

    runner = RunnerC(read_tools={"left", "right"})
    session = runner.run(None, model, {"left": read, "right": read}, stop=lambda _: bool(done))
    assert set(done) == {"left", "right"}
    assert len(session.spans) == 2


def test_third_runtime_has_different_defaults():
    runner = RunnerC()
    calls = []

    def failing(context, call):
        calls.append(call.name)
        raise TimeoutError("one attempt")

    with pytest.raises(TimeoutError):
        runner.run(None, lambda _, session: Reply([Call("read")]), {"read": failing})
    assert calls == ["read"]  # The second runtime retries once and returns failures to the model.
    session = runner.run(None, lambda _, session: Reply(value="draft"), {})
    assert session.turns == 6 and session.stop_reason == "turn_limit"
