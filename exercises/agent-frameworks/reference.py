"""Reference solution: carry the agent's semantics, including one call per phase."""

from runner_c import RunnerC

from refund_agent.runtimes.adapter import AdapterSettings, RefundAdapter


def mount(tools, model=None, settings=None):
    return RefundAdapter(tools, model, settings=settings or AdapterSettings(), runner_type=RunnerC)
