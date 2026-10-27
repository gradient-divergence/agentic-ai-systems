"""Student starter: mount RunnerC, then carry the missing no-retry rule."""

from dataclasses import replace

import runner_c

from refund_agent.runtimes.adapter import AdapterSettings, RefundAdapter


def mount(tools, model=None, settings=None):
    # TODO: a timeout must end this run; a caller may retry the whole run.
    settings = replace(settings or AdapterSettings(), no_retry=False)
    return RefundAdapter(tools, model, settings=settings, runner_type=runner_c.RunnerC)
