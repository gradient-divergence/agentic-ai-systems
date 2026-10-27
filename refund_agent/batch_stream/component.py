"""Chapter 31 registers its complete event-time module closure with release."""

import fcntl
import json
from contextlib import contextmanager
from datetime import timedelta
from pathlib import Path

from refund_agent.batch_stream.log import EventLog
from refund_agent.batch_stream.tools import EventTimeTools
from refund_agent.batch_stream.view import ALLOWED, MARGIN, duration
from refund_agent.release.extensions import Component, RuntimeStore, register, resource_guard
from refund_agent.release.journal import Journal

MODULE = "refund_agent.batch_stream.component"
RUN_JOURNAL = RuntimeStore(
    "../../.tmp-sol/batch-stream-run/journal.jsonl", "morgan", "append"
)
MODULES = tuple(
    f"refund_agent.batch_stream.{name}"
    for name in (
        "__init__",
        "log",
        "view",
        "tools",
        "windows",
        "authority",
        "replay",
        "generate",
        "suite",
    )
)


class RunJournal(Journal):
    @contextmanager
    def transaction(self):
        # Read through a separate handle; the guarded writer may seek only to EOF.
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as stream:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            with self.path.open(encoding="utf-8") as reader:
                rows = [json.loads(line) for line in reader]
            if any(row.get("entry") != i for i, row in enumerate(rows, 1)):
                raise ValueError("release journal sequence differs")
            yield rows, stream


def selector(*, margin=MARGIN, allowed_lateness=ALLOWED):
    return {
        "name": "batch-stream",
        "module": MODULE,
        "settings": {
            "margin_seconds": duration(margin).total_seconds(),
            "allowed_seconds": duration(allowed_lateness).total_seconds(),
        },
    }


def compose(*, base, kwargs, candidate, settings, mode, context):
    event_log = context.get("event_log")
    if not isinstance(event_log, EventLog):
        raise ValueError("batch_stream_event_log_required")
    base = type("BatchStreamCandidateTools", (EventTimeTools, base), {})
    return base, {
        **kwargs,
        "event_log": event_log,
        "margin": timedelta(seconds=settings["margin_seconds"]),
        "allowed_lateness": timedelta(seconds=settings["allowed_seconds"]),
        "batch_stream_fixture": context.get("batch_stream_fixture", False),
        "batch_stream_case": context.get("batch_stream_case"),
        "batch_stream_mode": mode,
    }


@resource_guard(0)
def run_release(candidate, suite):
    from refund_agent.release.rule import evaluate_release

    root = Path(__file__).resolve().parents[1]
    return evaluate_release(candidate, suite, RunJournal(root / RUN_JOURNAL.path))


from refund_agent.batch_stream import suite  # noqa: E402

register(
    Component(
        name="batch-stream",
        version=EventTimeTools.version,
        module=MODULE,
        modules=MODULES,
        order=31,
        stores=(RUN_JOURNAL,),
        hooks={
            "compose": compose,
            "cases": suite.cases,
            "run_case": suite.run_case,
            "run_loop_case": suite.run_loop_case,
            "fixture_context": suite.fixture_context,
            "changed_source": suite.changed_source,
            "before_run": suite.before_run,
            "after_run": suite.after_run,
        },
        slots={
            "31": "event-time absence, return window and correction modes; must-pass",
        },
    )
)
