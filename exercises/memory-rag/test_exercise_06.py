"""Acceptance checks for a reader's audit, without changing the chapter's demo.

Run with MEMORY_RAG_SOLUTION pointing to your completed starter file.
The shipped starter's unfinished audit is deliberately rejected by these checks.
"""

import json
import os
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path
from textwrap import indent

import pytest

from demos import memory_rag_demo as demo
from refund_agent import rules, split
from refund_agent.contracts import Policy
from refund_agent.data import generate as generator
from refund_agent.memory.memory import PolicyMemory
from refund_agent.memory.register import PolicyRegister, project


def training_payload():
    data, requests = demo.training_data()
    request_ids = {r.request_id for r in requests}
    order_ids = {f"o-{r.request_id.rsplit('-', 1)[1]}" for r in requests}
    # Only these rows cross the process boundary; no full dataset goes to the worker.
    view = data.model_copy(
        update={
            "requests": [],
            "orders": [row for row in data.orders if row.order_id in order_ids],
            "shipments": [row for row in data.shipments if row.order_id in order_ids],
            "retrievals": [row for row in data.retrievals if row.request_id in request_ids],
            "correct": [row for row in data.correct if row.request_id in request_ids],
            "public_scenarios": [key for key in data.public_scenarios if key in request_ids],
            "instructor_scenarios": [],
        }
    )
    return {
        "data": view.model_dump(mode="json"),
        "requests": [r.model_dump(mode="json") for r in requests],
        "cut": split.CUTS[0].isoformat(),
    }


def load(path, *, audit=True):
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-B",
            str(Path(__file__).with_name("exercise_worker.py")),
            str(Path(path).resolve()),
        ],
        input=json.dumps({**training_payload(), "audit": audit}),
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    receipt = json.loads(result.stdout)
    if "error" in receipt:
        error_type = (
            NotImplementedError if receipt["error"] == "NotImplementedError" else AssertionError
        )
        raise error_type(receipt["message"])
    return receipt


def register_state(register):
    snapshot = register.snapshot(demo.utc(10, 1))
    return json.loads(
        json.dumps(
            {
                "records": [asdict(record) for record in register.records],
                "texts": dict(snapshot.texts),
            },
            default=lambda value: value.isoformat(),
        )
    )


def fixture_register(late=None):
    fixture = json.loads(Path(__file__).with_name("late_policy_06.json").read_text())
    register = PolicyRegister()
    data, _ = demo.training_data()
    announcements = [(p.effective_at - timedelta(days=7), p) for p in data.policies]
    announcements.append(
        (
            datetime.fromisoformat(fixture["recorded_at"]),
            late or Policy.model_validate_json(json.dumps(fixture["policy"])),
        )
    )
    for recorded_at, policy in sorted(announcements, key=lambda row: row[0]):
        register.announce(policy, recorded_at)
    return register


def changed_rows():
    # A hand-checked reference result, not a reader's implementation of the audit.
    return [
        {
            "request_id": "r-7-0469",
            "version_then": "2026-05-01",
            "version_now": "2026-06-10",
            "right_then": True,
            "wrong_now": True,
        }
    ]


def _completed_reader_source():
    """A real audit for testing submitted files; the public starter stays unfinished."""
    source = Path(__file__).with_name("starter_06.py").read_text()
    body = """    from demos.memory_rag_demo import request_state
    from refund_agent import rules
    from refund_agent.memory.register import project

    data, requests = training_data()
    rows = []
    for request in requests:
        def decision(known_at):
            policies = project(register.snapshot(known_at), known_at)
            policy = max(
                (p for p in policies if p.effective_at <= request.at),
                key=lambda p: p.effective_at,
            )
            return policy.version, rules.draft(request_state(data, request, policy))

        then, before = decision(request.at)
        now, after = decision(utc(10, 1))
        if (before.outcome, before.reason) != (after.outcome, after.reason):
            rows.append({
                "request_id": request.request_id,
                "version_then": then,
                "version_now": now,
                "right_then": True,
                "wrong_now": True,
            })
    return rows"""
    return source.replace(
        '    raise NotImplementedError("Query both knowledge times and compare the structured decisions")',
        body,
    )


def test_completed_reader_file_is_accepted(tmp_path):
    path = tmp_path / "solution.py"
    path.write_text(_completed_reader_source())
    result = load(path)
    assert result["rows"] == changed_rows()
    acceptance(result)


def test_completed_reader_using_namespace_imports_is_accepted(tmp_path):
    path = tmp_path / "solution.py"
    path.write_text(
        _completed_reader_source()
        .replace("from demos", "from companion.demos")
        .replace("from refund_agent", "from companion.refund_agent")
    )
    acceptance(load(path))


@pytest.mark.parametrize("mistake", ["wrong-policy", "unchanged-row", "mutated-register"])
def test_incorrect_submitted_audits_are_rejected(tmp_path, mistake):
    source = _completed_reader_source()
    if mistake == "wrong-policy":
        source = source.replace("return_days=25", "return_days=20").replace(
            "clause_text('eligibility', 25)", "clause_text('eligibility', 20)"
        )
    elif mistake == "unchanged-row":
        source = source.replace(
            "    return rows", "    return rows + [dict(request_id='r-7-0085')]"
        )
    else:
        source = source.replace(
            "    return rows", "    register.retain(utc(10, 1))\n    return rows"
        )
    path = tmp_path / "solution.py"
    path.write_text(source)
    with pytest.raises(AssertionError):
        acceptance(load(path))


LEAK_PROBES = {
    "dataset": "leaked = training_data()[0].requests",
    "dataset-copy": "leaked = training_data()[0].model_copy().requests",
    "demo-generator": "from demos import memory_rag_demo as demo\nleaked = demo.generate().requests",
    "source-generator": (
        "from refund_agent.data.generate import generate\nleaked = generate().requests"
    ),
    "namespace-generator": (
        "from companion.refund_agent.data.generate import generate\n"
        "from refund_agent import split\n"
        "leaked = [r for r in generate().requests if r.at >= split.CUTS[0]]\n"
        "assert (sum(r.at < split.CUTS[1] for r in leaked), "
        "sum(r.at >= split.CUTS[1] for r in leaked)) == (81, 121)"
    ),
    "namespace-data-package": "from companion.refund_agent.data import generate",
    "data-package": "from refund_agent import data",
    "data-history": "import refund_agent.data.history",
    "data-live": "import companion.refund_agent.data.live",
    "relative-generator": (
        "from importlib import import_module\n"
        "leaked = import_module('.generate', 'companion.refund_agent.data').generate().requests"
    ),
    "arbitrary-generator-name": (
        "import sys\n"
        "sys.path.insert(0, {data_parent!r})\n"
        "from data.generate import generate\nleaked = generate().requests"
    ),
    "validation": (
        "from refund_agent import split\n"
        "leaked = split.time_split(training_data()[1], split.CUTS, timedelta(0))['validation']"
    ),
    "golden": (
        "from refund_agent import split\n"
        "leaked = split.time_split(training_data()[1], split.CUTS, timedelta(0))['golden']"
    ),
    "namespace-split": "from companion.refund_agent import split",
    "relative-split": (
        "from importlib import import_module\n"
        "leaked = import_module('.split', 'companion.refund_agent')"
    ),
    "golden-set": (
        "from refund_agent.split import GoldenSet\n"
        "leaked = GoldenSet(dict(name='probe'), (), ()).consult('exercise-06', 'audit')"
    ),
    "namespace-demo-generator": (
        "from companion.demos import memory_rag_demo as demo\nleaked = demo.generate().requests"
    ),
    "namespace-training-data": (
        "from companion.demos.memory_rag_demo import training_data as other_training\n"
        "leaked = other_training()[0].requests"
    ),
    "file-loader-generator": (
        "import importlib.util\n"
        "spec = importlib.util.spec_from_file_location('another_generator', {generator!r})\n"
        "module = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(module)\nleaked = module.generate().requests"
    ),
    "exercise-api": (
        "from importlib import import_module\n"
        "api = import_module('companion.exercises.memory-rag.test_exercise_06')\n"
        "leaked = api.demo.training_data()[0].requests"
    ),
    "shipped-data-path": "from pathlib import Path\nleaked = Path({shipped!r}).read_bytes()",
    "cached-data-path": "from pathlib import Path\nleaked = Path({cached!r}).read_text()",
    "cached-io-open": "import io\nleaked = io.open({cached!r}).read()",
    "cached-os-open": (
        "import os\nfd = os.open({cached!r}, os.O_RDONLY)\n"
        "try:\n    leaked = os.read(fd, 100)\nfinally:\n    os.close(fd)"
    ),
    "native-data-reader": "import numpy as np\nleaked = np.fromfile({cached!r}, dtype=np.uint8)",
    "nested-process": "import os\nos.system('true')",
}


@pytest.fixture(scope="module")
def held_out_file(tmp_path_factory):
    cache = tmp_path_factory.mktemp("cached-rows")
    generator.write_data(generator.generate(), cache)
    return cache / "data.json"


@pytest.mark.parametrize("route", LEAK_PROBES)
@pytest.mark.parametrize("phase", ["import", "audit"])
def test_subprocess_boundary_rejects_ordinary_leaks(tmp_path, held_out_file, route, phase):
    root = Path(__file__).resolve().parents[3]
    probe = LEAK_PROBES[route].format(
        data_parent=str(root / "companion/refund_agent"),
        generator=str(root / "companion/refund_agent/data/generate.py"),
        shipped=str(root / "companion/refund_agent/data/positive.json"),
        cached=str(held_out_file),
    )
    if route != "namespace-generator":
        # A caught denial must still fail acceptance; it cannot erase the attempted read.
        probe = (
            "try:\n"
            + indent(probe, "    ")
            + "\nexcept (AssertionError, ImportError, PermissionError):\n    pass\n"
        )
    source = _completed_reader_source()
    if phase == "import":
        source += "\n" + probe
    else:
        source = source.replace(
            "def audit(register):\n",
            "def audit(register):\n" + indent(probe.rstrip() + "\n", "    "),
        )
    path = tmp_path / "solution.py"
    path.write_text(source)
    with pytest.raises(AssertionError, match="held-out"):
        acceptance(load(path))


def test_exercise_dataset_copies_and_serializers_expose_only_training_records(tmp_path):
    check = """data, requests = training_data()
assert len(requests) == 398
assert all(r.at < utc(7, 1) for r in requests)
request_ids = {r.request_id for r in requests}
order_ids = {f"o-{r.request_id.rsplit('-', 1)[1]}" for r in requests}
for view in (data, data.model_copy(), type(data).model_validate_json(data.model_dump_json())):
    rows = view.model_dump()
    assert rows['requests'] == []
    assert {r['order_id'] for r in rows['orders']} == order_ids
    assert {r['order_id'] for r in rows['shipments']} == order_ids
    assert {r['request_id'] for r in rows['retrievals']} == request_ids
    assert {r['request_id'] for r in rows['correct']} == request_ids
    assert set(rows['public_scenarios']) <= request_ids
    assert rows['instructor_scenarios'] == []
"""
    path = tmp_path / "solution.py"
    path.write_text(_completed_reader_source() + "\n" + check)
    acceptance(load(path))


def acceptance(solution):
    expected = fixture_register()
    original = register_state(expected)
    assert solution["before"] == original, "register must match the prescribed late-policy fixture"
    rows = solution["rows"]
    data, requests = demo.training_data()
    changed = {}
    for request in requests:

        def decision(known_at):
            policies = project(expected.snapshot(known_at), known_at)
            policy = max(
                (p for p in policies if p.effective_at <= request.at), key=lambda p: p.effective_at
            )
            return policy, rules.draft(demo.request_state(data, request, policy))

        then, before = decision(request.at)
        now, after = decision(demo.utc(10, 1))
        if (before.outcome, before.reason) != (after.outcome, after.reason):
            changed[request.request_id] = {
                "request_id": request.request_id,
                "version_then": then.version,
                "version_now": now.version,
                "right_then": True,
                "wrong_now": True,
            }
    assert len(rows) == len(changed), "list exactly the changed drafts"
    assert {r["request_id"]: r for r in rows} == changed
    assert all(r["right_then"] is True and r["wrong_now"] is True for r in rows)
    assert solution["after"] == original, "audit must keep register records and texts unchanged"
    accepted = PolicyMemory(expected, demo.utc(6, 11))
    assert accepted.retrieve("damaged", demo.utc(6, 13), 1).refusal is None
    assert accepted.retrieve("damaged", demo.utc(6, 18), 1).refusal is None
    assert (
        accepted.retrieve("damaged", demo.utc(6, 18) + timedelta(seconds=1), 1).refusal
        == "beyond_notice"
    )


def test_untouched_starter_is_rejected():
    with pytest.raises(NotImplementedError):
        acceptance(load(Path(__file__).with_name("starter_06.py")))


def test_reader_solution_if_supplied():
    path = os.environ.get("MEMORY_RAG_SOLUTION")
    if path:
        acceptance(load(Path(path)))
    else:
        # No solution is supplied in the public companion; check its independent register setup.
        starter = load(Path(__file__).with_name("starter_06.py"), audit=False)
        assert starter["before"] == starter["after"] == register_state(fixture_register())
