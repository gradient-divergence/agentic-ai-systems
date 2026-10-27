"""Run one exercise submission with a training-only import and file view."""

import contextlib
import importlib.abc
import importlib.machinery
import importlib.util
import json
import os
import sys
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[3]
COMPANION = ROOT / "companion"
sys.path[:0] = [str(ROOT), str(COMPANION)]

# Trusted setup imports definitions, never generates requests in this process.
from demos import memory_rag_demo as demo  # noqa: E402
from refund_agent.data.generate import Dataset  # noqa: E402

utc, request_state = demo.utc, demo.request_state
PolicyRegister, Request = demo.PolicyRegister, demo.c.Request
del demo

# The reader needs Python helpers and model definitions, never native data readers.
SAFE_IMPORTS = frozenset(
    "abc builtins codecs collections contextlib copy copyreg csv dataclasses datetime decimal encodings enum "
    "functools hashlib importlib inspect io itertools json math numbers operator os pathlib "
    "pydantic pydantic_core annotated_types re runpy statistics string struct sys textwrap "
    "threading time traceback linecache tokenize token types typing typing_extensions warnings weakref zoneinfo __future__ "
    "_collections_abc _csv _datetime _decimal _io _json _strptime _thread _weakrefset".split()
) | frozenset(sys.builtin_module_names)


def register_state(register):
    snapshot = register.snapshot(utc(10, 1))
    return {
        "records": [asdict(record) for record in register.records],
        "texts": dict(snapshot.texts),
    }


class Boundary(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def __init__(self, payload, solution):
        self.violations = []
        self.solution = solution
        self.libraries = [Path(p).resolve() for p in (sys.prefix, sys.base_prefix)]
        safe_names = (
            "refund_agent.contracts",
            "refund_agent.rules",
            "refund_agent.memory.register",
            "refund_agent.retrieval.clauses",
        )
        self.modules = {name: sys.modules[name] for name in safe_names}
        self.sources = {Path(module.__file__).resolve() for module in self.modules.values()}
        self.sources.add(COMPANION / "exercises/memory-rag/late_policy_06.json")
        self.sources.add(Path(importlib.util.cache_from_source(str(solution))).resolve())
        view = self.training_view(payload)
        self.modules["demos.memory_rag_demo"] = view
        for name in ("refund_agent", "refund_agent.memory", "refund_agent.retrieval", "demos"):
            package = ModuleType(name)
            package.__path__ = []
            self.modules[name] = package
        self.modules.update({"companion." + name: module for name, module in self.modules.items()})
        package = ModuleType("companion")
        package.__path__ = []
        self.modules["companion"] = package

    def refuse(self, route):
        message = f"exercise read held-out data via {route}"
        self.violations.append(message)
        raise AssertionError(message)

    def training_view(self, payload):
        boundary = self

        class TrainingDataset(Dataset):
            def __getattribute__(self, name):
                if name == "requests":
                    boundary.refuse("data.requests; use training_data()[1]")
                return super().__getattribute__(name)

        data = TrainingDataset.model_validate_json(json.dumps(payload["data"]))
        requests = [Request.model_validate_json(json.dumps(row)) for row in payload["requests"]]
        assert not data.model_dump()["requests"]
        assert all(request.at < datetime.fromisoformat(payload["cut"]) for request in requests)

        def training_data(seed=7):
            if seed != 7:
                self.refuse("training_data with a different cohort")
            return data.model_copy(deep=True), [r.model_copy(deep=True) for r in requests]

        def demonstration_register():
            register = PolicyRegister()
            for policy in data.policies:
                register.announce(policy, policy.effective_at - timedelta(days=7))
            return register

        def unavailable(name):
            if name.startswith("__"):
                raise AttributeError(name)
            self.refuse(f"exercise API {name}")

        view = ModuleType("demos.memory_rag_demo")
        view.training_data = training_data
        view.demonstration_register = demonstration_register
        view.request_state = request_state
        view.utc = utc
        view.__getattr__ = unavailable
        return view

    def readable(self, path):
        if any(path.is_relative_to(root) for root in self.libraries):
            return True
        if path == self.solution or path in self.sources:
            return True
        if path.is_relative_to(COMPANION):
            return False
        # Reader helper modules are code; companion/cache data is never an input.
        return path.is_relative_to(self.solution.parent) and path.suffix in {".py", ".pyc"}

    def find_spec(self, fullname, path=None, target=None):
        if fullname in self.modules:
            module = self.modules[fullname]
            return importlib.machinery.ModuleSpec(
                fullname, self, is_package=hasattr(module, "__path__")
            )
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if fullname.split(".")[0] not in SAFE_IMPORTS:
            origin = Path(spec.origin).resolve() if spec and spec.origin else None
            if not (
                origin
                and origin.is_relative_to(self.solution.parent)
                and not origin.is_relative_to(COMPANION)
                and origin.suffix == ".py"
            ):
                self.refuse(f"import {fullname}")
        if spec and spec.origin and spec.origin not in {"built-in", "frozen"}:
            if not self.readable(Path(spec.origin).resolve()):
                self.refuse(f"import {fullname}")
        return None

    def create_module(self, spec):
        return self.modules[spec.name]

    def exec_module(self, module):
        pass

    def audit_read(self, event, args):
        if event in {
            "subprocess.Popen",
            "os.system",
            "os.posix_spawn",
            "os.fork",
            "os.forkpty",
            "os.exec",
        }:
            self.refuse(f"unguarded process {event}")
        if event != "open" or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        if args[2] & os.O_ACCMODE == os.O_WRONLY:
            return
        path = Path(os.fsdecode(args[0])).resolve()
        if not self.readable(path):
            self.refuse(f"file {path}")

    def install(self):
        # Drop real packages and their attributes as well as the generator/split modules.
        for name, module in list(sys.modules.items()):
            origin = getattr(module, "__file__", None)
            if name != "__main__" and (
                name.split(".")[0] not in SAFE_IMPORTS
                or (origin and Path(origin).resolve().is_relative_to(COMPANION))
            ):
                del sys.modules[name]
        sys.path[:] = [p for p in sys.path if p not in {str(ROOT), str(COMPANION)}]
        sys.path.insert(0, str(self.solution.parent))
        sys.meta_path.insert(0, self)
        sys.addaudithook(self.audit_read)


def main():
    payload = json.load(sys.stdin)
    path = Path(sys.argv[1]).resolve()
    boundary = Boundary(payload, path)
    boundary.install()
    try:
        with contextlib.redirect_stdout(sys.stderr):
            spec = importlib.util.spec_from_file_location("memory_exercise_solution", path)
            if spec is None or spec.loader is None:
                raise ValueError("solution must be an importable Python file")
            solution = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(solution)
            register = solution.late_register()
            before = register_state(register)
            rows = solution.audit(register) if payload["audit"] else None
            after = register_state(register)
        if boundary.violations:
            raise AssertionError(boundary.violations[0])
        result = {"before": before, "rows": rows, "after": after}
    except Exception as error:
        result = {"error": type(error).__name__, "message": str(error)}
        if boundary.violations:
            result = {"error": "AssertionError", "message": boundary.violations[0]}
    print(json.dumps(result, default=lambda value: value.isoformat()))


if __name__ == "__main__":
    main()
