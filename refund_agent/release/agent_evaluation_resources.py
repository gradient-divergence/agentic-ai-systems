"""Enforce selected component file manifests from the first package import."""

import _thread
import asyncio
import builtins
import csv
import ctypes
import hashlib
import importlib
import io
import mmap
import os
import re
import subprocess
import sys
import sysconfig
import threading
import zipfile
import zipimport
from base64 import urlsafe_b64encode
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import lru_cache, wraps
from importlib import metadata
from importlib.machinery import (
    BYTECODE_SUFFIXES,
    EXTENSION_SUFFIXES,
    SOURCE_SUFFIXES,
    ExtensionFileLoader,
    ModuleSpec,
    PathFinder,
    SourceFileLoader,
    SourcelessFileLoader,
)
from importlib.util import resolve_name
from multiprocessing.pool import Pool
from multiprocessing.process import BaseProcess
from pathlib import Path
from weakref import WeakValueDictionary

# Initialize the trusted interpreter's Python API handle before component auditing.
del ctypes

# region chapter29:core-resources
IMPLEMENTATION_FILES = (
    "__init__.py",
    "adaptation/__init__.py",
    "data/__init__.py",
    "memory/__init__.py",
    "offers/__init__.py",
    "offpolicy/__init__.py",
    "retrieval/__init__.py",
    "scenarios/__init__.py",
    "shadow/__init__.py",
    "agent/__init__.py",
    "release/__init__.py",
    "runtimes/__init__.py",
    "verifier/__init__.py",
    "offpolicy/estimators.py",
    "offpolicy/reward_model.py",
    "offpolicy/rule.py",
    "offpolicy/support.py",
    "data/positive.json",
    "release/candidate.py",
    "release/extensions.py",
    "release/agent_evaluation_resources.py",
    "release/requirement.json",
    "release/suite.py",
    "release/graders.py",
    "release/rule.py",
    "release/__main__.py",
    "runtimes/swap.py",
    "runtimes/faults.py",
    "runtimes/adapter.py",
    "runtimes/runner.py",
    "baseline.py",
    "evaluation.py",
    "tools.py",
    "rules.py",
    "contracts.py",
    "model/__init__.py",
    "data/generate.py",
    "labels.py",
    "split.py",
    "memory/support.py",
    "memory/assemble.py",
    "memory/register.py",
    "memory/memory.py",
    "adaptation/suite.py",
    "adaptation/arms.py",
    "adaptation/drafter.py",
    "adaptation/prompt.py",
    "adaptation/experiment.py",
    "retrieval/index.py",
    "retrieval/encoders.py",
    "retrieval/clauses.py",
    "retrieval/questions.py",
    "scenarios/suite.py",
    "scenarios/model.py",
    "scenarios/materialize.py",
    "scenarios/checks.py",
    "scenarios/tokens.py",
    "verifier/independent.py",
    "verifier/record.py",
    "verifier/experiment.py",
    "verifier/fixture.py",
    "verifier/agreeing.py",
    "offpolicy/policies.py",
    "offpolicy/record.py",
    "offers/notices.py",
    "release/owners.json",
    "release/journal.py",
    "shadow/__main__.py",
    "shadow/tools.py",
    "shadow/compare.py",
    "shadow/offers.py",
    "data/live.py",
    "../demos/data/adaptation_selection_record.json",
    "../demos/data/adaptation_report_05.json",
    "../demos/data/adaptation_model_choice_record.json",
    "../demos/data/practical_rl_evaluation_record.json",
    "../book_outputs/_fnd.multi-agent-systems.arms-01.txt",
    "../book_outputs/_fnd.memory-rag.funnel-01.txt",
    "../demos/reasoning_planning_demo.py",
    "selection.py",
    "offers/policies.py",
    "verifier/shared_error.py",
)
# endregion chapter29:core-resources


def core_paths():
    root = Path(__file__).resolve().parents[1]
    paths = tuple(root / name for name in IMPLEMENTATION_FILES)
    return _resolved_core(tuple((path, file_identity(path)) for path in paths))


def file_identity(path):
    """Watch content edits, replacements and links, including a restored mtime."""
    try:
        stat, link = path.stat(), path.lstat()
        return (
            stat.st_mtime_ns,
            stat.st_size,
            stat.st_ctime_ns,
            stat.st_dev,
            stat.st_ino,
            stat.st_nlink,
            link.st_ctime_ns,
            link.st_ino,
        )
    except OSError:
        return None  # The actual read retains its original error and audit checks.


@lru_cache(maxsize=16)
def _resolved_core(identities):
    return frozenset(path.resolve() for path, _ in identities)


# region chapter29:resource-reads
READS = {}
WRITES = {}
IMPORTS = {}
ACCESS = {}
CORE_MODULES = frozenset(
    "refund_agent"
    if name == "__init__.py"
    else "refund_agent." + name.removesuffix(".py").replace("/", ".").removesuffix(".__init__")
    for name in IMPLEMENTATION_FILES
    if name.endswith(".py") and not name.startswith("..")
)
_SCOPE = ContextVar("release_resource_scope", default=None)
_GUARDED_OPEN = ContextVar("release_guarded_open", default=False)
_REHEARSAL_PROCESS = ContextVar("release_rehearsal_process", default=None)
_FIXTURE = ContextVar("release_fixture_construction", default=None)
_CONSTRUCTED_EXECUTION = ContextVar("release_constructed_execution", default=None)
_APPEND_HANDLES = WeakValueDictionary()
_LIBRARIES = tuple(
    Path(sysconfig.get_path(key)).resolve()
    for key in ("stdlib", "platstdlib", "purelib", "platlib")
)
_CHAPTER_ROOT = Path(__file__).resolve().parent
# Installed numerical, model, data and tooling imports already admitted by the contract.
_APPROVED_DISTRIBUTIONS = frozenset(
    {
        "numpy",
        "pandas",
        "pydantic",
        "scikit-learn",
        "scipy",
        "duckdb",
        "polars",
        "annotated-types",
        "pydantic-core",
        "typing-extensions",
        "typing-inspection",
        "python-dateutil",
        "pytz",
        "tzdata",
        "six",
        "joblib",
        "threadpoolctl",
        "polars-runtime-32",
        "absl-py",
        "contourpy",
        "coverage",
        "cycler",
        "fonttools",
        "immutabledict",
        "iniconfig",
        "kiwisolver",
        "matplotlib",
        "ortools",
        "packaging",
        "pillow",
        "pluggy",
        "protobuf",
        "pygments",
        "pyparsing",
        "pytest",
        "pytest-cov",
        "ruff",
        "setuptools",
        "ty",
    }
)
_UNHASHED_METADATA_FILES = frozenset({"RECORD", "INSTALLER", "REQUESTED", "direct_url.json"})


# region chapter29:import-scope
def _module_location(name):
    """Locate cached or static modules without executing package initializers."""
    search, origin = None, None
    parts = name.split(".")
    for i, part in enumerate(parts):
        if i and search is None:
            return None, None  # A non-package's attributes are not module dependencies.
        prefix = ".".join(parts[: i + 1])
        module = sys.modules.get(prefix)
        spec = (
            getattr(module, "__spec__", None)
            if module is not None
            # A dotted namespace spec requires a parent in sys.modules.
            else PathFinder.find_spec(part, search)
        )
        if module is None and spec is None:
            parent = sys.modules.get(".".join(parts[:i]))
            if parent is not None and part in vars(parent):
                return None, None
            # Check and record the known parent, never an invented attribute module.
            return ".".join(parts[:i]), origin
        found = getattr(module, "__file__", None) or getattr(spec, "origin", None)
        locations = getattr(module, "__path__", None)
        if locations is None:
            locations = getattr(spec, "submodule_search_locations", None)
        try:
            search = list(locations) if locations is not None else None
        except KeyError:
            # A cached namespace can outlive its parent; keep its concrete paths.
            search = list(locations._path)
        origin = found or (next(iter(search), None) if search else None)
    return prefix, origin


def module_origin(name):
    """Return the origin used by first-party dependency checks."""
    return _module_location(name)[1]


def first_party_path(origin):
    if origin in (None, "built-in", "frozen"):
        return None
    path = Path(origin).absolute()
    resolved = path.resolve()
    if any(path.is_relative_to(root) or resolved.is_relative_to(root) for root in _LIBRARIES):
        return None
    return resolved


def _core_import(frame):
    module = sys.modules.get(frame.f_globals.get("__name__"))
    if module is None or module.__name__ not in CORE_MODULES:
        return False
    function = vars(module).get(frame.f_code.co_name)
    while function is not None:
        if getattr(function, "__code__", None) is frame.f_code:
            return frame.f_globals is function.__globals__
        function = getattr(function, "__wrapped__", None)
    return False


def _import_access(name, caller):
    if _core_import(caller):
        return  # An actual core function's import is not the calling component's dependency.
    dependency, origin = _module_location(name)
    path = first_party_path(origin)
    if path is None:
        return
    frame = sys._getframe(1)
    while frame is not None:
        owner = (frame.f_globals.get("__name__"), frame.f_globals.get("__file__"))
        if (
            owner[0]
            and owner[0].startswith("refund_agent.")
            and owner[0] not in CORE_MODULES
            and owner[1]
        ):
            owner = (owner[0], Path(owner[1]).resolve())
            access = ACCESS.get(owner)
            if _SCOPE.get() is None and access is None:
                IMPORTS.setdefault(owner, set()).add(dependency)
                return
            if access is not None:
                code, _ = _SCOPE.get() or access
                if path not in code:
                    raise ValueError("component import of undeclared first-party dependency")
                return
        frame = frame.f_back


def install_import_guards():
    original_import, original_module = builtins.__import__, importlib.import_module

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        absolute = (
            resolve_name("." * level + name, (globals or {}).get("__package__")) if level else name
        )
        caller = sys._getframe(1)
        _import_access(absolute, caller)
        for item in fromlist or ():
            if item != "*":
                _import_access(absolute + "." + item, caller)
        module = original_import(name, globals, locals, fromlist, level)
        _install_cffi_guard()
        return module

    def guarded_module(name, package=None):
        absolute = resolve_name(name, package) if name.startswith(".") else name
        _import_access(absolute, sys._getframe(1))
        module = original_module(name, package)
        _install_cffi_guard()
        return module

    builtins.__import__, importlib.import_module = guarded_import, guarded_module
    builtins.open = _guard_open(builtins.open)
    io.open = _guard_open(io.open)
    os.open = _guard_os_open(os.open)
    native_os = sys.modules.get(os.name)
    if native_os is not None:
        native_os.open = os.open
    _install_worker_guards()
    _install_cffi_guard()


def _install_cffi_guard():
    ffi = getattr(sys.modules.get("cffi"), "FFI", None)
    if ffi is None or ffi.dlopen.__module__ == __name__:
        return
    original = ffi.dlopen

    def guarded(self, name, *args, **kwargs):
        # cffi does not emit CPython's ctypes audit events.
        sys.audit("cffi.dlopen", name)
        return original(self, name, *args, **kwargs)

    ffi.dlopen = guarded


def _guard_os_open(original):
    def guarded(path, flags, mode=0o777, *, dir_fd=None):
        owner, access = _component_access()
        if owner is not None and dir_fd is not None:
            # CPython's open event omits dir_fd; use the real destination for its audit.
            path = (
                None
                if dir_fd == -1 and not os.path.isabs(path)
                else _mutation_path(path, dir_fd, *(access or ((), {})))
            )
            if path is None:
                raise ValueError(
                    "component capability refused: os.open with dir_fd has unresolved destination"
                )
            sys.audit("release.os.open", path, flags, dir_fd)
            dir_fd = None
        return original(path, flags, mode, dir_fd=dir_fd)

    return guarded


def _install_worker_guards():
    def guard(original, capability):
        @wraps(original)
        def guarded(*args, **kwargs):
            # Refuse at the call site, before a worker or coroutine loses this stack.
            sys.audit("release.worker", capability)
            return original(*args, **kwargs)

        return guarded

    for target, names, label in (
        (_thread, ("start_new_thread", "start_new", "start_joinable_thread"), "_thread"),
        (threading, ("_start_new_thread", "_start_joinable_thread"), "threading"),
        (threading.Thread, ("start",), "threading.Thread"),
        (ThreadPoolExecutor, ("__init__", "submit"), "concurrent.futures.ThreadPoolExecutor"),
        (ProcessPoolExecutor, ("__init__", "submit"), "concurrent.futures.ProcessPoolExecutor"),
        (BaseProcess, ("start",), "multiprocessing.Process"),
        (asyncio, ("to_thread",), "asyncio"),
        (asyncio.BaseEventLoop, ("run_in_executor",), "asyncio.BaseEventLoop"),
    ):
        for name in names:
            original = getattr(target, name, None)
            if original is not None:
                setattr(target, name, guard(original, f"{label}.{name}"))

    def pool_new(cls, *args, **kwargs):
        # Pool's destructor requires initialized fields; refuse before allocation.
        sys.audit("release.worker", f"{cls.__module__}.{cls.__name__}")
        return object.__new__(cls)

    Pool.__new__ = staticmethod(pool_new)  # ThreadPool inherits the same constructor.


# endregion chapter29:import-scope


@contextmanager
def scope(paths, stores=None):
    token = _SCOPE.set((frozenset(paths), dict(stores or {})))
    try:
        yield
    finally:
        _SCOPE.reset(token)


def _read_path(path):
    if not isinstance(path, (str, bytes, os.PathLike)):
        return None  # An inherited file descriptor has no manifest identity.
    path = Path(os.fsdecode(path)).resolve()
    if path.suffix == ".pyc" and path.parent.name == "__pycache__":
        from importlib.util import source_from_cache

        try:
            path = Path(source_from_cache(str(path)))
        except ValueError:
            pass  # Test runners may use their own compiled-code cache names.
    return path


def _metadata_reader(checking_path):
    def _metadata_bytes(path):
        # Only the verifier's exact file read may bypass its own recursive audit.
        token = checking_path.set(path)
        try:
            return path.read_bytes()
        finally:
            checking_path.reset(token)

    return _metadata_bytes


def _metadata_records(read_bytes):
    def _metadata_record_allows(path, directory):
        record = (directory / "RECORD").resolve()
        if record.parent != directory or not record.is_file():
            return False
        try:
            if record.stat().st_nlink > 1:
                return False
            rows = list(csv.reader(io.StringIO(read_bytes(record).decode("utf-8"))))
            if any(len(row) != 3 for row in rows):
                return False
            matches = []
            for row in rows:
                listed = Path(os.path.abspath(directory.parent / row[0]))
                if listed == path and listed.is_relative_to(directory):
                    matches.append(row)
            if len(matches) != 1:
                return False
            return _metadata_matches(path, directory, read_bytes(path), matches[0][1])
        except (OSError, UnicodeError, ValueError, csv.Error):
            return False

    return _metadata_record_allows


def _metadata_matches(path, directory, content, digest):
    if zipfile.is_zipfile(io.BytesIO(content)):
        return False
    if not digest:
        return path.parent == directory and path.name in _UNHASHED_METADATA_FILES
    algorithm, separator, expected = digest.partition("=")
    if not separator or not expected:
        return False
    current = (
        urlsafe_b64encode(hashlib.new(algorithm, content).digest()).decode("ascii").rstrip("=")
    )
    return current == expected


def _approved_metadata_path(path, directory):
    if not path.exists():
        raise FileNotFoundError(2, "No such file or directory", str(path))
    stem = directory.name.rsplit(".", 1)[0]
    stem = re.sub(r"-py\d+(?:\.\d+)?$", "", stem)
    name = re.sub(r"-v?\d.*$", "", stem)
    name = re.sub(r"[-_.]+", "-", name).lower()
    return name in _APPROVED_DISTRIBUTIONS and path.is_file() and path.stat().st_nlink <= 1


def _metadata_scope(checking_path, record_allows):
    def _external_metadata_read(path):
        """Exempt only approved, RECORD-listed, unmodified installed metadata."""
        if path == checking_path.get():
            return True
        if not any(
            path.is_relative_to(Path(prefix).resolve()) for prefix in (sys.prefix, sys.base_prefix)
        ):
            return False
        directory = next(
            (p for p in path.parents if p.name.endswith((".dist-info", ".egg-info"))), None
        )
        if directory is not None:
            return _approved_metadata_path(path, directory) and record_allows(path, directory)
        frame = sys._getframe(1)
        while frame is not None:
            if frame.f_code is metadata.FastPath.zip_children.__code__:
                # Discovery may probe missing paths or ordinary files, never read an archive.
                token = checking_path.set(path)
                try:
                    return not zipfile.is_zipfile(path)
                finally:
                    checking_path.reset(token)
            frame = frame.f_back
        return False

    return _external_metadata_read


_checking_metadata_path = ContextVar("release_metadata_check_path", default=None)
_read_metadata_bytes = _metadata_reader(_checking_metadata_path)
_external_metadata_read = _metadata_scope(
    _checking_metadata_path, _metadata_records(_read_metadata_bytes)
)
# Verification read authority stays inside closures; their factories are unavailable.
del (
    _checking_metadata_path,
    _read_metadata_bytes,
    _metadata_reader,
    _metadata_records,
    _metadata_scope,
)


def _external_loader_read(path):
    """Exempt discovered library code and its narrowly scoped metadata reads."""
    if not isinstance(path, (str, bytes, os.PathLike)):
        return False
    path = Path(os.fsdecode(path)).resolve()
    if _external_metadata_read(path):
        return True
    frame = sys._getframe(1)
    while frame is not None:
        if frame.f_code in (
            SourceFileLoader.get_code.__code__,
            SourcelessFileLoader.get_code.__code__,
            ExtensionFileLoader.create_module.__code__,
            ExtensionFileLoader.exec_module.__code__,
        ):
            selected = _loader_files(frame)
            if selected is None:
                return False
            origin, files = selected
            return path in files and any(origin.is_relative_to(root) for root in _LIBRARIES)
        if frame.f_code is zipimport._read_directory.__code__:
            # A missing interpreter archive is a lookup probe with no data bytes.
            return not path.exists() and any(
                path.is_relative_to(Path(prefix).resolve())
                for prefix in (sys.prefix, sys.base_prefix)
            )
        frame = frame.f_back
    return False


def _loader_context(frame):
    loader = frame.f_locals["self"]
    suffixes = {
        SourceFileLoader: SOURCE_SUFFIXES,
        SourcelessFileLoader: BYTECODE_SUFFIXES,
        ExtensionFileLoader: EXTENSION_SUFFIXES,
    }.get(type(loader))
    caller = frame.f_back
    if frame.f_code is ExtensionFileLoader.exec_module.__code__:
        loading = caller
    else:
        expected = (
            importlib._bootstrap.module_from_spec.__code__
            if frame.f_code is ExtensionFileLoader.create_module.__code__
            else SourceFileLoader.exec_module.__code__
        )
        loading = caller.f_back if caller is not None and caller.f_code is expected else None
    return loader, suffixes, loading


def _discovered_loader_spec(loader, suffixes, loading):
    discovery = loading.f_back if loading is not None else None
    spec = loading.f_locals.get("spec") if loading is not None else None
    if not (
        suffixes
        and loading is not None
        and loading.f_code is importlib._bootstrap._load_unlocked.__code__
        and discovery is not None
        and discovery.f_code is importlib._bootstrap._find_and_load_unlocked.__code__
        and discovery.f_back is not None
        and discovery.f_back.f_code is importlib._bootstrap._find_and_load.__code__
        and type(spec) is ModuleSpec
        and discovery.f_locals.get("spec") is spec
        and discovery.f_locals.get("name") == spec.name == loader.name
        and spec.loader is loader
        and spec.has_location
        and spec.origin == loader.path
        and spec.origin.endswith(tuple(suffixes))
        and spec.name != "refund_agent"
        and not spec.name.startswith("refund_agent.")
    ):
        return None
    return spec


def _loader_files(frame):
    loader, suffixes, loading = _loader_context(frame)
    spec = _discovered_loader_spec(loader, suffixes, loading)
    if spec is None:
        return None
    origin = Path(spec.origin).resolve()
    files = {origin}
    if (
        type(loader) is SourceFileLoader
        and spec.cached
        and spec.cached == frame.f_locals.get("bytecode_path")
    ):
        files.add(Path(spec.cached).resolve())
    return origin, files


def _store_mode(path, stores):
    permission = stores.get(path)
    if permission == "read-write" and path.exists():
        if any(mode == "append" and p.exists() and path.samefile(p) for p, mode in stores.items()):
            return "append"  # A writable hard link must not weaken the same store's mode.
    return permission


def permitted(path, flags, code, stores):
    writing = bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND))
    permission = _store_mode(path, stores)
    if not writing:
        return path in code or permission is not None
    return permission == "read-write" or (
        permission == "append" and bool(flags & os.O_APPEND) and not flags & os.O_TRUNC
    )


# region chapter29:append-handles
class _AppendHandle:
    """Keep the store's append capability on retained and nested handles."""

    def __init__(self, stream, owner=None):
        self._stream = stream
        self._owner = owner
        self._reader = None
        _APPEND_HANDLES[stream.fileno()] = stream

    def __getattr__(self, name):
        value = getattr(self._stream, name)
        return _AppendHandle(value, self._owner or self) if name in {"buffer", "raw"} else value

    def __enter__(self):
        self._stream.__enter__()
        return self

    def __exit__(self, *exception):
        self.close()

    def close(self):
        try:
            self._stream.close()
        finally:
            reader = (self._owner or self)._reader
            if reader is not None:
                reader.close()

    def fileno(self):
        self._stream.fileno()  # Preserve the closed-handle check.
        owner = self._owner or self
        if owner._reader is None:
            # A native/cached pwrite cannot obey our Python wrapper's append mode.
            if isinstance(owner._stream.name, int):
                raise ValueError("append runtime store cannot expose an inherited descriptor")
            owner._reader = io.open(owner._stream.name, "rb")
            _APPEND_HANDLES[owner._reader.fileno()] = owner._reader
        return owner._reader.fileno()

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._stream)

    def truncate(self, size=None):
        raise ValueError("append runtime store cannot be truncated")

    def seek(self, offset, whence=os.SEEK_SET):
        if offset != 0 or whence != os.SEEK_END:
            raise ValueError("append runtime store can seek only to its end")
        return self._stream.seek(offset, whence)

    def detach(self):
        raise ValueError("append runtime store cannot detach its writable handle")


def _guard_open(original):
    def guarded(file, mode="r", *args, **kwargs):
        token = _GUARDED_OPEN.set(True)
        try:
            stream = original(file, mode, *args, **kwargs)
        finally:
            _GUARDED_OPEN.reset(token)
        owner, access = _component_access()
        if "a" in mode and owner is not None:
            if access is None or _store_mode(_read_path(file), access[1]) == "append":
                return _AppendHandle(stream)
        return stream

    return guarded


def _append_descriptor(fd):
    stream = _APPEND_HANDLES.get(fd)
    if stream is not None and not stream.closed:
        return True
    _, access = _component_access()
    if access is None:
        return False
    target = os.fstat(fd)
    return any(
        mode == "append" and p.exists() and os.path.samestat(target, p.stat())
        for p, mode in access[1].items()
    )


def _component_access():
    frame, observed = sys._getframe(1), None
    while frame is not None:
        name = frame.f_globals.get("__name__", "")
        if name.startswith("refund_agent.") and name != __name__ and name not in CORE_MODULES:
            origin = frame.f_globals.get("__file__")
            if origin:
                owner = (name, Path(origin).resolve())
                observed = observed or owner
                if owner in ACCESS:
                    return owner, _SCOPE.get() or ACCESS[owner]
        frame = frame.f_back
    return observed, None


def _environment_path(path):
    if not isinstance(path, (str, bytes, os.PathLike)):
        return False
    path = Path(os.fsdecode(path))
    roots = (*_LIBRARIES, Path(sys.prefix).resolve(), Path(sys.base_prefix).resolve())
    # Protect both a link's location and its target.
    return any(
        candidate.is_relative_to(root)
        for candidate in (Path(os.path.abspath(path)), path.resolve())
        for root in roots
    )


def _directory_path(descriptor):
    """Resolve a directory descriptor through the OS, never through manifest guesses."""
    try:
        if sys.platform == "darwin":
            import fcntl

            name = fcntl.fcntl(descriptor, fcntl.F_GETPATH, bytes(1024)).split(b"\0", 1)[0]
        elif sys.platform.startswith("linux"):
            name = os.readlink(f"/proc/self/fd/{descriptor}")
        else:
            return None
        path = Path(os.fsdecode(name)).resolve()
        if path.is_dir() and os.path.samestat(os.fstat(descriptor), path.stat()):
            return path
    except (OSError, ValueError):
        pass
    return None


def _mutation_path(path, directory_fd, code, stores):
    candidates = set(code) | stores.keys()
    if isinstance(path, int):
        target = os.fstat(path)
        return next(
            (p for p in candidates if p.exists() and os.path.samestat(target, p.stat())), None
        )
    path = Path(os.fsdecode(path))
    if not path.is_absolute() and directory_fd not in (None, -1):
        parent = _directory_path(directory_fd)
        if parent is None:
            return None
        path = parent / path
    return Path(os.path.abspath(path))


@contextmanager
def fixture_scope(candidate, data, mode):
    """Issue an opaque fixture token only inside the core's own construction."""
    from refund_agent.release.suite import _fixture_tools

    caller = sys._getframe(2)  # contextlib.__enter__ is immediately above the driver.
    fixture_driver = (
        caller.f_code is _fixture_tools.__code__
        and caller.f_globals is _fixture_tools.__globals__
        and caller.f_back.f_globals is _fixture_tools.__globals__
    )
    constructed_driver = (
        caller.f_code is run_constructed_request.__code__
        and caller.f_globals is run_constructed_request.__globals__
    )
    if not (fixture_driver or constructed_driver):
        raise ValueError("fixture capability requires its core driver")
    capability = object()
    token = _FIXTURE.set((capability, candidate, data, mode, caller))
    try:
        yield capability
    finally:
        _FIXTURE.reset(token)


@dataclass(frozen=True)
class ConstructedDataset:
    """Explicit provenance for simulated data; real and absent marks are refused."""

    data: object
    source: str


@dataclass(frozen=True)
class ConstructedRun:
    """Detached observations only: no Tools, session, approval or reusable token."""

    trace: object
    payments: tuple
    explanations: dict
    held_requests: frozenset


class _ConstructedExecution:
    def __init__(self, driver, snapshot, request):
        self.driver, self.active = driver, True
        self.tools, self.data = None, None
        self.snapshot, self.request = snapshot, request.model_dump()
        self.log, self.events = None, None

    def ready(self, tools):
        from refund_agent import contracts as c

        if tools is not self.tools:
            raise ValueError("constructed run tools differ from core initialization")
        self.check()
        self.log = getattr(tools, "event_log", None)
        self.events = c._detach(self.log.entries) if self.log is not None else None
        if self.log is not None:
            open_view = tools._event_view
            tools._event_view = lambda state: self.event_view(state, open_view)

    def check(self, request=None):
        frame = sys._getframe(1)
        while frame is not None and frame is not self.driver:
            frame = frame.f_back
        if not self.active or frame is None:
            raise ValueError("constructed run execution scope expired")
        if request is not None and request.model_dump() != self.request:
            raise ValueError("constructed run request snapshot differs")
        if self.tools._data is not self.data or self.data.model_dump() != self.snapshot:
            raise ValueError("constructed run data snapshot differs")
        if self.log is not None and (
            self.tools.event_log is not self.log or self.log.entries != self.events
        ):
            raise ValueError("constructed run event log snapshot differs")

    @contextmanager
    def event_view(self, state, open_view):
        from refund_agent.batch_stream.view import shipment_as_of

        self.check(state.request)
        at = state.execution_at or state.request.at
        shipments = (
            [
                shipment_as_of(self.log, order.order_id, at).model_dump()
                for order in self.data.orders
            ]
            if self.log.arrived_by(at)
            else []
        )
        data, snapshot = self.data, self.snapshot
        try:
            with open_view(state):
                self.data = self.tools._data
                self.snapshot = {**snapshot, "shipments": shipments}
                self.check(state.request)
                yield
        finally:
            self.data, self.snapshot = data, snapshot
        self.check(state.request)


def _constructed_state_call(execution, method, state_position):
    @wraps(method)
    def guarded(*args, **kwargs):
        state = kwargs.get("state")
        if state is None:
            state = args[state_position]
        execution.check(state.request)
        return method(*args, **kwargs)

    return guarded


def _bind_constructed_execution(tools):
    execution = _CONSTRUCTED_EXECUTION.get()
    if execution is None:
        return
    if execution.tools is not None:
        raise ValueError("constructed run tools already initialized")
    execution.tools, execution.data = tools, tools._data
    tools._constructed_execution = execution
    execution.check()
    boundary = tools._boundary

    @contextmanager
    def scoped_boundary(state, operation):
        execution.check(state.request)
        with boundary(state, operation):
            yield

    tools._boundary = scoped_boundary
    tools._execute = _constructed_state_call(execution, tools._execute, 1)
    tools._commit = _constructed_state_call(execution, tools._commit, 0)


@contextmanager
def _constructed_execution(snapshot, request):
    driver = sys._getframe(2)
    if (
        driver.f_code is not run_constructed_request.__code__
        or driver.f_globals is not run_constructed_request.__globals__
    ):
        raise ValueError("constructed run execution scope requires its core driver")
    execution = _ConstructedExecution(driver, snapshot, request)
    token = _CONSTRUCTED_EXECUTION.set(execution)
    try:
        yield execution
    finally:
        # Retained tools and callables keep their checks, but never their authority.
        execution.active, execution.driver = False, None
        _CONSTRUCTED_EXECUTION.reset(token)


def run_constructed_request(dataset, candidate, request, *, mode="execute", at=None, **context):
    """Run generated records in the core's simulated ledger, under every normal guard."""
    from refund_agent import contracts as c
    from refund_agent.data.generate import Dataset
    from refund_agent.release.candidate import build_tools, run_request, validate_final_decision

    if (
        type(dataset) is not ConstructedDataset
        or dataset.source != "generated"
        or type(dataset.data) is not Dataset
    ):
        raise ValueError("constructed run requires an explicitly generated dataset")
    if {"canary_session", "release_fixture"} & context.keys():
        raise ValueError("constructed run grants no operational authority")
    data = dataset.data.model_copy(deep=True, update={"correct": []})
    request = c._validated(c.Request, request)
    if request not in data.requests:
        raise ValueError("request is outside the generated dataset")
    snapshot = data.model_dump()
    with _constructed_execution(snapshot, request) as execution:
        with fixture_scope(candidate, data, mode) as capability:
            tools = build_tools(data, candidate, mode, **{**context, "release_fixture": capability})
        execution.ready(tools)
        trace = run_request(candidate, tools, request, at=at)
        validate_final_decision(trace, tools, request)
        execution.check(request)
        return ConstructedRun(
            c._detach(trace),
            tuple(tools.payments),
            c._detach(getattr(tools, "explanations", {})),
            frozenset(getattr(tools, "held_requests", ())),
        )


def fixture_active(capability, candidate, mode):
    """A fixture token admits only its original Tools construction, never a grant."""
    from refund_agent.release.candidate import build_tools

    bound = _FIXTURE.get()
    if bound is None or capability is not bound[0] or candidate is not bound[1] or mode != bound[3]:
        return False
    frame = sys._getframe(1)
    while frame is not None:
        if (
            frame.f_code is build_tools.__wrapped__.__code__
            and frame.f_globals is build_tools.__wrapped__.__globals__
        ):
            return (
                frame.f_locals["data"] is bound[2]
                and frame.f_locals["candidate"] is candidate
                and frame.f_locals["mode"] == mode
                and frame.f_back.f_code is build_tools.__code__
                and frame.f_back.f_globals is build_tools.__globals__
                and frame.f_back.f_back is bound[4]
            )
        frame = frame.f_back
    return False


@lru_cache(maxsize=1)
def rehearsal_restart():
    """Issue the restart capability to the actual rehearsal driver, never its name."""
    from refund_agent.canary.cloud_native_deployment_rehearsal import run_rehearsal

    driver = run_rehearsal.__wrapped__
    code, namespace = driver.__code__, driver.__globals__

    def restart(command):
        caller = sys._getframe(1)
        if caller.f_code is not code or caller.f_globals is not namespace:
            raise ValueError("rehearsal restart capability requires its driver")
        token = _REHEARSAL_PROCESS.set(caller)
        try:
            return subprocess.run(command, check=False)
        finally:
            _REHEARSAL_PROCESS.reset(token)

    return restart


def _rehearsal_process():
    caller = _REHEARSAL_PROCESS.get()
    if caller is None:
        return False
    frame = sys._getframe(1)
    while frame is not None:
        if frame is caller:
            return True
        name = frame.f_globals.get("__name__", "")
        if name.startswith("refund_agent.") and name not in CORE_MODULES:
            return False  # A nested component cannot borrow the driver's capability.
        frame = frame.f_back
    return False


def _audit_component_capability(event, arguments):
    """Refuse unnecessary teaching capabilities; this is not hostile-code isolation."""
    process = event in {"subprocess.Popen", "pty.spawn"} or event.startswith(
        ("os.system", "os.exec", "os.spawn", "os.fork", "os.posix_spawn")
    )
    native = event.startswith(("ctypes.", "cffi.dlopen"))
    worker = event in {
        "release.worker",
        "_thread.start_new_thread",
        "_thread.start_joinable_thread",
    }
    if not (
        process or native or worker or event in {"release.os.open", "mmap.__new__", "os.mkdir"}
    ):
        return False
    owner, access = _component_access()
    if owner is None:
        return False
    if process and _rehearsal_process():
        return False
    if worker:
        capability = arguments[0] if event == "release.worker" else event
        raise ValueError(f"component capability refused: worker delegation ({capability})")
    if process or native:
        capability = "process spawning" if process else "native loading or calls"
        raise ValueError(f"component capability refused: {capability} ({event})")
    return _audit_file_capability(event, arguments, access)


def _audit_file_capability(event, arguments, access):
    if event == "mmap.__new__":
        if arguments[2] != mmap.ACCESS_READ:
            raise ValueError("component capability refused: writable mmap")
        return False  # Retain the earlier append-descriptor check, including read mappings.
    if event == "release.os.open":
        path, flags, _ = arguments
        writing = bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND))
        if writing and (
            access is None
            or _environment_path(path)
            or not permitted(_read_path(path), flags, *access)
        ):
            raise ValueError(
                "component capability refused: os.open with dir_fd outside a declared writable store"
            )
        return True  # The following open event authorizes and records the resolved read.
    return _audit_mkdir(arguments, access)


def _audit_mkdir(arguments, access):
    path, _, directory_fd = arguments
    roots = (
        *_LIBRARIES,
        Path(sys.prefix).resolve(),
        Path(sys.base_prefix).resolve(),
        _CHAPTER_ROOT,
    )
    code, stores = access or (core_paths(), {})
    path = _mutation_path(path, directory_fd, set(code) | {p / "_" for p in roots}, stores)
    if path is None:
        raise ValueError("component capability refused: os.mkdir with unresolved destination")
    if _environment_path(path) or any(
        candidate.is_relative_to(root)
        for candidate in (path, path.resolve())
        for root in (_CHAPTER_ROOT, *core_paths())
    ):
        raise ValueError("component capability refused: os.mkdir under a protected root")
    return True


_RESOURCE_EVENTS = frozenset(
    {
        "open",
        "import",
        "os.truncate",
        "os.remove",
        "os.rename",
        "os.rmdir",
        "os.link",
        "os.symlink",
        "os.chmod",
        "fcntl.fcntl",
        "mmap.__new__",
    }
)


def _audit_resource_read(event, arguments):
    if event == "mmap.__new__" and arguments[0] == -1:
        _audit_component_capability(event, arguments)
        return  # Anonymous read-only memory has no descriptor or file identity.
    if event in {"os.truncate", "fcntl.fcntl", "mmap.__new__"} and isinstance(arguments[0], int):
        if _append_descriptor(arguments[0]):
            raise ValueError(
                f"append runtime store forbids destructive descriptor operations ({event})"
            )
    if _audit_component_capability(event, arguments):
        return
    if event not in _RESOURCE_EVENTS:
        return
    if event == "import" and arguments[1] is None:
        return  # Discovery has no file; native loading emits the resolved filename.
    if event in {"fcntl.fcntl", "mmap.__new__"}:
        return
    owner, access = _component_access()
    if owner is None:
        return
    if event in {"open", "import"}:
        _audit_open(event, arguments, owner, access)
    else:
        _audit_mutation(event, arguments, access)


def _audit_open(event, arguments, owner, access):
    path, mode, flags = arguments if event == "open" else (arguments[1], "r", os.O_RDONLY)
    writing = bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND))
    if writing and not isinstance(path, (str, bytes, os.PathLike)):
        raise ValueError("component write to inherited descriptor is refused")
    if writing and _environment_path(path):
        raise ValueError("component cannot mutate the installed environment")
    if not writing and _external_loader_read(path):
        return
    path = _read_path(path)
    append = (
        _store_mode(path, access[1]) == "append"
        if access is not None
        else bool(flags & os.O_APPEND)
    )
    if writing and append and (not _GUARDED_OPEN.get() or mode is None):
        raise ValueError("append runtime store requires a guarded writable handle")
    if access is not None:
        if not permitted(path, flags, *access):
            raise ValueError(f"component access to undeclared resource or store mode: {path}")
        return
    if writing:
        WRITES.setdefault(owner, set()).add((path, flags))
    else:
        READS.setdefault(owner, set()).add(path)


def _audit_mutation(event, arguments, access):
    if access is None:
        raise ValueError("component mutation before runtime store authorization")
    code, stores = access
    if event in {"os.rename", "os.link"}:
        source, target, source_fd, target_fd = arguments
        paths = ((source, source_fd), (target, target_fd))
    elif event == "os.symlink":
        paths = ((arguments[1], arguments[2]),)
    elif event == "os.chmod":
        paths = ((arguments[0], arguments[2]),)
    else:
        paths = ((arguments[0], None if event == "os.truncate" else arguments[1]),)
    for path, directory_fd in paths:
        path = _mutation_path(path, directory_fd, code, stores)
        if _environment_path(path):
            raise ValueError("component cannot mutate the installed environment")
        path = _read_path(path)
        if _store_mode(path, stores) != "read-write":
            raise ValueError(f"component mutation of undeclared resource or store mode: {path}")


# endregion chapter29:append-handles


# endregion chapter29:resource-reads
