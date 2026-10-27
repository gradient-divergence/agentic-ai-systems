"""Explicit component registration; only selected modules enter release identity.

A component registers from its own module, declaring its complete module/data
closure. There is no package scan, implicit stage selection or absent-code substitute.
Hooks receive keyword arguments; their signatures are documented in the README.
"""

import ast
import os
import sys
from dataclasses import dataclass, field
from functools import lru_cache, wraps
from hashlib import sha256
from importlib import import_module
from importlib.util import resolve_name
from pathlib import Path
from types import BuiltinFunctionType, FunctionType, MethodType, ModuleType

from refund_agent.release import agent_evaluation_resources as resources
from refund_agent.selection import fingerprint

HOOKS = frozenset(
    {
        "compose",
        "wrap_model",
        "before_run",
        "after_run",
        "cases",
        "validate_suite",
        "select_case",
        "run_case",
        "run_loop_case",
        "fixture_context",
        "changed_source",
        "approval_context",
        "admit_shadow",
        "start_checks",
        "bundle_items",
        "promotion_requirements",
        "reasons",
    }
)


# region chapter29:runtime-stores
@dataclass(frozen=True)
class RuntimeStore:
    path: str
    owner: str
    mode: str


def _stores(plugin):
    from refund_agent.release.candidate import PACKAGE
    from refund_agent.release.rule import owner_ids

    result = {}
    for store in plugin.stores:
        if (
            not isinstance(store, RuntimeStore)
            or not store.path
            or store.owner not in owner_ids()
            or store.mode not in {"read", "append", "read-write"}
        ):
            raise ValueError("runtime store needs a path, listed owner and valid mode")
        path = (PACKAGE / store.path).resolve()
        if "exploration_bandits_package" in path.parts or path in result or path.is_dir():
            raise ValueError("runtime store path is sealed, duplicated or a directory")
        result[path] = store.mode
    return result


# endregion chapter29:runtime-stores


@dataclass(frozen=True)
class Component:
    name: str
    version: str
    module: str
    modules: tuple[str, ...] = ()
    files: tuple[str, ...] = ()
    order: int = 30
    outermost: bool = False
    hooks: dict = field(default_factory=dict)
    slots: dict = field(default_factory=dict)
    stores: tuple[RuntimeStore, ...] = field(default=(), kw_only=True)


_REGISTRY = {}
_CLOSURES = {}
_DYNAMIC_NAMES = frozenset(
    {"__import__", "eval", "exec", "getattr", "globals", "locals", "vars", "__builtins__"}
)
_DYNAMIC_ATTRIBUTES = frozenset(
    {
        "modules",
        "__dict__",
        "__globals__",
        "__closure__",
        "__self__",
        "__getattribute__",
        "__subclasses__",
        "__mro__",
        "__bases__",
        "__code__",
    }
)


def _load(module):
    if not module.startswith("refund_agent."):
        raise ValueError("component module must be under refund_agent")
    try:
        return import_module(module)
    except ImportError as error:
        raise ValueError("selected component module is unavailable") from error


def register(component):
    if (
        not component.name
        or not component.version
        or not component.module.startswith("refund_agent.")
    ):
        raise ValueError("component needs a name, version and refund_agent module")
    if set(component.hooks) - HOOKS or any(not callable(h) for h in component.hooks.values()):
        raise ValueError("unknown or non-callable extension hook")
    if any(
        h.__module__ not in (component.module,) + component.modules
        for h in component.hooks.values()
    ):
        raise ValueError("hook module must be included in the component code closure")
    previous = _REGISTRY.get(component.name)
    if previous is not None and previous != component:
        raise ValueError("component already registered with another definition")
    _REGISTRY[component.name] = component
    return component


def selections(specs):
    """Normalize names or {name, module, settings} selectors in composition order."""
    result, seen = [], set()
    for spec in specs:
        spec = {"name": spec} if isinstance(spec, str) else dict(spec)
        name = spec["name"]
        if spec.get("module"):
            _load(spec["module"])  # The module registers itself.
        if name not in _REGISTRY or name in seen:
            raise ValueError("component must be registered and selected once")
        plugin = _REGISTRY[name]
        if spec.get("module", plugin.module) != plugin.module:
            raise ValueError("selected component module differs from registration")
        seen.add(name)
        result.append({"name": name, "module": plugin.module, "settings": spec.get("settings", {})})
    if sum(_REGISTRY[s["name"]].outermost for s in result) > 1:
        raise ValueError("only one component may own the outermost scope boundary")
    return tuple(sorted(result, key=lambda s: (_REGISTRY[s["name"]].order, s["name"])))


# region chapter29:component-closure
def _module_path(name):
    if name in sys.modules and getattr(sys.modules[name], "__file__", None):
        return Path(sys.modules[name].__file__)
    from refund_agent.release.candidate import PACKAGE

    path = PACKAGE.joinpath(*name.split(".")[1:])
    for candidate in (path.with_suffix(".py"), path / "__init__.py"):
        if candidate.is_file():
            return candidate
    return None


@lru_cache(maxsize=256)
def _syntax(content, filename):
    return ast.parse(content, filename=filename)


def _static_imports_only(tree):
    """Reject dynamic loaders/cache access even when aliased or already cached."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] in {"importlib", "builtins"}:
                    raise ValueError("component dynamic imports are refused; use static imports")
        if isinstance(node, ast.ImportFrom) and (
            (node.module or "").split(".")[0] in {"importlib", "builtins"}
            or node.module == "sys"
            and any(a.name == "modules" for a in node.names)
        ):
            raise ValueError("component indirect imports or module cache access are refused")
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Name)
            and node.id in _DYNAMIC_NAMES
            or isinstance(node, ast.Attribute)
            and node.attr in _DYNAMIC_ATTRIBUTES
        ):
            raise ValueError("component indirect imports or module cache access are refused")


def _external_dependency(name):
    """Allow installed libraries/stdlib, refuse untracked first-party helpers."""
    if resources.first_party_path(resources.module_origin(name)) is not None:
        raise ValueError("first-party dependency outside refund_agent is an undeclared import")


def _referenced_modules(name):
    """Cached first-party modules/callables are dependencies even without imports."""
    module = sys.modules.get(name)
    if module is None:
        return ()
    result = set()
    for key, value in vars(module).items():
        if key.startswith("__"):
            continue
        if value is sys.modules:
            raise ValueError("component module cache reference is an undeclared import")
        if isinstance(value, ModuleType):
            dependency = value.__name__
            if dependency in {"sys", "importlib", "builtins", "inspect", "operator"}:
                raise ValueError("component import/cache reflection is refused")
        elif isinstance(value, (FunctionType, MethodType, BuiltinFunctionType, type)):
            dependency = value.__module__
        else:
            continue
        if not dependency or dependency == name:
            continue
        if dependency == "refund_agent" or dependency.startswith("refund_agent."):
            result.add(dependency)
        else:
            _external_dependency(dependency)
    return tuple(sorted(result))


def _closure(plugin, *, selection=None):
    """Follow local imports and package initializers without importing or scanning.

    Component.files is the complete resource manifest, independent of suffix or
    nesting. Undeclared import-time and subsequent reads refuse the component.
    """
    core = resources.core_paths()
    key = (plugin.name, selection)
    cached = _CLOSURES.get(key)
    if cached and cached["plugin"] is plugin:
        if cached["state"] == _closure_state(
            plugin, cached["state"]["modules"], core, cached["external"]
        ):
            for path in cached["reads"]:
                sys.audit("open", str(path), "r", os.O_RDONLY)
            _authorize_closure(cached["paths"], core, cached["allowed"], _stores(plugin))
            return dict(cached["sources"])
    sources, paths, allowed, external, state = _build_closure(plugin, core)
    _authorize_closure(paths, core, allowed, _stores(plugin))
    if state != _closure_state(plugin, state["modules"], core, external):
        raise ValueError("component closure changed during construction")
    _remember_closure(key, plugin, sources, paths, allowed, external, state)
    return sources


def _closure_state(plugin, names, core, external):
    from refund_agent.release.candidate import PACKAGE

    for dependency in external:
        _external_dependency(dependency)
    files = tuple(
        (name, PACKAGE / name, (PACKAGE / name).resolve(), resources.file_identity(PACKAGE / name))
        for name in sorted(set(plugin.files))
    )
    return dict(
        modules={name: _closure_module_state(name, core) for name in names},
        files=files,
        fixed=tuple((path, resources.file_identity(path)) for path in sorted(core)),
        reader=Path.read_bytes,
    )


def _closure_module_state(name, core):
    path = _module_path(name)
    if path is None:
        return None  # Unresolved static imports must be resolved again on cache hits.
    resolved = path.resolve()
    references = _referenced_modules(name) if resolved not in core else ()
    observed = tuple(sorted(resources.IMPORTS.get((name, resolved), ())))
    return path, resolved, resources.file_identity(path), references, observed


def _closure_read(path, identity):
    before = resources.file_identity(path)
    if before != identity:
        raise ValueError("component closure changed during construction")
    content = path.read_bytes()
    if before != resources.file_identity(path):
        raise ValueError("component closure changed during read")
    return content


def _remember_closure(key, plugin, sources, paths, allowed, external, state):
    if len(_CLOSURES) >= 256:
        _CLOSURES.pop(next(iter(_CLOSURES)))
    _CLOSURES[key] = dict(
        plugin=plugin,
        sources=dict(sources),
        paths=paths,
        allowed=allowed,
        external=external,
        state=state,
        reads=tuple(state["modules"][name][0] for name in paths)
        + tuple(path for _, path, _, _ in state["files"]),
    )


def _build_closure(plugin, core):
    sources, paths, pending = {}, {}, list((plugin.module,) + plugin.modules)
    external = set()
    state = _closure_state(plugin, (), core, external)
    while pending:
        name = pending.pop()
        if name in sources:
            continue
        path, tree = _closure_source(plugin, name, core, sources, paths, pending, state)
        if path.resolve() in core:
            continue  # The entire fixed core is fingerprinted independently.
        pending.extend(_closure_dependencies(name, path, tree, core, external, state))
    allowed = frozenset(_closure_files(state["files"], sources, core) | set(paths.values()))
    return sources, paths, allowed, external, state


def _closure_source(plugin, name, core, sources, paths, pending, state):
    if name != "refund_agent" and not name.startswith("refund_agent."):
        raise ValueError("component dependency must be under refund_agent")
    if name not in state["modules"]:
        state["modules"][name] = _closure_module_state(name, core)
    snapshot = state["modules"][name]
    if snapshot is None:
        raise ValueError("selected component dependency is unavailable")
    path, resolved, identity, _, _ = snapshot
    if name == plugin.module and resolved in core:
        raise ValueError("component must own a module outside the fixed core")
    if "exploration_bandits_package" in path.parts:
        raise ValueError("sealed modules cannot enter component closure")
    content = _closure_read(path, identity)
    paths[name] = resolved
    sources[name] = sha256(content).hexdigest()
    parent = name.rpartition(".")[0]
    while parent:
        pending.append(parent)
        parent = parent.rpartition(".")[0]
    return path, _syntax(content, str(path))


def _closure_dependencies(name, path, tree, core, external, state):
    _static_imports_only(tree)
    dependencies = list(state["modules"][name][3])
    for dependency in state["modules"][name][4]:
        if dependency == "refund_agent" or dependency.startswith("refund_agent."):
            dependencies.append(dependency)
        else:
            _closure_external(dependency, external)
    package = name if path.name == "__init__.py" else name.rpartition(".")[0]
    loader_names = {"import_module", "__import__"} | {
        alias.asname or alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module == "importlib"
        for alias in node.names
        if alias.name == "import_module"
    }
    for node in ast.walk(tree):
        imports = _node_imports(node, package, loader_names, path, core)
        for dependency in imports:
            if not isinstance(dependency, str):
                continue
            local = dependency == "refund_agent" or dependency.startswith("refund_agent.")
            if dependency and not local:
                _closure_external(dependency, external)
            if local:
                if dependency not in state["modules"]:
                    state["modules"][dependency] = _closure_module_state(dependency, core)
                if state["modules"][dependency] is not None:
                    dependencies.append(dependency)
    return dependencies


def _closure_external(dependency, external):
    _external_dependency(dependency)
    external.add(dependency)


def _node_imports(node, package, loader_names, path, core):
    if isinstance(node, ast.Import):
        return [alias.name for alias in node.names]
    if isinstance(node, ast.ImportFrom):
        base = (
            resolve_name("." * node.level + (node.module or ""), package)
            if node.level
            else node.module or ""
        )
        return [base] + [base + "." + alias.name for alias in node.names]
    if isinstance(node, ast.Call) and (
        isinstance(node.func, ast.Name)
        and node.func.id in loader_names
        or isinstance(node.func, ast.Attribute)
        and node.func.attr == "import_module"
    ):
        if not node.args or not isinstance(node.args[0], ast.Constant):
            if path in core:
                return []  # The fixed core's loader is already fingerprinted.
            raise ValueError("component dynamic import needs a literal module")
        imported = node.args[0].value
        return [
            resolve_name(imported, package)
            if isinstance(imported, str) and imported.startswith(".")
            else imported
        ]
    return []


def _closure_files(files, sources, core):
    resource_paths = set(core)
    for name, raw, path, identity in files:
        if str(name) in sources:
            raise ValueError("resource identity conflicts with module identity")
        if "exploration_bandits_package" in path.parts:
            raise ValueError("sealed resources cannot enter component closure")
        sources[str(name)] = sha256(_closure_read(raw, identity)).hexdigest()
        resource_paths.add(path)
    return resource_paths


def _authorize_closure(paths, core, allowed, stores):
    if allowed & stores.keys():
        raise ValueError("runtime store cannot also be declared as code or configuration")
    for name, path in paths.items():
        if path not in core:
            owner = (name, path)
            if resources.READS.get(owner, set()) - allowed - stores.keys():
                raise ValueError("component read of undeclared resource before selection")
            if any(
                not resources.permitted(path, flags, allowed, stores)
                for path, flags in resources.WRITES.get(owner, ())
            ):
                raise ValueError(
                    "component write outside authorized runtime store before selection"
                )
            resources.ACCESS[owner] = (allowed, stores)


def records(specs, *, candidate=None):
    result = []
    for spec in selections(specs):
        plugin = _REGISTRY[spec["name"]]
        sources = _closure(plugin, selection=(candidate, fingerprint(spec)))
        files = set(plugin.files)
        result.append(
            {
                **spec,
                "version": plugin.version,
                "modules": sorted(k for k in sources if k not in files),
                "files": sorted(files),
                "stores": [
                    vars(store).copy() for store in sorted(plugin.stores, key=lambda s: s.path)
                ],
                "fingerprint": _closure_fingerprint(tuple(sorted(sources.items()))),
            }
        )
    return result


@lru_cache(maxsize=256)
def _closure_fingerprint(sources):
    return fingerprint(dict(sources))


# endregion chapter29:component-closure


def active(candidate):
    declared = candidate.tools_composition.get("components", ())
    if list(declared) != records(declared, candidate=candidate.fingerprint):
        raise ValueError("selected component differs from the released candidate")
    return [(_REGISTRY[r["name"]], r["settings"]) for r in declared]


# region chapter29:resource-scope
def resource_guard(candidate_position):
    """Keep delegated reads inside this candidate's complete selected closure."""

    def decorate(function):
        @wraps(function)
        def guarded(*args, **kwargs):
            candidate = (
                args[candidate_position] if len(args) > candidate_position else kwargs["candidate"]
            )
            declared = candidate.tools_composition.get("components", ())
            if not declared:
                return function(*args, **kwargs)
            paths, stores = _declared_resources(declared)
            with resources.scope(paths, stores):
                return function(*args, **kwargs)

        return guarded

    return decorate


def _declared_resources(declared):
    from refund_agent.release.candidate import PACKAGE

    paths, stores = set(resources.core_paths()), {}
    for record in declared:
        for name in record["modules"]:
            path = _module_path(name)
            if path is None:
                raise ValueError("selected component dependency is unavailable")
            paths.add(path.resolve())
        paths.update((PACKAGE / name).resolve() for name in record["files"])
        for store in record.get("stores", ()):
            path = (PACKAGE / store["path"]).resolve()
            if path in stores and stores[path] != store["mode"]:
                raise ValueError("selected runtime store declarations conflict")
            stores[path] = store["mode"]
    if paths & stores.keys():
        raise ValueError("runtime store cannot also be declared as code or configuration")
    return paths, stores


# endregion chapter29:resource-scope


# region chapter29:hook-snapshots
def _invoke(hook, **context):
    from refund_agent.release.suite import detached_case

    case = context.get("case")
    if case is None:
        return hook(**context)
    case = detached_case(case)
    expected = fingerprint(case.record())
    try:
        return hook(**{**context, "case": case})
    finally:
        if fingerprint(case.record()) != expected:
            raise ValueError("evaluation manifest fixture changed during hook execution")


# endregion chapter29:hook-snapshots


@resource_guard(0)
def apply(candidate, hook, *, value, **context):
    """Ordered transforms; each hook must return the next value."""
    if hook == "after_run":
        from refund_agent.release.candidate import validate_final_decision

        validate_final_decision(value, context["tools"], context["request"])
    for plugin, settings in active(candidate):
        if hook in plugin.hooks:
            value = _invoke(
                plugin.hooks[hook], value=value, candidate=candidate, settings=settings, **context
            )
            if hook == "after_run":
                validate_final_decision(value, context["tools"], context["request"])
    return value


@resource_guard(0)
def handle(candidate, hook, **context):
    """One handler may claim a case; None explicitly leaves it to the core."""
    handled = []
    for plugin, settings in active(candidate):
        if hook in plugin.hooks:
            execution = context.get("execution")
            before = execution.runs if execution is not None else 0
            result = _invoke(plugin.hooks[hook], candidate=candidate, settings=settings, **context)
            if hook in {"run_case", "run_loop_case"}:
                if result is not None and (result is not True or execution is None):
                    raise ValueError(
                        "case handler must schedule execution through the core context"
                    )
                if execution is not None and (
                    result is True
                    and execution.runs == before
                    or result is None
                    and execution.runs != before
                ):
                    raise ValueError("case handler must claim its scheduled execution")
            if result is not None:
                handled.append(result)
    if len(handled) > 1:
        raise ValueError("multiple components claimed the same extension operation")
    return handled[0] if handled else None


@resource_guard(0)
def notify(candidate, hook, **context):
    for plugin, settings in active(candidate):
        if hook in plugin.hooks:
            _invoke(plugin.hooks[hook], candidate=candidate, settings=settings, **context)


@resource_guard(0)
def compose(candidate, base, kwargs, *, mode, context, outermost):
    for plugin, settings in active(candidate):
        if plugin.outermost == outermost and "compose" in plugin.hooks:
            supplied = base
            base, kwargs = plugin.hooks["compose"](
                base=base,
                kwargs=kwargs,
                candidate=candidate,
                settings=settings,
                mode=mode,
                context=context,
            )
            if not isinstance(base, type) or not issubclass(base, supplied):
                raise ValueError("composition must wrap the supplied base")
            if not isinstance(kwargs, dict):
                raise ValueError("composition must return constructor keyword arguments")
    return base, kwargs


def extend_suite(suite, specs):
    from dataclasses import replace

    specs = selections(specs)
    cases = list(suite.cases)
    for spec in specs:
        plugin = _REGISTRY[spec["name"]]
        _closure(plugin)
        if "cases" in plugin.hooks:
            cases.extend(plugin.hooks["cases"](settings=spec["settings"]))
    if len({c.case_id for c in cases}) != len(cases):
        raise ValueError("duplicate component case identity")
    return replace(suite, cases=tuple(cases), components=specs)


def validate_suite(suite):
    before = fingerprint([case.record() for case in suite.cases])
    for spec in selections(suite.components):
        plugin = _REGISTRY[spec["name"]]
        if "cases" in plugin.hooks:
            required = plugin.hooks["cases"](settings=spec["settings"])
            for case in required:
                if case.must_pass and sum(c.record() == case.record() for c in suite.cases) != 1:
                    raise ValueError("selected component must-pass case missing or changed")
        if "validate_suite" in plugin.hooks:
            plugin.hooks["validate_suite"](suite=suite, settings=spec["settings"])
        if fingerprint([case.record() for case in suite.cases]) != before:
            raise ValueError("evaluation manifest changed during suite validation")


def suite_slots(specs):
    result = {}
    for spec in selections(specs):
        slots = _REGISTRY[spec["name"]].slots
        if result.keys() & slots.keys():
            raise ValueError("duplicate component manifest slot")
        result.update(slots)
    return result


@resource_guard(0)
def contributions(candidate, hook, **context):
    """Return named, ordered data/check contributions for later chapter consumers.

    Canary start reads start_checks on every start. Bundle/promotion builders and
    the reason classifier consume their corresponding contributions, without
    editing each other's shared files or introducing an optional admission bypass.
    """
    if hook not in {"start_checks", "bundle_items", "promotion_requirements", "reasons"}:
        raise ValueError("unknown contribution hook")
    return {
        plugin.name: plugin.hooks[hook](candidate=candidate, settings=settings, **context)
        for plugin, settings in active(candidate)
        if hook in plugin.hooks
    }
