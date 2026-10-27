"""Reuse unchanged integrity evidence; every dependency change still refuses."""

import os
import sys
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from types import ModuleType

import pytest

from refund_agent.release import agent_evaluation_resources as resources
from refund_agent.release import extensions
from refund_agent.release.candidate import build_candidate
from refund_agent.release.suite import build_suite


@pytest.fixture
def selected(tmp_path, monkeypatch):
    monkeypatch.setattr(extensions, "_REGISTRY", dict(extensions._REGISTRY))
    package = ModuleType("refund_agent._cache_test")
    package.__path__ = [str(tmp_path)]
    names = {
        "initializer": package.__name__,
        "component": package.__name__ + ".component",
        "helper": package.__name__ + ".helper",
    }
    paths, modules = {}, {}
    for kind, name in names.items():
        path = tmp_path / ("__init__.py" if kind == "initializer" else kind + ".py")
        path.write_text("from . import helper\n" if kind == "component" else "value = 1\n")
        module = package if kind == "initializer" else ModuleType(name)
        module.__file__ = str(path)
        monkeypatch.setitem(sys.modules, name, module)
        paths[kind], modules[kind] = path, module
    paths["resource"] = tmp_path / "policy.json"
    paths["resource"].write_text('{"value": 1}\n')
    plugin = extensions.register(
        extensions.Component("cache-test", "1", names["component"], files=(str(paths["resource"]),))
    )
    suite = build_suite(trials=1, components=[plugin.name])
    candidate = build_candidate(suite)
    return candidate, suite, plugin, paths, modules


def test_unchanged_hooks_do_not_reread_component_closure(selected, monkeypatch):
    candidate, _, _, paths, _ = selected
    original, reads = Path.read_bytes, []

    def read(path):
        if path in paths.values():
            reads.append(path)
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", read)
    extensions.active(candidate)
    reads.clear()
    for _ in range(3):
        assert extensions.active(candidate)[0][0].name == "cache-test"
    assert reads == []


def test_core_store_validation_import_keeps_calling_component_closure(selected, monkeypatch):
    _, _, plugin, paths, modules = selected
    module = modules["component"]
    owner = (module.__name__, paths["component"].resolve())
    monkeypatch.delitem(resources.ACCESS, owner)
    monkeypatch.setitem(resources.IMPORTS, owner, set())
    monkeypatch.setattr(extensions, "_CLOSURES", {})
    module.extensions = extensions
    source = "def select(plugin):\n    return extensions._closure(plugin)\n"
    exec(compile(source, str(paths["component"]), "exec"), module.__dict__)

    first = module.select(plugin)
    assert first == module.select(plugin)
    assert first[module.__name__] == sha256(paths["component"].read_bytes()).hexdigest()


def test_component_store_import_after_snapshot_still_refuses_construction(selected, monkeypatch):
    _, _, plugin, paths, modules = selected
    module = modules["component"]
    owner = (module.__name__, paths["component"].resolve())
    monkeypatch.delitem(resources.ACCESS, owner)
    monkeypatch.setitem(resources.IMPORTS, owner, set())
    monkeypatch.setattr(extensions, "_CLOSURES", {})
    source = "def import_owner():\n    from refund_agent.release.rule import owner_ids\n"
    exec(compile(source, str(paths["component"]), "exec"), module.__dict__)
    original = extensions._stores

    def stores(component):
        module.import_owner()
        return original(component)

    monkeypatch.setattr(extensions, "_stores", stores)
    with pytest.raises(ValueError, match="closure changed during construction"):
        extensions._closure(plugin)


@pytest.mark.parametrize("kind", ["initializer", "component", "helper", "resource"])
def test_each_closure_file_invalidates_released_candidate(selected, kind):
    candidate, suite, _, paths, _ = selected
    paths[kind].write_bytes(paths[kind].read_bytes() + b"\n")
    with pytest.raises(ValueError, match="released candidate"):
        extensions.active(candidate)
    assert build_candidate(suite).fingerprint != candidate.fingerprint


def test_same_size_edit_with_restored_mtime_invalidates(selected):
    candidate, _, _, paths, _ = selected
    path = paths["helper"]
    before = path.stat()
    path.write_bytes(path.read_bytes().replace(b"1", b"2"))
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    with pytest.raises(ValueError, match="released candidate"):
        extensions.active(candidate)


def test_deleted_dependency_is_not_hidden_by_cache(selected):
    candidate, _, _, paths, _ = selected
    paths["helper"].unlink()
    with pytest.raises(FileNotFoundError):
        extensions.active(candidate)


@pytest.mark.parametrize("replacement", ["file", "symlink"])
def test_replaced_dependency_invalidates(selected, tmp_path, replacement):
    candidate, _, _, paths, _ = selected
    target = tmp_path / "replacement.py"
    target.write_text("value = 2\n")
    if replacement == "file":
        target.replace(paths["helper"])
    else:
        paths["helper"].unlink()
        paths["helper"].symlink_to(target)
    with pytest.raises(ValueError, match="released candidate"):
        extensions.active(candidate)


def test_changed_component_manifest_invalidates(selected, tmp_path, monkeypatch):
    candidate, _, plugin, _, _ = selected
    extra = tmp_path / "extra.json"
    extra.write_text("{}\n")
    monkeypatch.setitem(
        extensions._REGISTRY, plugin.name, replace(plugin, files=(*plugin.files, str(extra)))
    )
    with pytest.raises(ValueError, match="released candidate"):
        extensions.active(candidate)


def test_changed_selection_is_not_reused(selected):
    candidate, suite, plugin, _, _ = selected
    selected_suite = replace(
        suite,
        components=(
            {"name": plugin.name, "module": plugin.module, "settings": {"variant": "other"}},
        ),
    )
    other = build_candidate(selected_suite)
    assert extensions.active(other)[0][1] == {"variant": "other"}
    assert extensions.active(candidate)[0][1] == {}
    assert other.fingerprint != candidate.fingerprint


def test_mutated_candidate_record_is_rechecked(selected):
    candidate, _, _, _, _ = selected
    extensions.active(candidate)
    candidate.tools_composition["components"][0]["fingerprint"] = "0" * 64
    with pytest.raises(ValueError, match="released candidate"):
        extensions.active(candidate)


def test_new_in_memory_dependency_is_not_hidden_by_cache(selected, tmp_path, monkeypatch):
    candidate, _, _, _, modules = selected
    helper = ModuleType("refund_agent._cache_added")
    helper.__file__ = str(tmp_path / "added.py")
    Path(helper.__file__).write_text("value = 1\n")
    monkeypatch.setitem(sys.modules, helper.__name__, helper)
    modules["component"].added = helper
    with pytest.raises(ValueError, match="released candidate"):
        extensions.active(candidate)


def test_import_observation_is_not_hidden_by_cache(selected, tmp_path, monkeypatch):
    candidate, _, _, paths, modules = selected
    helper = ModuleType("refund_agent._cache_observed")
    helper.__file__ = str(tmp_path / "observed.py")
    Path(helper.__file__).write_text("value = 1\n")
    monkeypatch.setitem(sys.modules, helper.__name__, helper)
    owner = (modules["component"].__name__, paths["component"].resolve())
    monkeypatch.setitem(resources.IMPORTS, owner, {helper.__name__})
    with pytest.raises(ValueError, match="released candidate"):
        extensions.active(candidate)


def test_observed_undeclared_read_is_rechecked_on_cache_hit(selected, tmp_path, monkeypatch):
    candidate, _, _, paths, modules = selected
    owner = (modules["component"].__name__, paths["component"].resolve())
    monkeypatch.setitem(resources.READS, owner, {tmp_path / "undeclared.json"})
    with pytest.raises(ValueError, match="undeclared resource before selection"):
        extensions.active(candidate)


def test_observed_undeclared_write_is_rechecked_on_cache_hit(selected, tmp_path, monkeypatch):
    candidate, _, _, paths, modules = selected
    owner = (modules["component"].__name__, paths["component"].resolve())
    monkeypatch.setitem(resources.WRITES, owner, {(tmp_path / "undeclared.json", os.O_WRONLY)})
    with pytest.raises(ValueError, match="outside authorized runtime store before selection"):
        extensions.active(candidate)


def test_changed_external_import_origin_is_rechecked(selected, tmp_path, monkeypatch):
    _, _, plugin, paths, _ = selected
    library = tmp_path / "library"
    library.mkdir()
    helper = ModuleType("_cache_external")
    helper.__file__ = str(library / "external.py")
    Path(helper.__file__).write_text("value = 1\n")
    monkeypatch.setitem(sys.modules, helper.__name__, helper)
    monkeypatch.setattr(resources, "_LIBRARIES", (*resources._LIBRARIES, library))
    paths["component"].write_text("import _cache_external\nfrom . import helper\n")
    extensions._closure(plugin)
    helper.__file__ = str(tmp_path / "external.py")
    Path(helper.__file__).write_text("value = 1\n")
    with pytest.raises(ValueError, match="first-party dependency outside refund_agent"):
        extensions._closure(plugin)


def test_warm_cache_cannot_serve_an_undeclared_component_read(selected):
    candidate, _, plugin, _, modules = selected
    modules["component"].__dict__["extensions"] = extensions
    exec(
        "def read():\n    return extensions._closure(plugin)\n",
        {**modules["component"].__dict__, "plugin": plugin},
        namespace := {},
    )
    extensions._closure(plugin)
    with resources.scope(resources.core_paths()):
        with pytest.raises(ValueError, match="undeclared resource"):
            namespace["read"]()


def test_returned_record_cannot_poison_cached_fingerprint(selected):
    candidate, _, _, _, _ = selected
    record = extensions.records(candidate.tools_composition["components"])
    record[0]["modules"].clear()
    record[0]["fingerprint"] = "0" * 64
    assert extensions.active(candidate)[0][0].name == "cache-test"


def test_unchanged_core_fingerprints_do_not_reread_files(selected, monkeypatch):
    from refund_agent.release.candidate import PACKAGE, source_fingerprint

    original, reads = Path.read_bytes, []

    def read(path):
        reads.append(path)
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", read)
    expected = source_fingerprint("tools.py", "rules.py")
    reads.clear()
    assert source_fingerprint("tools.py", "rules.py") == expected
    assert not {PACKAGE / "tools.py", PACKAGE / "rules.py"} & set(reads)


def test_cached_fingerprint_still_authorizes_the_current_reader(selected):
    from refund_agent.release.candidate import source_fingerprint

    _, _, _, paths, modules = selected
    source_fingerprint(str(paths["resource"]))
    namespace = {
        **modules["component"].__dict__,
        "source_fingerprint": source_fingerprint,
        "path": str(paths["resource"]),
    }
    exec("def read():\n    return source_fingerprint(path)\n", namespace)
    with resources.scope(resources.core_paths()):
        with pytest.raises(ValueError, match="undeclared resource"):
            namespace["read"]()


@pytest.mark.parametrize("kind", ["initializer", "component", "helper", "resource"])
@pytest.mark.parametrize("when", ["read", "publication"])
def test_concurrent_closure_edit_is_refused(selected, monkeypatch, kind, when):
    candidate, _, plugin, paths, modules = selected
    target = paths[kind]
    original_read, original_build = Path.read_bytes, extensions._build_closure
    edited = False

    def read(path):
        nonlocal edited
        content = original_read(path)
        if when == "read" and path == target and not edited:
            path.write_bytes(content + b"\n")
            edited = True
        return content

    def build(*args):
        nonlocal edited
        result = original_build(*args)
        if when == "publication" and not edited:
            target.write_bytes(original_read(target) + b"\n")
            edited = True
        return result

    monkeypatch.setattr(Path, "read_bytes", read)
    monkeypatch.setattr(extensions, "_build_closure", build)
    reason = "read" if when == "read" else "construction"
    with pytest.raises(ValueError, match="closure changed during " + reason):
        extensions._closure(plugin)
    source = str(target) if kind == "resource" else modules[kind].__name__
    assert extensions._closure(plugin)[source] == sha256(original_read(target)).hexdigest()
    with pytest.raises(ValueError, match="released candidate"):
        extensions.active(candidate)


@pytest.mark.parametrize(
    "statement", ["import refund_agent._cache_optional", "from refund_agent import _cache_optional"]
)
def test_optional_local_import_target_appearing_invalidates(
    selected, tmp_path, monkeypatch, statement
):
    from refund_agent.release import candidate as candidate_module

    candidate, suite, plugin, paths, _ = selected
    paths["component"].write_text(f"try:\n    {statement}\nexcept ImportError:\n    pass\n")
    monkeypatch.setattr(candidate_module, "PACKAGE", tmp_path)
    before = extensions._closure(plugin)
    candidate = replace(
        candidate,
        tools_composition={
            **candidate.tools_composition,
            "components": extensions.records(suite.components),
        },
    )
    assert extensions.active(candidate)[0][0] is plugin
    target = tmp_path / "_cache_optional.py"
    target.write_text("value = 1\n")
    after = extensions._closure(plugin)
    assert "refund_agent._cache_optional" not in before
    assert after["refund_agent._cache_optional"] == sha256(target.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="released candidate"):
        extensions.active(candidate)
    changed = replace(
        candidate,
        tools_composition={
            **candidate.tools_composition,
            "components": extensions.records(suite.components),
        },
    )
    assert changed.fingerprint != candidate.fingerprint
    assert extensions.active(changed)[0][0] is plugin


def test_changed_module_resolution_invalidates(selected, tmp_path):
    candidate, suite, _, _, modules = selected
    target = tmp_path / "moved.py"
    target.write_text("value = 2\n")
    modules["helper"].__file__ = str(target)
    with pytest.raises(ValueError, match="released candidate"):
        extensions.active(candidate)
    assert build_candidate(suite).fingerprint != candidate.fingerprint


def test_changed_byte_reader_invalidates(selected, monkeypatch):
    candidate, suite, _, paths, _ = selected
    extensions.active(candidate)
    original = Path.read_bytes

    def read(path):
        content = original(path)
        return content.replace(b"1", b"2") if path == paths["helper"] else content

    monkeypatch.setattr(Path, "read_bytes", read)
    with pytest.raises(ValueError, match="released candidate"):
        extensions.active(candidate)
    assert build_candidate(suite).fingerprint != candidate.fingerprint


def test_changed_fixed_core_membership_is_rechecked(selected, monkeypatch):
    candidate, _, _, paths, _ = selected
    extensions.active(candidate)
    core = resources.core_paths() | {paths["component"].resolve()}
    monkeypatch.setattr(resources, "core_paths", lambda: core)
    with pytest.raises(ValueError, match="outside the fixed core"):
        extensions.active(candidate)


def test_new_external_from_import_target_is_rechecked(selected, tmp_path, monkeypatch):
    _, _, plugin, paths, _ = selected
    library, local = tmp_path / "library", tmp_path / "local"
    library.mkdir()
    local.mkdir()
    namespace = ModuleType("_cache_external_namespace")
    namespace.__path__ = [str(library), str(local)]
    monkeypatch.setitem(sys.modules, namespace.__name__, namespace)
    monkeypatch.setattr(resources, "_LIBRARIES", (*resources._LIBRARIES, library))
    paths["component"].write_text(
        "try:\n    from _cache_external_namespace import optional\nexcept ImportError:\n    pass\n"
    )
    extensions._closure(plugin)
    (local / "optional.py").write_text("value = 1\n")
    with pytest.raises(ValueError, match="first-party dependency outside refund_agent"):
        extensions._closure(plugin)
