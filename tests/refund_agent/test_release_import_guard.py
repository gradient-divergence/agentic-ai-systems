"""Origin discovery must preserve allowed imports and first-party refusals."""

import subprocess
import sys
from types import ModuleType

import pytest

from refund_agent.release.agent_evaluation_resources import module_origin


def test_guard_allows_cold_namespace_and_setuptools_imports_and_keeps_refusal(tmp_path):
    program = f"""
import sys
from pathlib import Path
from types import ModuleType
from refund_agent.release import agent_evaluation_resources as resources

resources.install_import_guards()
assert 'scripts' not in sys.modules and 'setuptools' not in sys.modules
import demos.regions as snippets
from setuptools._distutils.errors import DistutilsError
assert snippets.__name__ == 'demos.regions'
assert issubclass(DistutilsError, Exception)

component = ModuleType('refund_agent._test_namespace_import')
path = Path({str(tmp_path / "component.py")!r})
source = 'def undeclared():\\n    import demos.regions\\n'
path.write_text(source)
component.__file__ = str(path)
exec(compile(source, str(path), 'exec'), component.__dict__)
resources.ACCESS[(component.__name__, path.resolve())] = (frozenset([path.resolve()]), {{}})
try:
    component.undeclared()
except ValueError as error:
    assert str(error) == 'component import of undeclared first-party dependency'
else:
    raise AssertionError('undeclared first-party import succeeded')
"""
    result = subprocess.run(
        [sys.executable, "-c", program], capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_guard_allows_cached_namespace_child_without_its_parent(tmp_path):
    (tmp_path / "_guard_cached_namespace" / "child").mkdir(parents=True)
    program = f"""
import sys
sys.path.insert(0, {str(tmp_path)!r})
from _guard_cached_namespace import child
sys.modules.pop('_guard_cached_namespace')
from _guard_cached_namespace import child
assert child.__name__ == '_guard_cached_namespace.child'
sys.modules.pop('_guard_cached_namespace')

import refund_agent
from _guard_cached_namespace import child
assert child.__name__ == '_guard_cached_namespace.child'
"""
    result = subprocess.run(
        [sys.executable, "-c", program], capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    "source,dependency",
    [
        (
            "from refund_agent._guard_cold_exports import Public",
            "refund_agent._guard_cold_exports",
        ),
        (
            "import refund_agent._guard_cold_exports.__init__",
            "refund_agent._guard_cold_exports.__init__",
        ),
    ],
)
def test_cold_package_attributes_and_real_modules_keep_their_dependency_names(
    tmp_path, source, dependency
):
    package = tmp_path / "_guard_cold_exports"
    package.mkdir()
    (package / "__init__.py").write_text("class Public:\n    pass\n")
    program = f"""
from pathlib import Path
from types import ModuleType
import refund_agent
from refund_agent.release import agent_evaluation_resources as resources
refund_agent.__path__.append({str(tmp_path)!r})
component = ModuleType('refund_agent._guard_cold_owner')
path = Path({str(tmp_path / "owner.py")!r})
path.write_text({source!r})
component.__file__ = str(path)
exec(compile({source!r}, str(path), 'exec'), component.__dict__)
assert resources.IMPORTS[(component.__name__, path.resolve())] == {{{dependency!r}}}
"""
    result = subprocess.run(
        [sys.executable, "-c", program], capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_origin_lookup_resolves_unimported_parents_without_initializers(tmp_path, monkeypatch):
    namespace = tmp_path / "_guard_namespace"
    package = namespace / "package"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("raise AssertionError('initializer executed')\n")
    helper = package / "helper.py"
    helper.write_text("value = 1\n")
    monkeypatch.syspath_prepend(str(tmp_path))

    assert module_origin("_guard_namespace.package.helper") == str(helper)
    assert "_guard_namespace" not in sys.modules
    assert "_guard_namespace.package" not in sys.modules


def test_unknown_descendant_retains_first_party_parent_origin(tmp_path, monkeypatch):
    namespace = tmp_path / "_guard_unknown"
    namespace.mkdir()
    monkeypatch.syspath_prepend(str(tmp_path))

    assert module_origin("_guard_unknown.missing.child") == str(namespace)


@pytest.mark.parametrize(
    "module_name,attribute",
    [
        ("refund_agent.release.agent_evaluation_resources", "metadata"),
        ("refund_agent", "import_guard_marker"),
    ],
)
def test_from_import_tracks_the_module_without_inventing_attribute_dependencies(
    tmp_path, monkeypatch, module_name, attribute
):
    import refund_agent
    from refund_agent.release import agent_evaluation_resources as resources

    monkeypatch.setattr(refund_agent, "import_guard_marker", object(), raising=False)
    monkeypatch.setattr(resources, "IMPORTS", {})
    component = ModuleType("refund_agent._test_attribute_import")
    path = tmp_path / "attribute_import.py"
    source = f"from {module_name} import {attribute}\n"
    path.write_text(source)
    component.__file__ = str(path)

    exec(compile(source, str(path), "exec"), component.__dict__)

    assert resources.IMPORTS[(component.__name__, path.resolve())] == {module_name}
