"""Check sealed-execution opt-ins by reading Python source, never package data."""

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SEALED_TESTS = {
    "companion/tests/refund_agent/test_offers.py": {
        "test_committed_package_pins_windows_logger_and_every_learning_update",
    },
    "companion/tests/refund_agent/test_offpolicy.py": {
        "test_cap_zero_package_reports_only_fixed_rule_as_judged",
    },
    "companion/tests/test_practical_rl_demo.py": {
        "test_printed_listings_run_in_chapter_order_without_hidden_setup",
        "test_stored_output_regenerates_byte_for_byte",
        "test_fit_and_candidate_preparation_read_no_frozen_outcomes_or_golden",
        "test_offer_evaluation_logs_each_sealed_read_and_truth_only_after_record",
        "test_truth_consultation_is_refused_without_written_record",
    },
    "companion/exercises/practical-rl/test_acceptance.py": {
        "test_candidate_c_record_and_single_journaled_read",
    },
    "companion/tests/test_typed_decisions.py": {
        "test_seed_twenty_is_unused_by_any_other_history_draw",
    },
}


def _references_sealed(node: ast.AST) -> bool:
    for child in ast.walk(node):
        if isinstance(child, ast.Constant) and isinstance(child.value, str):
            if "exploration_bandits_package" in child.value or "PACKAGE_DIRECTORY" in child.value:
                return True
        if isinstance(child, (ast.Name, ast.Attribute)):
            name = child.id if isinstance(child, ast.Name) else child.attr
            if name in {"PACKAGE_DIRECTORY", "prepare_offer_evaluation", "copy_package"}:
                return True
        if isinstance(child, ast.ImportFrom):
            if any(alias.name in {"PACKAGE_DIRECTORY", "copy_package"} for alias in child.names):
                return True
        if isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute):
            if child.func.attr == "rglob" and ast.unparse(child.func.value) == "ROOT":
                return True
    return False


def _sealed_functions(tree: ast.Module) -> set[str]:
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    sealed = {node.name for node in functions if _references_sealed(node)}
    # Follow local helper calls and fixture arguments without importing the modules.
    while True:
        callers = {
            node.name
            for node in functions
            if any(
                (isinstance(child, ast.Name) and child.id in sealed)
                or (isinstance(child, ast.arg) and child.arg in sealed)
                for child in ast.walk(node)
            )
        }
        if callers <= sealed:
            return sealed
        sealed |= callers


@pytest.mark.skipif(
    not (ROOT.parent / "_quarto.yml").is_file(),
    reason="requires the book repository",
)
def test_canonical_sealed_paths_are_confined_to_opt_in_tests() -> None:
    problems = []
    seen = set()
    for directory in (ROOT / "tests", ROOT / "exercises", ROOT.parent / "scripts"):
        assert directory.is_dir(), f"missing required directory: {directory}"
        for path in sorted(directory.rglob("*.py")):
            if path == Path(__file__) or not (
                path.name.startswith("test_") or path.name == "conftest.py"
            ):
                continue
            relative = path.relative_to(ROOT.parent).as_posix()
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=relative)
            sealed = _sealed_functions(tree)
            expected = SEALED_TESTS.get(relative, set())
            actual = {name for name in sealed if name.startswith("test_")}
            if actual != expected or (sealed and relative not in SEALED_TESTS):
                problems.append(
                    f"{relative}: unexpected sealed access: {sorted(actual ^ expected)}"
                )
            for node in tree.body:
                if isinstance(node, ast.FunctionDef) and node.name in actual:
                    if "pytest.mark.sealed" not in map(ast.unparse, node.decorator_list):
                        problems.append(
                            f"{relative}:{node.lineno}: {node.name} needs sealed opt-in"
                        )
                elif not isinstance(node, ast.FunctionDef) and _references_sealed(node):
                    problems.append(f"{relative}:{node.lineno}: sealed access at module scope")
            seen.add(relative)
    assert SEALED_TESTS.keys() <= seen
    assert not problems, "\n".join(problems)
