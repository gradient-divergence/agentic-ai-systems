"""Require explicit authorization before tests access the canonical sealed package.

In a standalone clone of the companion, outside the book's own repository, three lists in tests/ci/
also apply: the tests that need the book's repository are skipped, and the tests that are known to
fail, and those that compare floating-point results byte for byte, are marked as expected failures,
so that they stay visible in every run.
"""

import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent
STANDALONE = not (ROOT.parent / "_quarto.yml").is_file()


def _listed(name: str) -> set[str]:
    path = ROOT / "tests" / "ci" / name
    if not path.is_file():
        return set()
    return {
        line.split("#", 1)[0].strip()
        for line in path.read_text().splitlines()
        if line.split("#", 1)[0].strip()
    }


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    if STANDALONE:
        book_only, known = _listed("needs-book-repo.txt"), _listed("known-failures.txt")
        machine = _listed("machine-sensitive.txt")
        book = pytest.mark.skip(reason="requires the book repository")
        failing = pytest.mark.xfail(
            reason="known failure, listed in tests/ci/known-failures.txt", strict=False
        )
        digits = pytest.mark.xfail(
            reason="compares floating-point results byte for byte; another CPU can differ in the last digits",
            strict=False,
        )
        for item in items:
            if item.nodeid in book_only:
                item.add_marker(book)
            elif item.nodeid in known:
                item.add_marker(failing)
            elif item.nodeid in machine:
                item.add_marker(digits)
    if os.environ.get("FND_ALLOW_SEALED") == "1":
        return
    skip = pytest.mark.skip(
        reason="canonical sealed evidence requires a separately authorized run (FND_ALLOW_SEALED=1)"
    )
    for item in items:
        if item.get_closest_marker("sealed") is not None:
            item.add_marker(skip)
