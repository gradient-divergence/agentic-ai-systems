"""Require explicit authorization before tests access the canonical sealed package."""

import os

import pytest


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    if os.environ.get("FND_ALLOW_SEALED") == "1":
        return
    skip = pytest.mark.skip(
        reason="canonical sealed evidence requires a separately authorized run (FND_ALLOW_SEALED=1)"
    )
    for item in items:
        if item.get_closest_marker("sealed") is not None:
            item.add_marker(skip)
