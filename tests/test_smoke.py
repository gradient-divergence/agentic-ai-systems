from pydantic.version import VERSION


def test_pydantic_version_is_v2() -> None:
    assert VERSION.startswith("2.")
