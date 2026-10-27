import socket
from collections.abc import Iterator

import pytest

from refund_agent.data.generate import Dataset, generate


@pytest.fixture(autouse=True)
def no_sockets(monkeypatch: pytest.MonkeyPatch) -> None:
    def blocked(*args: object, **kwargs: object) -> None:
        pytest.fail("refund-agent tests must not open sockets")

    monkeypatch.setattr(socket, "socket", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)


@pytest.fixture(scope="session")
def corpus() -> Iterator[Dataset]:
    yield generate(7)
