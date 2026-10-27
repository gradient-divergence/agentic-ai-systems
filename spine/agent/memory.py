"""Lightweight memory hooks for spine demos."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class MemoryRecord(BaseModel):
    key: str
    value: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class MemoryStore:
    def __init__(self) -> None:
        self._records: list[MemoryRecord] = []

    def add(self, record: MemoryRecord) -> None:
        self._records.append(record)

    def search(self, query: str, limit: int = 5) -> list[MemoryRecord]:
        matches = [
            record
            for record in self._records
            if query.lower() in record.key.lower() or query.lower() in record.value.lower()
        ]
        return matches[:limit]


def lookup_memory(store: MemoryStore, query: str, limit: int = 5) -> list[MemoryRecord]:
    return store.search(query, limit=limit)
