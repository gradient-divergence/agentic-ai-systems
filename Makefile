.PHONY: help all sync lint format type test

help:
	@echo "Targets:"
	@echo "  all    - Run lint, type checks, and tests"
	@echo "  sync   - Install dependencies (all extras + dev tools)"
	@echo "  lint   - Run ruff checks"
	@echo "  format - Run ruff formatter"
	@echo "  type   - Run ty type checker"
	@echo "  test   - Run pytest"

all: lint type test

sync:
	uv sync --all-extras

lint:
	uv run --offline --no-sync --extra dev ruff check .

format:
	uv run --offline --no-sync --extra dev ruff format .

type:
	uv run --offline --no-sync --extra dev ty check --python-version 3.10 .

test:
	uv run --offline --no-sync --extra dev pytest
