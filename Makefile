.PHONY: help all check sync lint format type test quick

# These four files keep the import order the book was built with, so ruff's import sort skips them.
FROZEN_IMPORTS := demos/part7_8_demo.py demos/reinforcement_learning_demo.py \
	refund_agent/security/security_privacy_attacks.py refund_agent/security/security_privacy_credentials.py
RUFF_FROZEN := $(foreach f,$(FROZEN_IMPORTS),--extend-per-file-ignores $(f):I001)

help:
	@echo "Targets:"
	@echo "  check  - Run what CI runs on every push: lint and the quick tests"
	@echo "  sync   - Install the pinned dependencies, all extras and dev tools (uv.lock is left as it is)"
	@echo "  lint   - Run ruff checks"
	@echo "  quick  - Run the quick tests listed in tests/ci/quick.txt"
	@echo "  test   - Run the whole pytest suite (it takes hours on one machine)"
	@echo "  type   - Run the ty type checker (an early release; not part of CI)"
	@echo "  format - Run the ruff formatter (it rewrites code the book prints: use it on new code only)"
	@echo "  all    - Run lint, type checks and tests"

all: lint type test

check: lint quick

sync:
	uv sync --frozen --all-extras

lint:
	uv run --offline --no-sync --extra dev ruff check . $(RUFF_FROZEN)

format:
	uv run --offline --no-sync --extra dev ruff format .

type:
	uv run --offline --no-sync --extra dev ty check --python-version 3.10 .

quick:
	@bash tests/ci/run-files.sh tests/ci/quick.txt

test:
	uv run --offline --no-sync --extra dev pytest
