# vgi-statsmodels — dev and test targets.
#
# Usage:
#   make test       # unit/integration (pytest) + end-to-end SQL (haybarn-unittest)
#   make test-unit  # pytest only
#   make test-sql   # DuckDB sqllogictest .test files via haybarn-unittest
#
# test-sql is self-contained: it points VGI_STATSMODELS_WORKER at the worker run
# as a uv stdio subprocess (exactly how DuckDB drives it after ATTACH) and runs
# the files under test/sql/. haybarn-unittest is a uv tool:
#   uv tool install haybarn-unittest   # installs ~/.local/bin/haybarn-unittest

# Worker command DuckDB uses for ATTACH (overridable). Use the installed console
# script in the project venv (what CI uses) rather than `uv run <script>`: the
# latter resolves a separate PEP-723 env that can cache a stale SDK, whereas the
# venv script always runs the synced vgi-python.
WORKER_STDIO    ?= $(CURDIR)/.venv/bin/vgi-statsmodels

# haybarn-unittest lives in the uv tools bin; keep it on PATH.
HAYBARN_BIN     ?= $(HOME)/.local/bin
TEST_DIR         = .
TEST_PATTERN     = test/sql/*

.PHONY: test test-unit test-sql lint

test: test-unit test-sql

test-unit:
	uv run pytest -q

test-sql:
	PATH="$(HAYBARN_BIN):$$PATH" \
		VGI_STATSMODELS_WORKER="$(WORKER_STDIO)" \
		haybarn-unittest --test-dir "$(TEST_DIR)" "$(TEST_PATTERN)"

lint:
	uv run ruff format --check .
	uv run ruff check .
	uv run mypy vgi_statsmodels/
	uvx pydoclint vgi_statsmodels/
