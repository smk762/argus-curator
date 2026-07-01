.DEFAULT_GOAL := help
DIST := dist
UV := uv

.PHONY: help install dev lint fmt test build clean smoke check schema

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install:  ## Create .venv + editable install (core deps only)
	$(UV) venv
	$(UV) pip install -e .

dev:  ## Create .venv + editable install with dev + cli + server extras
	$(UV) venv
	$(UV) pip install -e ".[dev,cli,server]"

lint:  ## Run ruff linter
	$(UV) run --no-sync ruff check src/ tests/

fmt:  ## Auto-format with ruff
	$(UV) run --no-sync ruff format src/ tests/
	$(UV) run --no-sync ruff check --fix src/ tests/

test:  ## Run pytest
	$(UV) run --no-sync pytest --tb=short -q

schema:  ## Regenerate the committed wire-contract JSON Schema
	$(UV) run --no-sync argus-curator schema

build: clean  ## Build sdist + wheel into dist/
	$(UV) build
	@echo ""
	@ls -lh $(DIST)/

clean:  ## Remove build artifacts
	rm -rf $(DIST) build src/*.egg-info src/argus_curator/*.egg-info

smoke: build  ## Build wheel, install in throwaway venv, smoke-test import
	$(eval TMPVENV := $(shell mktemp -d))
	$(UV) venv $(TMPVENV)/venv
	$(UV) pip install --python $(TMPVENV)/venv $(DIST)/*.whl
	$(TMPVENV)/venv/bin/python -c \
		"from argus_curator import scan_folder, __version__; print(f'argus-curator {__version__} OK')"
	rm -rf $(TMPVENV)

check: lint test build  ## Full local CI: lint + test + build
	@echo ""
	@echo "All checks passed."
