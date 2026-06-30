.DEFAULT_GOAL := help
DIST := dist

.PHONY: help install dev lint fmt test build clean smoke check

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install:  ## Editable install (core deps only)
	pip install -e .

dev:  ## Editable install with dev + cli + server extras
	pip install -e ".[dev,cli,server]"

lint:  ## Run ruff linter
	ruff check src/ tests/

fmt:  ## Auto-format with ruff
	ruff format src/ tests/
	ruff check --fix src/ tests/

test:  ## Run pytest
	pytest --tb=short -q

build: clean  ## Build sdist + wheel into dist/
	python -m build
	@echo ""
	@ls -lh $(DIST)/

clean:  ## Remove build artifacts
	rm -rf $(DIST) build src/*.egg-info src/argus_curator/*.egg-info

smoke: build  ## Build wheel, install in tmp venv, smoke-test import
	$(eval TMPVENV := $(shell mktemp -d))
	python -m venv $(TMPVENV)
	$(TMPVENV)/bin/pip install --quiet $(DIST)/*.whl
	$(TMPVENV)/bin/python -c \
		"from argus_curator import scan_folder, __version__; print(f'argus-curator {__version__} OK')"
	rm -rf $(TMPVENV)

check: lint test build  ## Full local CI: lint + test + build
	@echo ""
	@echo "All checks passed."
