IMAGE ?= us-west1-docker.pkg.dev/plainsightai-prod/premium-filters/filter-sam3-detector
.PHONY: help install install-dev test lint format clean

help:
	@echo "Available targets:"
	@echo "  install      - Install the package"
	@echo "  install-dev  - Install with development dependencies"
	@echo "  test         - Run tests (pytest); pass PYTEST_ARGS= for extras (e.g. --cov=filter_sam3_detector)"
	@echo "  lint         - Check code quality"
	@echo "  format       - Format code"
	@echo "  clean        - Clean build artifacts"

install:
	pip install -e ".[dev]"

install-dev:
	pip install -e ".[dev]"

test:
	python -m pytest -v tests/ $(PYTEST_ARGS)

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff format .
	uv run ruff check --fix .

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache
	rm -rf .ruff_cache
	rm -rf htmlcov
	find . -type d -name __pycache__ -exec rm -r {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

