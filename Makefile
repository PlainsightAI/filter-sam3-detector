IMAGE ?= us-west1-docker.pkg.dev/plainsightai-prod/premium-filters/filter-sam3-detector
VERSION ?= $(shell cat VERSION 2>/dev/null | tr -d '[:space:]' | sed 's/^v//')
HF_SECRET ?= sam3-hf-token
HF_SECRET_PROJECT ?= plainsightai-prod

.PHONY: help install install-dev test test-coverage lint format build-wheel build-image publish-image clean

help:
	@echo "Available targets:"
	@echo "  install       - Install the package"
	@echo "  install-dev   - Install with development dependencies"
	@echo "  test          - Run tests (pytest); pass PYTEST_ARGS= for extras (e.g. --cov=filter_sam3_detector)"
	@echo "  test-coverage - Run tests with junit + coverage XML/JSON (used by Testmo composite action)"
	@echo "  lint          - Check code quality"
	@echo "  format        - Format code"
	@echo "  build-wheel   - Build Python wheel into dist/ (used by publish-python-wheel action)"
	@echo "  build-image   - Build Docker image \$$(IMAGE):\$$(VERSION) (pulls HF token from Secret Manager)"
	@echo "  publish-image - Push Docker image \$$(IMAGE):\$$(VERSION)"
	@echo "  clean         - Clean build artifacts"

install:
	pip install -e ".[dev]"

install-dev:
	pip install -e ".[dev]"

test:
	python -m pytest -v tests/ $(PYTEST_ARGS)

test-coverage:
	@mkdir -p Reports
	@python -m pytest -v --cov=filter_sam3_detector --junitxml=Reports/coverage.xml --cov-report=json:Reports/coverage.json tests/

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff format .
	uv run ruff check --fix .

build-wheel:
	python -m pip install --upgrade build
	python -m build --wheel

# The Dockerfile's SAM3 weight snapshot step requires `--mount=type=secret,id=hf_token`.
# Honor a pre-set HF_TOKEN env (e.g. passed through from the workflow's secrets block)
# and otherwise fall back to GCP Secret Manager, mirroring cloudbuild.yaml's approach.
# The reusable premium-release workflow authenticates to GCP before invoking this
# target, so `gcloud secrets versions access` works without extra setup as long as
# the workflow's service account has roles/secretmanager.secretAccessor on the secret.
build-image:
	@set -e; \
	if [ -z "$$HF_TOKEN" ]; then \
		HF_TOKEN=$$(gcloud secrets versions access latest --secret=$(HF_SECRET) --project=$(HF_SECRET_PROJECT)); \
	fi; \
	[ -n "$$HF_TOKEN" ] || { echo "HF_TOKEN not set and failed to fetch $(HF_SECRET) from Secret Manager"; exit 1; }; \
	TMPFILE=$$(mktemp); \
	trap 'rm -f "$$TMPFILE"' EXIT; \
	printf '%s' "$$HF_TOKEN" > "$$TMPFILE"; \
	DOCKER_BUILDKIT=1 docker build \
		-t $(IMAGE):$(VERSION) \
		--secret id=hf_token,src="$$TMPFILE" \
		.

publish-image:
	docker push $(IMAGE):$(VERSION)

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache
	rm -rf .ruff_cache
	rm -rf htmlcov
	find . -type d -name __pycache__ -exec rm -r {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

