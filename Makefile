IMAGE ?= us-west1-docker.pkg.dev/plainsightai-prod/premium-filters/filter-sam3-detector
# VERSION keeps the "v" prefix when set (matches the release_tag output the
# reusable workflow passes through as an env var); DOCKER_TAG strips it for
# image tagging, matching cloudbuild.yaml's `sed 's/^v//'` convention.
VERSION ?= $(shell cat VERSION 2>/dev/null | tr -d '[:space:]')
DOCKER_TAG ?= $(VERSION:v%=%)
HF_SECRET ?= sam3-hf-token
# Secret lives in plainsightai-dev (the premium-release SA in prod can't reach
# it without a cross-project IAM grant we explicitly chose not to create;
# release CI threads HF_TOKEN through as a repo-level GH Actions secret
# instead). This default is only hit when someone runs `make build-image`
# locally with ADC into dev.
HF_SECRET_PROJECT ?= plainsightai-dev

.PHONY: help install test test-coverage lint format build-wheel build-image publish-image clean

help:
	@echo "Available targets:"
	@echo "  install       - Install the package with dev dependencies"
	@echo "  test          - Run tests (pytest); pass PYTEST_ARGS= for extras (e.g. --cov=filter_sam3_detector)"
	@echo "  test-coverage - Run tests with junit + coverage XML/JSON (used by Testmo composite action)"
	@echo "  lint          - Check code quality"
	@echo "  format        - Format code"
	@echo "  build-wheel   - Build Python wheel into dist/ (used by publish-python-wheel action)"
	@echo "  build-image   - Build Docker image \$$(IMAGE):\$$(DOCKER_TAG) (pulls HF token from Secret Manager)"
	@echo "  publish-image - Push Docker image \$$(IMAGE):\$$(DOCKER_TAG)"
	@echo "  clean         - Clean build artifacts"

install:
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
		-t $(IMAGE):$(DOCKER_TAG) \
		--secret id=hf_token,src="$$TMPFILE" \
		.

# Requires the caller to have docker auth configured for $(IMAGE)'s registry
# (e.g. `gcloud auth configure-docker us-west1-docker.pkg.dev` for GAR).
# The reusable premium-release workflow runs this in its `Configure Docker for GAR`
# step before invoking this target; direct callers must do the equivalent.
publish-image:
	docker push $(IMAGE):$(DOCKER_TAG)

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache
	rm -rf .ruff_cache
	rm -rf htmlcov
	find . -type d -name __pycache__ -exec rm -r {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

