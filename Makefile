# filter-sam3-detector Makefile
# OpenFilter SAM3 object detection filter

REPO_NAME_SNAKECASE ?= filter_sam3_detector
REPO_NAME_PASCALCASE ?= FilterSAM3Detector

# Pipeline for local testing with text prompt
PIPELINE := \
	- VideoIn --sources file://data/sample.mp4!loop \
	- $(REPO_NAME_SNAKECASE).filter.$(REPO_NAME_PASCALCASE) --text_prompt "$(TEXT_PROMPT)" \
	- Webvis

.PHONY: install run test docker-build docker-run publish clean help

help:
	@echo "filter-sam3-detector - SAM3 Object Detection Filter"
	@echo ""
	@echo "Usage:"
	@echo "  make install        Install the filter with dev dependencies"
	@echo "  make run            Run locally with CLI (requires TEXT_PROMPT=...)"
	@echo "  make test           Run tests"
	@echo "  make docker-build   Build Docker image for local development"
	@echo "  make docker-run     Run with docker-compose"
	@echo "  make publish        Build and publish to OpenFilter registry"
	@echo "  make clean          Remove build artifacts"
	@echo ""
	@echo "Examples:"
	@echo "  TEXT_PROMPT='person' make run"
	@echo "  CONFIDENCE_THRESHOLD=0.7 TEXT_PROMPT='car' make docker-run"

install:
	pip install -e .[dev] \
		--index-url https://python.openfilter.io/simple \
		--extra-index-url https://pypi.org/simple

run:
	@if [ -z "$(TEXT_PROMPT)" ]; then \
		echo "Error: TEXT_PROMPT is required. Usage: TEXT_PROMPT='person' make run"; \
		exit 1; \
	fi
	openfilter run $(PIPELINE)

test:
	pytest -vv -s tests/

docker-build:
	docker compose -f docker-compose.local.yaml build

docker-run:
	docker compose -f docker-compose.local.yaml up

docker-down:
	docker compose -f docker-compose.local.yaml down

# Publishing workflow
publish: clean
	@echo "Building distribution..."
	python -m build
	@echo "Publishing to OpenFilter registry..."
	twine upload --repository-url https://python.openfilter.io/ dist/*
	@echo "Done! Package published to https://python.openfilter.io/simple/filter-sam3-detector/"

clean:
	rm -rf build/ dist/ *.egg-info/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
