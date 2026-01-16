# Changelog
SAM3 Detector filter release notes

## [Unreleased]

## v0.1.1 - 2026-01-16

### Fixed
- Cloud Build: Fix GAR authentication for dev tag builds by using separate cloud-sdk step
- Cloud Build: Fix shell variable escaping for Cloud Build substitution variables
- Cloud Build: Strip 'v' prefix from VERSION file to ensure consistent Docker tags
- VERSION: Add 'v' prefix to match RELEASE.md format for GitHub Actions version check

### Added
- GitHub Actions: Add PR check for RELEASE.md and VERSION file consistency

## v0.1.0 - 2024-12-29

### Added
- Initial Release: new SAM3 detector filter
- Open-set object detection with text prompts
- Exemplar-based detection (few-shot learning)
- Support for bounding boxes, masks, and confidence scores
- GPU and CPU support (CUDA, CPU, MPS)
- Integration with OpenFilter framework
- Example scripts for common use cases
- Comprehensive documentation
