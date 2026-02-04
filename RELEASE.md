# Changelog
SAM3 Detector filter release notes

## [Unreleased]

## v0.1.3 - 2026-02-04

### Added
- Reference image prompts: `FILTER_REF_IMAGES` and `FILTER_REF_IMAGES_NEGATIVE` for positive exemplars (pasted bottom-left) and negative exemplars (pasted bottom-right); text prompt required when using ref images. Optional `ref_margin` and `ref_gap` for positioning (env: `FILTER_REF_MARGIN`, `FILTER_REF_GAP`).

## v0.1.2 - 2026-01-23

### Added
- Streaming video processor with detection throttling
- Text embedding caching and backbone sharing optimization
- Frame ID extraction and protege-compatible output
- SAM3 multi-output mode with `prompt_sets` configuration

### Changed
- Dev builds now push to prod registry

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
