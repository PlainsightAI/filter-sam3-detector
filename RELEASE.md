# Changelog
SAM3 Detector filter release notes

## [Unreleased]


## v0.1.6 - 2026-03-17
### Added
- Add filename to output filter subject data

## v0.1.5 - 2026-03-11
### Added
- Quick start guide focused on compose-first onboarding with detached commands and three runnable examples: `FILTER_TEXT_PROMPT`, `FILTER_TEXT_PROMPTS`, and positive/negative reference images.
- Optional utility script `scripts/convert_detections_jsonl_to_coco.py` to export `detections.jsonl` into COCO-style JSON (`images`, `annotations`, `categories` with `score`).
- Automatic COCO export on filter shutdown when `FILTER_OUTPUT_PATH` is configured (`FILTER_AUTO_EXPORT_COCO=true` by default).

### Changed
- Docker compose examples now surface get-started usage and output locations more clearly.
- Main compose example now accepts `VIDEO_PATH` and `FILTER_TEXT_PROMPTS`, writes `FILTER_OUTPUT_PATH`, and defaults to non-temporal get-started flow.

### Fixed
- **prompt_sets frame saving** (FILTER-349): `_process_multi_output` now saves original frames (once per frame) and annotated frames (per prompt set) when `FILTER_FRAMES_OUTPUT_DIR` / `FILTER_ANNOTATED_FRAMES_OUTPUT_DIR` are configured

## [Unreleased]

## v0.1.4 - 2026-02-24
### Added
- **Reference box prompts**: detection using positive and/or negative bounding boxes on the original image (SAM3-style geometric prompts). Set `FILTER_POSITIVE_BOXES` and/or `FILTER_NEGATIVE_BOXES` to a JSON array of `[x, y, w, h]` boxes in pixels; text prompt is optional. Visualization: green = positive ref, red = negative ref, blue = detections.

### Fixed
- Detection confidence: use scores from kept detections only (fix alignment when state scores include sub-threshold)
- Ref-images without SAM3: explicit branch with warning and frame forwarded unchanged (avoid AttributeError on processor)

### Changed
- Ref images: load and resize once in setup(); use cached PIL images per frame (no disk read per frame)
- Ref boxes: cache normalized boxes per resolution; recompute only when frame size changes

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
