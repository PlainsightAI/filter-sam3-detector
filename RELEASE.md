# Changelog
SAM3 Detector filter release notes

## v0.1.13 - 2026-04-20

### Changed
- Add create-release.yaml for GAR premium publishing
- Add shared security-scan workflow
- Remove old version-check.yaml
- Add Makefile IMAGE for premium-filters/
- Bump openfilter to >=0.1.27
- Update docker-compose.yaml to openfilter 0.1.27

## v0.1.12 - 2026-04-19
### Fixed
- **Air-gapped deploys** (FILTER-422): set `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` in the Dockerfile so `huggingface_hub` skips HEAD revalidation against `huggingface.co` and serves the baked-in SAM3 cache directly. Unblocks `docker run --network=none` and offline deployments.

## v0.1.11 - 2026-04-19
### Changed
- SPDX license expression in `pyproject.toml` normalized to `Apache-2.0 AND LicenseRef-SAM`; redundant `License ::` classifiers dropped.
- `.dockerignore` now allow-lists `LICENSING.md` so redistribution compliance files ship with built images.
- Pinned `numpy>=1.26.4,<3` via `[tool.uv] override-dependencies` to keep transitive resolution stable across SAM3 / torch wheels.

### CI
- `version-check.yaml` gates `check-release-log` behind a `dorny/paths-filter` step; pure docs/CI-only PRs (e.g. `.github/**`, `*.md`, `.dockerignore`-only edits) no longer require a VERSION/RELEASE.md bump. Mirrors the pattern in `PlainsightAI/protege-ml`.

## v0.1.10 - 2026-04-08
### Added
- Bfloat16 mixed-precision inference for image path via persistent `torch.autocast` context, matching SAM3 video path pattern (FILTER-394)
- `FILTER_MIXED_PRECISION` config flag (default: true on CUDA, no-op on CPU/MPS)

## v0.1.9 - 2026-04-06
### Added
- **Batched backbone inference** (FILTER-369): `process_batch()` runs the SAM3 vision backbone on accumulated frames in a single `set_image_batch()` call, then fans out per-frame grounding. Configurable via `FILTER_BATCH_SIZE` and `FILTER_ACCUMULATE_TIMEOUT_MS` (requires openfilter >= 0.1.16).

### Removed
- Vestigial `multiprocessing.set_start_method("spawn")` workaround (vidgear removed from openfilter).

## v0.1.8 - 2026-03-31
### Added
- **Cross-class overlap detection** (`ConfusionDetector`): new `filter_sam3_detector/confusion_detector.py` module that computes pairwise IoU between detections from different text prompts and flags near-identical regions (default threshold: IoU ≥ 0.95).
- **`FILTER_REMOVE_OVERLAP`** (default `false`): opt-in shutdown pass that keeps the highest-confidence detection per cross-class overlapping cluster and writes `detections_cleaned.jsonl`. Same-class boxes are unchanged (still handled by per-prompt NMS).
- **`FILTER_CONFUSION_IOU_THRESHOLD`** (default `0.95`): configurable IoU gate for overlap detection and removal.
- **Shutdown summary**: at end-of-run, logs cross-class **overlap pair** counts (before / after / removed) and **detection** totals (before / after / removed), plus cleaned JSONL path when `FILTER_REMOVE_OVERLAP=true`.
- **`scripts/analyze_confusions.py`**: standalone post-processing script that reads `detections.jsonl`, aggregates per-pair confusion statistics (rate, avg/max IoU, example frames), and emits tiered resolution guidance (`text` or `json` output).
- **`docs/filter-remove-overlap.md`**: operator walkthrough for `FILTER_TEXT_PROMPTS=car,truck` + `FILTER_REMOVE_OVERLAP=true` with expected JSONL and shutdown log.
- **Visualization (`FILTER_VISUALIZE`)**: annotated frames and viz topic now draw the **detection class label** (`label` / `class` / `class_name`) on each box in addition to the score, with a **stable color per class** so multi-prompt runs (e.g. `car` vs `truck`) are easy to read in Webvis and saved annotated frames.

### Changed
- `QUICKSTART.md` Example 2 now references `FILTER_REMOVE_OVERLAP` and links to `docs/filter-remove-overlap.md`.
- Confusion detection is auto-enabled (stats only, no removal) when `FILTER_TEXT_PROMPTS` contains more than one class; single-prompt runs see zero overhead.
- **Shutdown order:** cross-prompt overlap finalize runs **before** automatic COCO export. When `FILTER_REMOVE_OVERLAP=true` and `detections_cleaned.jsonl` is written, **`labels_coco.json`** is generated from the **cleaned** JSONL (otherwise from the primary `detections.jsonl`).

## v0.1.7 - 2026-03-25
### Added
- Dual licensing documentation (`LICENSING.md`) and updated README badge
- License files (`LICENSE`, `LICENSING.md`) now copied into Docker images for redistribution compliance
- PyPI metadata updated with dual license expression and license file bundling

## v0.1.6 - 2026-03-17
### Added
- Quick start guide focused on compose-first onboarding with detached commands and runnable examples: `FILTER_TEXT_PROMPT`, `FILTER_TEXT_PROMPTS`, `FILTER_POSITIVE_BOXES`, and `FILTER_REF_IMAGES`.
- Optional utility script `scripts/convert_detections_jsonl_to_coco.py` to export `detections.jsonl` into COCO-style JSON (`images`, `annotations`, `categories` with `score`).
- Automatic COCO export on filter shutdown when `FILTER_OUTPUT_PATH` is configured (`FILTER_AUTO_EXPORT_COCO` opt-in).

### Changed
- Docker compose examples now surface get-started usage and output locations more clearly.
- Main compose example now accepts `VIDEO_PATH` and prompt variants, writes `FILTER_OUTPUT_PATH`, and defaults to non-temporal get-started flow.

## v0.1.5 - 2026-03-11
### Added
- Add filename to output filter subject data

### Fixed
- **prompt_sets frame saving** (FILTER-349): `_process_multi_output` now saves original frames (once per frame) and annotated frames (per prompt set) when `FILTER_FRAMES_OUTPUT_DIR` / `FILTER_ANNOTATED_FRAMES_OUTPUT_DIR` are configured

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
