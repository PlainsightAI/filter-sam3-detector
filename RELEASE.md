# Changelog
SAM3 Detector filter release notes

## [Unreleased]

### Fixed
- `docker-compose.yaml`: the default video mount pointed at `./data/sample-video.mp4`, which does not exist in the repo. It now points at the bundled `./data/car.mp4`, so `docker compose up` works without setting `VIDEO_PATH` first. `docker-compose.test.yaml` hard-mounted the same missing file and now uses the bundled clip too.
- `README.md`: the documented flow was `cp your_video.mp4 data/sample-video.mp4` followed by `docker compose up`, which only worked because of that stale default. It now leads with the bundled clip, so the block runs as written, and shows `VIDEO_PATH` as the custom-video override, so a custom video is actually used instead of being silently ignored.
- `.env.example`: default `FILTER_TEXT_PROMPT` was `post`, which matches the bundled PNG rather than the default video. Now `car`. `QUICKSTART.md` carried the same mismatch in Example 1 and in the Python-script example.
- Multi-prompt examples used comma-separated values (`car,truck`), which parse as a single prompt: `prompt_delimiter` defaults to `###`. Corrected across `QUICKSTART.md`, `.env.example`, `docs/filter-remove-overlap.md` and `docs/plan-sam-stabilization.md`.
- `docker-compose.yaml` used `${FILTER_TEXT_PROMPT:-car}`, which substitutes when the variable is unset **or empty**, so it overrode the deliberately-empty prompt that QUICKSTART Examples 2, 3 and 4 set to run on prompts, boxes or reference images. It is `${FILTER_TEXT_PROMPT-car}` now, substituting only when unset.
- `docker-compose.yaml` hardcoded `FILTER_ENABLE_TEMPORAL_INTERVALS: "false"` as a literal, so neither `.env` nor the shell could turn temporal intervals on and the README's documented run produced plain detection. It reads from the environment now, and the README block names the variable, says the image tag comes from `SAM3_DETECTOR_VERSION` rather than `latest`, and points at `./results`, which is the volume compose actually mounts.
- `docker-compose.yaml` defaulted `FILTER_TEXT_PROMPT` to the empty string, so a bare `docker compose up -d` in a clean checkout took the filter's no-prompt branch: it warned and emitted nothing, which reads as a broken pipeline rather than a missing setting. It now defaults to `car`, matching the bundled `./data/car.mp4`. `FILTER_TEXT_PROMPTS` was also absent from the `environment:` block, so setting it inline on the command line, the style the quickstart demonstrates, dropped it silently; it is declared now.
- `docker-compose.yaml` required an untracked `.env`, so the bare `docker compose up` documented in `README.md` and `QUICKSTART.md` failed in a clean checkout before the reader reached the `cp .env.example .env` step. The env file is optional now (`required: false`); every value it can carry already has a default.

- `README.md`: the Method 2 walk-through pointed at webvis on port `8001`; compose publishes `8002`, so the documented URL answered nothing.
- `README.md`: the Method 2 environment table carried names and defaults the filter does not have. `FILTER_HALF_LIFE` is `FILTER_TEMPORAL_HALF_LIFE` and its default is unset (`None`) rather than `5.0`, and `FILTER_TEMPORAL_PRESENCE_THRESHOLD` is `0.5` rather than `0.4`. The same two wrong values appeared in three separate tables.
- `README.md`: two Output Format blocks described the interval file as a JSON document wrapping the intervals with a `total_frames` count. It is ndjson, one interval per line, and `to_dict` (`temporal_intervals.py:51-59`) emits exactly five keys, none of them `total_frames`.
- `README.md`: several blocks promised intervals written to `output/intervals.json` on routes that write nothing. Nothing writes on that path unless `temporal_streaming_mode` is set alongside `temporal_output_json_path`, and `finalize()` closes the streaming handle without a non-streaming dump, so a reader following those blocks got an empty directory and no error.

### Added
- `QUICKSTART.md`: an input-video section naming the bundled clip and two public sample videos, with a table mapping each clip to the prompt variable and value that actually yields its classes.

## v0.1.31 - 2026-08-20

### Changed

- Build the filter image on `openfilter-base:py3.14` (was `py3.11`). The published wheel supports Python 3.14, so the image now ships 3.14. Running on 3.10–3.13 is unaffected.

## v0.1.30 - 2026-08-18

### Changed

- Update the openfilter dependency to 1.3.0
- Add Python 3.13 and 3.14 support: raise the `requires-python` ceiling to `<3.15`; the CI test matrix now runs 3.10–3.14.

## v0.1.29 - 2026-08-11

### Changed

- Build on `openfilter-base` instead of `pytorch/pytorch:*-cuda12.8-*-runtime`: the CUDA base was never apt-upgraded (OS-package CVEs). torch/torchvision are pinned to `>=2.9,<2.10` / `>=0.24,<0.25`, whose wheels bundle CUDA 12.8 (cu128) — what Blackwell (sm_120) needs — so Blackwell support rides on the torch wheel, not the base image. The pin is deliberate: an unpinned torch now resolves to 2.13.x (CUDA 13/cu13, no cu12 runtime); torch 2.9.1+cu128 is validated on Blackwell (RTX 5060) via the lab GPU smoke.
- Update the openfilter dependency to 1.2.2

## v0.1.28 - 2026-08-05

### Fixed
- `Dockerfile`: coerce an empty `hf_token` secret to `None` before downloading HF assets. When the `HF_TOKEN` secret is absent (Dependabot/fork PRs), the `--mount=type=secret` still creates an empty file, so `token` became `""` and was passed to `snapshot_download` for the public `kernels-community/cv-utils` kernel. `huggingface_hub` then emitted an invalid `Bearer ` (empty) auth header, failing the build with `httpx.LocalProtocolError: Illegal header value b'Bearer '` and breaking `release / dry-run-publish`. Now anonymous downloads work when no token is present.

## v0.1.27 - 2026-08-04

### Changed
- Update the openfilter dependency to `>=1.2.1`
- Point the `docker-compose.yaml` utility images at `openfilter-{video-in,webvis}:1.2.1` and pin the filter's own image default to the release version.
- Bump `actions/checkout` to `v7` in the `apply-rulesets` workflow (latest major, Node24 runtime).

## v0.1.26 - 2026-07-30

### Changed
- Grant `id-token: write` in `create-release.yaml` so the public release workflow will be able to produce a keyless (cosign) SBOM attestation once the shared SBOM steps land (PlainsightAI/gh-actions-public#32). Inert until then — this release publishes without an attestation.
- Update openfilter to 1.2.0 and the `av` pin to `~=17.1.0` (av 16→17) to match it.
- Replace the abandoned `decord` video reader with PyAV (`av`, already a dependency) in the vendored SAM3 `load_video_frames_from_video_file`. `decord` 0.6.0 and the `eva-decord` fork bundle a stale ffmpeg 4.x (CVE-2026-40962 class); PyAV uses ffmpeg 8.x. Removes `decord`/`eva-decord` from the dependency lists. Unblocks dropping the shared CVE-2026-40962 ignore (PlainsightAI/gh-actions-public#30).
- `detect_objects_video` example: `--prompt` now accepts multiple values (`--prompt "cup" "bowl"` or repeated `--prompt` flags), wiring them into the detector's `text_prompts`. Simplified the pipeline to write JSONL directly via the detector's built-in `output_path` (with annotated frames being written to `annotated_frames_output_dir` opt-in via `--visualize`), removing the `Recorder` and `ImageOut` sink filters.

## v0.1.25 - 2026-07-28

### Changed
- Add Blackwell (RTX PRO 6000 / sm_120) support: move the Docker base to `torch 2.10.0+cu128` so torch/torchvision ship sm_120 kernels. The previous `2.12.1+cu126` base (set in #50) crashed on Blackwell with `cudaErrorNoKernelImageForDevice` on the first GPU op and returned empty detections. What forced the move is cu128, not a specific version (`pytorch/pytorch` has no `2.12.1-cuda12.8` tag). 2.10.0-cu128 is chosen over 2.11.0 because it keeps `sm_70`: its compiled arch set is sm_70/75/80/86/90/100/120, so it adds Blackwell (sm_120) without dropping Volta/V100 (sm_70), which `2.11.0-cu128` drops. Stays on Ubuntu 24.04 / Python 3.12. Validated on real hardware: SAM3 detects on RTX PRO 6000 (Blackwell) and A10 (no regression).

## v0.1.24 - 2026-07-26

### Changed
- `README.md`: use the exact HyperLabel™ wording specified by Shanker for the filter family.

## v0.1.23 - 2026-07-25

### Changed
- `README.md`: document this filter as part of the Plainsight **Hyperlabel** family of filters.

### Fixed
- `pyproject.toml`: pin `opentelemetry-resourcedetector-gcp==1.11.0a0` via `override-dependencies`. `openfilter` requires `>=1.11.0a0,<1.12.dev0`, a range only pre-releases satisfy; uv accepted them while the package had no stable release, but `1.13.0` shipped on 2026-07-22 and resolution started failing in the Docker build (`dry-run-publish`) with "No solution found when resolving dependencies". Pinning the pre-release keeps the fix scoped to this one package instead of enabling pre-releases globally.

## v0.1.22 - 2026-07-16
### Added 
- Transformer's Sam3VideoModel support enabled
- Occasionally prunes session state to keep gpu memory under wraps. 
- Deprecated `FILTER_VIDEO_DETECTION_INTERVAL` and `FILTER_VIDEO_MIN_TRACKING_CONFIDENCE` are now explicitly warned on during config normalization and ignored; they no longer throttle video inference or change tracking confidence.

### Removed
- Streaming video processor and related tests removed

## v0.1.21 - 2026-07-09

### Fixed
- Guard the SAM3 weight-bake step in `Dockerfile` against a missing/empty `HF_TOKEN`. Dependabot and fork PR builds run without repo secrets, so the mounted `hf_token` secret is empty; the build now skips `snapshot_download` in that case instead of sending an illegal `Bearer ` header and failing the `dry-run-publish` check. The real publish path is unaffected and still bakes weights.

## v0.1.20 - 2026-06-19

### Added
 - **opt-in torch.compile for SAM3 vision backbone ([FILTER-373](https://plainsight-ai.atlassian.net/browse/FILTER-373))**
- **Multiplex grounding ([FILTER-374](https://plainsight-ai.atlassian.net/browse/FILTER-374))**: Multi-prompt detection now batches all prompts into a single decoder pass. Peak VRAM logic limits memory via fallback.

## v0.1.19 - 2026-06-03

### Added
- Register `FilterSAM3DetectorOutput` output schema under `openfilter.filter_runtime.shapes.DetectionSet` with `$id: https://schemas.plainsight.ai/filters/sam3-detector/v1` and data key `"detections"`.
- Add schema-compliance unit tests to cover coordinate validation and extra field pruning in `tests/test_filter_sam3_detector.py`.

### Changed
- **BREAKING**: `TemporalIntervalFilter` default configuration for `label_field` has been changed from `None` (track all as one) to `"label"` (track per-class). Existing standalone deployments that relied on `None` for single-signal tracking must be updated to explicitly set `label_field: null`.
- **BREAKING**: `output_boxes=False` and `output_scores=False` are no longer supported. The filter raises `ValueError` at startup if either is set; the canonical `FilterSAM3DetectorOutput` schema requires both fields.
- **BREAKING**: Multi-output mode now publishes detections to `frame.data["detections"]` as a canonical DetectionSet dictionary (`{"items": [...]}`) rather than a flat list. Downstream aggregators that previously iterated `frame.data["detections"]` directly must be updated to expect the new schema.
- Upgrade openfilter SDK package dependency to version 1.1.0.
- Upgrade openfilter SDK package dependency to version 1.1.1.
- Migrate `_extract_detections_from_state` to output canonical `bbox`, `label`, and `mask` structures.
- Transition frame processing to write the canonical detections to the top-level `frame.data["detections"]` path (legacy meta dual-writes retained for unmigrated consumers).
- Standardize `.jsonl` output records to follow the canonical detections schema format.
- Update downstream internal consumers (`confusion_detector.py`, `temporal_intervals.py`, and `coco_export.py`) to support the new schema structure.

### Removed
- (No removals - legacy protege-compatible dual-writes were restored to ensure backward compatibility for unmigrated consumers.)

## v0.1.18 - 2026-04-29
Enhances text_prompts parsing in FilterSAM3Detector with configurable delimiters and prompt→label mapping.

### Changed
- Added prompt_delimiter and class_delimiter
- Support class|||prompt format (e.g. "vehicle|||car###animal|||cat")
- Normalize prompts into list + prompt_label_map
- Validate delimiters and reject duplicate mappings
- Include both label and prompt in output

## v0.1.17 - 2026-04-29

### Fixed
- Fix Docker build secret format (use `secret-envs` format for `docker/build-push-action`)

## v0.1.16 - 2026-04-28

### Changed
- **Distribution channel pivoted to Docker Hub.** Image is now published to `plainsightai/openfilter-sam3-detector` (publicly pullable, no auth) instead of the GAR `premium-filters/` path. PM confirmed the filter is classified public — source has been Apache-2.0 + LicenseRef-SAM since v0.1.7, and the prebuilt artifact distribution now matches.
- **Release workflow flipped to the public reusable workflow** (`PlainsightAI/gh-actions-public/.github/workflows/filter-release.yaml`) instead of `gh-actions/filter-release-premium.yaml`. The public workflow publishes the wheel to PyPI in addition to the Docker image — first PyPI publish for `filter-sam3-detector`.
- **`cloudbuild.yaml` removed.** Cloud Build was silently double-publishing every release to Docker Hub alongside the GAR pipeline (with version-label drift on `:0.1.13` and a digest mismatch on `:0.1.15`); GitHub Actions is now the single source of truth.
- `docker-compose.yaml`, `examples/pipelines/raw-detections.yaml`, README Method 2, and QUICKSTART all reference the new Docker Hub path. The GAR auth prerequisites have been dropped from QUICKSTART and README.
- `Makefile` `IMAGE` default switched from the GAR premium path to `plainsightai/openfilter-sam3-detector`. Local `make build-image` / `make publish-image` now target Docker Hub.

### CI
- `create-release.yaml` passes `platforms: linux/amd64` to the public reusable workflow. The default `linux/amd64,linux/arm64` matrix would fail at the base-image pull because `pytorch/pytorch:*-cuda*` ships amd64-only across every CUDA tag.
- HF_TOKEN is forwarded into the publish-docker job via `forward_secrets_as_env` and mounted as the `id=hf_token` BuildKit secret via `build_secrets`, so the gated SAM3 weights are still pulled and baked at build time.

## v0.1.15 - 2026-04-23

### CI
- Consolidate `make test` onto a single pytest path; coverage flags pass through `PYTEST_ARGS=` (e.g. `make test PYTEST_ARGS=--cov=filter_sam3_detector`). Drops the stdlib `unittest discover` invocation.
- Drop duplicate `test.yaml` workflow — `release / run-tests` already gates every PR via the reusable filter-release workflow.
- Apply main-branch merge gate via `.github/rulesets/main.json` + `apply-rulesets.yaml` (rulesets-as-code). Branch protection now lives in-tree and self-applies on pushes to main.
- Add `build-wheel`, `build-image`, `publish-image` Makefile targets so the reusable premium-release workflow's wheel + image publish jobs complete end-to-end (no more `make: *** No rule to make target 'build-wheel'`). `build-image` honors a pre-set `HF_TOKEN` env and otherwise fetches `sam3-hf-token` from GCP Secret Manager, matching `cloudbuild.yaml`'s approach. New `DOCKER_TAG` variable strips the `v` prefix from `VERSION` so image tags match `cloudbuild.yaml`'s convention. Drop redundant `install-dev` target (identical to `install`). Add `Makefile` to the release workflow's `source-paths` so future Makefile-only changes trigger the release-log check.

### Changed
- Loosen `[dev]` pins (`setuptools`, `wheel`, `pytest`, `pytest-cov`) from `==` to `~=` so patch-level fixes are picked up while keeping the current minor cap. Protects `release / run-tests` (which installs via `make install` → `pip install -e ".[dev]"`) from future pytest 9 / setuptools 80 surprises.

## v0.1.14 - 2026-04-21

### Changed
- Bump openfilter SDK to >=0.1.30
- Add source-paths release gate to CI workflow
- Add test-coverage Makefile target for Testmo composite action

### Fixed
- Fix test_config_defaults: update expected model_id from `facebook/sam2-hiera-large` to `facebook/sam3`
- Fix test_output_json_file: auto-enable streaming_mode when output_json_path + emit_on_complete are set
- Add `debug: False` to SAM3 filter defaults

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
