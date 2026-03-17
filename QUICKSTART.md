# Quick Start

This guide focuses on first-run success with Docker Compose and clear, reproducible examples.

You will run one of these pipelines:

- `docker-compose.yaml` for text prompt examples (`FILTER_TEXT_PROMPT` and `FILTER_TEXT_PROMPTS`)
- `docker-compose.exemplar.yaml` for positive/negative reference images (and optional ref boxes)

## Prerequisites

- Docker and Docker Compose plugin
- NVIDIA GPU runtime configured (for default CUDA flow)
- A local input video path (no download step required)

## Example 1: Single prompt (`FILTER_TEXT_PROMPT`)

Use this when detecting one class.

```bash
VIDEO_PATH=/absolute/path/to/video.mp4 \
FILTER_TEXT_PROMPT=car \
docker compose -f docker-compose.yaml up -d
```

Open Webvis at `http://localhost:8001`.

Stop when done:

```bash
docker compose -f docker-compose.yaml down
```

Outputs (host):

- `./output/detections.jsonl`
- `./output/labels_coco.jsonl` (generated automatically on shutdown)

## Example 2: Multi prompt (`FILTER_TEXT_PROMPTS`)

Use this when detecting multiple classes in a single run.

```bash
VIDEO_PATH=/absolute/path/to/video.mp4 \
FILTER_TEXT_PROMPTS="avocado,chicken,salmon,egg" \
FILTER_CONFIDENCE_THRESHOLD=0.2 \
docker compose -f docker-compose.yaml up -d
```

Open Webvis at `http://localhost:8001`.

Stop when done:

```bash
docker compose -f docker-compose.yaml down
```

Outputs (host):

- `./output/detections.jsonl`
- `./output/labels_coco.jsonl` (generated automatically on shutdown)

## Example 3: Positive/negative reference images

Use this when you want visual guidance from positive and negative image references.

```bash
VIDEO_PATH=/absolute/path/to/video.mp4 \
FILTER_TEXT_PROMPT="small plastic cup with lid" \
FILTER_REF_IMAGES="/absolute/path/to/positive1.jpg,/absolute/path/to/positive2.jpg" \
FILTER_REF_IMAGES_NEGATIVE="/absolute/path/to/negative1.jpg" \
docker compose -f docker-compose.exemplar.yaml up -d
```

Open Webvis at `http://localhost:8002`.

Stop when done:

```bash
docker compose -f docker-compose.exemplar.yaml down
```

Outputs (host, default):

- `./results/detections.jsonl`
- `./results/labels_coco.jsonl` (generated automatically on shutdown)
- `./results/frames/`

## Test-case style verification

For each run, verify with the same checklist:

1. **Input**
   - Confirm `VIDEO_PATH` is the intended file.
2. **Pipeline**
   - Confirm compose file and env vars match the example.
3. **Output**
   - Confirm `detections.jsonl` exists and is non-empty.
   - Confirm Webvis displays detections consistent with prompts or ref images.
4. **Comparison**
   - Compare representative frames from your run against your expected result set (for example, chosen frame numbers or snapshots maintained by your team).

## Optional: Manual COCO export

COCO output is generated automatically when the filter stops. If you need to re-export manually:

```bash
python scripts/convert_detections_jsonl_to_coco.py \
  --input ./output/detections.jsonl \
  --output ./output/labels_coco.jsonl
```

Use `./results/detections.jsonl` as input when running `docker-compose.exemplar.yaml`.
