# Sweetgreen Pipeline Integration

This guide covers testing the SAM3 detector as a drop-in replacement for protege model filters in the Sweetgreen pipeline.

## Prerequisites

1. **Docker** with NVIDIA GPU support
2. **gcloud CLI** authenticated with access to `plainsightai-dev` and `plainsightai-prod` projects
3. **gsutil** for downloading test videos

### Docker Registry Authentication

Authenticate with Google Artifact Registry to pull production images:

```bash
# Authenticate with GAR (required for video_in, aggregator, webvis images)
gcloud auth configure-docker us-west1-docker.pkg.dev

# Verify authentication
docker pull us-west1-docker.pkg.dev/plainsightai-prod/oci/video_in:v1.4.25
```

## Getting Test Data

Download and prepare the Sweetgreen test video:

```bash
# Download from GCS (requires access to protege-artifacts-production bucket)
gsutil cp gs://protege-artifacts-production/customer-sweetgreen/Shared-Videos/TestVideo-2025-11-18.mp4 data/input.mp4

# Extract a clip starting at 1:01 (avoids initial setup frames)
# The full video is ~50 minutes at 30fps (92,762 frames)
ffmpeg -ss 00:01:01 -i data/input.mp4 -c copy -avoid_negative_ts make_zero data/sample-video.mp4
```

**Note:** The sample video is ~1.6GB. At 5fps processing rate, the full video takes ~50 minutes to process (~15,000 frames). For quick testing, you can create a shorter clip:

```bash
# Create a 2-minute test clip (starts at 1:01, runs for 2 minutes)
ffmpeg -ss 00:01:01 -t 00:02:00 -i data/input.mp4 -c copy data/sample-video.mp4
```

## Building the Local Image

Build the SAM3 detector image locally for testing:

```bash
# Get HuggingFace token from GCP secrets
HF_TOKEN=$(gcloud secrets versions access latest --secret=sam3-hf-token --project=plainsightai-dev)

# Build the image
docker build --secret id=hf_token,env=HF_TOKEN -t filter-sam3-detector:local .
```

## Running the Pipeline

### Local Testing (without event sink)

```bash
# With local build
SAM3_IMAGE=filter-sam3-detector:local \
  docker compose -f sweetgreen.yaml up --abort-on-container-exit

# With remote image (0.2.0-dev tag)
docker compose -f sweetgreen.yaml up --abort-on-container-exit
```

### Production Mode (with event sink)

```bash
# Set required environment variables
export FILTER_API_ENDPOINT="https://api.prod.plainsight.tech/filter-pipelines/sweetgreen-<location>/events?project=<project-id>"
export FILTER_API_TOKEN="<your-token>"

# Run with event sink profile
docker compose -f sweetgreen.yaml --profile event-sink up --abort-on-container-exit
```

### Viewing Results

- **Web visualization**: http://localhost:8020
- **JSONL output**: `cat output/detections.jsonl | jq .`

## Architecture

### Multi-Output Mode (sweetgreen.yaml)

A single SAM3 instance handles all detection tasks using **multi-output mode**. The model loads once, processes each frame once through the expensive backbone pass, then runs each prompt set against the cached image features:

```
                        ┌─────────────────────┐
                        │      video_in       │
                        │  (emits _filter ID) │
                        └─────────┬───────────┘
                                  │
                                  ▼
                        ┌─────────────────────┐
                        │    sam3_detector    │
                        │   (multi-output)    │
                        │                     │
                        │ prompt_sets:        │
                        │  - bowl_detector    │
                        │  - chit_detector    │
                        │  - ingredients      │
                        │  - dressing         │
                        └─────────┬───────────┘
                                  │
              ┌───────────┬───────┼───────┬───────────┐
              │           │       │       │           │
              ▼           ▼       ▼       ▼           │
            main        chit    bowl   dressing      │
              │           │       │     _cups        │
              └───────────┴───────┴───────┘          │
                                  │                  │
                                  ▼                  │
                        ┌─────────────────────┐      │
                        │     aggregator      │◄─────┘
                        │  (real production)  │ (receives _filter)
                        └─────────┬───────────┘
                                  │
                    ┌─────────────┴─────────────┐
                    │                           │
                    ▼                           ▼
            ┌─────────────┐           ┌─────────────────┐
            │   webvis    │           │  event_sink     │
            │  (always)   │           │ (--profile      │
            └─────────────┘           │  event-sink)    │
                                      └─────────────────┘
```

**Why multi-output mode?**
- **Memory efficient**: One GPU, one model load (~16GB VRAM)
- **Fast**: Backbone pass runs once per frame, then each prompt set uses cached features
- **Production-compatible**: Outputs match the 4-topic structure expected by the aggregator

### What SAM3 Replaces

| Production Filter | SAM3 Replacement | Text Prompts |
|-------------------|------------------|--------------|
| `filter-bowl-detector` | `sam3_detector` (single) or `sam3_bowl_detector` (multi) | bowl |
| `filter-chit-detector` | `sam3_detector` (single) or `sam3_chit_detector` (multi) | chit |
| `filter_protege_bowl` | `sam3_detector` (single) or `sam3_ingredient_classifier` (multi) | avocado, chicken, blackened_chicken, steak, egg, tofu, portobello |
| `filter_protege_dressing_cups` | `sam3_detector` (single) or `sam3_dressing_detector` (multi) | dressing_cup |
| `filter_crop` | N/A | (SAM3 detects on full frame, no crop needed) |
| `filter_sweetgreen_ocr` | N/A | (OCR not replaceable by detection model) |

## Output Format

The JSONL output follows the event-sink format compatible with downstream systems:

```json
{
  "filter_name": "SAM3Detector",
  "topic": "main",
  "data": {
    "id": 0,
    "meta": {
      "detections": [
        {
          "id": 1,
          "class": "bowl",
          "score": 0.49,
          "box": [537, 852, 1491, 2017]
        },
        {
          "id": 2,
          "class": "chicken",
          "score": 0.68,
          "box": [642, 1178, 739, 1259]
        }
      ],
      "classification": {
        "classes": ["chicken", "bowl"],
        "confidences": [0.68, 0.49],
        "architecture": "sam3"
      },
      "frame_id": 0
    }
  }
}
```

### Key Properties

- **Frame IDs**: Extracted from the `_filter` topic (TI-130) emitted by VideoIn. Each frame has a unique, incrementing ID starting from 0.
- **Detection IDs**: Globally unique across all frames. If frame 0 has IDs 1-39, frame 1 starts at 40.
- **Class names**: SAM3 outputs exact prompt text (e.g., "chicken", "bowl"). The aggregator aliases these to canonical names (e.g., "roasted_chicken").

## Verifying Output

### Check Frame ID Uniqueness

```bash
# Each frame ID should appear exactly once
cat output/detections.jsonl | jq -r '.data.id' | sort -n | uniq -c | head -20
```

### Check Detection ID Uniqueness

```bash
# Should return nothing if all IDs are unique
cat output/detections.jsonl | jq -r '.data.meta.detections[].id' | sort -n | uniq -c | awk '$1 > 1'
```

### Summary Statistics

```bash
echo "Total frames: $(wc -l < output/detections.jsonl)"
echo "Total detections: $(cat output/detections.jsonl | jq -r '.data.meta.detections | length' | awk '{s+=$1}END{print s}')"
echo "Max detection ID: $(cat output/detections.jsonl | jq -r '.data.meta.detections[].id' | sort -n | tail -1)"
```

## Production Endpoints

| Location | Endpoint |
|----------|----------|
| 11th & Pine | `https://api.prod.plainsight.tech/filter-pipelines/sweetgreen-11th-pine/events?project=4a5de1d9-aef0-44a4-9cfe-071528b11d88` |
| La Brea | `https://api.prod.plainsight.tech/filter-pipelines/sweetgreen-labrea/events?project=4a5de1d9-aef0-44a4-9cfe-071528b11d88` |
| Totem Lake | `https://api.prod.plainsight.tech/filter-pipelines/sweetgreen-totem-lake/events?project=4a5de1d9-aef0-44a4-9cfe-071528b11d88` |

## Comparison with Production Topology

The production topology (`external-sweetgreen/scripts/docker-compose.prod.yaml`) uses:

```yaml
FILTER_SOURCES: '["tcp://filter-bowl-detector:5620;main",
                  "tcp://filter_sweetgreen_ocr:6000;chit",
                  "tcp://filter_protege_bowl:6002;bowl",
                  "tcp://filter_protege_dressing_cups:6008;bowl>dressing_cups"]'
```

### Single-GPU SAM3 (sweetgreen.yaml)

A single SAM3 instance provides all topics:

```yaml
FILTER_SOURCES: '["tcp://sam3_detector:5551;main",
                  "tcp://sam3_detector:5551;main>bowl",
                  "tcp://sam3_detector:5551;main>dressing_cups"]'
```

### Multi-GPU SAM3 (sweetgreen-multi-gpu.yaml)

Dedicated SAM3 instances match production port structure:

```yaml
FILTER_SOURCES: '["tcp://sam3_bowl_detector:5620;main",
                  "tcp://sam3_chit_detector:5720;chit",
                  "tcp://sam3_ingredient_classifier:6002;bowl",
                  "tcp://sam3_dressing_detector:6008;bowl>dressing_cups"]'
```

Both configurations maintain the same topic routing (`main`, `bowl`, `dressing_cups`) that the aggregator expects.

## Troubleshooting

### Docker Registry Authentication Failed

```bash
# Re-authenticate with GAR
gcloud auth configure-docker us-west1-docker.pkg.dev

# For OpenFilter container registry (event sink image)
# This uses the containers.openfilter.io registry which is public
```

### GPU Not Available

```bash
# Check NVIDIA Docker runtime is installed
docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu22.04 nvidia-smi

# If this fails, install nvidia-container-toolkit:
# https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html
```

### Model Download Issues

```bash
# Verify HuggingFace token is accessible
gcloud secrets versions access latest --secret=sam3-hf-token --project=plainsightai-dev

# Check token validity
curl -H "Authorization: Bearer $HF_TOKEN" https://huggingface.co/api/whoami
```

### Aggregator Not Receiving Data

```bash
# Check topic routing in aggregator logs
docker compose -f sweetgreen.yaml logs filter_sweet_green_subject_data_aggregator | grep "incoming bundle keys"
```

Expected output:
```
[AGG] incoming bundle keys: ['main', 'bowl', 'dressing_cups']
[AGG] matched_keys=['bowl', 'dressing_cups']
```

### Event Sink Connection Issues

```bash
# Test API endpoint connectivity
curl -v -X POST "$FILTER_API_ENDPOINT" \
  -H "Authorization: Bearer $FILTER_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '[]'

# Check event sink logs
docker compose -f sweetgreen.yaml --profile event-sink logs filter_event_sink
```

### Video File Issues

```bash
# Check video file exists and is readable
ls -la data/sample-video.mp4

# Verify video metadata
ffprobe data/sample-video.mp4

# Check frame count
ffprobe -v error -select_streams v:0 -count_packets \
  -show_entries stream=nb_read_packets -of csv=p=0 data/sample-video.mp4
```
