# filter-sam3-detector

[![OpenFilter](https://img.shields.io/badge/OpenFilter-v0.1.0-blue)](https://openfilter.io)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)

OpenFilter implementation for SAM3 (Segment Anything Model 3) open-set object detection.

## Overview

This filter performs zero-shot and few-shot object detection using SAM3. It supports two prompting modes:

- **Text prompts**: Natural language descriptions (e.g., "person", "car", "red apple")
- **Image exemplars**: Few-shot learning with cropped example images

## Quick Start

### Installation

```bash
pip install filter-sam3-detector \
    --index-url https://python.openfilter.io/simple \
    --extra-index-url https://pypi.org/simple
```

### Running with Docker Compose

Create a `docker-compose.yaml`:

```yaml
services:
  video_in:
    image: containers.openfilter.io/plainsightai/openfilter-video-in:v0.1.10
    environment:
      FILTER_SOURCES: file:///video.mp4!sync!loop
      FILTER_OUTPUTS: tcp://*
    volumes:
      - ./my_video.mp4:/video.mp4:ro

  sam3_detector:
    image: containers.openfilter.io/plainsightai/filter-sam3-detector:v0.1.0
    environment:
      FILTER_SOURCES: tcp://video_in
      FILTER_OUTPUTS: tcp://*
      FILTER_TEXT_PROMPT: "person"
      FILTER_CONFIDENCE_THRESHOLD: "0.7"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  webvis:
    image: containers.openfilter.io/plainsightai/openfilter-webvis:v0.1.10
    environment:
      FILTER_SOURCES: tcp://sam3_detector
    ports:
      - 8001:8000
```

Run with:

```bash
docker compose up
# View results at http://localhost:8001
```

### Running with OpenFilter CLI

```bash
openfilter run \
  - VideoIn --sources 'file://video.mp4!loop' \
  - filter_sam3_detector.filter.FilterSAM3Detector --text_prompt 'person' \
  - Webvis
```

## Configuration

All parameters can be set via environment variables with the `FILTER_` prefix:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text_prompt` | str | None | Text prompt for detection (e.g., "person", "car") |
| `exemplars_path` | str | None | Path to directory with exemplar images |
| `model_id` | str | `facebook/sam2-hiera-large` | HuggingFace model ID |
| `device` | str | `cuda` | Device: "cuda", "cpu", or "mps" |
| `confidence_threshold` | float | 0.5 | Minimum confidence for detections |
| `mask_threshold` | float | 0.5 | Threshold for mask binarization |
| `max_detections` | int | 100 | Maximum detections per frame |
| `output_masks` | bool | true | Include segmentation masks in output |
| `output_boxes` | bool | true | Include bounding boxes in output |
| `output_scores` | bool | true | Include confidence scores in output |
| `output_label` | str | `sam3_detections` | Key in frame.data['meta'] |
| `visualize` | bool | false | Draw detections on output frames |
| `debug` | bool | false | Enable debug logging |

### Environment Variable Examples

```bash
export FILTER_TEXT_PROMPT="red apple"
export FILTER_CONFIDENCE_THRESHOLD=0.7
export FILTER_DEVICE=cuda
export FILTER_VISUALIZE=true
```

## Prompting Modes

### Text Prompts

Use natural language to describe what to detect:

```yaml
environment:
  FILTER_TEXT_PROMPT: "person wearing a hat"
```

### Image Exemplars

Provide a directory of cropped images showing examples of what to detect:

```yaml
environment:
  FILTER_EXEMPLARS_PATH: /exemplars
volumes:
  - ./my_exemplars:/exemplars:ro
```

The exemplar directory should contain cropped images (JPG/PNG) of the objects you want to detect. SAM3 will encode these and use them for few-shot detection.

## Output Format

Detections are stored in `frame.data['meta'][output_label]`:

```python
[
    {
        "box": [x1, y1, x2, y2],  # Bounding box coordinates
        "score": 0.95,            # Confidence score
        "mask": [[...]]           # Binary mask (if output_masks=true)
    },
    ...
]
```

## Development

```bash
# Clone and install
git clone <repository-url>
cd filter-sam3-detector
make install

# Run tests
make test

# Build Docker image
make docker-build

# Run with docker-compose
TEXT_PROMPT="person" make docker-run
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines.

## Publishing

To publish a new version to the OpenFilter registry:

```bash
# Update VERSION file
echo "0.2.0" > VERSION

# Build and publish
make publish
```

## License

Apache-2.0 - See [LICENSE](LICENSE) for details.
