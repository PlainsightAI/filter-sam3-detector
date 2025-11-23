# filter_sam3_detector

OpenFilter implementation for SAM3 (Segment Anything Model 3) object detection with open-set capabilities.

## Overview

This filter performs object detection using SAM3 with two prompting modes:

- **Text prompts**: Natural language descriptions (e.g., "person", "car")
- **Image exemplars**: Few-shot learning with bounding box examples

## Installation

```bash
uv pip install -e .
```

## Usage

### As a module

```bash
uv run python -m filter_sam3_detector --help
```

### Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_id` | str | `"facebook/sam3"` | HuggingFace model ID or local path |
| `device` | str | `"cuda"` | Device: "cuda", "cpu", or "mps" |
| `text_prompt` | str | None | Text prompt for detection |
| `exemplars_path` | str | None | Path to JSON file with exemplar boxes |
| `confidence_threshold` | float | 0.5 | Minimum confidence for detections |
| `mask_threshold` | float | 0.5 | Threshold for mask binarization |
| `max_detections` | int | 100 | Maximum detections per frame |
| `output_masks` | bool | True | Output segmentation masks |
| `output_boxes` | bool | True | Output bounding boxes |
| `output_scores` | bool | True | Output confidence scores |
| `output_label` | str | `"sam3_detections"` | Key in frame.data['meta'] |
| `visualize` | bool | False | Draw detections on output frames |
| `debug` | bool | False | Enable debug logging |

### Environment Variables

All configuration parameters can be set via environment variables with the `FILTER_` prefix:

```bash
export FILTER_TEXT_PROMPT="person"
export FILTER_CONFIDENCE_THRESHOLD=0.7
export FILTER_DEVICE=cuda
```

### Exemplar Format

JSON file with bounding boxes:

```json
{
  "boxes": [[x1, y1, x2, y2], [x1, y1, x2, y2]],
  "labels": [1, 1]
}
```

Or simple format (all positive):

```json
[[x1, y1, x2, y2], [x1, y1, x2, y2]]
```

## Output

Detections are stored in `frame.data['meta'][output_label]`:

```python
[
  {
    "box": [x1, y1, x2, y2],
    "score": 0.95,
    "mask": [[...]]  # Binary mask as 2D array
  },
  ...
]
```

## Development

```bash
# Install with dev dependencies
uv pip install -e ".[dev]"

# Run tests
uv run pytest
```

## License

Apache-2.0
