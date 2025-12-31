# Examples

This directory contains example scripts and notebooks demonstrating various use cases for the SAM3 detector filter.

## Scripts

| Script | Description |
|--------|-------------|
| `detect_objects_video.py` | Basic video object detection pipeline with JSONL export |
| `run_detection.py` | Single video detection with text prompts |
| `run_detection_batch.py` | Batch processing of multiple videos |
| `run_detection_images.py` | Image-based detection (single object type) |
| `run_detection_images_multi_objects.py` | Multi-object detection on images |
| `run_detection_multi_objects.py` | Multi-object video detection |
| `run_detection_with_order_ui.py` | Detection with custom UI overlay |
| `jsonl_to_coco.py` | Convert JSONL detections to COCO format |

## Notebooks

The `notebooks/` directory contains Jupyter notebooks with interactive examples:

| Notebook | Description |
|----------|-------------|
| `detect_image.ipynb` | Step-by-step image detection tutorial |
| `demo_salad_detection.ipynb` | Food item detection example |
| `demo_bowl_detection.ipynb` | Container detection example |
| `demo_order_ui.ipynb` | Custom UI overlay example |

## Quick Start

### Basic Video Detection

```bash
python detect_objects_video.py \
    --video input.mp4 \
    --prompt "person" \
    --output-dir ./results \
    --confidence 0.5
```

### Batch Processing

```bash
python run_detection_batch.py \
    --videos videos/*.mp4 \
    --prompt "car" \
    --output-dir ./batch_results
```

### Image Detection

```bash
python run_detection_images.py \
    -i images/ \
    -o "person" \
    --confidence 0.3
```

## Creating Your Own Pipelines

See the main [README.md](../README.md) for documentation on creating custom docker-compose pipelines with the SAM3 detector filter.
