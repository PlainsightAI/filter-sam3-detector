# filter-sam3-detector

[![License: Apache 2.0 + SAM](https://img.shields.io/badge/License-Apache%202.0%20%2B%20SAM-blue.svg)](LICENSING.md)

OpenFilter implementation for SAM3 (Segment Anything Model 3) object detection with open-set capabilities.

This filter is a part of Plainsight's **HyperLabel™**, which enables accelerated/automated annotation, model retraining and deployment promotion.

## Features

- **Open-Set Detection**: Detect objects not in standard training datasets
- **Dual Prompting Modes**: Text prompts or exemplar images (few-shot learning)
- **Reference Box Prompts**: Positive/negative bounding boxes on the original image (SAM3-style geometric prompts; optional text prompt)
- **Flexible Output**: Bounding boxes, segmentation masks, and confidence scores
- **GPU Acceleration**: CUDA, CPU, and MPS (Apple Silicon) support
- **Real-time Processing**: Processes video streams in real-time
- **Pipeline Integration**: Works seamlessly with OpenFilter pipeline architecture
- **Environment Configuration**: Full configuration through environment variables
- **Performance Optimized**: Configurable detection limits, resolution control
- **Fault Tolerant**: Handles errors gracefully, forwards frames on failure
- **Cost Efficient**: Local inference, no API costs

## Architecture

The filter follows the OpenFilter pattern with three main stages:

### Stage Responsibilities

| Stage | Responsibility |
|-------|----------------|
| `setup()` | Load SAM3 model from HuggingFace; load and process exemplar images; initialize device (CUDA/CPU/MPS) |
| `process()` | Core operation: run SAM3 inference on frames; extract detections; attach results to frame metadata |
| `shutdown()` | Clean up resources (release model, clear GPU memory) when filter stops |

### Data Signature

The filter returns processed frames with the following data structure:

**Frame Metadata:**
- Original frame data preserved
- Detection results added to `frame.data['meta'][output_label]`:
  ```python
  [
    {
      "box": [x1, y1, x2, y2],  # Bounding box coordinates
      "score": 0.95,            # Confidence score (0.0-1.0)
      "mask": [[...]]           # Binary mask as 2D array (optional)
    },
    ...
  ]
  ```

## Installation

See [INSTALL.md](INSTALL.md) for detailed installation instructions.

**Quick install:**
```bash
# Clone repository
git clone <repository-url>
cd filter-sam3-detector

# Install package
uv pip install -e .

# Or with development dependencies
uv pip install -e ".[dev]"
```

## Get Started

For a first run with Docker Compose, including examples for:

- `FILTER_TEXT_PROMPT`
- `FILTER_TEXT_PROMPTS`
- positive reference boxes and reference images

use [`QUICKSTART.md`](QUICKSTART.md).

The quick start uses detached compose commands:

```bash
docker compose -f docker-compose.yaml up -d
```

## Configuration

1. Copy the example environment file:
```bash
cp .env.example .env
```

2. Edit `.env` file with your configuration:
```bash
# Prompt configuration (choose one)
FILTER_TEXT_PROMPT=person                    # Text prompt for detection
FILTER_EXEMPLARS_PATH=./exemplars/           # Path to exemplar images directory
# FILTER_POSITIVE_BOXES='[[x,y,w,h],...]'    # Reference boxes (positive), JSON array of [x,y,w,h] in pixels
# FILTER_NEGATIVE_BOXES='[[x,y,w,h],...]'    # Reference boxes (negative), JSON array of [x,y,w,h] in pixels

# Model configuration
FILTER_MODEL_ID=facebook/sam3                # HuggingFace model ID
FILTER_DEVICE=cuda                           # Device: cuda, cpu, or mps

# Detection parameters
FILTER_CONFIDENCE_THRESHOLD=0.5              # Minimum confidence (0.0-1.0)
FILTER_MASK_THRESHOLD=0.5                    # Mask binarization threshold
FILTER_MAX_DETECTIONS=100                    # Maximum detections per frame

# Output configuration
FILTER_OUTPUT_MASKS=true                     # Output segmentation masks
FILTER_OUTPUT_BOXES=true                     # Output bounding boxes
FILTER_OUTPUT_SCORES=true                    # Output confidence scores
FILTER_OUTPUT_LABEL=sam3_detections          # Key in frame.data['meta']

# Visualization and debugging
FILTER_VISUALIZE=false                       # Draw detections on frames
# FILTER_VIZ_TOPIC=viz                       # When set: main=original+meta, this topic=drawn frame+meta
FILTER_DEBUG=false                           # Enable debug logging
```

### Configuration Matrix

| Variable | Type | Default | Required | Notes |
|----------|------|---------|----------|-------|
| `text_prompt` | string | None | No* | Natural language description (e.g., "person", "car") |
| `exemplars_path` | string | None | No* | Path to directory with exemplar images |
| `model_id` | string | "facebook/sam3" | No | HuggingFace model ID or local path |
| `device` | string | "cuda" | No | Device: "cuda", "cpu", or "mps" |
| `confidence_threshold` | float | 0.5 | No | Minimum confidence (0.0-1.0) |
| `mask_threshold` | float | 0.5 | No | Mask binarization threshold (0.0-1.0) |
| `max_detections` | int | 100 | No | Maximum detections per frame |
| `output_masks` | bool | true | No | Output segmentation masks |
| `output_boxes` | bool | true | No | Output bounding boxes |
| `output_scores` | bool | true | No | Output confidence scores |
| `output_label` | string | "sam3_detections" | No | Key for storing results |
| `visualize` | bool | false | No | Draw detections on output frames |
| `viz_topic` | string | "" | No | When set (e.g. `viz`), main gets original frame + meta; this topic gets drawn frame + meta. Empty = legacy (visualize draws on main). |
| `ref_images` | string | None | No | Comma-separated paths for positive ref images (pasted on composite). Ignored when `positive_boxes` or `negative_boxes` are set. |
| `ref_images_negative` | string | None | No | Comma-separated paths for negative ref images. Ignored when ref boxes are set. |
| `composite_topic` | string | "" | No | When set (e.g. `composite`), publish the composite image (frame + refs) on this topic when REF_IMGS are in use. |
| `debug` | bool | false | No | Enable debug logging |

\* When using `positive_boxes` or `negative_boxes`, a text prompt is optional (the model can use the placeholder "visual"). Otherwise either `text_prompt` or `exemplars_path` must be provided. When using REF_IMGS (ref images), a text prompt is required; REF_IMGS are disabled when ref boxes are set.

### Reference box prompts

In single-output mode you can add reference bounding boxes on the **original image** (no composite): set `FILTER_POSITIVE_BOXES` and/or `FILTER_NEGATIVE_BOXES` to a **JSON array** of boxes, each box `[x, y, width, height]` in pixels. Positive boxes encourage detections similar to those regions; negative boxes suppress them. Example in `.env`:

```bash
FILTER_POSITIVE_BOXES="[[480, 290, 110, 360], [370, 280, 115, 375]]"
FILTER_NEGATIVE_BOXES="[[100, 100, 50, 200]]"
```

Text prompt is optional when using reference boxes. With `FILTER_VISUALIZE=true`, positive ref boxes are drawn in green, negative in red, and detections in blue.

**Rule: when `FILTER_POSITIVE_BOXES` or `FILTER_NEGATIVE_BOXES` are set, reference images (REF_IMGS) are not used** — only the reference-boxes mode on the original image is applied. Set REF_IMGS only when you are not using ref boxes.

### Reference images (REF_IMGS)

You can pass **reference images** (positive and/or negative) that are pasted on a composite (frame + refs) for visual prompting. Set `FILTER_REF_IMAGES` and/or `FILTER_REF_IMAGES_NEGATIVE` to comma-separated paths (files or directories; directories are expanded to image files). A text prompt is required when using REF_IMGS. To view the composite image in the pipeline, set `FILTER_COMPOSITE_TOPIC=composite` and ensure the filter outputs include the composite topic (e.g. in Webvis you can open `/composite`).

## Usage

### Method 1: Using Example Scripts (Recommended)

Scripts read configuration from environment variables (e.g. from a `.env` file). Copy `env.example` to `.env` and set at least `VIDEO_PATH` and `FILTER_TEXT_PROMPT`.

#### Object Detection with Text Prompts

Scripts read configuration from environment variables (use a `.env` file or pass them inline):

```bash
# Set in .env: VIDEO_PATH, FILTER_TEXT_PROMPT, FILTER_OUTPUT_DIR, etc.
python scripts/filter_object_detection.py
```

Or pass variables inline:

```bash
# Detect people in a video
VIDEO_PATH=input.mp4 FILTER_TEXT_PROMPT=person FILTER_OUTPUT_DIR=./results \
  FILTER_CONFIDENCE_THRESHOLD=0.5 python scripts/filter_object_detection.py

# Detect cars with visualization
VIDEO_PATH=traffic.mp4 FILTER_TEXT_PROMPT=car FILTER_CONFIDENCE_THRESHOLD=0.6 \
  FILTER_VISUALIZE=true FILTER_OUTPUT_DIR=./cars python scripts/filter_object_detection.py

# Process multiple videos (run once per video)
VIDEO_PATH=video1.mp4 FILTER_TEXT_PROMPT=dog FILTER_OUTPUT_DIR=./detections \
  python scripts/filter_object_detection.py
# Then VIDEO_PATH=video2.mp4 ... and VIDEO_PATH=video3.mp4 ...
```

Optional: `FILTER_VIDEO_LOOP=true` keeps the video looping so frames are still available after the model loads (~14s); useful for short videos.

#### Detection with Reference Boxes

Use positive and/or negative reference bounding boxes on the frame (SAM3-style geometric prompts) with or without a text prompt:

```bash
# In .env set: VIDEO_PATH, and FILTER_POSITIVE_BOXES and/or FILTER_NEGATIVE_BOXES (JSON arrays of [x,y,w,h])
# Optional: FILTER_TEXT_PROMPT for text-guided detection
python scripts/filter_object_detection_exemplar.py
```

**Reference boxes:** Set `FILTER_POSITIVE_BOXES` and/or `FILTER_NEGATIVE_BOXES` to a JSON array of boxes, each `[x, y, width, height]` in pixels. Example: `FILTER_POSITIVE_BOXES="[[480, 290, 110, 360]]"`. Text prompt is optional. With `FILTER_VISUALIZE=true`, ref boxes are drawn in green (positive) and red (negative), detections in blue.

### Method 2: Docker Pipeline

Run the complete detection pipeline with Docker Compose. The prebuilt image is published to Docker Hub at `plainsightai/openfilter-sam3-detector` and is publicly pullable — no auth required.

```bash
# 1. Point VIDEO_PATH at your video (compose defaults to the bundled ./data/car.mp4)
export VIDEO_PATH=$(pwd)/your_video.mp4

# 2. Compose pulls on first run, so this step is optional. Model weights are
#    baked in, so no HF_TOKEN at runtime. SAM3_DETECTOR_VERSION pins the image
#    and compose defaults it to 0.1.29, so pull that rather than `latest`, which
#    the run below does not use.
docker pull plainsightai/openfilter-sam3-detector:0.1.29

# 3. Run the pipeline
FILTER_TEXT_PROMPT="person" docker compose up

# 4. View results at http://localhost:8002 (webvis)
# Detection output lands under ./results. Temporal intervals are off: turning them
# on takes FILTER_ENABLE_TEMPORAL_INTERVALS=true, and persisting them takes two
# more keys in .env. See "Persisting intervals" below.
```

<details>
<summary>Build from source instead</summary>

```bash
# Required to download the gated SAM3 weights at build time.
export HF_TOKEN="your_huggingface_token"

# HF_TOKEN is passed as a BuildKit secret, not an env var or build arg,
# so it never ends up in an image layer. Compose does not forward the token,
# so invoke `docker build` directly and tag it to match docker-compose.yaml.
docker build --secret id=hf_token,env=HF_TOKEN \
  -t plainsightai/openfilter-sam3-detector:latest .
```

</details>

**Pipeline Architecture:**
```
video_in → sam3_detector (with integrated temporal intervals) → webvis
               ↓
         intervals in frame metadata (not persisted, see below)
```

**Requirements:**
- Docker with NVIDIA Container Toolkit, and the Compose plugin at **2.24 or newer**:
  `docker-compose.yaml` uses the `env_file` `path` / `required` mapping, which older
  Compose cannot parse, so every command here fails outright rather than degrading.
  Check with `docker compose version`.
- CUDA-compatible GPU (sm_50+ including RTX 50-series/Blackwell)
- HuggingFace account with access to gated models

**Environment Variables:**
| Variable | Default | Description |
|----------|---------|-------------|
| `HF_TOKEN` | - | HuggingFace token for model access (build-time) |
| `FILTER_TEXT_PROMPT` | `car` from compose; the filter's own default is unset | What to detect |
| `FILTER_TEMPORAL_HALF_LIFE` | unset (`None`) | EMA decay rate (frames) |
| `FILTER_TEMPORAL_PRESENCE_THRESHOLD` | `0.5` | EMA threshold for presence detection |

> Note: SAM3 weights are baked into the image at build time, and the container runs with `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1`. No network or `HF_TOKEN` is needed at runtime — the image is safe to run with `--network=none`.

**Interval shape.** This is what the tracker builds in memory. Nothing writes it
to disk on this route: `temporal_intervals.py:322` opens an output file only when
`temporal_streaming_mode` is set as well as `temporal_output_json_path`, and
`finalize()` closes the streaming handle without a non-streaming dump. Both are
reachable through `.env`, which compose still loads (it is optional, not
inert): set `FILTER_TEMPORAL_OUTPUT_JSON_PATH` and `FILTER_TEMPORAL_STREAMING_MODE`
alongside `FILTER_ENABLE_TEMPORAL_INTERVALS`. Neither is declared in the
`environment:` block, so the shell alone will not carry them.

**Persisting intervals.** The path has to land inside the mounted volume, which is
`/output` in the container (`docker-compose.yaml:84` maps `${FILTER_OUTPUT_DIR:-./results}`
onto it). A relative path, or anything outside `/output`, is created inside the
container by `IntervalTracker` and disappears with it, with no error to say so:

```bash
FILTER_ENABLE_TEMPORAL_INTERVALS=true
FILTER_TEMPORAL_STREAMING_MODE=true
FILTER_TEMPORAL_OUTPUT_JSON_PATH=/output/intervals.json
```
The file is **ndjson**: one interval per line, flushed as each interval closes
(`temporal_intervals.py:415-422`). There is no wrapper object and no
`total_frames`; `to_dict` (`:51-59`) emits exactly these five keys.

```
{"start_frame": 23, "end_frame": 69, "label": "person", "present": true, "confidence": 0.95}
{"start_frame": 88, "end_frame": 140, "label": "person", "present": true, "confidence": 0.91}
```

### Method 3: Using as a Standalone Filter

```bash
# Set environment variables
export FILTER_TEXT_PROMPT="person"
export FILTER_CONFIDENCE_THRESHOLD=0.7
export FILTER_DEVICE=cuda
export FILTER_SOURCES="tcp://127.0.0.1:5555"
export FILTER_OUTPUTS="tcp://127.0.0.1:5556"

# Run the filter
filter-sam3-detector
```

### Method 4: Using in Python Code

```python
from filter_sam3_detector import FilterSAM3Detector
from openfilter.filter_runtime.filter import Filter
from openfilter.filter_runtime.filters.video_in import VideoIn
from openfilter.filter_runtime.filters.recorder import Recorder

# Define pipeline
filters = [
    (VideoIn, {
        "sources": "file://input.mp4",
        "outputs": ["tcp://127.0.0.1:5555"],
    }),
    (FilterSAM3Detector, {
        "sources": "tcp://127.0.0.1:5555",
        "outputs": ["tcp://127.0.0.1:5556"],
        "text_prompt": "person",
        "confidence_threshold": 0.5,
        "device": "cuda",
    }),
    (Recorder, {
        "sources": "tcp://127.0.0.1:5556",
        "path": "detections.jsonl",
        "format": "jsonl",
    }),
]

# Run pipeline
Filter.run_multi(filters)
```

## Temporal Interval Detection

Convert noisy per-frame detections into stable presence/absence intervals using EMA smoothing.

### Quick Start (Docker - Recommended)

```bash
# Run the integrated pipeline with temporal intervals turned on. They are off by
# default: FILTER_ENABLE_TEMPORAL_INTERVALS is what switches them.
# SAM3_DETECTOR_VERSION pins the image; compose defaults it to 0.1.29, not `latest`.
# Point VIDEO_PATH at your file (compose defaults to the bundled ./data/car.mp4)
export VIDEO_PATH=$(pwd)/your_video.mp4

FILTER_ENABLE_TEMPORAL_INTERVALS=true FILTER_TEXT_PROMPT="person" docker compose up

# Output lands under ./results, which is the volume compose mounts
```

### Quick Start (Python Script)

```bash
# Run on any video with custom prompts
uv run python scripts/run_temporal_intervals.py video.mp4 \
    --prompts "person,hand,cup" \
    --output results.json
```

### Integrated Mode (Recommended)

Enable temporal intervals directly in the SAM3 detector - no separate filter needed:

```python
from filter_sam3_detector import FilterSAM3Detector

# Single filter with integrated temporal tracking
pipeline = [
    (FilterSAM3Detector, {
        "text_prompt": "person",
        "output_label": "detections",
        # Integrated temporal intervals
        "enable_temporal_intervals": True,
        "temporal_streaming_mode": True,  # Emit incrementally
        "temporal_half_life": 5.0,
        "temporal_presence_threshold": 0.4,
        "temporal_output_json_path": "intervals.json",
    }),
]
```

### Separate Filter Mode (Legacy)

For pipelines requiring separate filter stages:

```python
from filter_sam3_detector import FilterSAM3Detector
from filter_sam3_detector.temporal_intervals import TemporalIntervalFilter

# SAM3 detector -> Temporal interval filter
pipeline = [
    (FilterSAM3Detector, {
        "text_prompt": "person",
        "output_label": "detections",
    }),
    (TemporalIntervalFilter, {
        "detection_key": "detections",
        "half_life": 5.0,           # EMA responsiveness (frames)
        "presence_threshold": 0.4,  # Detection threshold
        "output_json_path": "intervals.json",
    }),
]
```

### Output Format

ndjson, one interval per line as each closes. No wrapper object and no
`total_frames`: `to_dict` (`temporal_intervals.py:51-59`) emits exactly these
five keys.

```
{"start_frame": 20, "end_frame": 150, "label": "person", "present": true, "confidence": 0.92}
{"start_frame": 151, "end_frame": 180, "label": "person", "present": false, "confidence": 0.15}
```

### Configuration Options

| Option | Default | Description |
|--------|---------|-------------|
| `enable_temporal_intervals` | false | Enable integrated temporal tracking |
| `temporal_streaming_mode` | false | Emit intervals incrementally (vs. at end) |
| `temporal_half_life` | 5.0 | Frames for 50% EMA decay |
| `temporal_presence_threshold` | 0.4 | EMA score to trigger presence |
| `temporal_output_json_path` | None | Path to write intervals JSON |
| `temporal_emit_on_change` | true | Only emit when state changes |

## Usage Scenarios

### 1. Person Detection

Set in `.env`: `VIDEO_PATH`, `FILTER_TEXT_PROMPT=person`, `FILTER_OUTPUT_DIR`, `FILTER_CONFIDENCE_THRESHOLD=0.6`. Then:

```bash
python scripts/filter_object_detection.py
```

### 2. Vehicle Detection

Set `VIDEO_PATH`, `FILTER_TEXT_PROMPT=car`, `FILTER_OUTPUT_DIR`, `FILTER_RESIZE=480`. Then run `python scripts/filter_object_detection.py`.

### 3. Detection with Reference Boxes

Use bounding boxes on the frame as positive/negative prompts (with or without text):

```bash
# In .env: VIDEO_PATH, FILTER_POSITIVE_BOXES='[[x,y,w,h],...]', FILTER_NEGATIVE_BOXES (optional), FILTER_TEXT_PROMPT (optional)
python scripts/filter_object_detection_exemplar.py
```

### 4. Pipeline Integration

Combine with other OpenFilter filters:

```python
from openfilter.filter_runtime.filter import Filter
from openfilter.filter_runtime.filters.video_in import VideoIn
from openfilter.filter_runtime.filters.resize import Resize
from openfilter.filter_runtime.filters.recorder import Recorder
from filter_sam3_detector import FilterSAM3Detector

filters = [
    (VideoIn, {"sources": "file://input.mp4"}),
    (Resize, {"width": 640, "height": 480}),  # Pre-processing
    (FilterSAM3Detector, {"text_prompt": "person"}),
    (Recorder, {"path": "output.jsonl"}),
]

Filter.run_multi(filters)
```

## Output Format

Detections are stored in `frame.data['meta'][output_label]`:

```python
[
  {
    "box": [x1, y1, x2, y2],  # Bounding box coordinates
    "score": 0.95,            # Confidence score (0.0-1.0)
    "mask": [[...]]           # Binary mask as 2D array (if output_masks=True)
  },
  ...
]
```

When using the Recorder filter, detections are saved in JSONL format:

```json
{
  "frame_id": 0,
  "meta": {
    "sam3_detections": [
      {
        "box": [100, 150, 200, 250],
        "score": 0.95,
        "mask": [[0, 0, 1, 1, ...]]
      }
    ]
  }
}
```

## Performance Tips

### Image Processing
- **Resize Videos**: Use `--resize 480` for faster processing
- **Limit Detections**: Reduce `FILTER_MAX_DETECTIONS` for better performance
- **Disable Masks**: Set `FILTER_OUTPUT_MASKS=false` to save memory

### Device Selection
- **Use GPU**: Set `FILTER_DEVICE=cuda` for 10-50x speedup
- **CPU Fallback**: Automatically falls back to CPU if GPU unavailable
- **Apple Silicon**: Use `FILTER_DEVICE=mps` on macOS

### Confidence Thresholds
- **Text Prompts**: Default `0.5` works well
- **Exemplar-Based**: Use `0.3` for better recall
- **High Precision**: Use `0.7` or higher
- **High Recall**: Use `0.3` or lower

## Development

### Project Structure

```
filter-sam3-detector/
├── filter_sam3_detector/
│   ├── __init__.py
│   └── filter.py              # Main filter implementation
├── scripts/                   # Example usage scripts
│   ├── filter_object_detection.py       # Video pipeline (text prompt)
│   ├── filter_object_detection_exemplar.py  # Video pipeline (reference boxes + optional text)
│   └── run_temporal_intervals.py
├── examples/                  # Additional examples
│   └── detect_objects_video.py
├── docs/                      # Documentation
│   ├── API.md
│   ├── configuration.md
│   ├── advanced-usage.md
│   └── performance.md
├── tests/                     # Test files
│   ├── test_filter.py
│   └── test_integration.py
├── sam3/                      # Vendorized SAM3 library
├── env.example               # Environment configuration example
└── pyproject.toml           # Project dependencies
```

### Key Dependencies

- `openfilter[all]>=0.1.0` - Filter framework
- `torch>=2.0.0` - PyTorch for model inference
- `torchvision>=0.15.0` - Image processing
- `transformers>=4.40.0` - HuggingFace model loading
- `opencv-python>=4.8.0` - Image manipulation
- `pillow>=10.0.0` - Image processing
- `numpy>=1.24.0` - Numerical operations

### Testing

```bash
# Run tests
make test

# Run tests with coverage (pass extra pytest args via PYTEST_ARGS)
make test PYTEST_ARGS="--cov=filter_sam3_detector --cov-report=term"

# Check code quality
make lint

# Format code
make format
```

## Known Issues

### Exemplar-Based Detection Not Working

**Status**: Bug in `_load_exemplar_images()` - backbone output format handling is incorrect.

**Symptoms**: When using `exemplars_path`, you may see warnings like:
```
WARNING  Failed to load exemplar example.jpg: 'NoneType' object is not subscriptable
ERROR    No exemplar images could be loaded
```

**Root Cause**: The code at `filter.py:853-858` doesn't properly handle the SAM3 backbone output format. The backbone returns features in a different structure than expected.

**Workaround**: Use text prompts (`text_prompt`) instead of exemplar images until this is fixed.

**Tracking**: This issue affects the few-shot learning functionality. Text-based detection works correctly.

## Troubleshooting

### Model Loading Issues

**Problem**: Model fails to load or takes too long

**Solutions**:
- Ensure you have sufficient GPU memory (recommended: 8GB+)
- Use CPU mode if GPU is unavailable: `--device cpu`
- Check internet connection (model downloads from HuggingFace on first use)
- Verify CUDA installation: `nvidia-smi`

### No Detections Found

**Problem**: Filter runs but finds no objects

**Solutions**:
- Lower confidence threshold: `--confidence 0.3`
- Try different text prompts (be more specific or more general)
- For exemplar-based: ensure exemplar images are clear and representative
- Check that input video has the objects you're looking for

### Out of Memory Errors

**Problem**: CUDA out of memory errors

**Solutions**:
- Resize input: `--resize 480`
- Reduce max detections: `export FILTER_MAX_DETECTIONS=50`
- Disable masks: `export FILTER_OUTPUT_MASKS=false`
- Use CPU mode: `--device cpu` (slower but uses less memory)

### Import Errors

**Problem**: `ImportError: cannot import name 'FilterSAM3Detector'`

**Solutions**:
- Ensure package is installed: `uv pip install -e .`
- Check Python version (requires 3.10+)
- Verify all dependencies are installed
- Reinstall: `uv pip install -e . --force-reinstall`

### Slow Processing

**Problem**: Processing is very slow

**Solutions**:
- Use GPU: `--device cuda`
- Resize videos: `--resize 480`
- Reduce max detections
- Disable masks if not needed
- Process fewer frames (use sample rate in video input)

### Performance Optimization

To improve processing speed:
1. Use GPU acceleration (`FILTER_DEVICE=cuda`)
2. Resize inputs to appropriate resolution (`--resize 480`)
3. Limit detections (`FILTER_MAX_DETECTIONS=50`)
4. Disable unused outputs (masks if not needed)
5. Use smaller model variant (if available)

## Documentation

For more detailed information, configuration examples, and advanced usage scenarios, see the comprehensive documentation:

- [Installation Guide](INSTALL.md) - Detailed installation instructions
- [Quick Start Guide](QUICKSTART.md) - Get started in minutes
- [API Reference](docs/API.md) - Complete API documentation
- [Configuration Guide](docs/configuration.md) - Configuration options
- [Advanced Usage](docs/advanced-usage.md) - Advanced patterns and examples
- [Performance Tuning](docs/performance.md) - Optimization guide
- [Scripts Documentation](scripts/README.md) - Example scripts usage

## License

This project uses dual licensing. The filter wrapper code is licensed under **Apache 2.0**, and the vendorized SAM3 library (`sam3/`) is licensed under the **SAM License**, which includes trade control restrictions. See [LICENSING.md](LICENSING.md) for full details.

## References

- [SAM3 Paper](https://arxiv.org/abs/2406.05663)
- [OpenFilter Documentation](https://openfilter.io)
- [SAM3 GitHub](https://github.com/facebookresearch/sam3)
